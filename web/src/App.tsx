import { useCallback, useEffect, useRef, useState } from 'react';
import { HalEye, type HalState } from './HalEye';
import {
  decodeFramedPcm,
  HalAudioPlayer,
  type MicMode,
  type ServerMessage,
} from './halClient';

interface ChatEntry {
  id: number;
  speaker: 'YOU' | 'HAL' | 'SYSTEM' | 'ERROR';
  text: string;
  interrupted?: boolean;
}

type CaptureOwner = { kind: 'space' } | { kind: 'pointer'; pointerId: number } | { kind: 'toggle' };

interface BrowserCaptureSession {
  gen: number;
  stream: MediaStream | null;
  source: MediaStreamAudioSourceNode | null;
  processor: ScriptProcessorNode | null;
  sinkGain: GainNode | null;
  chunks: Float32Array[];
  samples: number;
  sampleRate: number;
}

const TAP_THRESHOLD_MS = 280;
const MIN_CAPTURE_DURATION_S = 0.2;
const MAX_CAPTURE_DURATION_S = 29.5;
const MAX_TEXT_CHARS = 4000;
const MAX_LOG_ENTRIES = 200;
const GRILLE_SLOTS = 7;

function float32ToBase64(samples: Float32Array): string {
  const bytes = new Uint8Array(samples.buffer, samples.byteOffset, samples.byteLength);
  let binary = '';
  const step = 0x8000;
  for (let i = 0; i < bytes.length; i += step) {
    binary += String.fromCharCode(...bytes.subarray(i, i + step));
  }
  return btoa(binary);
}

export default function App() {
  const [connected, setConnected] = useState(false);
  const [ready, setReady] = useState(false);
  const [halState, setHalState] = useState<HalState>('loading');
  const [statusText, setStatusText] = useState('CONNECTING TO DISCOVERY ONE...');
  const [models, setModels] = useState<string[]>([]);
  const [currentModel, setCurrentModel] = useState('');
  const [steps, setSteps] = useState<number>(6);
  const [sampleRate, setSampleRate] = useState<number>(48000);
  const [localMicAllowed, setLocalMicAllowed] = useState<boolean>(true);
  const [micMode, setMicMode] = useState<MicMode>('pipewire');
  const [metrics, setMetrics] = useState('');
  const [entries, setEntries] = useState<ChatEntry[]>([]);
  const [streamingHal, setStreamingHal] = useState('');
  const [typedText, setTypedText] = useState('');
  const [pttEngaged, setPttEngaged] = useState(false);
  const [latched, setLatched] = useState(false);
  const [captureElapsedS, setCaptureElapsedS] = useState(0);
  const [audioSuspended, setAudioSuspended] = useState(false);
  const [muted, setMuted] = useState<boolean>(() => {
    try {
      return window.localStorage.getItem('ioghal.muted') === '1';
    } catch {
      return false;
    }
  });
  const [canReplay, setCanReplay] = useState(false);
  const [showScrollJump, setShowScrollJump] = useState(false);

  const wsRef = useRef<WebSocket | null>(null);
  const playerRef = useRef<HalAudioPlayer | null>(null);
  const pendingTurnRef = useRef<boolean>(false);
  const activeTurnIdRef = useRef<number>(0);
  const nextTurnIdRef = useRef<number>(1);
  const activeCaptureIdRef = useRef<number>(0);
  const nextCaptureIdRef = useRef<number>(1);
  const levelRef = useRef<number>(0);
  const captureProgressRef = useRef<number>(0);
  const streamingHalRef = useRef<string>('');
  const hadConnectedOnceRef = useRef<boolean>(false);

  const micModeRef = useRef<MicMode>(micMode);
  const captureMicModeRef = useRef<MicMode>('pipewire');
  const readyRef = useRef<boolean>(ready);
  const pttActiveRef = useRef<boolean>(false);
  const latchedRef = useRef<boolean>(false);
  const awaitingMicPermRef = useRef<boolean>(false);
  const pressStartMsRef = useRef<number>(0);
  const ownerRef = useRef<CaptureOwner | null>(null);
  const spaceDownRef = useRef<boolean>(false);
  const barPointerIdRef = useRef<number | null>(null);

  const browserGenRef = useRef<number>(0);
  const browserSessionRef = useRef<BrowserCaptureSession | null>(null);

  const entryIdRef = useRef<number>(1);
  const transcriptRef = useRef<HTMLDivElement | null>(null);
  const logEndRef = useRef<HTMLDivElement | null>(null);
  const pinnedBottomRef = useRef<boolean>(true);
  const inputRef = useRef<HTMLInputElement | null>(null);
  const grilleRefs = useRef<(HTMLSpanElement | null)[]>([]);

  useEffect(() => {
    micModeRef.current = micMode;
  }, [micMode]);

  useEffect(() => {
    readyRef.current = ready;
  }, [ready]);

  useEffect(() => {
    streamingHalRef.current = streamingHal;
  }, [streamingHal]);

  // Instantiate WebAudio player once
  if (playerRef.current === null) {
    playerRef.current = new HalAudioPlayer({
      onPlaybackIdle: () => {
        setCanReplay(playerRef.current?.hasReplay ?? false);
        if (!pttActiveRef.current) {
          pendingTurnRef.current = false;
          setHalState('idle');
          setStatusText('ALL CIRCUITS OPERATIONAL — HOLD OR TAP SPACE TO TALK');
        }
      },
      onSuspendedChange: (suspended) => setAudioSuspended(suspended),
    });
    playerRef.current.setMuted(muted);
  }

  useEffect(() => {
    playerRef.current?.setMuted(muted);
    try {
      window.localStorage.setItem('ioghal.muted', muted ? '1' : '0');
    } catch {
      // ignore storage errors
    }
  }, [muted]);

  // Smart auto-scroll: only scroll when user is near the bottom
  useEffect(() => {
    if (pinnedBottomRef.current) {
      logEndRef.current?.scrollIntoView({ behavior: 'smooth' });
      setShowScrollJump(false);
    } else if (entries.length > 0 || streamingHal) {
      setShowScrollJump(true);
    }
  }, [entries, streamingHal]);

  const appendEntry = useCallback(
    (speaker: ChatEntry['speaker'], text: string, interrupted = false) => {
      if (!text.trim()) return;
      setEntries((prev) => {
        const next = [...prev, { id: entryIdRef.current++, speaker, text, interrupted }];
        return next.length > MAX_LOG_ENTRIES ? next.slice(-MAX_LOG_ENTRIES) : next;
      });
    },
    [],
  );

  const commitInterruptedHal = useCallback(() => {
    const partial = streamingHalRef.current.trim();
    if (partial) {
      appendEntry('HAL', partial, true);
    }
    streamingHalRef.current = '';
    setStreamingHal('');
  }, [appendEntry]);

  const mintTurnId = useCallback(() => {
    const tid = nextTurnIdRef.current++;
    activeTurnIdRef.current = tid;
    return tid;
  }, []);

  const retireTurn = useCallback(() => {
    activeTurnIdRef.current = nextTurnIdRef.current++;
    pendingTurnRef.current = false;
  }, []);

  const mintCaptureId = useCallback(() => {
    const cid = nextCaptureIdRef.current++;
    activeCaptureIdRef.current = cid;
    return cid;
  }, []);

  const teardownBrowserSession = useCallback((discardChunks: boolean) => {
    browserGenRef.current += 1;
    awaitingMicPermRef.current = false;
    const sess = browserSessionRef.current;
    browserSessionRef.current = null;
    if (!sess) return null;
    if (sess.processor) {
      sess.processor.onaudioprocess = null;
      try {
        sess.processor.disconnect();
      } catch {
        // ignore
      }
      sess.processor = null;
    }
    if (sess.source) {
      try {
        sess.source.disconnect();
      } catch {
        // ignore
      }
      sess.source = null;
    }
    if (sess.sinkGain) {
      try {
        sess.sinkGain.disconnect();
      } catch {
        // ignore
      }
      sess.sinkGain = null;
    }
    sess.stream?.getTracks().forEach((t) => t.stop());
    sess.stream = null;
    if (discardChunks) {
      sess.chunks = [];
      sess.samples = 0;
      return null;
    }
    return sess;
  }, []);

  const resetLocalCaptureFlags = useCallback(() => {
    pttActiveRef.current = false;
    latchedRef.current = false;
    awaitingMicPermRef.current = false;
    ownerRef.current = null;
    spaceDownRef.current = false;
    barPointerIdRef.current = null;
    activeCaptureIdRef.current = 0;
    captureProgressRef.current = 0;
    setPttEngaged(false);
    setLatched(false);
    setCaptureElapsedS(0);
    levelRef.current = 0;
  }, []);

  const sendJson = useCallback((payload: Record<string, unknown>) => {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(payload));
    }
  }, []);

  const uploadBrowserSession = useCallback(
    (sess: BrowserCaptureSession | null) => {
      if (!sess || sess.samples === 0) {
        setHalState('idle');
        setStatusText('ALL CIRCUITS OPERATIONAL — HOLD OR TAP SPACE TO TALK');
        return;
      }
      const durS = sess.samples / Math.max(sess.sampleRate, 1);
      if (durS < MIN_CAPTURE_DURATION_S) {
        setHalState('idle');
        setStatusText('AUDIO TOO SHORT — HOLD OR TAP SPACE TO TALK');
        return;
      }
      const merged = new Float32Array(sess.samples);
      let offset = 0;
      for (const c of sess.chunks) {
        merged.set(c, offset);
        offset += c.length;
      }
      const turnId = mintTurnId();
      pendingTurnRef.current = true;
      setHalState('thinking');
      setStatusText('TRANSCRIBING SPEECH...');
      sendJson({
        type: 'audio_b64',
        turn_id: turnId,
        pcm_b64: float32ToBase64(merged),
        sample_rate: sess.sampleRate,
      });
    },
    [mintTurnId, sendJson],
  );

  const finishRecording = useCallback(() => {
    if (!pttActiveRef.current) return;
    const modeUsed = captureMicModeRef.current;
    const cid = activeCaptureIdRef.current;
    resetLocalCaptureFlags();

    if (modeUsed === 'browser') {
      const sess = teardownBrowserSession(false);
      uploadBrowserSession(sess);
    } else {
      const turnId = mintTurnId();
      pendingTurnRef.current = true;
      setHalState('thinking');
      setStatusText('TRANSCRIBING SPEECH...');
      sendJson({ type: 'ptt_stop', turn_id: turnId, capture_id: cid });
    }
  }, [mintTurnId, resetLocalCaptureFlags, sendJson, teardownBrowserSession, uploadBrowserSession]);

  // Keep finishRecording accessible inside ws.onmessage without tearing down socket
  const finishRecordingRef = useRef(finishRecording);
  useEffect(() => {
    finishRecordingRef.current = finishRecording;
  }, [finishRecording]);

  // Capture elapsed timer & progress ring while recording
  useEffect(() => {
    if (!pttEngaged) return;
    const startedAt = performance.now();
    const id = window.setInterval(() => {
      const sec = Math.min(MAX_CAPTURE_DURATION_S, (performance.now() - startedAt) / 1000);
      captureProgressRef.current = sec / MAX_CAPTURE_DURATION_S;
      setCaptureElapsedS(sec);
    }, 100);
    return () => window.clearInterval(id);
  }, [pttEngaged]);

  // Drive playback RMS + live 7-slot Discovery One speaker grille VU meter
  useEffect(() => {
    let rafId = 0;
    const pump = () => {
      if (!document.hidden) {
        const player = playerRef.current;
        if (player && player.isPlaying) {
          levelRef.current = player.sampleLevel();
        } else if (!pttActiveRef.current) {
          levelRef.current = 0;
        }
        const lvl = levelRef.current;
        const active = pttActiveRef.current || Boolean(player?.isPlaying);
        const center = (GRILLE_SLOTS - 1) / 2;
        for (let i = 0; i < GRILLE_SLOTS; i++) {
          const el = grilleRefs.current[i];
          if (!el) continue;
          if (!active) {
            el.style.opacity = '';
            el.style.transform = '';
          } else {
            const dist = Math.abs(i - center) / center;
            const slotEnergy = Math.max(0.18, Math.min(1, lvl * 1.45 - dist * 0.45 + 0.25));
            el.style.opacity = String(0.35 + slotEnergy * 0.65);
            el.style.transform = `scaleX(${0.65 + slotEnergy * 0.35})`;
          }
        }
      }
      rafId = requestAnimationFrame(pump);
    };
    rafId = requestAnimationFrame(pump);
    return () => cancelAnimationFrame(rafId);
  }, []);

  // Dispose audio resources on unmount
  useEffect(() => {
    return () => {
      teardownBrowserSession(true);
      playerRef.current?.dispose();
      resetLocalCaptureFlags();
    };
  }, [resetLocalCaptureFlags, teardownBrowserSession]);

  // WebSocket lifecycle with exponential backoff reconnect
  useEffect(() => {
    let alive = true;
    let reconnectTimer: number | null = null;
    let backoffMs = 1000;

    const connect = () => {
      const proto = window.location.protocol === 'https:' ? 'wss' : 'ws';
      const url = `${proto}://${window.location.host}/ws`;
      const ws = new WebSocket(url);
      ws.binaryType = 'arraybuffer';
      wsRef.current = ws;

      ws.onopen = () => {
        if (!alive || wsRef.current !== ws) return;
        backoffMs = 1000;
        setConnected(true);
        if (hadConnectedOnceRef.current) {
          setMetrics('');
          appendEntry('SYSTEM', 'Reconnected to Discovery One — fresh conversation session started.');
        }
        hadConnectedOnceRef.current = true;
      };

      ws.onclose = () => {
        if (!alive || wsRef.current !== ws) return;
        setConnected(false);
        setReady(false);
        teardownBrowserSession(true);
        playerRef.current?.stop();
        resetLocalCaptureFlags();
        retireTurn();
        setStreamingHal('');
        setHalState('loading');
        setStatusText('CONNECTION LOST — RECONNECTING...');
        const delay = backoffMs;
        backoffMs = Math.min(8000, Math.round(backoffMs * 1.6));
        reconnectTimer = window.setTimeout(connect, delay);
      };

      ws.onmessage = (event) => {
        if (!alive || wsRef.current !== ws) return;
        if (event.data instanceof ArrayBuffer) {
          const decoded = decodeFramedPcm(event.data);
          if (decoded && decoded.turnId === activeTurnIdRef.current) {
            playerRef.current?.enqueueChunk(decoded.pcm);
          }
          return;
        }

        let msg: ServerMessage;
        try {
          msg = JSON.parse(String(event.data)) as ServerMessage;
        } catch {
          return;
        }
        if (!msg || typeof msg !== 'object') return;

        const msgTurnId = 'turn_id' in msg && typeof msg.turn_id === 'number' ? msg.turn_id : null;
        const isStaleTurn = msgTurnId !== null && msgTurnId !== activeTurnIdRef.current;
        const msgCaptureId =
          'capture_id' in msg && typeof msg.capture_id === 'number' ? msg.capture_id : null;
        const isStaleCapture =
          msgCaptureId !== null && msgCaptureId !== activeCaptureIdRef.current;

        switch (msg.type) {
          case 'system_status': {
            setReady(Boolean(msg.ready));
            if (typeof msg.local_mic_allowed === 'boolean') {
              setLocalMicAllowed(msg.local_mic_allowed);
              if (!msg.local_mic_allowed && micModeRef.current === 'pipewire') {
                setMicMode('browser');
              }
            }
            if (typeof msg.steps === 'number') setSteps(msg.steps);
            if (msg.sample_rate) {
              setSampleRate(Number(msg.sample_rate));
              playerRef.current?.setSampleRate(Number(msg.sample_rate));
            }
            if (Array.isArray(msg.models) && msg.models.length > 0) {
              setModels(msg.models);
            }
            if (msg.model) setCurrentModel(msg.model);
            if (
              !pttActiveRef.current &&
              !pendingTurnRef.current &&
              !(playerRef.current?.isPlaying ?? false)
            ) {
              setHalState(msg.state);
              setStatusText(msg.stage || '');
            }
            if (msg.error) appendEntry('ERROR', msg.error);
            break;
          }
          case 'capture_failed': {
            if (isStaleCapture) break;
            teardownBrowserSession(true);
            resetLocalCaptureFlags();
            pendingTurnRef.current = false;
            setHalState('idle');
            setStatusText('MICROPHONE ERROR — TRY MIC: BROWSER');
            appendEntry('ERROR', msg.message || 'Microphone capture failed.');
            break;
          }
          case 'capture_limit': {
            if (isStaleCapture) break;
            if (pttActiveRef.current) {
              appendEntry('SYSTEM', msg.message || 'Recording limit reached.');
              if (captureMicModeRef.current === 'pipewire') {
                finishRecordingRef.current();
              }
            }
            break;
          }
          case 'state': {
            if (isStaleTurn || isStaleCapture) break;
            if (msg.state === 'idle' && (pttActiveRef.current || playerRef.current?.isPlaying)) {
              break;
            }
            if (msg.state === 'idle') {
              pendingTurnRef.current = false;
            }
            setHalState(msg.state);
            if (msg.state === 'listening' && latchedRef.current) {
              setStatusText('LISTENING (LOCKED) — PRESS SPACE AGAIN TO SEND · ESC TO CANCEL');
            } else if (msg.detail) {
              setStatusText(msg.detail);
            }
            break;
          }
          case 'mic_level': {
            if (
              !isStaleCapture &&
              pttActiveRef.current &&
              captureMicModeRef.current === 'pipewire'
            ) {
              levelRef.current = Number(msg.level) || 0;
            }
            break;
          }
          case 'transcript': {
            if (isStaleTurn) break;
            appendEntry(msg.speaker, msg.text);
            break;
          }
          case 'audio_start': {
            if (isStaleTurn) break;
            playerRef.current?.beginTurn(msg.sample_rate);
            setStreamingHal('');
            break;
          }
          case 'hal_partial': {
            if (isStaleTurn) break;
            setStreamingHal(msg.text || '');
            break;
          }
          case 'audio_end': {
            if (isStaleTurn) break;
            const cancelled = Boolean(msg.cancelled);
            const failed = Boolean(msg.failed);
            playerRef.current?.finishTurn(cancelled, failed);
            setCanReplay(playerRef.current?.hasReplay ?? false);
            if (!cancelled && !failed && msg.text) {
              appendEntry('HAL', msg.text);
            } else if ((cancelled || failed) && msg.text) {
              appendEntry('HAL', msg.text, true);
            }
            setStreamingHal('');
            if (!(playerRef.current?.isPlaying ?? false) && !pttActiveRef.current) {
              pendingTurnRef.current = false;
              setHalState('idle');
              setStatusText(
                failed
                  ? 'TURN ERROR — HOLD OR TAP SPACE TO TALK'
                  : 'ALL CIRCUITS OPERATIONAL — HOLD OR TAP SPACE TO TALK',
              );
            }
            break;
          }
          case 'metrics': {
            if (isStaleTurn) break;
            const rtfPart = typeof msg.rtf === 'number' ? ` · RTF ${msg.rtf}x` : '';
            setMetrics(`STT ${msg.stt_s}s · FIRST CHUNK ${msg.first_chunk_s}s${rtfPart}`);
            break;
          }
          case 'cleared': {
            setEntries([]);
            setStreamingHal('');
            setMetrics('');
            appendEntry('SYSTEM', 'Conversation memory cleared. I still remember you, Charles.');
            break;
          }
          case 'error': {
            if (isStaleTurn) break;
            appendEntry('ERROR', msg.message || 'Unknown error');
            // Turn-scoped errors clear pending turn & reset idle UI so a bad upload or
            // mid-turn failure never leaves the console stuck in thinking/speaking.
            // Non-turn errors (e.g. invalid set_model) must NOT clobber an active turn.
            if (msgTurnId !== null) {
              pendingTurnRef.current = false;
              playerRef.current?.stop();
              if (!pttActiveRef.current) {
                setHalState('idle');
                setStatusText('TURN ERROR — HOLD OR TAP SPACE TO TALK');
              }
            }
            break;
          }
        }
      };
    };

    connect();
    return () => {
      alive = false;
      if (reconnectTimer !== null) clearTimeout(reconnectTimer);
      wsRef.current?.close();
    };
  }, [appendEntry, resetLocalCaptureFlags, retireTurn, teardownBrowserSession]);

  const startBrowserCapture = useCallback(async () => {
    const gen = ++browserGenRef.current;
    awaitingMicPermRef.current = true;
    let acquiredStream: MediaStream | null = null;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });
      acquiredStream = stream;
      awaitingMicPermRef.current = false;
      if (gen !== browserGenRef.current || !pttActiveRef.current) {
        stream.getTracks().forEach((t) => t.stop());
        return;
      }
      const player = playerRef.current;
      if (!player) {
        stream.getTracks().forEach((t) => t.stop());
        return;
      }
      const running = await player.resumeNow();
      if (!running || gen !== browserGenRef.current || !pttActiveRef.current) {
        stream.getTracks().forEach((t) => t.stop());
        if (!running && gen === browserGenRef.current) {
          resetLocalCaptureFlags();
          setHalState('idle');
          setStatusText('AUDIO CONTEXT SUSPENDED — CLICK ENABLE AUDIO');
        }
        return;
      }
      const ctx = player.ensureContext();
      const source = ctx.createMediaStreamSource(stream);
      const processor = ctx.createScriptProcessor(2048, 1, 1);
      // Zero-gain sink prevents mic bleed to speakers while keeping ScriptProcessor alive
      const sinkGain = ctx.createGain();
      sinkGain.gain.value = 0;

      const session: BrowserCaptureSession = {
        gen,
        stream,
        source,
        processor,
        sinkGain,
        chunks: [],
        samples: 0,
        sampleRate: ctx.sampleRate || 48000,
      };
      const maxSamples = Math.floor(session.sampleRate * MAX_CAPTURE_DURATION_S);
      browserSessionRef.current = session;

      processor.onaudioprocess = (ev) => {
        if (browserSessionRef.current !== session || !pttActiveRef.current) return;
        const ch = ev.inputBuffer.getChannelData(0);
        const remaining = maxSamples - session.samples;
        if (remaining <= 0) return;
        const copy =
          ch.length > remaining ? new Float32Array(ch.subarray(0, remaining)) : new Float32Array(ch);
        session.chunks.push(copy);
        session.samples += copy.length;
        let sumSq = 0;
        for (let i = 0; i < copy.length; i++) sumSq += copy[i] * copy[i];
        levelRef.current = Math.min(1, Math.sqrt(sumSq / copy.length) * 6.0);
        if (session.samples >= maxSamples) {
          appendEntry('SYSTEM', `Reached ${Math.floor(MAX_CAPTURE_DURATION_S)}s recording limit.`);
          finishRecording();
        }
      };

      source.connect(processor);
      processor.connect(sinkGain);
      sinkGain.connect(ctx.destination);
      setHalState('listening');
      setStatusText(
        latchedRef.current
          ? 'LISTENING (LOCKED · BROWSER MIC) — PRESS SPACE AGAIN TO SEND · ESC TO CANCEL'
          : 'LISTENING (BROWSER MIC) — RELEASE OR PRESS SPACE TO SEND',
      );
    } catch (err) {
      awaitingMicPermRef.current = false;
      acquiredStream?.getTracks().forEach((t) => t.stop());
      if (gen !== browserGenRef.current) return;
      teardownBrowserSession(true);
      resetLocalCaptureFlags();
      appendEntry('ERROR', `Browser mic failed: ${String(err)}.`);
      if (localMicAllowed) {
        setMicMode('pipewire');
      }
      setHalState('idle');
      setStatusText('ALL CIRCUITS OPERATIONAL — HOLD OR TAP SPACE TO TALK');
    }
  }, [appendEntry, finishRecording, localMicAllowed, resetLocalCaptureFlags, teardownBrowserSession]);

  const beginRecording = useCallback(
    (owner: CaptureOwner) => {
      if (!readyRef.current || pttActiveRef.current) return;
      playerRef.current?.ensureContext();
      playerRef.current?.stop();
      commitInterruptedHal();
      retireTurn();

      const mode = micModeRef.current;
      const cid = mintCaptureId();
      captureMicModeRef.current = mode;
      pttActiveRef.current = true;
      latchedRef.current = owner.kind === 'toggle';
      ownerRef.current = owner;
      pressStartMsRef.current = performance.now();
      setPttEngaged(true);
      setLatched(owner.kind === 'toggle');
      setHalState('listening');
      setStatusText(
        owner.kind === 'toggle'
          ? 'LISTENING (LOCKED) — PRESS SPACE AGAIN TO SEND · ESC TO CANCEL'
          : 'LISTENING — RELEASE (OR PRESS SPACE) TO SEND',
      );

      if (mode === 'browser') {
        sendJson({ type: 'interrupt', silent: true });
        void startBrowserCapture();
      } else {
        sendJson({ type: 'ptt_start', capture_id: cid });
      }
    },
    [commitInterruptedHal, mintCaptureId, retireTurn, sendJson, startBrowserCapture],
  );

  const cancelAll = useCallback(
    (sendInterrupt = true) => {
      playerRef.current?.stop();
      teardownBrowserSession(true);
      commitInterruptedHal();
      resetLocalCaptureFlags();
      const cancelTid = mintTurnId();
      pendingTurnRef.current = false;
      if (sendInterrupt) {
        sendJson({ type: 'interrupt', turn_id: cancelTid });
      }
      setHalState('idle');
      setStatusText('ALL CIRCUITS OPERATIONAL — HOLD OR TAP SPACE TO TALK');
    },
    [commitInterruptedHal, mintTurnId, resetLocalCaptureFlags, sendJson, teardownBrowserSession],
  );

  const sameOwner = (a: CaptureOwner | null, b: CaptureOwner): boolean => {
    if (!a || a.kind !== b.kind) return false;
    if (a.kind === 'pointer' && b.kind === 'pointer') {
      return a.pointerId === b.pointerId;
    }
    return true;
  };

  const handlePressStart = useCallback(
    (owner: CaptureOwner) => {
      if (!readyRef.current) return;
      if (pttActiveRef.current) {
        if (latchedRef.current) {
          finishRecording();
        }
        return;
      }
      beginRecording(owner);
    },
    [beginRecording, finishRecording],
  );

  const handlePressEnd = useCallback(
    (owner: CaptureOwner) => {
      if (!pttActiveRef.current || latchedRef.current) return;
      if (!sameOwner(ownerRef.current, owner)) return;
      const heldMs = performance.now() - pressStartMsRef.current;
      if (heldMs < TAP_THRESHOLD_MS) {
        latchedRef.current = true;
        ownerRef.current = { kind: 'toggle' };
        setLatched(true);
        setStatusText('LISTENING (LOCKED) — PRESS SPACE AGAIN TO SEND · ESC TO CANCEL');
      } else {
        finishRecording();
      }
    },
    [finishRecording],
  );

  const handlePointerAbort = useCallback(
    (pointerId: number) => {
      if (!pttActiveRef.current || latchedRef.current) return;
      if (!sameOwner(ownerRef.current, { kind: 'pointer', pointerId })) return;
      cancelAll();
    },
    [cancelAll],
  );

  const handleKeyboardToggle = useCallback(() => {
    if (!readyRef.current) return;
    if (pttActiveRef.current) {
      finishRecording();
    } else {
      beginRecording({ kind: 'toggle' });
    }
  }, [beginRecording, finishRecording]);

  // Global Spacebar, Escape, blur, and tab-visibility lifecycle
  useEffect(() => {
    const isNonPttInteractiveFocused = () => {
      const el = document.activeElement;
      if (!(el instanceof HTMLElement)) return false;
      if (el.dataset.pttControl === 'true') return false;
      const tag = el.tagName;
      return (
        tag === 'INPUT' ||
        tag === 'TEXTAREA' ||
        tag === 'SELECT' ||
        tag === 'BUTTON' ||
        tag === 'A' ||
        el.isContentEditable
      );
    };

    const onKeyDown = (e: KeyboardEvent) => {
      if (e.code === 'Escape') {
        const el = document.activeElement;
        if (el instanceof HTMLSelectElement) {
          el.blur();
        }
        if (
          pttActiveRef.current ||
          pendingTurnRef.current ||
          (playerRef.current?.isPlaying ?? false) ||
          !(playerRef.current?.isServerDone ?? true)
        ) {
          e.preventDefault();
          e.stopPropagation();
          cancelAll();
        }
        return;
      }

      if (e.code !== 'Space') return;
      if (e.ctrlKey || e.metaKey || e.altKey) return;
      if (isNonPttInteractiveFocused()) return;

      e.preventDefault();
      e.stopPropagation();
      if (e.repeat || spaceDownRef.current) return;
      spaceDownRef.current = true;
      handlePressStart({ kind: 'space' });
    };

    const onKeyUp = (e: KeyboardEvent) => {
      if (e.code !== 'Space') return;
      if (!spaceDownRef.current) return;
      e.preventDefault();
      e.stopPropagation();
      spaceDownRef.current = false;
      handlePressEnd({ kind: 'space' });
    };

    const onBlur = () => {
      spaceDownRef.current = false;
      // Do not abort while the browser's own mic permission prompt holds focus
      if (awaitingMicPermRef.current) return;
      if (pttActiveRef.current && ownerRef.current?.kind === 'space' && !latchedRef.current) {
        cancelAll();
      }
    };

    const onVisibilityChange = () => {
      if (document.hidden && pttActiveRef.current && !awaitingMicPermRef.current) {
        cancelAll();
      }
    };

    window.addEventListener('keydown', onKeyDown, { capture: true, passive: false });
    window.addEventListener('keyup', onKeyUp, { capture: true, passive: false });
    window.addEventListener('blur', onBlur);
    document.addEventListener('visibilitychange', onVisibilityChange);
    return () => {
      window.removeEventListener('keydown', onKeyDown, { capture: true });
      window.removeEventListener('keyup', onKeyUp, { capture: true });
      window.removeEventListener('blur', onBlur);
      document.removeEventListener('visibilitychange', onVisibilityChange);
    };
  }, [cancelAll, handlePressEnd, handlePressStart]);

  const submitText = (e: React.FormEvent) => {
    e.preventDefault();
    const text = typedText.trim();
    if (!text || !ready || !connected || pttActiveRef.current) return;
    if (text.length > MAX_TEXT_CHARS) {
      appendEntry('ERROR', `Message exceeds ${MAX_TEXT_CHARS} characters.`);
      return;
    }
    playerRef.current?.ensureContext();
    playerRef.current?.stop();
    commitInterruptedHal();
    setTypedText('');
    inputRef.current?.blur();
    const turnId = mintTurnId();
    pendingTurnRef.current = true;
    setHalState('thinking');
    setStatusText(`PROCESSING (${currentModel})...`);
    sendJson({ type: 'text', turn_id: turnId, text });
  };

  const handleReset = (e: React.MouseEvent<HTMLButtonElement>) => {
    cancelAll(false);
    const resetTid = mintTurnId();
    sendJson({ type: 'reset', turn_id: resetTid });
    e.currentTarget.blur();
  };

  const handleEnableAudio = async () => {
    const ok = await playerRef.current?.resumeNow();
    if (ok) {
      playerRef.current?.flushPrebuffer();
      if (playerRef.current?.isPlaying && !pttActiveRef.current) {
        setHalState('speaking');
        setStatusText('HAL 9000 SPEAKING...');
      }
    }
  };

  const handleReplay = async () => {
    if (
      pttActiveRef.current ||
      pendingTurnRef.current ||
      halState === 'thinking' ||
      halState === 'speaking' ||
      !playerRef.current?.hasReplay
    ) {
      return;
    }
    const ok = await playerRef.current.resumeNow();
    if (!ok || pttActiveRef.current || pendingTurnRef.current || !playerRef.current?.hasReplay) {
      return;
    }
    if (playerRef.current.replayLast()) {
      setHalState('speaking');
      setStatusText('REPLAYING LAST TRANSMISSION...');
    }
  };

  const handleTranscriptScroll = () => {
    const el = transcriptRef.current;
    if (!el) return;
    const distance = el.scrollHeight - el.scrollTop - el.clientHeight;
    const nearBottom = distance < 48;
    pinnedBottomRef.current = nearBottom;
    if (nearBottom) setShowScrollJump(false);
  };

  const canAbort =
    connected && (pttEngaged || halState === 'thinking' || halState === 'speaking');

  const pttLabel = !pttEngaged
    ? 'HOLD OR CLICK [ SPACE ] TO TALK'
    : latched
      ? `LOCKED ${captureElapsedS.toFixed(1)}s · SPACE TO SEND`
      : `TRANSMITTING ${captureElapsedS.toFixed(1)}s · RELEASE`;

  return (
    <div className="shell">
      <main className="layout">
        <section className="faceplate" aria-label="IOGHAL Optical Unit">
          <div className="nameplate">
            <span className="nameplate-hal">IOG</span>
            <span className="nameplate-num">HAL</span>
          </div>

          <HalEye
            state={halState}
            levelRef={levelRef}
            disabled={!ready || !connected}
            engaged={pttEngaged}
            latched={latched}
            captureProgressRef={captureProgressRef}
            onPointerStart={(pid) => handlePressStart({ kind: 'pointer', pointerId: pid })}
            onPointerEnd={(pid) => handlePressEnd({ kind: 'pointer', pointerId: pid })}
            onPointerCancel={handlePointerAbort}
            onKeyboardToggle={handleKeyboardToggle}
          />

          <div className={`grille grille-${halState}`} aria-hidden="true">
            {Array.from({ length: GRILLE_SLOTS }).map((_, i) => (
              <span
                key={i}
                ref={(el) => {
                  grilleRefs.current[i] = el;
                }}
                className="grille-slot"
              />
            ))}
          </div>

          <div className="ptt-actions">
            <button
              type="button"
              data-ptt-control="true"
              className={`ptt-bar ${pttEngaged ? 'ptt-active' : ''}`}
              disabled={!ready || !connected}
              aria-pressed={pttEngaged}
              onPointerDown={(e) => {
                if (!ready || !connected || e.button !== 0 || barPointerIdRef.current !== null) {
                  return;
                }
                e.preventDefault();
                barPointerIdRef.current = e.pointerId;
                try {
                  e.currentTarget.setPointerCapture(e.pointerId);
                } catch {
                  // ignore
                }
                handlePressStart({ kind: 'pointer', pointerId: e.pointerId });
              }}
              onPointerUp={(e) => {
                if (barPointerIdRef.current !== e.pointerId) return;
                e.preventDefault();
                barPointerIdRef.current = null;
                handlePressEnd({ kind: 'pointer', pointerId: e.pointerId });
                try {
                  e.currentTarget.releasePointerCapture(e.pointerId);
                } catch {
                  // ignore
                }
              }}
              onPointerCancel={(e) => {
                if (barPointerIdRef.current !== e.pointerId) return;
                barPointerIdRef.current = null;
                handlePointerAbort(e.pointerId);
              }}
              onLostPointerCapture={(e) => {
                if (barPointerIdRef.current !== e.pointerId) return;
                barPointerIdRef.current = null;
                handlePointerAbort(e.pointerId);
              }}
              onClick={(e) => {
                if (e.detail !== 0) return;
                handleKeyboardToggle();
              }}
            >
              <span className="ptt-dot" />
              {pttLabel}
            </button>

            {canAbort && (
              <button
                type="button"
                className="abort-btn"
                title="Interrupt HAL or cancel active recording (Esc)"
                onClick={(e) => {
                  cancelAll();
                  e.currentTarget.blur();
                }}
              >
                ESC
              </button>
            )}
          </div>
        </section>

        <section className="console">
          <header className="console-header">
            <div className="status-block">
              <span className={`led led-${connected ? halState : 'error'}`} />
              <div>
                <div className="status-label" aria-live="polite">
                  {statusText}
                </div>
                <div className="telemetry-sub">
                  VOXCPM2 {Math.round(sampleRate / 1000)}KHZ · STEPS {steps} · WHISPER LARGE-V3
                  {metrics ? ` · ${metrics}` : ''}
                </div>
              </div>
            </div>

            <div className="controls">
              {audioSuspended && (
                <button
                  type="button"
                  className="ctrl-btn ctrl-warn"
                  onClick={() => void handleEnableAudio()}
                >
                  ENABLE AUDIO
                </button>
              )}

              <select
                aria-label="LLM Model"
                className="ctrl-select"
                disabled={!connected}
                value={currentModel}
                onChange={(e) => {
                  const val = e.target.value;
                  setCurrentModel(val);
                  sendJson({ type: 'set_model', model: val });
                  e.target.blur();
                }}
              >
                {!currentModel && <option value="">Connecting…</option>}
                {(models.length ? models : currentModel ? [currentModel] : []).map((m) => (
                  <option key={m} value={m}>
                    {m}
                  </option>
                ))}
              </select>

              <select
                aria-label="Voice Diffusion Steps"
                className="ctrl-select"
                disabled={!connected}
                title="VoxCPM2 diffusion steps (lower is faster, higher is richer)"
                value={steps}
                onChange={(e) => {
                  const val = Number(e.target.value);
                  setSteps(val);
                  sendJson({ type: 'set_steps', steps: val });
                  e.target.blur();
                }}
              >
                <option value={4}>4 STEPS (FAST)</option>
                <option value={6}>6 STEPS</option>
                <option value={8}>8 STEPS</option>
                <option value={10}>10 STEPS (HQ)</option>
              </select>

              <button
                type="button"
                className="ctrl-btn"
                disabled={pttEngaged || !localMicAllowed}
                title={
                  localMicAllowed
                    ? 'Switch between local PipeWire microphone and browser WebAudio microphone'
                    : 'Local host microphone is disabled on non-loopback bindings'
                }
                onClick={(e) => {
                  if (pttActiveRef.current) return;
                  setMicMode((prev) => (prev === 'pipewire' ? 'browser' : 'pipewire'));
                  e.currentTarget.blur();
                }}
              >
                MIC: {micMode === 'pipewire' ? 'LOCAL' : 'BROWSER'}
              </button>

              <button
                type="button"
                className="ctrl-btn"
                aria-pressed={muted}
                title="Mute or unmute HAL voice playback"
                onClick={(e) => {
                  setMuted((prev) => !prev);
                  e.currentTarget.blur();
                }}
              >
                {muted ? 'MUTED' : 'AUDIO'}
              </button>

              {canReplay && (
                <button
                  type="button"
                  className="ctrl-btn"
                  disabled={pttEngaged || halState === 'thinking' || halState === 'speaking'}
                  title="Replay the last completed HAL transmission"
                  onClick={(e) => {
                    handleReplay();
                    e.currentTarget.blur();
                  }}
                >
                  REPLAY
                </button>
              )}

              <button
                type="button"
                className="ctrl-btn"
                disabled={!connected}
                onClick={handleReset}
              >
                RESET
              </button>
            </div>
          </header>

          <div
            ref={transcriptRef}
            className="transcript"
            role="log"
            aria-live="polite"
            onScroll={handleTranscriptScroll}
          >
            {entries.length === 0 && !streamingHal && (
              <div className="empty-hint">
                <p>DISCOVERY ONE · LOGIC MEMORY TERMINAL</p>
                <p>
                  {ready
                    ? 'Hold SPACEBAR while speaking (or tap SPACEBAR once to latch and tap again to send). Press ESC to interrupt.'
                    : 'Warming up CUDA neural subsystems. Spacebar Push-to-Talk will unlock automatically.'}
                </p>
              </div>
            )}

            {entries.map((entry) => (
              <div key={entry.id} className={`msg msg-${entry.speaker.toLowerCase()}`}>
                <span className="msg-tag">
                  {entry.speaker}
                  {entry.interrupted ? ' · INTERRUPTED' : ''}
                </span>
                <p className="msg-text">{entry.text}</p>
              </div>
            ))}

            {streamingHal && (
              <div className="msg msg-hal msg-streaming">
                <span className="msg-tag">HAL</span>
                <p className="msg-text">{streamingHal}</p>
              </div>
            )}
            <div ref={logEndRef} />
          </div>

          {showScrollJump && (
            <button
              type="button"
              className="scroll-jump"
              onClick={() => {
                pinnedBottomRef.current = true;
                setShowScrollJump(false);
                logEndRef.current?.scrollIntoView({ behavior: 'smooth' });
              }}
            >
              ↓ LATEST TRANSMISSION
            </button>
          )}

          <form className="composer" onSubmit={submitText}>
            <input
              ref={inputRef}
              type="text"
              aria-label="Type a transmission to HAL"
              maxLength={MAX_TEXT_CHARS}
              value={typedText}
              disabled={!ready || !connected || pttEngaged}
              onChange={(e) => setTypedText(e.target.value)}
              placeholder={
                pttEngaged
                  ? 'Finish or cancel microphone capture before typing...'
                  : 'Or type a transmission to HAL and press Enter...'
              }
            />
            <button
              type="submit"
              disabled={!ready || !connected || pttEngaged || !typedText.trim()}
            >
              SEND
            </button>
          </form>
        </section>
      </main>
    </div>
  );
}
