/**
 * Reusable WebSocket protocol + WebAudio streaming playback engine for IOGHAL.
 *
 * Host React or TypeScript applications can import ``HalAudioPlayer`` and
 * protocol types directly to embed IOGHAL into a larger UI.
 */

export type HalState = 'loading' | 'idle' | 'listening' | 'thinking' | 'speaking' | 'error';

export type MicMode = 'pipewire' | 'browser';

export interface SystemStatusMessage {
  type: 'system_status';
  ready: boolean;
  error: string | null;
  stage: string;
  state: HalState;
  model?: string;
  models: string[];
  ollama_available?: boolean;
  llama_server_available?: boolean;
  local_mic_allowed?: boolean;
  sample_rate: number;
  max_duration_s?: number;
  steps?: number;
}

export interface HalMetricsMessage {
  type: 'metrics';
  turn_id?: number;
  stt_s: number;
  first_chunk_s: number;
  rtf?: number;
  web_search_s?: number;
  llm_first_sentence_s?: number;
  tts_first_chunk_s?: number;
}

export type ServerMessage =
  | SystemStatusMessage
  | HalMetricsMessage
  | { type: 'state'; turn_id?: number; capture_id?: number; state: HalState; detail?: string }
  | { type: 'capture_failed'; capture_id?: number; message?: string }
  | { type: 'capture_limit'; capture_id?: number; message?: string }
  | { type: 'mic_level'; capture_id?: number; level: number }
  | { type: 'transcript'; turn_id?: number; speaker: 'YOU' | 'HAL' | 'SYSTEM' | 'ERROR'; text: string }
  | { type: 'audio_start'; turn_id?: number; sample_rate?: number }
  | { type: 'hal_partial'; turn_id?: number; text: string }
  | { type: 'audio_end'; turn_id?: number; text?: string; cancelled?: boolean; failed?: boolean }
  | { type: 'cleared'; turn_id?: number }
  | { type: 'error'; turn_id?: number; message?: string };

// VoxCPM2 streams ~160ms chunks at ~0.33x RTF (~53ms/chunk), so 150ms starts on chunk 1
// with 3x headroom while widening automatically to 300ms if a genuine underrun occurs.
const INITIAL_PREBUFFER_SEC = 0.15;
const REBUFFER_PREBUFFER_SEC = 0.30;
// 8ms = 3 WebAudio render quanta @ 48 kHz (2.67ms each): tight start with zero underrun risk.
const SCHEDULE_LEAD_SEC = 0.008;
const PCM_HEADER_BYTES = 4;

/** Parse a binary WebSocket PCM frame prefixed with a 4-byte little-endian uint32 turn_id. */
export function decodeFramedPcm(buffer: ArrayBuffer): { turnId: number; pcm: Float32Array } | null {
  if (buffer.byteLength <= PCM_HEADER_BYTES) return null;
  const payloadBytes = buffer.byteLength - PCM_HEADER_BYTES;
  if (payloadBytes % 4 !== 0) return null;
  const view = new DataView(buffer);
  const turnId = view.getUint32(0, true);
  // Zero-copy aligned Float32Array view (PCM_HEADER_BYTES === 4 is Float32-aligned)
  const pcm = new Float32Array(buffer, PCM_HEADER_BYTES, payloadBytes / 4);
  return { turnId, pcm };
}

export interface AudioPlayerCallbacks {
  onPlaybackIdle: () => void;
  onSuspendedChange: (suspended: boolean) => void;
  onUnderrun?: (count: number) => void;
}

/**
 * Streaming WebAudio PCM player with prebuffering, smooth gain ramps, mute,
 * autoplay suspension tracking, and last-turn replay buffer.
 */
export class HalAudioPlayer {
  private ctx: AudioContext | null = null;
  private gain: GainNode | null = null;
  private analyser: AnalyserNode | null = null;
  private analyserData: Uint8Array<ArrayBuffer> = new Uint8Array(new ArrayBuffer(256));
  private activeSources = new Set<AudioBufferSourceNode>();
  private prebuffer: Float32Array[] = [];
  private prebufferSamples = 0;
  private playbackStarted = false;
  private serverDone = true;
  private nextPlayTime = 0;
  private sampleRate = 48000;
  private muted = false;
  private currentTurnChunks: Float32Array[] = [];
  private lastReplayPcm: Float32Array | null = null;
  private lastReplaySampleRate = 48000;
  private underrunCount = 0;
  private playbackGen = 0;
  private callbacks: AudioPlayerCallbacks;

  constructor(callbacks: AudioPlayerCallbacks) {
    this.callbacks = callbacks;
  }

  get isPlaying(): boolean {
    return this.activeSources.size > 0;
  }

  get hasPendingPrebuffer(): boolean {
    return this.prebufferSamples > 0;
  }

  get isServerDone(): boolean {
    return this.serverDone;
  }

  get hasReplay(): boolean {
    return this.lastReplayPcm !== null && this.lastReplayPcm.length > 0;
  }

  get isMuted(): boolean {
    return this.muted;
  }

  setSampleRate(sr: number): void {
    if (sr > 0) this.sampleRate = sr;
  }

  setMuted(muted: boolean): void {
    this.muted = muted;
    if (this.gain && this.ctx && this.ctx.state !== 'closed') {
      const now = this.ctx.currentTime;
      this.gain.gain.cancelScheduledValues(now);
      this.gain.gain.setTargetAtTime(muted ? 0 : 1, now, 0.015);
    }
  }

  ensureContext(): AudioContext {
    if (!this.ctx || this.ctx.state === 'closed') {
      let ctx: AudioContext;
      try {
        ctx = new AudioContext({ sampleRate: 48000 });
      } catch {
        ctx = new AudioContext();
      }
      const gain = ctx.createGain();
      gain.gain.value = this.muted ? 0 : 1;
      const analyser = ctx.createAnalyser();
      analyser.fftSize = 256;
      gain.connect(analyser);
      analyser.connect(ctx.destination);

      ctx.onstatechange = () => {
        this.callbacks.onSuspendedChange(ctx.state === 'suspended');
      };

      this.ctx = ctx;
      this.gain = gain;
      this.analyser = analyser;
      this.analyserData = new Uint8Array(new ArrayBuffer(analyser.fftSize));
    }

    if (this.ctx.state === 'suspended') {
      void this.ctx
        .resume()
        .then(() => {
          if (this.ctx) {
            this.callbacks.onSuspendedChange(this.ctx.state === 'suspended');
          }
        })
        .catch(() => {
          this.callbacks.onSuspendedChange(true);
        });
    } else {
      this.callbacks.onSuspendedChange(false);
    }

    return this.ctx;
  }

  async resumeNow(): Promise<boolean> {
    const ctx = this.ensureContext();
    if (ctx.state === 'suspended') {
      try {
        await ctx.resume();
      } catch {
        this.callbacks.onSuspendedChange(true);
        return false;
      }
    }
    const running = ctx.state === 'running';
    this.callbacks.onSuspendedChange(!running);
    return running;
  }

  beginTurn(sampleRate?: number): void {
    this.stop();
    if (sampleRate && sampleRate > 0) {
      this.sampleRate = sampleRate;
    }
    this.serverDone = false;
    this.underrunCount = 0;
    this.currentTurnChunks = [];
  }

  enqueueChunk(pcm: Float32Array): void {
    if (this.serverDone || pcm.length === 0) return;
    this.currentTurnChunks.push(pcm);

    const ctx = this.ensureContext();
    // If context is suspended or playback starved, accumulate prebuffer cleanly
    if (ctx.state === 'suspended') {
      this.prebuffer.push(pcm);
      this.prebufferSamples += pcm.length;
      this.callbacks.onSuspendedChange(true);
      return;
    }

    const targetSec = this.underrunCount > 0 ? REBUFFER_PREBUFFER_SEC : INITIAL_PREBUFFER_SEC;
    const targetSamples = Math.floor(this.sampleRate * targetSec);

    if (!this.playbackStarted) {
      this.prebuffer.push(pcm);
      this.prebufferSamples += pcm.length;
      if (this.prebufferSamples >= targetSamples) {
        this.flushPrebuffer();
      }
      return;
    }

    // Re-prebuffer on genuine starvation (all scheduled sources finished while turn still streaming)
    if (this.activeSources.size === 0 && this.nextPlayTime <= ctx.currentTime) {
      this.underrunCount += 1;
      this.callbacks.onUnderrun?.(this.underrunCount);
      this.playbackStarted = false;
      this.prebuffer.push(pcm);
      this.prebufferSamples += pcm.length;
      if (this.prebufferSamples >= Math.floor(this.sampleRate * REBUFFER_PREBUFFER_SEC)) {
        this.flushPrebuffer();
      }
      return;
    }

    this.schedulePcm(pcm, false);
  }

  finishTurn(cancelled: boolean, failed: boolean): void {
    this.serverDone = true;
    if (cancelled || failed) {
      this.stop();
      this.currentTurnChunks = [];
      return;
    }
    if (this.currentTurnChunks.length > 0) {
      const total = this.currentTurnChunks.reduce((acc, c) => acc + c.length, 0);
      const merged = new Float32Array(total);
      let off = 0;
      for (const c of this.currentTurnChunks) {
        merged.set(c, off);
        off += c.length;
      }
      this.lastReplayPcm = merged;
      this.lastReplaySampleRate = this.sampleRate;
      this.currentTurnChunks = [];
    }
    this.flushPrebuffer();
  }

  flushPrebuffer(): void {
    if (this.prebufferSamples === 0) return;
    const ctx = this.ensureContext();
    if (ctx.state === 'suspended') {
      this.callbacks.onSuspendedChange(true);
      return;
    }
    let merged: Float32Array;
    if (this.prebuffer.length === 1) {
      merged = this.prebuffer[0];
    } else {
      merged = new Float32Array(this.prebufferSamples);
      let offset = 0;
      for (const chunk of this.prebuffer) {
        merged.set(chunk, offset);
        offset += chunk.length;
      }
    }
    this.prebuffer = [];
    this.prebufferSamples = 0;
    this.playbackStarted = true;
    this.schedulePcm(merged, true);
  }

  replayLast(): boolean {
    if (!this.lastReplayPcm || this.lastReplayPcm.length === 0) return false;
    const ctx = this.ensureContext();
    if (ctx.state !== 'running') {
      this.callbacks.onSuspendedChange(true);
      return false;
    }
    this.stop();
    this.sampleRate = this.lastReplaySampleRate;
    this.serverDone = true;
    this.playbackStarted = true;
    this.schedulePcm(this.lastReplayPcm, true);
    return true;
  }

  private schedulePcm(pcm: Float32Array, applyFadeIn: boolean): void {
    if (pcm.length === 0) return;
    const ctx = this.ensureContext();
    const gain = this.gain;
    if (!gain) return;

    const now = ctx.currentTime;
    const hadGap = this.nextPlayTime <= now;
    const samples = applyFadeIn || hadGap ? this.withShortFadeIn(pcm) : pcm;
    const audioBuf = ctx.createBuffer(1, samples.length, this.sampleRate);
    audioBuf.copyToChannel(samples, 0);

    const src = ctx.createBufferSource();
    src.buffer = audioBuf;
    src.connect(gain);

    // Preserve exact future cursor (no artificial gaps); add 15ms lead only when starting fresh/underrun
    const startAt = hadGap ? now + SCHEDULE_LEAD_SEC : this.nextPlayTime;
    this.nextPlayTime = startAt + audioBuf.duration;

    const myGen = this.playbackGen;
    this.activeSources.add(src);
    src.onended = () => {
      try {
        src.disconnect();
      } catch {
        // ignore
      }
      this.activeSources.delete(src);
      if (myGen !== this.playbackGen) return;
      if (this.activeSources.size === 0 && this.serverDone) {
        this.callbacks.onPlaybackIdle();
      }
    };
    src.start(startAt);
  }

  private withShortFadeIn(pcm: Float32Array): Float32Array {
    const fadeSamples = Math.min(pcm.length, Math.floor(this.sampleRate * 0.008));
    if (fadeSamples <= 1) return pcm;
    const copy = new Float32Array(pcm);
    for (let i = 0; i < fadeSamples; i++) {
      copy[i] *= i / fadeSamples;
    }
    return copy;
  }

  sampleLevel(): number {
    if (!this.analyser || this.activeSources.size === 0) return 0;
    this.analyser.getByteTimeDomainData(this.analyserData);
    let sumSq = 0;
    for (let i = 0; i < this.analyserData.length; i++) {
      const v = (this.analyserData[i] - 128) / 128;
      sumSq += v * v;
    }
    return Math.min(1, Math.sqrt(sumSq / this.analyserData.length) * 4.2);
  }

  stop(): void {
    this.playbackGen += 1;
    this.prebuffer = [];
    this.prebufferSamples = 0;
    this.playbackStarted = false;
    this.serverDone = true;

    const sources = Array.from(this.activeSources);
    this.activeSources.clear();
    this.nextPlayTime = 0;

    if (sources.length === 0) return;

    if (this.gain && this.ctx && this.ctx.state === 'running') {
      const now = this.ctx.currentTime;
      // 10ms anti-pop ramp before stopping active sources
      this.gain.gain.cancelScheduledValues(now);
      this.gain.gain.setValueAtTime(this.gain.gain.value, now);
      this.gain.gain.linearRampToValueAtTime(0, now + 0.01);
      const restore = this.muted ? 0 : 1;
      this.gain.gain.setValueAtTime(restore, now + 0.014);
      for (const src of sources) {
        src.onended = null;
        try {
          src.stop(now + 0.012);
        } catch {
          try {
            src.disconnect();
          } catch {
            // ignore
          }
        }
      }
      return;
    }

    for (const src of sources) {
      src.onended = null;
      try {
        src.stop();
        src.disconnect();
      } catch {
        // already stopped
      }
    }
  }

  dispose(): void {
    this.stop();
    if (this.ctx && this.ctx.state !== 'closed') {
      void this.ctx.close();
    }
    this.ctx = null;
    this.gain = null;
    this.analyser = null;
  }
}
