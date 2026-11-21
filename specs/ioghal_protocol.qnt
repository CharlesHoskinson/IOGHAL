// Executable Quint specification of the IOGHAL WebSocket + Push-to-Talk + Concurrent Pipeline protocol.
//
// Models the real client (web/src/App.tsx + HalAudioPlayer) and server
// (src/hal9000/server.py + HalPipeline + OllamaBrain + MicRecorder) across N concurrent browser tabs.
//
// What it covers:
// - Client PTT ownership (hold, tap-to-latch, Escape cancel, blur/hide, disconnect, Reset, Replay)
// - Wire turn_id ownership vs server internal turn_gen cancellation
// - capture_id scoping for PipeWire local mic start/stop/limit/failure
// - Optional live web-search stage (WorkerSearch) with timeout/error fallback into LLM
// - Concurrent LLM producer + TTS consumer bounded queue (WorkerPipeline) with cancel/error drain
// - Overlapping retired workers still holding GPU lock after barge-in or disconnect + async TTS warmup
// - Model switching (set_model) & system_status broadcasts during idle/active turns
// - Invalid audio_b64 / turn-scoped error recovery without stuck-thinking deadlocks
// - Strict audio framing: EvAudioStart -> EvPcmChunk* -> EvAudioEnd per active turn
//
// What it does NOT cover:
// - WebAudio sample-level DSP math, resampling, or CUDA kernel timings
// - Exact Ollama token/sentence splitting, web search HTML parsing, or Whisper acoustic accuracy
//
// Update this spec and re-run `quint typecheck`, `quint test`, and `quint run` before changing the
// wire protocol, PTT state machine, or concurrent pipeline orchestration.

module ioghal_protocol {
  pure val CLIENTS: Set[int] = Set(1, 2)
  pure val MAX_TURN_ID: int = 8
  pure val MAX_QUEUE: int = 3

  type UiState = UiIdle | UiListening | UiThinking | UiSpeaking
  type MicMode = ModeLocal | ModeBrowser
  type Owner = NoOwner | SpaceOwner | PointerOwner | ToggleOwner

  type WorkerStage = WorkerIdle | WorkerStt | WorkerSearch | WorkerPipeline

  type GpuHolder =
    | GpuFree
    | GpuWarmup
    | GpuWorker({ clientId: int, gen: int })

  type C2SMsg =
    | MsgPttStart({ cid: int })
    | MsgPttStop({ tid: int, cid: int })
    | MsgAudioStreamStart({ cid: int, valid: bool })
    | MsgAudioStreamChunk({ cid: int, valid: bool })
    | MsgAudioStreamEnd({ tid: int, cid: int })
    | MsgAudioStreamCancel({ cid: int })
    | MsgAudioB64({ tid: int, valid: bool })
    | MsgText({ tid: int })
    | MsgInterrupt({ tid: int, silent: bool })
    | MsgReset({ tid: int })
    | MsgSetModel({ valid: bool })

  type S2CMsg =
    | EvSystemStatus({ ui: UiState })
    | EvState({ tid: int, cid: int, ui: UiState })
    | EvCaptureFailed({ cid: int })
    | EvCaptureLimit({ cid: int })
    | EvAudioStart({ tid: int })
    | EvPcmChunk({ tid: int })
    | EvAudioEnd({ tid: int, cancelled: bool, failed: bool })
    | EvCleared({ tid: int })
    | EvError({ tid: int })

  type ClientState = {
    connected: bool,
    uiState: UiState,
    micMode: MicMode,
    captureMode: MicMode,
    pttActive: bool,
    latched: bool,
    owner: Owner,
    activeTurnId: int,
    nextTurnId: int,
    activeCaptureId: int,
    nextCaptureId: int,
    pendingTurn: bool,
    playingTurnId: int,
    serverAudioDone: bool,
    hasReplay: bool,
    replaying: bool,
    audioSuspended: bool,
    audioOpenTid: int,
    audioEndedTid: int,
    protocolViolation: bool,
    lastClearedTurnId: int,
  }

  type ServerConn = {
    connected: bool,
    turnGen: int,
    historyGen: int,
    workerStage: WorkerStage,
    workerWireTid: int,
    workerGen: int,
    workerHistoryGen: int,
    workerAudioStarted: bool,
    llmStreamActive: bool,
    llmThinkingSelf: bool,
    llmDone: bool,
    llmFailed: bool,
    queuedSentences: int,
    ttsHoldingGpu: bool,
    sttHoldingGpu: bool,
    micRecording: bool,
    micCaptureId: int,
    micLimitHit: bool,
    streamCaptureActive: bool,
    streamCaptureId: int,
    streamValid: bool,
    lastSubmittedTid: int,
    historyPoisoned: bool,
  }

  type HardwareState = {
    micOwner: int,       // 0 = none, else ClientId
    gpuHolder: GpuHolder,
    warmupDone: bool,
  }

  type Coverage = {
    pttTurn: bool,
    latchedTurn: bool,
    browserTurn: bool,
    bargeIn: bool,
    resetThenTurn: bool,
    captureLimit: bool,
    errorRecovery: bool,
    disconnectCleanup: bool,
    webSearchTurn: bool,
    webSearchFallback: bool,
    pipelineOverlap: bool,
    setModelMidTurn: bool,
    invalidAudioRecovery: bool,
    replayAfterTurn: bool,
    warmupGpuContention: bool,
    retiredWorkerGpuHandoff: bool,
    thinkingSelfBargeIn: bool,
    emptyThinkingRecovery: bool,
    replaySuspendedGuard: bool,
    rapidDoublePtt: bool,
    disconnectDuringActivePipeline: bool,
    browserStreamTurn: bool,
    browserStreamBargeIn: bool,
    browserStreamRapidDouble: bool,
  }

  var clients: int -> ClientState
  var servers: int -> ServerConn
  var c2s: int -> List[C2SMsg]
  var s2c: int -> List[S2CMsg]
  var hw: HardwareState
  var cov: Coverage

  pure val initClient: ClientState = {
    connected: true,
    uiState: UiIdle,
    micMode: ModeLocal,
    captureMode: ModeLocal,
    pttActive: false,
    latched: false,
    owner: NoOwner,
    activeTurnId: 0,
    nextTurnId: 1,
    activeCaptureId: 0,
    nextCaptureId: 1,
    pendingTurn: false,
    playingTurnId: 0,
    serverAudioDone: true,
    hasReplay: false,
    replaying: false,
    audioSuspended: false,
    audioOpenTid: 0,
    audioEndedTid: 0,
    protocolViolation: false,
    lastClearedTurnId: 0,
  }

  pure val initServer: ServerConn = {
    connected: true,
    turnGen: 0,
    historyGen: 0,
    workerStage: WorkerIdle,
    workerWireTid: 0,
    workerGen: 0,
    workerHistoryGen: 0,
    workerAudioStarted: false,
    llmStreamActive: false,
    llmThinkingSelf: false,
    llmDone: true,
    llmFailed: false,
    queuedSentences: 0,
    ttsHoldingGpu: false,
    sttHoldingGpu: false,
    micRecording: false,
    micCaptureId: 0,
    micLimitHit: false,
    streamCaptureActive: false,
    streamCaptureId: 0,
    streamValid: true,
    lastSubmittedTid: 0,
    historyPoisoned: false,
  }

  pure val initCov: Coverage = {
    pttTurn: false,
    latchedTurn: false,
    browserTurn: false,
    bargeIn: false,
    resetThenTurn: false,
    captureLimit: false,
    errorRecovery: false,
    disconnectCleanup: false,
    webSearchTurn: false,
    webSearchFallback: false,
    pipelineOverlap: false,
    setModelMidTurn: false,
    invalidAudioRecovery: false,
    replayAfterTurn: false,
    warmupGpuContention: false,
    retiredWorkerGpuHandoff: false,
    thinkingSelfBargeIn: false,
    emptyThinkingRecovery: false,
    replaySuspendedGuard: false,
    rapidDoublePtt: false,
    disconnectDuringActivePipeline: false,
    browserStreamTurn: false,
    browserStreamBargeIn: false,
    browserStreamRapidDouble: false,
  }

  pure def isClientGpuHolder(h: GpuHolder, id: int): bool =
    match h {
      | GpuWorker(w) => w.clientId == id
      | _ => false
    }

  pure def isExactGpuHolder(h: GpuHolder, id: int, gen: int): bool =
    match h {
      | GpuWorker(w) => w.clientId == id and w.gen == gen
      | _ => false
    }

  pure def isRetiredGpuHolder(h: GpuHolder, srvs: int -> ServerConn): bool =
    match h {
      | GpuWorker(w) => {
          val s = srvs.get(w.clientId)
          not(s.connected) or s.workerStage == WorkerIdle or s.workerGen != w.gen
        }
      | _ => false
    }

  action init: bool = all {
    clients' = CLIENTS.mapBy(_ => initClient),
    servers' = CLIENTS.mapBy(_ => initServer),
    c2s' = CLIENTS.mapBy(_ => List()),
    s2c' = CLIENTS.mapBy(_ => List()),
    hw' = { micOwner: 0, gpuHolder: GpuFree, warmupDone: false },
    cov' = initCov,
  }

  // --- Background Runtime Actions ---

  action runtimeWarmupAcquire: bool = all {
    not(hw.warmupDone),
    hw.gpuHolder == GpuFree,
    hw' = { ...hw, gpuHolder: GpuWarmup },
    clients' = clients,
    servers' = servers,
    c2s' = c2s,
    s2c' = s2c,
    cov' = { ...cov, warmupGpuContention: true },
  }

  action runtimeWarmupRelease: bool = all {
    hw.gpuHolder == GpuWarmup,
    hw' = { ...hw, gpuHolder: GpuFree, warmupDone: true },
    clients' = clients,
    servers' = servers,
    c2s' = c2s,
    s2c' = s2c,
    cov' = cov,
  }

  // A retired worker (superseded by barge-in, interrupt, reset, or disconnect) eventually
  // exits its `with gpu_lock:` block and releases the GPU without emitting any wire events.
  action retiredWorkerReleaseGpu: bool = all {
    isRetiredGpuHolder(hw.gpuHolder, servers),
    hw' = { ...hw, gpuHolder: GpuFree },
    clients' = clients,
    servers' = servers,
    c2s' = c2s,
    s2c' = s2c,
    cov' = { ...cov, retiredWorkerGpuHandoff: true },
  }

  // --- Client Actions (faithful to web/src/App.tsx) ---

  action clientBeginPtt(id: int, own: Owner): bool = {
    val c = clients.get(id)
    val nextTid = c.nextTurnId
    val cid = c.nextCaptureId
    val isToggle = own == ToggleOwner
    val msg = if (c.micMode == ModeBrowser) {
      MsgInterrupt({ tid: 0, silent: true })
    } else {
      MsgPttStart({ cid: cid })
    }
    all {
      c.connected,
      not(c.pttActive),
      c.nextTurnId < MAX_TURN_ID,
      c.nextCaptureId < MAX_TURN_ID,
      clients' = clients.put(id, {
        ...c,
        uiState: UiListening,
        captureMode: c.micMode,
        pttActive: true,
        latched: isToggle,
        owner: own,
        activeTurnId: nextTid,
        nextTurnId: nextTid + 1,
        activeCaptureId: cid,
        nextCaptureId: cid + 1,
        pendingTurn: false,
        playingTurnId: 0,
        serverAudioDone: true,
        replaying: false,
        audioOpenTid: 0,
      }),
      c2s' = c2s.put(id, c2s.get(id).append(msg)),
      servers' = servers,
      s2c' = s2c,
      hw' = hw,
      cov' = { ...cov, bargeIn: cov.bargeIn or c.playingTurnId != 0 or c.pendingTurn },
    }
  }

  // Models a rapid duplicate MsgPttStart arriving on the wire while already recording locally
  // (e.g., scripted client or reconnect/race). Server must restart mic cleanly & clear micLimitHit.
  // Models live browser mic streaming start (`audio_stream_start`), incremental binary PCM frames
  // (`audio_stream_chunk`), rapid double-start, and near-0ms `audio_stream_end` release.
  action clientBrowserStreamStart(id: int, validStart: bool): bool = {
    val c = clients.get(id)
    val cid = c.activeCaptureId
    all {
      c.connected,
      c.pttActive,
      c.captureMode == ModeBrowser,
      cid > 0,
      c2s.get(id).length() < 6,
      c2s' = c2s.put(id, c2s.get(id).append(MsgAudioStreamStart({ cid: cid, valid: validStart }))),
      clients' = clients,
      servers' = servers,
      s2c' = s2c,
      hw' = hw,
      cov' = cov,
    }
  }

  action clientBrowserStreamChunk(id: int, cid: int, validChunk: bool): bool = {
    val c = clients.get(id)
    all {
      c.connected,
      c.pttActive,
      c.captureMode == ModeBrowser,
      cid > 0,
      c2s.get(id).length() < 6,
      c2s' = c2s.put(id, c2s.get(id).append(MsgAudioStreamChunk({ cid: cid, valid: validChunk }))),
      clients' = clients,
      servers' = servers,
      s2c' = s2c,
      hw' = hw,
      cov' = cov,
    }
  }

  action clientFinishBrowserStream(id: int): bool = {
    val c = clients.get(id)
    val tid = c.nextTurnId
    val cid = c.activeCaptureId
    all {
      c.connected,
      c.pttActive,
      c.captureMode == ModeBrowser,
      c.nextTurnId < MAX_TURN_ID,
      clients' = clients.put(id, {
        ...c,
        uiState: UiThinking,
        pttActive: false,
        latched: false,
        owner: NoOwner,
        activeTurnId: tid,
        nextTurnId: tid + 1,
        activeCaptureId: 0,
        pendingTurn: true,
        audioOpenTid: 0,
      }),
      c2s' = c2s.put(id, c2s.get(id).append(MsgAudioStreamEnd({ tid: tid, cid: cid }))),
      servers' = servers,
      s2c' = s2c,
      hw' = hw,
      cov' = {
        ...cov,
        pttTurn: true,
        browserTurn: true,
        browserStreamTurn: true,
        resetThenTurn: cov.resetThenTurn or (c.lastClearedTurnId > 0),
      },
    }
  }

  action clientRapidDoublePtt(id: int): bool = {
    val c = clients.get(id)
    val cid = c.nextCaptureId
    all {
      c.connected,
      c.pttActive,
      c.captureMode == ModeLocal,
      c.nextCaptureId < MAX_TURN_ID,
      clients' = clients.put(id, {
        ...c,
        activeCaptureId: cid,
        nextCaptureId: cid + 1,
      }),
      c2s' = c2s.put(id, c2s.get(id).append(MsgPttStart({ cid: cid }))),
      servers' = servers,
      s2c' = s2c,
      hw' = hw,
      cov' = cov,
    }
  }

  action clientTapLatch(id: int): bool = {
    val c = clients.get(id)
    all {
      c.connected,
      c.pttActive,
      not(c.latched),
      c.owner == SpaceOwner or c.owner == PointerOwner,
      clients' = clients.put(id, { ...c, latched: true, owner: ToggleOwner }),
      servers' = servers,
      c2s' = c2s,
      s2c' = s2c,
      hw' = hw,
      cov' = { ...cov, latchedTurn: true },
    }
  }

  action clientFinishPtt(id: int, validUpload: bool): bool = {
    val c = clients.get(id)
    val tid = c.nextTurnId
    val cid = c.activeCaptureId
    val msg = if (c.captureMode == ModeBrowser) {
      MsgAudioB64({ tid: tid, valid: validUpload })
    } else {
      MsgPttStop({ tid: tid, cid: cid })
    }
    all {
      c.connected,
      c.pttActive,
      c.nextTurnId < MAX_TURN_ID,
      clients' = clients.put(id, {
        ...c,
        uiState: UiThinking,
        pttActive: false,
        latched: false,
        owner: NoOwner,
        activeTurnId: tid,
        nextTurnId: tid + 1,
        activeCaptureId: 0,
        pendingTurn: true,
        audioOpenTid: 0,
      }),
      c2s' = c2s.put(id, c2s.get(id).append(msg)),
      servers' = servers,
      s2c' = s2c,
      hw' = hw,
      cov' = {
        ...cov,
        pttTurn: true,
        browserTurn: cov.browserTurn or (c.captureMode == ModeBrowser),
        resetThenTurn: cov.resetThenTurn or (c.lastClearedTurnId > 0),
      },
    }
  }

  action clientSubmitText(id: int): bool = {
    val c = clients.get(id)
    val tid = c.nextTurnId
    all {
      c.connected,
      not(c.pttActive),
      c.nextTurnId < MAX_TURN_ID,
      clients' = clients.put(id, {
        ...c,
        uiState: UiThinking,
        activeTurnId: tid,
        nextTurnId: tid + 1,
        pendingTurn: true,
        playingTurnId: 0,
        serverAudioDone: true,
        replaying: false,
        audioOpenTid: 0,
      }),
      c2s' = c2s.put(id, c2s.get(id).append(MsgText({ tid: tid }))),
      servers' = servers,
      s2c' = s2c,
      hw' = hw,
      cov' = {
        ...cov,
        bargeIn: cov.bargeIn or (c.playingTurnId != 0),
        resetThenTurn: cov.resetThenTurn or (c.lastClearedTurnId > 0),
      },
    }
  }

  action clientCancelEscape(id: int): bool = {
    val c = clients.get(id)
    val tid = c.nextTurnId
    all {
      c.connected,
      c.pttActive or c.pendingTurn or c.playingTurnId != 0 or not(c.serverAudioDone) or c.replaying,
      c.nextTurnId < MAX_TURN_ID,
      clients' = clients.put(id, {
        ...c,
        uiState: UiIdle,
        pttActive: false,
        latched: false,
        owner: NoOwner,
        activeTurnId: tid,
        nextTurnId: tid + 1,
        activeCaptureId: 0,
        pendingTurn: false,
        playingTurnId: 0,
        serverAudioDone: true,
        replaying: false,
        audioOpenTid: 0,
      }),
      c2s' = c2s.put(id, c2s.get(id).append(MsgInterrupt({ tid: tid, silent: false }))),
      servers' = servers,
      s2c' = s2c,
      hw' = hw,
      cov' = { ...cov, bargeIn: true },
    }
  }

  action clientReset(id: int): bool = {
    val c = clients.get(id)
    val tid = c.nextTurnId
    all {
      c.connected,
      c.nextTurnId < MAX_TURN_ID,
      clients' = clients.put(id, {
        ...c,
        uiState: UiIdle,
        pttActive: false,
        latched: false,
        owner: NoOwner,
        activeTurnId: tid,
        nextTurnId: tid + 1,
        activeCaptureId: 0,
        pendingTurn: false,
        playingTurnId: 0,
        serverAudioDone: true,
        replaying: false,
        audioOpenTid: 0,
        lastClearedTurnId: tid,
      }),
      c2s' = c2s.put(id, c2s.get(id).append(MsgReset({ tid: tid }))),
      servers' = servers,
      s2c' = s2c,
      hw' = hw,
      cov' = cov,
    }
  }

  action clientSetModel(id: int, validModel: bool): bool = {
    val c = clients.get(id)
    all {
      c.connected,
      c2s.get(id).length() < 4,
      c2s' = c2s.put(id, c2s.get(id).append(MsgSetModel({ valid: validModel }))),
      clients' = clients,
      servers' = servers,
      s2c' = s2c,
      hw' = hw,
      cov' = {
        ...cov,
        setModelMidTurn: cov.setModelMidTurn or c.pendingTurn or c.pttActive or c.playingTurnId != 0,
      },
    }
  }

  action clientToggleMicMode(id: int): bool = {
    val c = clients.get(id)
    val nextMode = if (c.micMode == ModeLocal) ModeBrowser else ModeLocal
    all {
      c.connected,
      not(c.pttActive),
      clients' = clients.put(id, { ...c, micMode: nextMode }),
      servers' = servers,
      c2s' = c2s,
      s2c' = s2c,
      hw' = hw,
      cov' = cov,
    }
  }

  // Browser autoplay policy may suspend WebAudio when idle; clicking Enable Audio resumes it.
  action clientSetAudioSuspended(id: int, suspended: bool): bool = {
    val c = clients.get(id)
    all {
      c.connected,
      not(c.replaying),
      c.playingTurnId == 0,
      c.audioSuspended != suspended,
      clients' = clients.put(id, { ...c, audioSuspended: suspended }),
      servers' = servers,
      c2s' = c2s,
      s2c' = s2c,
      hw' = hw,
      cov' = { ...cov, replaySuspendedGuard: cov.replaySuspendedGuard or (suspended and c.hasReplay) },
    }
  }

  // Guarded replay action: only allowed when idle AND AudioContext is running (not suspended,
  // fixing Bug #5a in App.tsx/halClient.ts where replayLast() with a suspended AudioContext
  // left activeSources > 0 and stuck the UI in UiSpeaking).
  action clientStartReplay(id: int): bool = {
    val c = clients.get(id)
    all {
      c.connected,
      c.hasReplay,
      not(c.audioSuspended),
      not(c.pttActive),
      not(c.pendingTurn),
      c.playingTurnId == 0,
      c.uiState == UiIdle,
      clients' = clients.put(id, {
        ...c,
        uiState: UiSpeaking,
        replaying: true,
        serverAudioDone: true,
      }),
      servers' = servers,
      c2s' = c2s,
      s2c' = s2c,
      hw' = hw,
      cov' = { ...cov, replayAfterTurn: true },
    }
  }

  action clientPlaybackDrain(id: int): bool = {
    val c = clients.get(id)
    all {
      c.connected,
      (c.playingTurnId != 0 or c.replaying),
      c.serverAudioDone,
      clients' = clients.put(id, {
        ...c,
        playingTurnId: 0,
        replaying: false,
        pendingTurn: if (c.pttActive or c.replaying) c.pendingTurn else false,
        uiState: if (c.pttActive) c.uiState else UiIdle,
      }),
      servers' = servers,
      c2s' = c2s,
      s2c' = s2c,
      hw' = hw,
      cov' = cov,
    }
  }

  action clientReceiveS2C(id: int): bool = {
    val c = clients.get(id)
    val q = s2c.get(id)
    all {
      c.connected,
      q.length() > 0,
      val msg = q.head()
      val rest = q.tail()
      val nextC = match msg {
        | EvSystemStatus(p) => {
            if (not(c.pttActive) and not(c.pendingTurn) and c.playingTurnId == 0 and not(c.replaying)) {
              ...c,
              uiState: p.ui,
            } else c
          }
        | EvState(p) => {
            val staleT = p.tid != 0 and p.tid != c.activeTurnId
            val staleC = p.cid != 0 and p.cid != c.activeCaptureId
            val blockIdle = p.ui == UiIdle and (c.pttActive or c.playingTurnId != 0 or c.replaying)
            if (staleT or staleC or blockIdle) c
            else {
              ...c,
              uiState: p.ui,
              pendingTurn: if (p.ui == UiIdle) false else c.pendingTurn,
            }
          }
        | EvCaptureFailed(p) => {
            if (p.cid != 0 and p.cid != c.activeCaptureId) c
            else {
              ...c,
              pttActive: false,
              latched: false,
              owner: NoOwner,
              activeCaptureId: 0,
              pendingTurn: false,
              uiState: UiIdle,
            }
          }
        | EvCaptureLimit(p) => {
            if (p.cid != 0 and p.cid != c.activeCaptureId) c
            else if (c.pttActive and c.captureMode == ModeLocal and c.nextTurnId < MAX_TURN_ID) {
              ...c,
              uiState: UiThinking,
              pttActive: false,
              latched: false,
              owner: NoOwner,
              activeTurnId: c.nextTurnId,
              nextTurnId: c.nextTurnId + 1,
              activeCaptureId: 0,
              pendingTurn: true,
              audioOpenTid: 0,
            } else c
          }
        | EvAudioStart(p) => {
            if (p.tid != c.activeTurnId) c
            else {
              val bad = c.audioOpenTid == p.tid or c.audioEndedTid == p.tid
              {
                ...c,
                playingTurnId: 0,
                replaying: false,
                serverAudioDone: false,
                audioOpenTid: p.tid,
                protocolViolation: c.protocolViolation or bad,
              }
            }
          }
        | EvPcmChunk(p) => {
            if (p.tid != c.activeTurnId) c
            else {
              val bad = c.audioOpenTid != p.tid or c.serverAudioDone
              {
                ...c,
                playingTurnId: p.tid,
                replaying: false,
                uiState: UiSpeaking,
                protocolViolation: c.protocolViolation or bad,
              }
            }
          }
        | EvAudioEnd(p) => {
            if (p.tid != c.activeTurnId) c
            else {
              val bad = c.audioOpenTid != p.tid
              val stopPlay = p.cancelled or p.failed
              val nextPlaying = if (stopPlay) 0 else c.playingTurnId
              val goIdle = nextPlaying == 0 and not(c.pttActive)
              val completedAudio = not(p.cancelled) and not(p.failed) and c.playingTurnId == p.tid
              {
                ...c,
                serverAudioDone: true,
                playingTurnId: nextPlaying,
                hasReplay: c.hasReplay or completedAudio,
                audioOpenTid: 0,
                audioEndedTid: p.tid,
                protocolViolation: c.protocolViolation or bad,
                pendingTurn: if (goIdle) false else c.pendingTurn,
                uiState: if (goIdle) UiIdle else c.uiState,
              }
            }
          }
        | EvCleared(_) => c
        | EvError(p) => {
            // Turn-scoped errors (p.tid == activeTurnId) clear pendingTurn and return to UiIdle
            // unless recording or playing; non-turn errors (p.tid == 0, e.g., bad set_model) do NOT
            // clobber an active turn!
            if (p.tid == 0) c
            else if (p.tid != c.activeTurnId) c
            else {
              ...c,
              pendingTurn: false,
              playingTurnId: 0,
              replaying: false,
              serverAudioDone: true,
              audioOpenTid: 0,
              uiState: if (c.pttActive) c.uiState else UiIdle,
            }
          }
      }
      val extraC2S = match msg {
        | EvCaptureLimit(p) =>
            if ((p.cid == 0 or p.cid == c.activeCaptureId) and c.pttActive and c.captureMode == ModeLocal and c.nextTurnId < MAX_TURN_ID)
              List(MsgPttStop({ tid: c.nextTurnId, cid: c.activeCaptureId }))
            else List()
        | _ => List()
      }
      all {
        clients' = clients.put(id, nextC),
        s2c' = s2c.put(id, rest),
        c2s' = c2s.put(id, c2s.get(id).concat(extraC2S)),
        servers' = servers,
        hw' = hw,
        cov' = cov,
      }
    }
  }

  action clientDisconnect(id: int): bool = {
    val c = clients.get(id)
    val s = servers.get(id)
    all {
      c.connected,
      clients' = clients.put(id, { ...initClient, connected: false }),
      // Note: gpuHolder is NOT forcibly zeroed on disconnect! If a worker for `id` currently
      // holds the GPU lock, it becomes a retired GPU holder and releases via `retiredWorkerReleaseGpu`.
      servers' = servers.put(id, {
        ...s,
        connected: false,
        turnGen: s.turnGen + 1,
        workerStage: WorkerIdle,
        llmStreamActive: false,
        llmThinkingSelf: false,
        llmDone: true,
        llmFailed: false,
        queuedSentences: 0,
        ttsHoldingGpu: false,
        sttHoldingGpu: false,
        micRecording: false,
        micCaptureId: 0,
        micLimitHit: false,
        streamCaptureActive: false,
        streamCaptureId: 0,
        streamValid: true,
      }),
      c2s' = c2s.put(id, List()),
      s2c' = s2c.put(id, List()),
      hw' = {
        ...hw,
        micOwner: if (hw.micOwner == id) 0 else hw.micOwner,
      },
      cov' = {
        ...cov,
        disconnectCleanup: true,
        disconnectDuringActivePipeline:
          cov.disconnectDuringActivePipeline or
          (s.workerStage == WorkerPipeline and (s.ttsHoldingGpu or s.llmStreamActive)),
      },
    }
  }

  action clientReconnect(id: int): bool = {
    val c = clients.get(id)
    all {
      not(c.connected),
      // Wait for any retired GPU worker from the previous socket on `id` to finish releasing GPU
      not(isClientGpuHolder(hw.gpuHolder, id)),
      clients' = clients.put(id, initClient),
      servers' = servers.put(id, initServer),
      c2s' = c2s.put(id, List()),
      s2c' = s2c.put(id, List()),
      hw' = hw,
      cov' = cov,
    }
  }

  // --- Server Actions (faithful to src/hal9000/server.py + pipeline.py) ---

  action serverProcessC2S(id: int): bool = {
    val s = servers.get(id)
    val q = c2s.get(id)
    all {
      s.connected,
      q.length() > 0,
      val msg = q.head()
      val rest = q.tail()
      match msg {
        | MsgPttStart(p) => {
            val canAcquire = hw.micOwner == 0 or hw.micOwner == id
            val wasThinking = s.llmThinkingSelf
            if (canAcquire) all {
              servers' = servers.put(id, {
                ...s,
                turnGen: s.turnGen + 1,
                workerStage: WorkerIdle,
                llmStreamActive: false,
                llmThinkingSelf: false,
                ttsHoldingGpu: false,
                sttHoldingGpu: false,
                queuedSentences: 0,
                llmDone: true,
                llmFailed: false,
                micRecording: true,
                micCaptureId: p.cid,
                micLimitHit: false,
                streamCaptureActive: false,
                streamCaptureId: 0,
                streamValid: true,
              }),
              hw' = { ...hw, micOwner: id },
              s2c' = s2c.put(id, s2c.get(id).append(EvState({ tid: 0, cid: p.cid, ui: UiListening }))),
              c2s' = c2s.put(id, rest),
              clients' = clients,
              cov' = {
                ...cov,
                thinkingSelfBargeIn: cov.thinkingSelfBargeIn or wasThinking,
                rapidDoublePtt: cov.rapidDoublePtt or s.micRecording,
              },
            } else all {
              servers' = servers.put(id, {
                ...s,
                turnGen: s.turnGen + 1,
                workerStage: WorkerIdle,
                llmStreamActive: false,
                llmThinkingSelf: false,
                ttsHoldingGpu: false,
                sttHoldingGpu: false,
                queuedSentences: 0,
                llmDone: true,
                llmFailed: false,
              }),
              hw' = hw,
              s2c' = s2c.put(id, s2c.get(id).append(EvCaptureFailed({ cid: p.cid }))),
              c2s' = c2s.put(id, rest),
              clients' = clients,
              cov' = { ...cov, thinkingSelfBargeIn: cov.thinkingSelfBargeIn or wasThinking },
            }
          }
        | MsgPttStop(p) => {
            if (not(s.micRecording)) all {
              servers' = servers,
              hw' = hw,
              s2c' = s2c.put(id, s2c.get(id).append(EvState({ tid: p.tid, cid: 0, ui: UiIdle }))),
              c2s' = c2s.put(id, rest),
              clients' = clients,
              cov' = cov,
            } else {
              val nextGen = s.turnGen + 1
              all {
                servers' = servers.put(id, {
                  ...s,
                  turnGen: nextGen,
                  micRecording: false,
                  micCaptureId: 0,
                  micLimitHit: false,
                  workerStage: WorkerStt,
                  workerWireTid: p.tid,
                  workerGen: nextGen,
                  workerHistoryGen: s.historyGen,
                  workerAudioStarted: false,
                  llmStreamActive: false,
                  llmThinkingSelf: false,
                  llmDone: false,
                  llmFailed: false,
                  queuedSentences: 0,
                  ttsHoldingGpu: false,
                  sttHoldingGpu: false,
                  lastSubmittedTid: p.tid,
                }),
                hw' = { ...hw, micOwner: if (hw.micOwner == id) 0 else hw.micOwner },
                s2c' = s2c,
                c2s' = c2s.put(id, rest),
                clients' = clients,
                cov' = cov,
              }
            }
          }
        | MsgAudioStreamStart(p) => {
            val wasThinking = s.llmThinkingSelf
            val wasStreamActive = s.streamCaptureActive
            if (not(p.valid)) all {
              servers' = servers.put(id, {
                ...s,
                streamCaptureActive: false,
                streamCaptureId: 0,
                streamValid: true,
              }),
              hw' = hw,
              s2c' = s2c.put(id, s2c.get(id).append(EvCaptureFailed({ cid: p.cid }))),
              c2s' = c2s.put(id, rest),
              clients' = clients,
              cov' = cov,
            } else all {
              servers' = servers.put(id, {
                ...s,
                turnGen: s.turnGen + 1,
                workerStage: WorkerIdle,
                llmStreamActive: false,
                llmThinkingSelf: false,
                ttsHoldingGpu: false,
                sttHoldingGpu: false,
                queuedSentences: 0,
                llmDone: true,
                llmFailed: false,
                micRecording: false,
                micCaptureId: 0,
                micLimitHit: false,
                streamCaptureActive: true,
                streamCaptureId: p.cid,
                streamValid: true,
              }),
              hw' = { ...hw, micOwner: if (hw.micOwner == id) 0 else hw.micOwner },
              s2c' = s2c,
              c2s' = c2s.put(id, rest),
              clients' = clients,
              cov' = {
                ...cov,
                thinkingSelfBargeIn: cov.thinkingSelfBargeIn or wasThinking,
                browserStreamRapidDouble: cov.browserStreamRapidDouble or wasStreamActive,
              },
            }
          }
        | MsgAudioStreamChunk(p) => {
            val matched = s.streamCaptureActive and s.streamCaptureId == p.cid
            all {
              servers' = if (matched) servers.put(id, {
                ...s,
                streamValid: s.streamValid and p.valid,
              }) else servers,
              hw' = hw,
              s2c' = s2c,
              c2s' = c2s.put(id, rest),
              clients' = clients,
              cov' = cov,
            }
          }
        | MsgAudioStreamCancel(p) => {
            val shouldClear = s.streamCaptureActive and (p.cid == 0 or s.streamCaptureId == p.cid)
            all {
              servers' = if (shouldClear) servers.put(id, {
                ...s,
                streamCaptureActive: false,
                streamCaptureId: 0,
                streamValid: true,
              }) else servers,
              hw' = hw,
              s2c' = s2c,
              c2s' = c2s.put(id, rest),
              clients' = clients,
              cov' = cov,
            }
          }
        | MsgAudioStreamEnd(p) => {
            val matched = s.streamCaptureActive and s.streamCaptureId == p.cid
            if (not(matched)) all {
              servers' = servers.put(id, {
                ...s,
                streamCaptureActive: false,
                streamCaptureId: 0,
                streamValid: true,
              }),
              hw' = hw,
              s2c' = s2c.put(id, s2c.get(id).append(EvState({ tid: p.tid, cid: 0, ui: UiIdle }))),
              c2s' = c2s.put(id, rest),
              clients' = clients,
              cov' = cov,
            } else if (not(s.streamValid)) all {
              servers' = servers.put(id, {
                ...s,
                turnGen: s.turnGen + 1,
                workerStage: WorkerIdle,
                llmStreamActive: false,
                llmThinkingSelf: false,
                ttsHoldingGpu: false,
                sttHoldingGpu: false,
                queuedSentences: 0,
                llmDone: true,
                llmFailed: false,
                streamCaptureActive: false,
                streamCaptureId: 0,
                streamValid: true,
              }),
              hw' = hw,
              s2c' = s2c.put(id, s2c.get(id).concat(List(
                EvError({ tid: p.tid }),
                EvState({ tid: p.tid, cid: 0, ui: UiIdle }),
              ))),
              c2s' = c2s.put(id, rest),
              clients' = clients,
              cov' = { ...cov, invalidAudioRecovery: true },
            } else {
              val nextGen = s.turnGen + 1
              all {
                servers' = servers.put(id, {
                  ...s,
                  turnGen: nextGen,
                  micRecording: false,
                  micCaptureId: 0,
                  micLimitHit: false,
                  streamCaptureActive: false,
                  streamCaptureId: 0,
                  streamValid: true,
                  workerStage: WorkerStt,
                  workerWireTid: p.tid,
                  workerGen: nextGen,
                  workerHistoryGen: s.historyGen,
                  workerAudioStarted: false,
                  llmStreamActive: false,
                  llmThinkingSelf: false,
                  llmDone: false,
                  llmFailed: false,
                  queuedSentences: 0,
                  ttsHoldingGpu: false,
                  sttHoldingGpu: false,
                  lastSubmittedTid: p.tid,
                }),
                hw' = { ...hw, micOwner: if (hw.micOwner == id) 0 else hw.micOwner },
                s2c' = s2c,
                c2s' = c2s.put(id, rest),
                clients' = clients,
                cov' = { ...cov, browserStreamTurn: true },
              }
            }
          }
        | MsgAudioB64(p) => {
            if (not(p.valid)) all {
              // Invalid upload: emit turn-scoped error AND idle state so client never sticks in UiThinking
              servers' = servers.put(id, {
                ...s,
                turnGen: s.turnGen + 1,
                workerStage: WorkerIdle,
                llmStreamActive: false,
                llmThinkingSelf: false,
                ttsHoldingGpu: false,
                sttHoldingGpu: false,
                queuedSentences: 0,
                llmDone: true,
                llmFailed: false,
                micRecording: false,
                micCaptureId: 0,
                micLimitHit: false,
                streamCaptureActive: false,
                streamCaptureId: 0,
                streamValid: true,
              }),
              hw' = { ...hw, micOwner: if (hw.micOwner == id) 0 else hw.micOwner },
              s2c' = s2c.put(id, s2c.get(id).concat(List(
                EvError({ tid: p.tid }),
                EvState({ tid: p.tid, cid: 0, ui: UiIdle }),
              ))),
              c2s' = c2s.put(id, rest),
              clients' = clients,
              cov' = { ...cov, invalidAudioRecovery: true },
            } else {
              val nextGen = s.turnGen + 1
              all {
                servers' = servers.put(id, {
                  ...s,
                  turnGen: nextGen,
                  micRecording: false,
                  micCaptureId: 0,
                  micLimitHit: false,
                  streamCaptureActive: false,
                  streamCaptureId: 0,
                  streamValid: true,
                  workerStage: WorkerStt,
                  workerWireTid: p.tid,
                  workerGen: nextGen,
                  workerHistoryGen: s.historyGen,
                  workerAudioStarted: false,
                  llmStreamActive: false,
                  llmThinkingSelf: false,
                  llmDone: false,
                  llmFailed: false,
                  queuedSentences: 0,
                  ttsHoldingGpu: false,
                  sttHoldingGpu: false,
                  lastSubmittedTid: p.tid,
                }),
                hw' = { ...hw, micOwner: if (hw.micOwner == id) 0 else hw.micOwner },
                s2c' = s2c,
                c2s' = c2s.put(id, rest),
                clients' = clients,
                cov' = cov,
              }
            }
          }
        | MsgText(p) => {
            val nextGen = s.turnGen + 1
            val wasThinking = s.llmThinkingSelf
            all {
              servers' = servers.put(id, {
                ...s,
                turnGen: nextGen,
                micRecording: false,
                micCaptureId: 0,
                micLimitHit: false,
                streamCaptureActive: false,
                streamCaptureId: 0,
                streamValid: true,
                workerStage: WorkerSearch,
                workerWireTid: p.tid,
                workerGen: nextGen,
                workerHistoryGen: s.historyGen,
                workerAudioStarted: false,
                llmStreamActive: false,
                llmThinkingSelf: false,
                llmDone: false,
                llmFailed: false,
                queuedSentences: 0,
                ttsHoldingGpu: false,
                sttHoldingGpu: false,
                lastSubmittedTid: p.tid,
              }),
              hw' = { ...hw, micOwner: if (hw.micOwner == id) 0 else hw.micOwner },
              s2c' = s2c,
              c2s' = c2s.put(id, rest),
              clients' = clients,
              cov' = { ...cov, thinkingSelfBargeIn: cov.thinkingSelfBargeIn or wasThinking },
            }
          }
        | MsgInterrupt(p) => {
            val outEv = if (p.silent) List() else List(EvState({ tid: p.tid, cid: 0, ui: UiIdle }))
            val wasThinking = s.llmThinkingSelf
            val wasStreaming = s.streamCaptureActive
            all {
              servers' = servers.put(id, {
                ...s,
                turnGen: s.turnGen + 1,
                workerStage: WorkerIdle,
                llmStreamActive: false,
                llmThinkingSelf: false,
                ttsHoldingGpu: false,
                sttHoldingGpu: false,
                queuedSentences: 0,
                llmDone: true,
                llmFailed: false,
                micRecording: false,
                micCaptureId: 0,
                micLimitHit: false,
                streamCaptureActive: false,
                streamCaptureId: 0,
                streamValid: true,
              }),
              hw' = { ...hw, micOwner: if (hw.micOwner == id) 0 else hw.micOwner },
              s2c' = s2c.put(id, s2c.get(id).concat(outEv)),
              c2s' = c2s.put(id, rest),
              clients' = clients,
              cov' = {
                ...cov,
                thinkingSelfBargeIn: cov.thinkingSelfBargeIn or wasThinking,
                browserStreamBargeIn: cov.browserStreamBargeIn or wasStreaming,
              },
            }
          }
        | MsgReset(p) => {
            val wasThinking = s.llmThinkingSelf
            all {
              servers' = servers.put(id, {
                ...s,
                turnGen: s.turnGen + 1,
                historyGen: s.historyGen + 1,
                workerStage: WorkerIdle,
                llmStreamActive: false,
                llmThinkingSelf: false,
                ttsHoldingGpu: false,
                sttHoldingGpu: false,
                queuedSentences: 0,
                llmDone: true,
                llmFailed: false,
                micRecording: false,
                micCaptureId: 0,
                micLimitHit: false,
                streamCaptureActive: false,
                streamCaptureId: 0,
                streamValid: true,
              }),
              hw' = { ...hw, micOwner: if (hw.micOwner == id) 0 else hw.micOwner },
              s2c' = s2c.put(id, s2c.get(id).append(EvCleared({ tid: p.tid }))),
              c2s' = c2s.put(id, rest),
              clients' = clients,
              cov' = { ...cov, thinkingSelfBargeIn: cov.thinkingSelfBargeIn or wasThinking },
            }
          }
        | MsgSetModel(p) => {
            val outEv = if (p.valid) EvSystemStatus({ ui: UiIdle }) else EvError({ tid: 0 })
            all {
              servers' = servers,
              hw' = hw,
              s2c' = s2c.put(id, s2c.get(id).append(outEv)),
              c2s' = c2s.put(id, rest),
              clients' = clients,
              cov' = cov,
            }
          }
      }
    }
  }

  action serverMicHitLimit(id: int): bool = {
    val s = servers.get(id)
    all {
      s.connected,
      s.micRecording,
      not(s.micLimitHit),
      servers' = servers.put(id, { ...s, micLimitHit: true }),
      s2c' = s2c.put(id, s2c.get(id).append(EvCaptureLimit({ cid: s.micCaptureId }))),
      clients' = clients,
      c2s' = c2s,
      hw' = hw,
      cov' = { ...cov, captureLimit: true },
    }
  }

  // --- Worker Actions: STT (2-phase GPU acquire/release) ---

  action workerSttAcquireGpu(id: int): bool = {
    val s = servers.get(id)
    all {
      s.connected,
      s.workerStage == WorkerStt,
      not(s.sttHoldingGpu),
      s.workerGen == s.turnGen,
      hw.gpuHolder == GpuFree,
      servers' = servers.put(id, { ...s, sttHoldingGpu: true }),
      hw' = { ...hw, gpuHolder: GpuWorker({ clientId: id, gen: s.workerGen }) },
      clients' = clients,
      c2s' = c2s,
      s2c' = s2c,
      cov' = cov,
    }
  }

  action workerSttComplete(id: int, speechFound: bool): bool = {
    val s = servers.get(id)
    all {
      s.connected,
      s.workerStage == WorkerStt,
      s.sttHoldingGpu,
      isExactGpuHolder(hw.gpuHolder, id, s.workerGen),
      if (speechFound) all {
        servers' = servers.put(id, { ...s, workerStage: WorkerSearch, sttHoldingGpu: false }),
        s2c' = s2c.put(id, s2c.get(id).append(EvState({ tid: s.workerWireTid, cid: 0, ui: UiThinking }))),
      } else all {
        servers' = servers.put(id, { ...s, workerStage: WorkerIdle, sttHoldingGpu: false, llmDone: true }),
        s2c' = s2c.put(id, s2c.get(id).append(EvState({ tid: s.workerWireTid, cid: 0, ui: UiIdle }))),
      },
      hw' = { ...hw, gpuHolder: GpuFree },
      clients' = clients,
      c2s' = c2s,
      cov' = cov,
    }
  }

  // --- Worker Actions: Optional Live Web Search (non-fatal fallback into LLM) ---

  action workerSearchComplete(id: int, usedWeb: bool, searchFailed: bool): bool = {
    val s = servers.get(id)
    val searchEv = if (usedWeb) List(EvState({ tid: s.workerWireTid, cid: 0, ui: UiThinking })) else List()
    all {
      s.connected,
      s.workerStage == WorkerSearch,
      s.workerGen == s.turnGen,
      // Search failure is non-fatal: falls back to local LLM knowledge without failing the turn
      servers' = servers.put(id, {
        ...s,
        workerStage: WorkerPipeline,
        llmStreamActive: true,
        llmThinkingSelf: false,
        llmDone: false,
        llmFailed: false,
        queuedSentences: 0,
      }),
      s2c' = s2c.put(id, s2c.get(id).concat(searchEv)),
      clients' = clients,
      c2s' = c2s,
      hw' = hw,
      cov' = {
        ...cov,
        webSearchTurn: cov.webSearchTurn or (usedWeb and not(searchFailed)),
        webSearchFallback: cov.webSearchFallback or (usedWeb and searchFailed),
      },
    }
  }

  // --- Worker Actions: Concurrent LLM Producer (with Glimmer `to=self` phase) + TTS Consumer ---

  // Glimmer enters hidden `to=self` thinking phase before emitting any visible `to=user` sentence.
  action workerLlmEnterThinking(id: int): bool = {
    val s = servers.get(id)
    all {
      s.connected,
      s.workerStage == WorkerPipeline,
      s.workerGen == s.turnGen,
      s.llmStreamActive,
      not(s.llmThinkingSelf),
      not(s.llmDone),
      not(s.llmFailed),
      s.queuedSentences == 0,
      not(s.workerAudioStarted),
      servers' = servers.put(id, { ...s, llmThinkingSelf: true }),
      clients' = clients,
      c2s' = c2s,
      s2c' = s2c,
      hw' = hw,
      cov' = cov,
    }
  }

  // Glimmer exits `to=self` thinking and transitions to `to=user` visible speech.
  action workerLlmExitThinking(id: int): bool = {
    val s = servers.get(id)
    all {
      s.connected,
      s.workerStage == WorkerPipeline,
      s.workerGen == s.turnGen,
      s.llmStreamActive,
      s.llmThinkingSelf,
      not(s.llmDone),
      servers' = servers.put(id, { ...s, llmThinkingSelf: false }),
      clients' = clients,
      c2s' = c2s,
      s2c' = s2c,
      hw' = hw,
      cov' = cov,
    }
  }

  // Glimmer exhausts `num_predict` while still inside `to=self` hidden thinking (`reply=""`, 0 visible tokens).
  // `asError = true` models explicit turn-error surfacing; `asError = false` models clean empty-idle completion.
  action workerLlmExhaustedInThinking(id: int, asError: bool): bool = {
    val s = servers.get(id)
    all {
      s.connected,
      s.workerStage == WorkerPipeline,
      s.workerGen == s.turnGen,
      s.llmStreamActive,
      s.llmThinkingSelf,
      not(s.llmDone),
      s.queuedSentences == 0,
      not(s.workerAudioStarted),
      servers' = servers.put(id, {
        ...s,
        llmStreamActive: false,
        llmThinkingSelf: false,
        llmDone: true,
        llmFailed: asError,
      }),
      clients' = clients,
      c2s' = c2s,
      s2c' = s2c,
      hw' = hw,
      cov' = { ...cov, emptyThinkingRecovery: true },
    }
  }

  // LLM producer enqueues a completed visible `to=user` sentence (concurrently while TTS may hold GPU!)
  action workerLlmProduceSentence(id: int): bool = {
    val s = servers.get(id)
    all {
      s.connected,
      s.workerStage == WorkerPipeline,
      s.workerGen == s.turnGen,
      s.llmStreamActive,
      not(s.llmThinkingSelf),
      not(s.llmDone),
      not(s.llmFailed),
      s.queuedSentences < MAX_QUEUE,
      servers' = servers.put(id, { ...s, queuedSentences: s.queuedSentences + 1 }),
      clients' = clients,
      c2s' = c2s,
      s2c' = s2c,
      hw' = hw,
      cov' = { ...cov, pipelineOverlap: cov.pipelineOverlap or s.ttsHoldingGpu },
    }
  }

  // LLM producer finishes streaming sentences normally
  action workerLlmFinish(id: int): bool = {
    val s = servers.get(id)
    all {
      s.connected,
      s.workerStage == WorkerPipeline,
      s.workerGen == s.turnGen,
      not(s.llmThinkingSelf),
      not(s.llmDone),
      servers' = servers.put(id, { ...s, llmStreamActive: false, llmDone: true }),
      clients' = clients,
      c2s' = c2s,
      s2c' = s2c,
      hw' = hw,
      cov' = cov,
    }
  }

  // LLM producer encounters an Ollama/network error and signals consumer via sentinel
  action workerLlmError(id: int): bool = {
    val s = servers.get(id)
    all {
      s.connected,
      s.workerStage == WorkerPipeline,
      s.workerGen == s.turnGen,
      not(s.llmDone),
      servers' = servers.put(id, {
        ...s,
        llmStreamActive: false,
        llmThinkingSelf: false,
        llmDone: true,
        llmFailed: true,
      }),
      clients' = clients,
      c2s' = c2s,
      s2c' = s2c,
      hw' = hw,
      cov' = cov,
    }
  }

  // TTS consumer pops a sentence from queue and acquires GPU lock
  action workerTtsPopAndAcquireGpu(id: int): bool = {
    val s = servers.get(id)
    all {
      s.connected,
      s.workerStage == WorkerPipeline,
      s.workerGen == s.turnGen,
      not(s.ttsHoldingGpu),
      s.queuedSentences > 0,
      hw.gpuHolder == GpuFree,
      servers' = servers.put(id, {
        ...s,
        queuedSentences: s.queuedSentences - 1,
        ttsHoldingGpu: true,
      }),
      hw' = { ...hw, gpuHolder: GpuWorker({ clientId: id, gen: s.workerGen }) },
      clients' = clients,
      c2s' = c2s,
      s2c' = s2c,
      cov' = cov,
    }
  }

  // TTS consumer emits a PCM chunk (sending deferred EvAudioStart on first chunk)
  action workerTtsEmitChunk(id: int): bool = {
    val s = servers.get(id)
    val startEv = if (s.workerAudioStarted) List() else List(EvAudioStart({ tid: s.workerWireTid }))
    val chunkEv = List(EvPcmChunk({ tid: s.workerWireTid }))
    all {
      s.connected,
      s.workerStage == WorkerPipeline,
      s.workerGen == s.turnGen,
      s.ttsHoldingGpu,
      isExactGpuHolder(hw.gpuHolder, id, s.workerGen),
      servers' = servers.put(id, { ...s, workerAudioStarted: true }),
      s2c' = s2c.put(id, s2c.get(id).concat(startEv).concat(chunkEv)),
      hw' = hw,
      clients' = clients,
      c2s' = c2s,
      cov' = cov,
    }
  }

  // TTS consumer finishes current sentence and releases GPU lock
  action workerTtsFinishSentence(id: int): bool = {
    val s = servers.get(id)
    all {
      s.connected,
      s.workerStage == WorkerPipeline,
      s.workerGen == s.turnGen,
      s.ttsHoldingGpu,
      isExactGpuHolder(hw.gpuHolder, id, s.workerGen),
      servers' = servers.put(id, { ...s, ttsHoldingGpu: false }),
      hw' = { ...hw, gpuHolder: GpuFree },
      clients' = clients,
      c2s' = c2s,
      s2c' = s2c,
      cov' = cov,
    }
  }

  // Pipeline drains once LLM is done and queue is empty and GPU is released
  action workerPipelineFinish(id: int): bool = {
    val s = servers.get(id)
    val evs = if (s.llmFailed) {
      if (s.workerAudioStarted) List(
        EvAudioEnd({ tid: s.workerWireTid, cancelled: false, failed: true }),
        EvError({ tid: s.workerWireTid }),
        EvState({ tid: s.workerWireTid, cid: 0, ui: UiIdle }),
      ) else List(
        EvError({ tid: s.workerWireTid }),
        EvState({ tid: s.workerWireTid, cid: 0, ui: UiIdle }),
      )
    } else if (s.workerAudioStarted) {
      List(EvAudioEnd({ tid: s.workerWireTid, cancelled: false, failed: false }))
    } else {
      List(EvState({ tid: s.workerWireTid, cid: 0, ui: UiIdle }))
    }
    all {
      s.connected,
      s.workerStage == WorkerPipeline,
      s.workerGen == s.turnGen,
      s.llmDone,
      s.queuedSentences == 0,
      not(s.ttsHoldingGpu),
      servers' = servers.put(id, {
        ...s,
        workerStage: WorkerIdle,
        workerAudioStarted: false,
        llmStreamActive: false,
        llmThinkingSelf: false,
        llmFailed: false,
        historyPoisoned: s.historyPoisoned or (not(s.llmFailed) and s.workerAudioStarted and s.workerHistoryGen != s.historyGen),
      }),
      s2c' = s2c.put(id, s2c.get(id).concat(evs)),
      hw' = hw,
      clients' = clients,
      c2s' = c2s,
      cov' = { ...cov, errorRecovery: cov.errorRecovery or s.llmFailed },
    }
  }

  // Fatal STT or TTS exception while holding or not holding GPU: always releases GPU,
  // aborts any in-flight LLM stream (`brain.cancel_inflight()`), & signals error cleanly.
  action workerGpuFail(id: int): bool = {
    val s = servers.get(id)
    val holdsGpu = s.sttHoldingGpu or s.ttsHoldingGpu
    val evs = if (s.workerAudioStarted) List(
      EvAudioEnd({ tid: s.workerWireTid, cancelled: false, failed: true }),
      EvError({ tid: s.workerWireTid }),
      EvState({ tid: s.workerWireTid, cid: 0, ui: UiIdle }),
    ) else List(
      EvError({ tid: s.workerWireTid }),
      EvState({ tid: s.workerWireTid, cid: 0, ui: UiIdle }),
    )
    all {
      s.connected,
      s.workerStage == WorkerStt or s.workerStage == WorkerPipeline,
      s.workerGen == s.turnGen,
      holdsGpu implies isExactGpuHolder(hw.gpuHolder, id, s.workerGen),
      servers' = servers.put(id, {
        ...s,
        workerStage: WorkerIdle,
        workerAudioStarted: false,
        llmStreamActive: false,
        llmThinkingSelf: false,
        llmDone: true,
        llmFailed: false,
        queuedSentences: 0,
        ttsHoldingGpu: false,
        sttHoldingGpu: false,
      }),
      hw' = if (holdsGpu) { ...hw, gpuHolder: GpuFree } else hw,
      s2c' = s2c.put(id, s2c.get(id).concat(evs)),
      clients' = clients,
      c2s' = c2s,
      cov' = { ...cov, errorRecovery: true },
    }
  }

  action step: bool = {
    nondet id = CLIENTS.oneOf()
    any {
      runtimeWarmupAcquire,
      runtimeWarmupRelease,
      retiredWorkerReleaseGpu,
      clientBeginPtt(id, SpaceOwner),
      clientBeginPtt(id, PointerOwner),
      clientBeginPtt(id, ToggleOwner),
      clientBrowserStreamStart(id, true),
      clientBrowserStreamStart(id, false),
      clientBrowserStreamChunk(id, clients.get(id).activeCaptureId, true),
      clientBrowserStreamChunk(id, clients.get(id).activeCaptureId, false),
      clientFinishBrowserStream(id),
      clientRapidDoublePtt(id),
      clientTapLatch(id),
      clientFinishPtt(id, true),
      clientFinishPtt(id, false),
      clientSubmitText(id),
      clientCancelEscape(id),
      clientReset(id),
      clientSetModel(id, true),
      clientSetModel(id, false),
      clientToggleMicMode(id),
      clientSetAudioSuspended(id, true),
      clientSetAudioSuspended(id, false),
      clientStartReplay(id),
      clientPlaybackDrain(id),
      clientReceiveS2C(id),
      clientDisconnect(id),
      clientReconnect(id),
      serverProcessC2S(id),
      serverMicHitLimit(id),
      workerSttAcquireGpu(id),
      workerSttComplete(id, true),
      workerSttComplete(id, false),
      workerSearchComplete(id, false, false),
      workerSearchComplete(id, true, false),
      workerSearchComplete(id, true, true),
      workerLlmEnterThinking(id),
      workerLlmExitThinking(id),
      workerLlmExhaustedInThinking(id, true),
      workerLlmExhaustedInThinking(id, false),
      workerLlmProduceSentence(id),
      workerLlmFinish(id),
      workerLlmError(id),
      workerTtsPopAndAcquireGpu(id),
      workerTtsEmitChunk(id),
      workerTtsFinishSentence(id),
      workerPipelineFinish(id),
      workerGpuFail(id),
    }
  }

  // --- Protocol Safety Invariants ---

  // 1. At most one tab can hold the physical PipeWire microphone, matching hw.micOwner.
  val invExclusiveMicOwnership: bool = and {
    CLIENTS.filter(i => servers.get(i).micRecording).size() <= 1,
    CLIENTS.forall(i => servers.get(i).micRecording implies hw.micOwner == i),
  }

  // 2. Disconnected clients never hold the physical microphone.
  val invNoOrphanedMic: bool =
    CLIENTS.forall(i =>
      not(servers.get(i).connected) implies (hw.micOwner != i and not(servers.get(i).micRecording))
    )

  // 3. GPU mutual exclusion: at most one live worker flags holding GPU, and it matches hw.gpuHolder.
  val invGpuMutualExclusion: bool = and {
    CLIENTS.filter(i => servers.get(i).sttHoldingGpu or servers.get(i).ttsHoldingGpu).size() <= 1,
    CLIENTS.forall(i => {
      val s = servers.get(i)
      (s.sttHoldingGpu or s.ttsHoldingGpu) implies isExactGpuHolder(hw.gpuHolder, i, s.workerGen)
    }),
    CLIENTS.forall(i => not(servers.get(i).sttHoldingGpu and servers.get(i).ttsHoldingGpu)),
  }

  // 4. Client NEVER plays live or replay audio while actively recording (barge-in safety).
  val invNoAudioWhileRecording: bool =
    CLIENTS.forall(i =>
      clients.get(i).pttActive implies (clients.get(i).playingTurnId == 0 and not(clients.get(i).replaying))
    )

  // 5. Client UI NEVER drops back to Idle while pttActive is true.
  val invNeverIdleWhileRecording: bool =
    CLIENTS.forall(i => clients.get(i).pttActive implies clients.get(i).uiState != UiIdle)

  // 6. Client NEVER plays PCM belonging to a stale/cancelled turn.
  val invNoStalePcmPlayed: bool =
    CLIENTS.forall(i => {
      val c = clients.get(i)
      c.playingTurnId == 0 or c.playingTurnId == c.activeTurnId
    })

  // 7. Replay NEVER runs concurrently with a pending/live turn or recording.
  val invNoReplayDuringActiveTurn: bool =
    CLIENTS.forall(i => {
      val c = clients.get(i)
      c.replaying implies (not(c.pttActive) and not(c.pendingTurn) and c.playingTurnId == 0)
    })

  // 8. Strict audio stream framing per turn (EvAudioStart -> EvPcmChunk* -> EvAudioEnd).
  val invStrictAudioFraming: bool =
    CLIENTS.forall(i => not(clients.get(i).protocolViolation))

  // 9. Bounded producer-consumer sentence queue never overflows or goes negative.
  val invQueueBounded: bool =
    CLIENTS.forall(i => {
      val s = servers.get(i)
      s.queuedSentences >= 0 and s.queuedSentences <= MAX_QUEUE
    })

  // 10. Server echoes the client's wire turn_id unchanged across Reset and SetModel.
  val invWireTurnIdMatchesSubmitted: bool =
    CLIENTS.forall(i => {
      val s = servers.get(i)
      s.workerStage != WorkerIdle implies s.workerWireTid == s.lastSubmittedTid
    })

  // 11. Capture ownership is consistent: pttActive iff owner != NoOwner.
  val invCaptureOwnerConsistent: bool =
    CLIENTS.forall(i => clients.get(i).pttActive == (clients.get(i).owner != NoOwner))

  // 12. Quiescent liveness safety: when all queues are drained and no worker/audio is running,
  //     the client can NEVER be permanently stuck in UiThinking or UiSpeaking.
  val invNoStuckThinkingWhenQuiescent: bool =
    CLIENTS.forall(i => {
      val c = clients.get(i)
      val s = servers.get(i)
      val quiescent =
        c.connected and s.connected and
        c2s.get(i).length() == 0 and s2c.get(i).length() == 0 and
        s.workerStage == WorkerIdle and c.playingTurnId == 0 and not(c.replaying) and not(c.pttActive)
      quiescent implies (c.uiState != UiThinking and c.uiState != UiSpeaking)
    })

  // 13. Idle workers NEVER leak an active LLM HTTP stream or stay stuck in `to=self` thinking.
  val invNoLeakedLlmStreamWhenIdle: bool =
    CLIENTS.forall(i => {
      val s = servers.get(i)
      s.workerStage == WorkerIdle implies (not(s.llmStreamActive) and not(s.llmThinkingSelf))
    })

  // 14. Audio replay NEVER starts while WebAudio AudioContext is suspended (prevents stuck UiSpeaking).
  val invNoReplayWhileAudioSuspended: bool =
    CLIENTS.forall(i => clients.get(i).replaying implies not(clients.get(i).audioSuspended))

  // 15. Conversation history is NEVER poisoned by a post-reset stale worker commit.
  val invNoPoisonedHistory: bool =
    CLIENTS.forall(i => not(servers.get(i).historyPoisoned))

  // 16. Disconnected server connections NEVER leak an active browser streaming audio buffer.
  val invNoOrphanedStreamCapture: bool =
    CLIENTS.forall(i =>
      not(servers.get(i).connected) implies not(servers.get(i).streamCaptureActive)
    )

  val allInvariants: bool = and {
    invExclusiveMicOwnership,
    invNoOrphanedMic,
    invGpuMutualExclusion,
    invNoAudioWhileRecording,
    invNeverIdleWhileRecording,
    invNoStalePcmPlayed,
    invNoReplayDuringActiveTurn,
    invStrictAudioFraming,
    invQueueBounded,
    invWireTurnIdMatchesSubmitted,
    invCaptureOwnerConsistent,
    invNoStuckThinkingWhenQuiescent,
    invNoLeakedLlmStreamWhenIdle,
    invNoReplayWhileAudioSuspended,
    invNoPoisonedHistory,
    invNoOrphanedStreamCapture,
  }

  // --- Reachability Witnesses ---
  val wPttTurn: bool = cov.pttTurn
  val wLatchedTurn: bool = cov.latchedTurn
  val wBrowserTurn: bool = cov.browserTurn
  val wBargeIn: bool = cov.bargeIn
  val wResetThenTurn: bool = cov.resetThenTurn
  val wCaptureLimit: bool = cov.captureLimit
  val wErrorRecovery: bool = cov.errorRecovery
  val wDisconnectCleanup: bool = cov.disconnectCleanup
  val wWebSearchTurn: bool = cov.webSearchTurn
  val wWebSearchFallback: bool = cov.webSearchFallback
  val wPipelineOverlap: bool = cov.pipelineOverlap
  val wSetModelMidTurn: bool = cov.setModelMidTurn
  val wInvalidAudioRecovery: bool = cov.invalidAudioRecovery
  val wReplayAfterTurn: bool = cov.replayAfterTurn
  val wWarmupGpuContention: bool = cov.warmupGpuContention
  val wRetiredWorkerGpuHandoff: bool = cov.retiredWorkerGpuHandoff
  val wThinkingSelfBargeIn: bool = cov.thinkingSelfBargeIn
  val wEmptyThinkingRecovery: bool = cov.emptyThinkingRecovery
  val wReplaySuspendedGuard: bool = cov.replaySuspendedGuard
  val wRapidDoublePtt: bool = cov.rapidDoublePtt
  val wDisconnectDuringActivePipeline: bool = cov.disconnectDuringActivePipeline
  val wBrowserStreamTurn: bool = cov.browserStreamTurn
  val wBrowserStreamBargeIn: bool = cov.browserStreamBargeIn
  val wBrowserStreamRapidDouble: bool = cov.browserStreamRapidDouble
}
