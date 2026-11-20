// Executable Quint specification of the IOGHAL WebSocket + Push-to-Talk + Audio protocol.
//
// Models the real client (web/src/App.tsx + HalAudioPlayer) and server
// (src/hal9000/server.py + HalPipeline + MicRecorder) across N concurrent browser tabs.
//
// What it covers:
// - Client PTT ownership (hold, tap-to-latch, Escape cancel, blur/hide, disconnect, Reset)
// - Wire turn_id ownership vs server internal turn_gen cancellation
// - capture_id scoping for PipeWire local mic start/stop/limit/failure
// - Framed binary PCM delivery (turn_id header checked on client)
// - Single-owner host microphone lock & single-owner GPU lock across concurrent tabs
//
// What it does NOT cover:
// - WebAudio sample-level DSP math, resampling, or CUDA kernel timings
// - Exact Ollama token/sentence splitting or Whisper acoustic accuracy
//
// Update this spec and re-run `quint run` before changing the wire protocol or PTT state machine.

module ioghal_protocol {
  pure val CLIENTS: Set[int] = Set(1, 2)
  pure val MAX_TURN_ID: int = 8

  type UiState = UiIdle | UiListening | UiThinking | UiSpeaking
  type MicMode = ModeLocal | ModeBrowser
  type Owner = NoOwner | SpaceOwner | PointerOwner | ToggleOwner

  type WorkerStage = WorkerIdle | WorkerStt | WorkerLlm | WorkerTts

  type C2SMsg =
    | MsgPttStart({ cid: int })
    | MsgPttStop({ tid: int, cid: int })
    | MsgAudioB64({ tid: int })
    | MsgText({ tid: int })
    | MsgInterrupt({ tid: int, silent: bool })
    | MsgReset({ tid: int })

  type S2CMsg =
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
    playedStalePcm: bool,
    receivedResetDesync: bool,
    lastClearedTurnId: int,
  }

  type ServerConn = {
    connected: bool,
    turnGen: int,
    workerStage: WorkerStage,
    workerWireTid: int,
    workerGen: int,
    workerAudioStarted: bool,
    micRecording: bool,
    micCaptureId: int,
    micLimitHit: bool,
    lastSubmittedTid: int,
  }

  var clients: int -> ClientState
  var servers: int -> ServerConn
  var c2s: int -> List[C2SMsg]
  var s2c: int -> List[S2CMsg]
  var micOwner: int   // 0 = none, else ClientId
  var gpuOwner: int   // 0 = none, else ClientId

  // Action coverage flags for witnesses
  var sawPttTurn: bool
  var sawLatchedTurn: bool
  var sawBrowserTurn: bool
  var sawBargeIn: bool
  var sawResetThenTurn: bool
  var sawCaptureLimit: bool
  var sawErrorRecovery: bool
  var sawDisconnectCleanup: bool

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
    playedStalePcm: false,
    receivedResetDesync: false,
    lastClearedTurnId: 0,
  }

  pure val initServer: ServerConn = {
    connected: true,
    turnGen: 0,
    workerStage: WorkerIdle,
    workerWireTid: 0,
    workerGen: 0,
    workerAudioStarted: false,
    micRecording: false,
    micCaptureId: 0,
    micLimitHit: false,
    lastSubmittedTid: 0,
  }

  action init: bool = all {
    clients' = CLIENTS.mapBy(_ => initClient),
    servers' = CLIENTS.mapBy(_ => initServer),
    c2s' = CLIENTS.mapBy(_ => List()),
    s2c' = CLIENTS.mapBy(_ => List()),
    micOwner' = 0,
    gpuOwner' = 0,
    sawPttTurn' = false,
    sawLatchedTurn' = false,
    sawBrowserTurn' = false,
    sawBargeIn' = false,
    sawResetThenTurn' = false,
    sawCaptureLimit' = false,
    sawErrorRecovery' = false,
    sawDisconnectCleanup' = false,
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
      }),
      c2s' = c2s.put(id, c2s.get(id).append(msg)),
      servers' = servers,
      s2c' = s2c,
      micOwner' = micOwner,
      gpuOwner' = gpuOwner,
      sawPttTurn' = sawPttTurn,
      sawLatchedTurn' = sawLatchedTurn,
      sawBrowserTurn' = sawBrowserTurn,
      sawBargeIn' = (sawBargeIn or (c.playingTurnId != 0 or c.pendingTurn)),
      sawResetThenTurn' = sawResetThenTurn,
      sawCaptureLimit' = sawCaptureLimit,
      sawErrorRecovery' = sawErrorRecovery,
      sawDisconnectCleanup' = sawDisconnectCleanup,
    }
  }

  action clientTapLatch(id: int): bool = {
    val c = clients.get(id)
    all {
      c.connected,
      c.pttActive,
      not(c.latched),
      c.owner == SpaceOwner or c.owner == PointerOwner,
      clients' = clients.put(id, {
        ...c,
        latched: true,
        owner: ToggleOwner,
      }),
      servers' = servers,
      c2s' = c2s,
      s2c' = s2c,
      micOwner' = micOwner,
      gpuOwner' = gpuOwner,
      sawPttTurn' = sawPttTurn,
      sawLatchedTurn' = true,
      sawBrowserTurn' = sawBrowserTurn,
      sawBargeIn' = sawBargeIn,
      sawResetThenTurn' = sawResetThenTurn,
      sawCaptureLimit' = sawCaptureLimit,
      sawErrorRecovery' = sawErrorRecovery,
      sawDisconnectCleanup' = sawDisconnectCleanup,
    }
  }

  action clientFinishPtt(id: int): bool = {
    val c = clients.get(id)
    val tid = c.nextTurnId
    val cid = c.activeCaptureId
    val msg = if (c.captureMode == ModeBrowser) {
      MsgAudioB64({ tid: tid })
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
      }),
      c2s' = c2s.put(id, c2s.get(id).append(msg)),
      servers' = servers,
      s2c' = s2c,
      micOwner' = micOwner,
      gpuOwner' = gpuOwner,
      sawPttTurn' = true,
      sawLatchedTurn' = sawLatchedTurn,
      sawBrowserTurn' = (sawBrowserTurn or (c.captureMode == ModeBrowser)),
      sawBargeIn' = sawBargeIn,
      sawResetThenTurn' = (sawResetThenTurn or (c.lastClearedTurnId > 0)),
      sawCaptureLimit' = sawCaptureLimit,
      sawErrorRecovery' = sawErrorRecovery,
      sawDisconnectCleanup' = sawDisconnectCleanup,
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
      }),
      c2s' = c2s.put(id, c2s.get(id).append(MsgText({ tid: tid }))),
      servers' = servers,
      s2c' = s2c,
      micOwner' = micOwner,
      gpuOwner' = gpuOwner,
      sawPttTurn' = sawPttTurn,
      sawLatchedTurn' = sawLatchedTurn,
      sawBrowserTurn' = sawBrowserTurn,
      sawBargeIn' = (sawBargeIn or (c.playingTurnId != 0)),
      sawResetThenTurn' = (sawResetThenTurn or (c.lastClearedTurnId > 0)),
      sawCaptureLimit' = sawCaptureLimit,
      sawErrorRecovery' = sawErrorRecovery,
      sawDisconnectCleanup' = sawDisconnectCleanup,
    }
  }

  action clientCancelEscape(id: int): bool = {
    val c = clients.get(id)
    val tid = c.nextTurnId
    all {
      c.connected,
      c.pttActive or c.pendingTurn or c.playingTurnId != 0 or not(c.serverAudioDone),
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
      }),
      c2s' = c2s.put(id, c2s.get(id).append(MsgInterrupt({ tid: tid, silent: false }))),
      servers' = servers,
      s2c' = s2c,
      micOwner' = micOwner,
      gpuOwner' = gpuOwner,
      sawPttTurn' = sawPttTurn,
      sawLatchedTurn' = sawLatchedTurn,
      sawBrowserTurn' = sawBrowserTurn,
      sawBargeIn' = true,
      sawResetThenTurn' = sawResetThenTurn,
      sawCaptureLimit' = sawCaptureLimit,
      sawErrorRecovery' = sawErrorRecovery,
      sawDisconnectCleanup' = sawDisconnectCleanup,
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
        lastClearedTurnId: tid,
      }),
      // Matches App.tsx handleReset: single reset command (no double interrupt+reset)
      c2s' = c2s.put(id, c2s.get(id).append(MsgReset({ tid: tid }))),
      servers' = servers,
      s2c' = s2c,
      micOwner' = micOwner,
      gpuOwner' = gpuOwner,
      sawPttTurn' = sawPttTurn,
      sawLatchedTurn' = sawLatchedTurn,
      sawBrowserTurn' = sawBrowserTurn,
      sawBargeIn' = sawBargeIn,
      sawResetThenTurn' = sawResetThenTurn,
      sawCaptureLimit' = sawCaptureLimit,
      sawErrorRecovery' = sawErrorRecovery,
      sawDisconnectCleanup' = sawDisconnectCleanup,
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
      micOwner' = micOwner,
      gpuOwner' = gpuOwner,
      sawPttTurn' = sawPttTurn,
      sawLatchedTurn' = sawLatchedTurn,
      sawBrowserTurn' = sawBrowserTurn,
      sawBargeIn' = sawBargeIn,
      sawResetThenTurn' = sawResetThenTurn,
      sawCaptureLimit' = sawCaptureLimit,
      sawErrorRecovery' = sawErrorRecovery,
      sawDisconnectCleanup' = sawDisconnectCleanup,
    }
  }

  action clientPlaybackDrain(id: int): bool = {
    val c = clients.get(id)
    all {
      c.connected,
      c.playingTurnId != 0,
      c.serverAudioDone,
      clients' = clients.put(id, {
        ...c,
        playingTurnId: 0,
        pendingTurn: if (c.pttActive) c.pendingTurn else false,
        uiState: if (c.pttActive) c.uiState else UiIdle,
      }),
      servers' = servers,
      c2s' = c2s,
      s2c' = s2c,
      micOwner' = micOwner,
      gpuOwner' = gpuOwner,
      sawPttTurn' = sawPttTurn,
      sawLatchedTurn' = sawLatchedTurn,
      sawBrowserTurn' = sawBrowserTurn,
      sawBargeIn' = sawBargeIn,
      sawResetThenTurn' = sawResetThenTurn,
      sawCaptureLimit' = sawCaptureLimit,
      sawErrorRecovery' = sawErrorRecovery,
      sawDisconnectCleanup' = sawDisconnectCleanup,
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
        | EvState(p) => {
            val staleT = p.tid != 0 and p.tid != c.activeTurnId
            val staleC = p.cid != 0 and p.cid != c.activeCaptureId
            val blockIdle = p.ui == UiIdle and (c.pttActive or c.playingTurnId != 0)
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
              // Matches App.tsx: auto-calls finishRecording() on PipeWire capture_limit
              ...c,
              uiState: UiThinking,
              pttActive: false,
              latched: false,
              owner: NoOwner,
              activeTurnId: c.nextTurnId,
              nextTurnId: c.nextTurnId + 1,
              activeCaptureId: 0,
              pendingTurn: true,
            } else c
          }
        | EvAudioStart(p) => {
            if (p.tid != c.activeTurnId) c
            else {
              ...c,
              playingTurnId: 0,
              serverAudioDone: false,
            }
          }
        | EvPcmChunk(p) => {
            // Framed PCM check: decodeFramedPcm verifies header turn_id == activeTurnId
            if (p.tid != c.activeTurnId) c
            else {
              ...c,
              playingTurnId: p.tid,
              uiState: UiSpeaking,
            }
          }
        | EvAudioEnd(p) => {
            if (p.tid != c.activeTurnId) c
            else {
              val stopPlay = p.cancelled or p.failed
              val nextPlaying = if (stopPlay) 0 else c.playingTurnId
              val goIdle = nextPlaying == 0 and not(c.pttActive)
              {
                ...c,
                serverAudioDone: true,
                playingTurnId: nextPlaying,
                pendingTurn: if (goIdle) false else c.pendingTurn,
                uiState: if (goIdle and not(p.failed)) UiIdle else c.uiState,
              }
            }
          }
        | EvCleared(_) => c
        | EvError(p) => {
            if (p.tid != 0 and p.tid != c.activeTurnId) c
            else {
              ...c,
              pendingTurn: false,
              playingTurnId: 0,
              serverAudioDone: true,
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
        micOwner' = micOwner,
        gpuOwner' = gpuOwner,
        sawPttTurn' = sawPttTurn,
        sawLatchedTurn' = sawLatchedTurn,
        sawBrowserTurn' = sawBrowserTurn,
        sawBargeIn' = sawBargeIn,
        sawResetThenTurn' = sawResetThenTurn,
        sawCaptureLimit' = sawCaptureLimit,
        sawErrorRecovery' = sawErrorRecovery,
        sawDisconnectCleanup' = sawDisconnectCleanup,
      }
    }
  }

  action clientDisconnect(id: int): bool = {
    val c = clients.get(id)
    val s = servers.get(id)
    all {
      c.connected,
      clients' = clients.put(id, {
        ...initClient,
        connected: false,
      }),
      servers' = servers.put(id, {
        ...s,
        connected: false,
        turnGen: s.turnGen + 1,
        workerStage: WorkerIdle,
        micRecording: false,
        micCaptureId: 0,
        micLimitHit: false,
      }),
      c2s' = c2s.put(id, List()),
      s2c' = s2c.put(id, List()),
      micOwner' = if (micOwner == id) 0 else micOwner,
      gpuOwner' = if (gpuOwner == id) 0 else gpuOwner,
      sawPttTurn' = sawPttTurn,
      sawLatchedTurn' = sawLatchedTurn,
      sawBrowserTurn' = sawBrowserTurn,
      sawBargeIn' = sawBargeIn,
      sawResetThenTurn' = sawResetThenTurn,
      sawCaptureLimit' = sawCaptureLimit,
      sawErrorRecovery' = sawErrorRecovery,
      sawDisconnectCleanup' = true,
    }
  }

  action clientReconnect(id: int): bool = {
    val c = clients.get(id)
    val s = servers.get(id)
    all {
      not(c.connected),
      clients' = clients.put(id, initClient),
      servers' = servers.put(id, initServer),
      c2s' = c2s.put(id, List()),
      s2c' = s2c.put(id, List()),
      micOwner' = micOwner,
      gpuOwner' = gpuOwner,
      sawPttTurn' = sawPttTurn,
      sawLatchedTurn' = sawLatchedTurn,
      sawBrowserTurn' = sawBrowserTurn,
      sawBargeIn' = sawBargeIn,
      sawResetThenTurn' = sawResetThenTurn,
      sawCaptureLimit' = sawCaptureLimit,
      sawErrorRecovery' = sawErrorRecovery,
      sawDisconnectCleanup' = sawDisconnectCleanup,
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
            val canAcquire = micOwner == 0 or micOwner == id
            if (canAcquire) all {
              servers' = servers.put(id, {
                ...s,
                turnGen: s.turnGen + 1,
                micRecording: true,
                micCaptureId: p.cid,
                micLimitHit: false,
              }),
              micOwner' = id,
              s2c' = s2c.put(id, s2c.get(id).append(EvState({ tid: 0, cid: p.cid, ui: UiListening }))),
              c2s' = c2s.put(id, rest),
              clients' = clients,
              gpuOwner' = gpuOwner,
              sawPttTurn' = sawPttTurn,
              sawLatchedTurn' = sawLatchedTurn,
              sawBrowserTurn' = sawBrowserTurn,
              sawBargeIn' = sawBargeIn,
              sawResetThenTurn' = sawResetThenTurn,
              sawCaptureLimit' = sawCaptureLimit,
              sawErrorRecovery' = sawErrorRecovery,
              sawDisconnectCleanup' = sawDisconnectCleanup,
            } else all {
              servers' = servers.put(id, { ...s, turnGen: s.turnGen + 1 }),
              micOwner' = micOwner,
              s2c' = s2c.put(id, s2c.get(id).append(EvCaptureFailed({ cid: p.cid }))),
              c2s' = c2s.put(id, rest),
              clients' = clients,
              gpuOwner' = gpuOwner,
              sawPttTurn' = sawPttTurn,
              sawLatchedTurn' = sawLatchedTurn,
              sawBrowserTurn' = sawBrowserTurn,
              sawBargeIn' = sawBargeIn,
              sawResetThenTurn' = sawResetThenTurn,
              sawCaptureLimit' = sawCaptureLimit,
              sawErrorRecovery' = sawErrorRecovery,
              sawDisconnectCleanup' = sawDisconnectCleanup,
            }
          }
        | MsgPttStop(p) => {
            if (not(s.micRecording)) all {
              servers' = servers,
              micOwner' = micOwner,
              // Fix proved by Quint: always emit idle ack for p.tid so client never sticks in UiThinking
              s2c' = s2c.put(id, s2c.get(id).append(EvState({ tid: p.tid, cid: 0, ui: UiIdle }))),
              c2s' = c2s.put(id, rest),
              clients' = clients,
              gpuOwner' = gpuOwner,
              sawPttTurn' = sawPttTurn,
              sawLatchedTurn' = sawLatchedTurn,
              sawBrowserTurn' = sawBrowserTurn,
              sawBargeIn' = sawBargeIn,
              sawResetThenTurn' = sawResetThenTurn,
              sawCaptureLimit' = sawCaptureLimit,
              sawErrorRecovery' = sawErrorRecovery,
              sawDisconnectCleanup' = sawDisconnectCleanup,
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
                  workerAudioStarted: false,
                  lastSubmittedTid: p.tid,
                }),
                micOwner' = if (micOwner == id) 0 else micOwner,
                s2c' = s2c,
                c2s' = c2s.put(id, rest),
                clients' = clients,
                gpuOwner' = gpuOwner,
                sawPttTurn' = sawPttTurn,
                sawLatchedTurn' = sawLatchedTurn,
                sawBrowserTurn' = sawBrowserTurn,
                sawBargeIn' = sawBargeIn,
                sawResetThenTurn' = sawResetThenTurn,
                sawCaptureLimit' = sawCaptureLimit,
                sawErrorRecovery' = sawErrorRecovery,
                sawDisconnectCleanup' = sawDisconnectCleanup,
              }
            }
          }
        | MsgAudioB64(p) => {
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
                workerAudioStarted: false,
                lastSubmittedTid: p.tid,
              }),
              micOwner' = if (micOwner == id) 0 else micOwner,
              s2c' = s2c,
              c2s' = c2s.put(id, rest),
              clients' = clients,
              gpuOwner' = gpuOwner,
              sawPttTurn' = sawPttTurn,
              sawLatchedTurn' = sawLatchedTurn,
              sawBrowserTurn' = sawBrowserTurn,
              sawBargeIn' = sawBargeIn,
              sawResetThenTurn' = sawResetThenTurn,
              sawCaptureLimit' = sawCaptureLimit,
              sawErrorRecovery' = sawErrorRecovery,
              sawDisconnectCleanup' = sawDisconnectCleanup,
            }
          }
        | MsgText(p) => {
            val nextGen = s.turnGen + 1
            all {
              servers' = servers.put(id, {
                ...s,
                turnGen: nextGen,
                micRecording: false,
                micCaptureId: 0,
                micLimitHit: false,
                workerStage: WorkerLlm,
                workerWireTid: p.tid,
                workerGen: nextGen,
                workerAudioStarted: false,
                lastSubmittedTid: p.tid,
              }),
              micOwner' = if (micOwner == id) 0 else micOwner,
              s2c' = s2c,
              c2s' = c2s.put(id, rest),
              clients' = clients,
              gpuOwner' = gpuOwner,
              sawPttTurn' = sawPttTurn,
              sawLatchedTurn' = sawLatchedTurn,
              sawBrowserTurn' = sawBrowserTurn,
              sawBargeIn' = sawBargeIn,
              sawResetThenTurn' = sawResetThenTurn,
              sawCaptureLimit' = sawCaptureLimit,
              sawErrorRecovery' = sawErrorRecovery,
              sawDisconnectCleanup' = sawDisconnectCleanup,
            }
          }
        | MsgInterrupt(p) => {
            val outEv = if (p.silent) List() else List(EvState({ tid: p.tid, cid: 0, ui: UiIdle }))
            all {
              servers' = servers.put(id, {
                ...s,
                turnGen: s.turnGen + 1,
                micRecording: false,
                micCaptureId: 0,
                micLimitHit: false,
              }),
              micOwner' = if (micOwner == id) 0 else micOwner,
              s2c' = s2c.put(id, s2c.get(id).concat(outEv)),
              c2s' = c2s.put(id, rest),
              clients' = clients,
              gpuOwner' = gpuOwner,
              sawPttTurn' = sawPttTurn,
              sawLatchedTurn' = sawLatchedTurn,
              sawBrowserTurn' = sawBrowserTurn,
              sawBargeIn' = sawBargeIn,
              sawResetThenTurn' = sawResetThenTurn,
              sawCaptureLimit' = sawCaptureLimit,
              sawErrorRecovery' = sawErrorRecovery,
              sawDisconnectCleanup' = sawDisconnectCleanup,
            }
          }
        | MsgReset(p) => all {
            servers' = servers.put(id, {
              ...s,
              turnGen: s.turnGen + 1,
              micRecording: false,
              micCaptureId: 0,
              micLimitHit: false,
            }),
            micOwner' = if (micOwner == id) 0 else micOwner,
            s2c' = s2c.put(id, s2c.get(id).append(EvCleared({ tid: p.tid }))),
            c2s' = c2s.put(id, rest),
            clients' = clients,
            gpuOwner' = gpuOwner,
            sawPttTurn' = sawPttTurn,
            sawLatchedTurn' = sawLatchedTurn,
            sawBrowserTurn' = sawBrowserTurn,
            sawBargeIn' = sawBargeIn,
            sawResetThenTurn' = sawResetThenTurn,
            sawCaptureLimit' = sawCaptureLimit,
            sawErrorRecovery' = sawErrorRecovery,
            sawDisconnectCleanup' = sawDisconnectCleanup,
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
      micOwner' = micOwner,
      gpuOwner' = gpuOwner,
      sawPttTurn' = sawPttTurn,
      sawLatchedTurn' = sawLatchedTurn,
      sawBrowserTurn' = sawBrowserTurn,
      sawBargeIn' = sawBargeIn,
      sawResetThenTurn' = sawResetThenTurn,
      sawCaptureLimit' = true,
      sawErrorRecovery' = sawErrorRecovery,
      sawDisconnectCleanup' = sawDisconnectCleanup,
    }
  }

  action workerSttStep(id: int, speechFound: bool): bool = {
    val s = servers.get(id)
    all {
      s.connected,
      s.workerStage == WorkerStt,
      gpuOwner == 0,
      if (s.workerGen != s.turnGen) all {
        // Retired before emitting
        servers' = servers.put(id, { ...s, workerStage: WorkerIdle }),
        s2c' = s2c,
      } else if (speechFound) all {
        servers' = servers.put(id, { ...s, workerStage: WorkerLlm }),
        s2c' = s2c.put(id, s2c.get(id).append(EvState({ tid: s.workerWireTid, cid: 0, ui: UiThinking }))),
      } else all {
        servers' = servers.put(id, { ...s, workerStage: WorkerIdle }),
        s2c' = s2c.put(id, s2c.get(id).append(EvState({ tid: s.workerWireTid, cid: 0, ui: UiIdle }))),
      },
      clients' = clients,
      c2s' = c2s,
      micOwner' = micOwner,
      gpuOwner' = 0,
      sawPttTurn' = sawPttTurn,
      sawLatchedTurn' = sawLatchedTurn,
      sawBrowserTurn' = sawBrowserTurn,
      sawBargeIn' = sawBargeIn,
      sawResetThenTurn' = sawResetThenTurn,
      sawCaptureLimit' = sawCaptureLimit,
      sawErrorRecovery' = sawErrorRecovery,
      sawDisconnectCleanup' = sawDisconnectCleanup,
    }
  }

  action workerLlmToTts(id: int): bool = {
    val s = servers.get(id)
    all {
      s.connected,
      s.workerStage == WorkerLlm,
      if (s.workerGen != s.turnGen) all {
        servers' = servers.put(id, { ...s, workerStage: WorkerIdle }),
        s2c' = s2c,
      } else all {
        servers' = servers.put(id, { ...s, workerStage: WorkerTts }),
        s2c' = s2c,
      },
      clients' = clients,
      c2s' = c2s,
      micOwner' = micOwner,
      gpuOwner' = gpuOwner,
      sawPttTurn' = sawPttTurn,
      sawLatchedTurn' = sawLatchedTurn,
      sawBrowserTurn' = sawBrowserTurn,
      sawBargeIn' = sawBargeIn,
      sawResetThenTurn' = sawResetThenTurn,
      sawCaptureLimit' = sawCaptureLimit,
      sawErrorRecovery' = sawErrorRecovery,
      sawDisconnectCleanup' = sawDisconnectCleanup,
    }
  }

  action workerEmitTtsChunk(id: int): bool = {
    val s = servers.get(id)
    val startEv = if (s.workerAudioStarted) List() else List(EvAudioStart({ tid: s.workerWireTid }))
    val chunkEv = List(EvPcmChunk({ tid: s.workerWireTid }))
    all {
      s.connected,
      s.workerStage == WorkerTts,
      gpuOwner == 0,
      if (s.workerGen != s.turnGen) all {
        servers' = servers.put(id, { ...s, workerStage: WorkerIdle }),
        s2c' = s2c,
      } else all {
        servers' = servers.put(id, { ...s, workerAudioStarted: true }),
        s2c' = s2c.put(id, s2c.get(id).concat(startEv).concat(chunkEv)),
      },
      clients' = clients,
      c2s' = c2s,
      micOwner' = micOwner,
      gpuOwner' = 0,
      sawPttTurn' = sawPttTurn,
      sawLatchedTurn' = sawLatchedTurn,
      sawBrowserTurn' = sawBrowserTurn,
      sawBargeIn' = sawBargeIn,
      sawResetThenTurn' = sawResetThenTurn,
      sawCaptureLimit' = sawCaptureLimit,
      sawErrorRecovery' = sawErrorRecovery,
      sawDisconnectCleanup' = sawDisconnectCleanup,
    }
  }

  action workerFinishTts(id: int): bool = {
    val s = servers.get(id)
    val cancelled = s.workerGen != s.turnGen
    val endEv = if (s.workerAudioStarted and not(cancelled)) {
      List(EvAudioEnd({ tid: s.workerWireTid, cancelled: false, failed: false }))
    } else if (not(s.workerAudioStarted) and not(cancelled)) {
      List(EvState({ tid: s.workerWireTid, cid: 0, ui: UiIdle }))
    } else List()
    all {
      s.connected,
      s.workerStage == WorkerTts,
      servers' = servers.put(id, { ...s, workerStage: WorkerIdle, workerAudioStarted: false }),
      s2c' = s2c.put(id, s2c.get(id).concat(endEv)),
      clients' = clients,
      c2s' = c2s,
      micOwner' = micOwner,
      gpuOwner' = gpuOwner,
      sawPttTurn' = sawPttTurn,
      sawLatchedTurn' = sawLatchedTurn,
      sawBrowserTurn' = sawBrowserTurn,
      sawBargeIn' = sawBargeIn,
      sawResetThenTurn' = sawResetThenTurn,
      sawCaptureLimit' = sawCaptureLimit,
      sawErrorRecovery' = sawErrorRecovery,
      sawDisconnectCleanup' = sawDisconnectCleanup,
    }
  }

  action workerFail(id: int): bool = {
    val s = servers.get(id)
    val isLive = s.workerGen == s.turnGen
    val evs = if (not(isLive)) List()
      else if (s.workerAudioStarted) List(
        EvError({ tid: s.workerWireTid }),
        EvAudioEnd({ tid: s.workerWireTid, cancelled: false, failed: true }),
      ) else List(
        EvError({ tid: s.workerWireTid }),
        EvState({ tid: s.workerWireTid, cid: 0, ui: UiIdle }),
      )
    all {
      s.connected,
      s.workerStage != WorkerIdle,
      servers' = servers.put(id, { ...s, workerStage: WorkerIdle, workerAudioStarted: false }),
      s2c' = s2c.put(id, s2c.get(id).concat(evs)),
      clients' = clients,
      c2s' = c2s,
      micOwner' = micOwner,
      gpuOwner' = gpuOwner,
      sawPttTurn' = sawPttTurn,
      sawLatchedTurn' = sawLatchedTurn,
      sawBrowserTurn' = sawBrowserTurn,
      sawBargeIn' = sawBargeIn,
      sawResetThenTurn' = sawResetThenTurn,
      sawCaptureLimit' = sawCaptureLimit,
      sawErrorRecovery' = (sawErrorRecovery or isLive),
      sawDisconnectCleanup' = sawDisconnectCleanup,
    }
  }

  action step: bool = {
    nondet id = CLIENTS.oneOf()
    any {
      clientBeginPtt(id, SpaceOwner),
      clientBeginPtt(id, PointerOwner),
      clientBeginPtt(id, ToggleOwner),
      clientTapLatch(id),
      clientFinishPtt(id),
      clientSubmitText(id),
      clientCancelEscape(id),
      clientReset(id),
      clientToggleMicMode(id),
      clientPlaybackDrain(id),
      clientReceiveS2C(id),
      clientDisconnect(id),
      clientReconnect(id),
      serverProcessC2S(id),
      serverMicHitLimit(id),
      workerSttStep(id, true),
      workerSttStep(id, false),
      workerLlmToTts(id),
      workerEmitTtsChunk(id),
      workerFinishTts(id),
      workerFail(id),
    }
  }

  // --- Protocol Safety Invariants ---

  // 1. At most one tab can hold the physical PipeWire microphone, matching micOwner.
  val invExclusiveMicOwnership: bool = and {
    CLIENTS.filter(i => servers.get(i).micRecording).size() <= 1,
    CLIENTS.forall(i => servers.get(i).micRecording implies micOwner == i),
  }

  // 2. Disconnected clients never hold the physical microphone or GPU lock.
  val invNoOrphanedResources: bool =
    CLIENTS.forall(i =>
      not(servers.get(i).connected) implies (micOwner != i and gpuOwner != i and not(servers.get(i).micRecording))
    )

  // 3. Client NEVER plays audio while actively recording (barge-in safety).
  val invNoAudioWhileRecording: bool =
    CLIENTS.forall(i => clients.get(i).pttActive implies clients.get(i).playingTurnId == 0)

  // 4. Client UI NEVER drops back to Idle while pttActive is true (e.g. from browser-mic interrupt ack).
  val invNeverIdleWhileRecording: bool =
    CLIENTS.forall(i => clients.get(i).pttActive implies clients.get(i).uiState != UiIdle)

  // 5. Client NEVER plays PCM belonging to a stale/cancelled turn.
  val invNoStalePcmPlayed: bool =
    CLIENTS.forall(i => {
      val c = clients.get(i)
      c.playingTurnId == 0 or c.playingTurnId == c.activeTurnId
    })

  // 6. Server echoes the client's wire turn_id unchanged (Reset never desyncs subsequent turns).
  val invWireTurnIdMatchesSubmitted: bool =
    CLIENTS.forall(i => {
      val s = servers.get(i)
      s.workerStage != WorkerIdle implies s.workerWireTid == s.lastSubmittedTid
    })

  // 7. Capture ownership is consistent: pttActive iff owner != NoOwner.
  val invCaptureOwnerConsistent: bool =
    CLIENTS.forall(i => clients.get(i).pttActive == (clients.get(i).owner != NoOwner))

  // 8. Quiescent liveness safety: when all queues are drained and no worker/audio is running,
  //    the client can NEVER be permanently stuck in UiThinking.
  val invNoStuckThinkingWhenQuiescent: bool =
    CLIENTS.forall(i => {
      val c = clients.get(i)
      val s = servers.get(i)
      val quiescent =
        c.connected and s.connected and
        c2s.get(i).length() == 0 and s2c.get(i).length() == 0 and
        s.workerStage == WorkerIdle and c.playingTurnId == 0 and not(c.pttActive)
      quiescent implies c.uiState != UiThinking
    })

  val allInvariants: bool = and {
    invExclusiveMicOwnership,
    invNoOrphanedResources,
    invNoAudioWhileRecording,
    invNeverIdleWhileRecording,
    invNoStalePcmPlayed,
    invWireTurnIdMatchesSubmitted,
    invCaptureOwnerConsistent,
    invNoStuckThinkingWhenQuiescent,
  }

  // --- Reachability Witnesses ---
  val wPttTurn: bool = sawPttTurn
  val wLatchedTurn: bool = sawLatchedTurn
  val wBrowserTurn: bool = sawBrowserTurn
  val wBargeIn: bool = sawBargeIn
  val wResetThenTurn: bool = sawResetThenTurn
  val wCaptureLimit: bool = sawCaptureLimit
  val wErrorRecovery: bool = sawErrorRecovery
  val wDisconnectCleanup: bool = sawDisconnectCleanup
}
