// Executable Quint specification for the Muse Glimmer 30B voice runtime,
// VRAM/cgroup memory exclusivity, dual-mode routing, hidden `to=self` reasoning phase,
// single-slot HTTP cancellation, and KV prefix cache / history commit consistency.
//
// Models the real hardware and serving constraints on RTX Spark N1X (120 GiB unified LPDDR5X):
// - ollama.service systemd cgroup cap: MemoryMax=16G, MemorySwapMax=0 (`20-memory-cap.conf`)
// - heavy.slice cgroup budget: MemoryHigh=64G (MemoryMax=72G) hosting Whisper (~4G) + VoxCPM2 (~4G)
//   + optional direct CUDA 13 llama-server (~18-19G without vision mmproj)
// - OLLAMA_MAX_LOADED_MODELS=1 residency & warmup-vs-set_model race (`server.py:219-256`)
// - Hidden `to=self` reasoning -> `to=user` speaking transition, `num_predict` exhaustion,
//   ATEM channel filtering, and barge-in / TTS-failure slot cancellation (`brain.py`, `pipeline.py`)
// - History commit timing vs spoken TTS completion and KV prefix cache invariance
//
// Run:
//   quint typecheck specs/glimmer_runtime.qnt
//   quint test specs/glimmer_runtime_test.qnt
//   quint run specs/glimmer_runtime.qnt --invariant=allInvariants --max-samples=10000

module glimmer_runtime {
  pure val CLIENTS: Set[int] = Set(1, 2)
  pure val WHISPER_GB: int = 4
  pure val VOXCPM_GB: int = 4
  pure val OLLAMA_CGROUP_MAX_GB: int = 16
  pure val HEAVY_SLICE_MAX_GB: int = 64
  pure val MAX_GEN: int = 10

  type Backend = BackendNone | BackendOllama | BackendLlamaServer

  type ModelId =
    | ModelNone
    | ModelGlimmerQ4Llama     // 19 GB in heavy.slice (16.8G text + 1.6G DFlash, no mmproj)
    | ModelGlimmerQ3Ollama    // 15 GB in ollama.service (<=13.4G Q3 + 1.6G DFlash, no mmproj)
    | ModelGlimmerFastOllama  // 15 GB separate Ollama tag (demonstrates dual-mode thrashing)
    | ModelGlimmerQ4Ollama    // 20 GB in ollama.service (exceeds 16G cgroup cap!)
    | ModelGemma26bLlama      // 18 GB in heavy.slice
    | ModelQwen9bOllama       // 10 GB in ollama.service

  type ReasoningMode = ModeZeroThink | ModeLowThink

  type UiState = UiIdle | UiThinking | UiSpeaking

  type LlmPhase =
    | PhaseIdle
    | PhasePreStream         // Messages built, before opening HTTP stream (`brain.py:479-548`)
    | PhaseThinkingSelf      // Generating hidden `to=self` tokens (0 visible tokens yet)
    | PhaseSpeakingUser      // Streaming `to=user` visible tokens
    | PhaseRetryingZeroThink // Auto-retrying in ModeZeroThink after length-exhausted `to=self`
    | PhaseDrainingTts       // LLM finished, TTS still synthesizing queued sentences

  type SlotHolder =
    | SlotFree
    | SlotBusy({ clientId: int, gen: int, backend: Backend, zombie: bool })

  type WarmupReq = { seq: int, model: ModelId }

  type HistoryEntry = {
    clientId: int,
    gen: int,
    historyGen: int,
    spoken: bool,
    failed: bool,
    cancelled: bool,
  }

  type CudaActor =
    | CudaActorWhisperInit
    | CudaActorVoxcpmInit
    | CudaActorSttInfer
    | CudaActorTtsMegaCapture
    | CudaActorTtsStream

  type CudaState = {
    whisperBooted: bool,
    voxcpmBooted: bool,
    gpuLockHeld: bool,
    capturing: Set[CudaActor],
    runningOp: Set[CudaActor],
    capturedSteps: Set[int],
    capturingSteps: int,
    rngCorrupted: bool,
  }

  type RuntimeConfig = {
    singleModelDualMode: bool,
    enforceBackendExclusivity: bool,
    serializeWarmups: bool,
    checkCancelBeforeStream: bool,
    pipelineCancelsLlmOnFailure: bool,
    handleEmptyThinkingReply: bool,
    stripAtemSelfChannel: bool,
    commitHistoryAfterTts: bool,
    keepSystemPromptInvariant: bool,
    enforceCudaCaptureExclusivity: bool,
    scopedRetiredCancel: bool,
  }

  type ClientTurn = {
    selectedModel: ModelId,
    activeMode: ReasoningMode,
    uiState: UiState,
    turnGen: int,
    workerGen: int,
    historyGen: int,
    workerHistoryGen: int,
    phase: LlmPhase,
    cancelRequested: bool,
    retiredWorkerCleanupPending: bool,
    llmProducedSentence: bool,
    ttsSpokenSentence: bool,
    audioStarted: bool,
    orphanAudioStart: bool,
    historyCommittedThisTurn: bool,
    pendingCommitIndex: int, // -1 if none, else index in `history`
  }

  type Coverage = {
    zeroThinkTurn: bool,
    lowThinkTurn: bool,
    bargeInDuringThinking: bool,
    emptyThinkingRecovered: bool,
    backendSwitchClean: bool,
    ttsFailureCancelsLlm: bool,
    warmupSerialized: bool,
    resetMidTurn: bool,
    kvCachePreservedAcrossModes: bool,
    cudaGraphCaptureVerified: bool,
    nonDefaultStepsCaptured: bool,
    llmHttp500Recovered: bool,
    llmMidStreamDisconnectRecovered: bool,
    scopedRetiredCancelVerified: bool,
  }

  pure val fixedConfig: RuntimeConfig = {
    singleModelDualMode: true,
    enforceBackendExclusivity: true,
    serializeWarmups: true,
    checkCancelBeforeStream: true,
    pipelineCancelsLlmOnFailure: true,
    handleEmptyThinkingReply: true,
    stripAtemSelfChannel: true,
    commitHistoryAfterTts: true,
    keepSystemPromptInvariant: true,
    enforceCudaCaptureExclusivity: true,
    scopedRetiredCancel: true,
  }

  pure def modelBackend(m: ModelId): Backend =
    match m {
      | ModelNone => BackendNone
      | ModelGlimmerQ4Llama => BackendLlamaServer
      | ModelGemma26bLlama => BackendLlamaServer
      | _ => BackendOllama
    }

  pure def modelMemGb(m: ModelId): int =
    match m {
      | ModelNone => 0
      | ModelGlimmerQ4Llama => 19
      | ModelGlimmerQ3Ollama => 15
      | ModelGlimmerFastOllama => 15
      | ModelGlimmerQ4Ollama => 20
      | ModelGemma26bLlama => 18
      | ModelQwen9bOllama => 10
    }

  pure def isGlimmerModel(m: ModelId): bool =
    match m {
      | ModelGlimmerQ4Llama => true
      | ModelGlimmerQ3Ollama => true
      | ModelGlimmerFastOllama => true
      | ModelGlimmerQ4Ollama => true
      | _ => false
    }

  // Models legal to select under fixed vs unfixed residency rules
  pure def selectableModels(enforceExcl: bool): Set[ModelId] =
    if (enforceExcl) {
      Set(ModelGlimmerQ4Llama, ModelGlimmerQ3Ollama, ModelGemma26bLlama, ModelQwen9bOllama)
    } else {
      Set(
        ModelGlimmerQ4Llama,
        ModelGlimmerQ3Ollama,
        ModelGlimmerQ4Ollama,
        ModelGemma26bLlama,
        ModelQwen9bOllama,
      )
    }

  pure def isZombieSlot(s: SlotHolder): bool =
    match s {
      | SlotBusy(b) => b.zombie
      | SlotFree => false
    }

  pure def isSlotHeldBy(s: SlotHolder, cid: int, g: int): bool =
    match s {
      | SlotBusy(b) => b.clientId == cid and b.gen == g and not(b.zombie)
      | SlotFree => false
    }

  pure def isSlotFree(s: SlotHolder): bool =
    match s {
      | SlotFree => true
      | SlotBusy(_) => false
    }

  var cfg: RuntimeConfig
  var cuda: CudaState
  var turns: int -> ClientTurn
  var ollamaLoaded: ModelId
  var llamaServerLoaded: ModelId
  var activeGlobalModel: ModelId
  var llmSlot: SlotHolder
  var warmupSeq: int
  var pendingWarmups: Set[WarmupReq]
  var history: List[HistoryEntry]
  var lastKvPromptTag: int         // 0 = unset, 1 = invariant/zero, 2 = mutated-low-think
  var kvModeInvalidations: int
  var modeSwitchReloads: int
  var staleWarmupEvictions: int
  var silentEmptyDrops: int
  var ollamaCgroupOom: bool
  var dualBackendCollision: bool
  var atemLeakedToTts: bool
  var activePoolCrossKilled: bool
  var cov: Coverage

  pure val initCudaReady: CudaState = {
    whisperBooted: true,
    voxcpmBooted: true,
    gpuLockHeld: false,
    capturing: Set(),
    runningOp: Set(),
    capturedSteps: Set(6),
    capturingSteps: 0,
    rngCorrupted: false,
  }

  pure val initCudaBoot: CudaState = {
    whisperBooted: false,
    voxcpmBooted: false,
    gpuLockHeld: false,
    capturing: Set(),
    runningOp: Set(),
    capturedSteps: Set(),
    capturingSteps: 0,
    rngCorrupted: false,
  }

  pure def initTurn(m: ModelId): ClientTurn = {
    selectedModel: m,
    activeMode: ModeZeroThink,
    uiState: UiIdle,
    turnGen: 0,
    workerGen: 0,
    historyGen: 0,
    workerHistoryGen: 0,
    phase: PhaseIdle,
    cancelRequested: false,
    retiredWorkerCleanupPending: false,
    llmProducedSentence: false,
    ttsSpokenSentence: false,
    audioStarted: false,
    orphanAudioStart: false,
    historyCommittedThisTurn: false,
    pendingCommitIndex: -1,
  }

  pure val initCov: Coverage = {
    zeroThinkTurn: false,
    lowThinkTurn: false,
    bargeInDuringThinking: false,
    emptyThinkingRecovered: false,
    backendSwitchClean: false,
    ttsFailureCancelsLlm: false,
    warmupSerialized: false,
    resetMidTurn: false,
    kvCachePreservedAcrossModes: false,
    cudaGraphCaptureVerified: false,
    nonDefaultStepsCaptured: false,
    llmHttp500Recovered: false,
    llmMidStreamDisconnectRecovered: false,
    scopedRetiredCancelVerified: false,
  }

  action initBase(c: RuntimeConfig, startModel: ModelId, startCuda: CudaState): bool = all {
    cfg' = c,
    cuda' = startCuda,
    turns' = CLIENTS.mapBy(_ => initTurn(startModel)),
    ollamaLoaded' = if (modelBackend(startModel) == BackendOllama) startModel else ModelNone,
    llamaServerLoaded' = if (modelBackend(startModel) == BackendLlamaServer) startModel else ModelNone,
    activeGlobalModel' = startModel,
    llmSlot' = SlotFree,
    warmupSeq' = 0,
    pendingWarmups' = Set(),
    history' = List(),
    lastKvPromptTag' = 0,
    kvModeInvalidations' = 0,
    modeSwitchReloads' = 0,
    staleWarmupEvictions' = 0,
    silentEmptyDrops' = 0,
    ollamaCgroupOom' = (modelBackend(startModel) == BackendOllama and modelMemGb(startModel) > OLLAMA_CGROUP_MAX_GB),
    dualBackendCollision' = false,
    atemLeakedToTts' = false,
    activePoolCrossKilled' = false,
    cov' = initCov,
  }

  action initWithConfig(c: RuntimeConfig, startModel: ModelId): bool =
    initBase(c, startModel, initCudaReady)

  action initStartupWithConfig(c: RuntimeConfig, startModel: ModelId): bool =
    initBase(c, startModel, initCudaBoot)

  // Default initial state uses the fully fixed architecture with direct llama-server Glimmer Q4_K_M
  action init: bool = initWithConfig(fixedConfig, ModelGlimmerQ4Llama)

  // Startup initial state for verifying sequential WhisperSTT -> HalVoiceEngine boot
  action initStartup: bool = initStartupWithConfig(fixedConfig, ModelGlimmerQ4Llama)

  // --- Model Switching & Warmup Residency Actions ---

  // Client requests `set_model`. Under fixed config, model residency is runtime-coordinated
  // (no cross-tab eviction mid-turn, no >16G Ollama load, outgoing backend unloaded cleanly).
  action clientSetModel(id: int, target: ModelId): bool = {
    val t = turns.get(id)
    val targetBackend = modelBackend(target)
    val allowedTarget =
      target != ModelNone and
      (not(cfg.enforceBackendExclusivity) or targetBackend != BackendOllama or modelMemGb(target) <= OLLAMA_CGROUP_MAX_GB)
    // Under fixed config, backend switch waits for LLM slot to be idle so Tab 2 cannot evict Tab 1 mid-turn
    val slotReady = not(cfg.enforceBackendExclusivity) or isSlotFree(llmSlot)
    val nextSeq = warmupSeq + 1
    // Switching backend updates residency cleanly when enforceBackendExclusivity is true
    val nextOllama =
      if (targetBackend == BackendOllama) {
        if (cfg.serializeWarmups) ollamaLoaded else ollamaLoaded
      } else {
        if (cfg.enforceBackendExclusivity) ModelNone else ollamaLoaded
      }
    val nextLlama =
      if (targetBackend == BackendLlamaServer) target
      else if (cfg.enforceBackendExclusivity) ModelNone
      else llamaServerLoaded
    val collisionNow = nextOllama != ModelNone and nextLlama != ModelNone
    val didSwitchBackend =
      modelBackend(activeGlobalModel) != BackendNone and
      modelBackend(activeGlobalModel) != targetBackend and
      not(collisionNow)
    // When enforceBackendExclusivity is true, synchronize all tabs to the single active runtime model
    // so multi-tab sessions never thrash OLLAMA_MAX_LOADED_MODELS=1.
    val nextTurns =
      if (cfg.enforceBackendExclusivity) {
        CLIENTS.mapBy(i => { ...turns.get(i), selectedModel: target })
      } else {
        turns.put(id, { ...t, selectedModel: target })
      }
    val nextWarmups =
      if (targetBackend == BackendOllama) {
        if (cfg.serializeWarmups) Set({ seq: nextSeq, model: target })
        else pendingWarmups.union(Set({ seq: nextSeq, model: target }))
      } else {
        if (cfg.serializeWarmups) Set() else pendingWarmups
      }
    all {
      allowedTarget,
      slotReady,
      warmupSeq < MAX_GEN,
      turns' = nextTurns,
      activeGlobalModel' = target,
      ollamaLoaded' = nextOllama,
      llamaServerLoaded' = nextLlama,
      warmupSeq' = nextSeq,
      pendingWarmups' = nextWarmups,
      dualBackendCollision' = (dualBackendCollision or collisionNow),
      ollamaCgroupOom' = ollamaCgroupOom,
      llmSlot' = llmSlot,
      history' = history,
      lastKvPromptTag' = if (target != activeGlobalModel) 0 else lastKvPromptTag,
      kvModeInvalidations' = kvModeInvalidations,
      modeSwitchReloads' = modeSwitchReloads,
      staleWarmupEvictions' = staleWarmupEvictions,
      silentEmptyDrops' = silentEmptyDrops,
      atemLeakedToTts' = atemLeakedToTts,
      activePoolCrossKilled' = activePoolCrossKilled,
      cfg' = cfg,
      cuda' = cuda,
      cov' = {
        ...cov,
        backendSwitchClean: cov.backendSwitchClean or didSwitchBackend,
      },
    }
  }

  // An enqueued async Ollama warmup HTTP POST executes (`server.py:219-256`).
  // Without `serializeWarmups`, an older `seq < warmupSeq` POST can arrive AFTER a newer `set_model`
  // and evict the newly loaded model!
  action executeOllamaWarmup(req: WarmupReq): bool = {
    val isStale = req.seq < warmupSeq
    val cgroupExceeded = modelMemGb(req.model) > OLLAMA_CGROUP_MAX_GB
    val wouldEvictActive =
      (ollamaLoaded != ModelNone and ollamaLoaded != req.model) or
      (activeGlobalModel != req.model)
    val staleEviction = isStale and wouldEvictActive
    val nextLlama = if (cfg.enforceBackendExclusivity) ModelNone else llamaServerLoaded
    val collisionNow = not(cgroupExceeded) and nextLlama != ModelNone
    all {
      req.in(pendingWarmups),
      // Under serialized warmups, stale requests are discarded and warmups wait for idle slot
      not(cfg.serializeWarmups) or (not(isStale) and isSlotFree(llmSlot)),
      pendingWarmups' = pendingWarmups.exclude(Set(req)),
      ollamaLoaded' = if (cgroupExceeded) ollamaLoaded else req.model,
      llamaServerLoaded' = nextLlama,
      ollamaCgroupOom' = (ollamaCgroupOom or cgroupExceeded),
      dualBackendCollision' = (dualBackendCollision or collisionNow),
      staleWarmupEvictions' = staleWarmupEvictions + (if (staleEviction) 1 else 0),
      turns' = turns,
      activeGlobalModel' = activeGlobalModel,
      llmSlot' = llmSlot,
      warmupSeq' = warmupSeq,
      history' = history,
      lastKvPromptTag' = lastKvPromptTag,
      kvModeInvalidations' = kvModeInvalidations,
      modeSwitchReloads' = modeSwitchReloads,
      silentEmptyDrops' = silentEmptyDrops,
      atemLeakedToTts' = atemLeakedToTts,
      activePoolCrossKilled' = activePoolCrossKilled,
      cfg' = cfg,
      cuda' = cuda,
      cov' = { ...cov, warmupSerialized: cov.warmupSerialized or cfg.serializeWarmups },
    }
  }

  // --- Turn Lifecycle & Dual-Mode Routing ---

  // User submits a turn routed to either ModeZeroThink (casual/instant) or ModeLowThink (complex/web).
  action startTurn(id: int, mode: ReasoningMode): bool = {
    val t = turns.get(id)
    val nextGen = t.turnGen + 1
    // Naive dual-mode routing on Ollama switches between two model tags (thrashing OLLAMA_MAX_LOADED_MODELS=1!)
    val effectiveModel =
      if (not(cfg.singleModelDualMode) and modelBackend(t.selectedModel) == BackendOllama and isGlimmerModel(t.selectedModel)) {
        if (mode == ModeZeroThink) ModelGlimmerFastOllama else ModelGlimmerQ3Ollama
      } else {
        t.selectedModel
      }
    // Cancelling any prior active turn on `id` frees its HTTP stream if it was already open
    val nextSlot = match llmSlot {
      | SlotBusy(b) =>
          if (b.clientId == id and b.gen == t.workerGen) SlotFree
          else llmSlot
      | SlotFree => SlotFree
    }
    val wasThinking = t.phase == PhaseThinkingSelf
    all {
      t.turnGen < MAX_GEN,
      // Single-tab or idle-slot start; barge-in allowed on own active turn
      t.phase == PhaseIdle or t.phase == PhasePreStream or t.phase == PhaseThinkingSelf or t.phase == PhaseSpeakingUser or t.phase == PhaseDrainingTts,
      turns' = turns.put(id, {
        ...t,
        selectedModel: effectiveModel,
        activeMode: mode,
        uiState: UiThinking,
        turnGen: nextGen,
        workerGen: nextGen,
        workerHistoryGen: t.historyGen,
        phase: PhasePreStream,
        cancelRequested: false,
        retiredWorkerCleanupPending: t.phase != PhaseIdle or t.retiredWorkerCleanupPending,
        llmProducedSentence: false,
        ttsSpokenSentence: false,
        audioStarted: false,
        historyCommittedThisTurn: false,
        pendingCommitIndex: -1,
      }),
      llmSlot' = nextSlot,
      ollamaLoaded' = ollamaLoaded,
      llamaServerLoaded' = llamaServerLoaded,
      activeGlobalModel' = activeGlobalModel,
      warmupSeq' = warmupSeq,
      pendingWarmups' = pendingWarmups,
      history' = history,
      lastKvPromptTag' = lastKvPromptTag,
      kvModeInvalidations' = kvModeInvalidations,
      modeSwitchReloads' = modeSwitchReloads,
      staleWarmupEvictions' = staleWarmupEvictions,
      silentEmptyDrops' = silentEmptyDrops,
      ollamaCgroupOom' = ollamaCgroupOom,
      dualBackendCollision' = dualBackendCollision,
      atemLeakedToTts' = atemLeakedToTts,
      activePoolCrossKilled' = activePoolCrossKilled,
      cfg' = cfg,
      cuda' = cuda,
      cov' = {
        ...cov,
        bargeInDuringThinking: cov.bargeInDuringThinking or wasThinking,
      },
    }
  }

  // Models the exact `brain.py:479-548` race: a worker in `PhasePreStream` gets cancelled
  // (e.g. Escape / barge-in) BEFORE it creates `httpx.Client` and opens the HTTP stream.
  action cancelInPreStream(id: int): bool = {
    val t = turns.get(id)
    all {
      t.phase == PhasePreStream,
      not(t.cancelRequested),
      t.turnGen < MAX_GEN,
      turns' = turns.put(id, {
        ...t,
        uiState: UiIdle,
        turnGen: t.turnGen + 1,
        cancelRequested: true,
      }),
      llmSlot' = llmSlot,
      ollamaLoaded' = ollamaLoaded,
      llamaServerLoaded' = llamaServerLoaded,
      activeGlobalModel' = activeGlobalModel,
      warmupSeq' = warmupSeq,
      pendingWarmups' = pendingWarmups,
      history' = history,
      lastKvPromptTag' = lastKvPromptTag,
      kvModeInvalidations' = kvModeInvalidations,
      modeSwitchReloads' = modeSwitchReloads,
      staleWarmupEvictions' = staleWarmupEvictions,
      silentEmptyDrops' = silentEmptyDrops,
      ollamaCgroupOom' = ollamaCgroupOom,
      dualBackendCollision' = dualBackendCollision,
      atemLeakedToTts' = atemLeakedToTts,
      activePoolCrossKilled' = activePoolCrossKilled,
      cfg' = cfg,
      cuda' = cuda,
      cov' = cov,
    }
  }

  // Worker opens HTTP stream to Ollama or direct `llama-server` (`brain.py:540-548`).
  action llmOpenStream(id: int): bool = {
    val t = turns.get(id)
    val m = t.selectedModel
    val b = modelBackend(m)
    val cgroupExceeded = b == BackendOllama and modelMemGb(m) > OLLAMA_CGROUP_MAX_GB
    val modelEvictedOnOllama = b == BackendOllama and ollamaLoaded != ModelNone and ollamaLoaded != m
    val isModeThrash =
      modelEvictedOnOllama and isGlimmerModel(ollamaLoaded) and isGlimmerModel(m)
    val nextOllama =
      if (b == BackendOllama) (if (cgroupExceeded) ollamaLoaded else m)
      else if (cfg.enforceBackendExclusivity) ModelNone
      else ollamaLoaded
    val nextLlama =
      if (b == BackendLlamaServer) m
      else if (cfg.enforceBackendExclusivity) ModelNone
      else llamaServerLoaded
    val collisionNow = nextOllama != ModelNone and nextLlama != ModelNone
    // KV prefix check: mutating messages[0] between ZeroThink and LowThink invalidates KV cache
    val promptTag =
      if (cfg.keepSystemPromptInvariant) 1
      else if (t.activeMode == ModeZeroThink) 1
      else 2
    val kvInvalidated = lastKvPromptTag != 0 and lastKvPromptTag != promptTag
    val kvPreserved =
      cfg.keepSystemPromptInvariant and lastKvPromptTag == 1 and
      (t.activeMode == ModeLowThink or t.activeMode == ModeZeroThink)
    val nextPhase =
      if (isGlimmerModel(m) and t.activeMode == ModeLowThink) PhaseThinkingSelf
      else PhaseSpeakingUser
    all {
      t.phase == PhasePreStream,
      isSlotFree(llmSlot),
      if (t.cancelRequested) {
        if (cfg.checkCancelBeforeStream) {
          // Fixed: atomic `stop_check()` under `_lock` aborts BEFORE opening HTTP stream!
          all {
            turns' = turns.put(id, { ...t, phase: PhaseIdle, cancelRequested: false }),
            llmSlot' = SlotFree,
            ollamaLoaded' = ollamaLoaded,
            llamaServerLoaded' = llamaServerLoaded,
            ollamaCgroupOom' = ollamaCgroupOom,
            dualBackendCollision' = dualBackendCollision,
            modeSwitchReloads' = modeSwitchReloads,
            lastKvPromptTag' = lastKvPromptTag,
            kvModeInvalidations' = kvModeInvalidations,
            cov' = cov,
          }
        } else {
          // Bug 2a (`brain.py:540-548`): missed pre-stream `stop_check()` opens a ZOMBIE stream!
          all {
            turns' = turns.put(id, { ...t, phase: PhaseIdle }),
            llmSlot' = SlotBusy({ clientId: id, gen: t.workerGen, backend: b, zombie: true }),
            ollamaLoaded' = nextOllama,
            llamaServerLoaded' = nextLlama,
            ollamaCgroupOom' = (ollamaCgroupOom or cgroupExceeded),
            dualBackendCollision' = (dualBackendCollision or collisionNow),
            modeSwitchReloads' = modeSwitchReloads + (if (isModeThrash) 1 else 0),
            lastKvPromptTag' = promptTag,
            kvModeInvalidations' = kvModeInvalidations + (if (kvInvalidated) 1 else 0),
            cov' = cov,
          }
        }
      } else {
        all {
          turns' = turns.put(id, { ...t, phase: nextPhase }),
          llmSlot' = SlotBusy({ clientId: id, gen: t.workerGen, backend: b, zombie: false }),
          ollamaLoaded' = nextOllama,
          llamaServerLoaded' = nextLlama,
          ollamaCgroupOom' = (ollamaCgroupOom or cgroupExceeded),
          dualBackendCollision' = (dualBackendCollision or collisionNow),
          modeSwitchReloads' = modeSwitchReloads + (if (isModeThrash) 1 else 0),
          lastKvPromptTag' = promptTag,
          kvModeInvalidations' = kvModeInvalidations + (if (kvInvalidated) 1 else 0),
          cov' = {
            ...cov,
            zeroThinkTurn: cov.zeroThinkTurn or (t.activeMode == ModeZeroThink),
            lowThinkTurn: cov.lowThinkTurn or (t.activeMode == ModeLowThink),
            kvCachePreservedAcrossModes: cov.kvCachePreservedAcrossModes or kvPreserved,
          },
        }
      },
      activeGlobalModel' = activeGlobalModel,
      warmupSeq' = warmupSeq,
      pendingWarmups' = pendingWarmups,
      history' = history,
      staleWarmupEvictions' = staleWarmupEvictions,
      silentEmptyDrops' = silentEmptyDrops,
      atemLeakedToTts' = atemLeakedToTts,
      activePoolCrossKilled' = activePoolCrossKilled,
      cfg' = cfg,
      cuda' = cuda,
    }
  }

  // Hidden `to=self` reasoning phase transitions cleanly to `to=user` visible speech.
  // If raw ATEM tokens appear without `stripAtemSelfChannel`, `to=self` leaks into TTS!
  action llmThinkingToSpeaking(id: int, rawAtemChannel: bool): bool = {
    val t = turns.get(id)
    val leaked = rawAtemChannel and not(cfg.stripAtemSelfChannel)
    all {
      t.phase == PhaseThinkingSelf,
      isSlotHeldBy(llmSlot, id, t.workerGen),
      turns' = turns.put(id, { ...t, phase: PhaseSpeakingUser }),
      atemLeakedToTts' = (atemLeakedToTts or leaked),
      activePoolCrossKilled' = activePoolCrossKilled,
      llmSlot' = llmSlot,
      ollamaLoaded' = ollamaLoaded,
      llamaServerLoaded' = llamaServerLoaded,
      activeGlobalModel' = activeGlobalModel,
      warmupSeq' = warmupSeq,
      pendingWarmups' = pendingWarmups,
      history' = history,
      lastKvPromptTag' = lastKvPromptTag,
      kvModeInvalidations' = kvModeInvalidations,
      modeSwitchReloads' = modeSwitchReloads,
      staleWarmupEvictions' = staleWarmupEvictions,
      silentEmptyDrops' = silentEmptyDrops,
      ollamaCgroupOom' = ollamaCgroupOom,
      dualBackendCollision' = dualBackendCollision,
      cfg' = cfg,
      cuda' = cuda,
      cov' = cov,
    }
  }

  // Glimmer hits `num_predict` / timeout while still in `to=self` thinking -> 0 visible tokens (`reply=""`).
  action llmExhaustedInThinking(id: int): bool = {
    val t = turns.get(id)
    all {
      t.phase == PhaseThinkingSelf,
      isSlotHeldBy(llmSlot, id, t.workerGen),
      if (cfg.handleEmptyThinkingReply) {
        // Fixed: retry once in zero-think mode (prefilling empty `to=self`) without emitting orphan audio_start!
        all {
          turns' = turns.put(id, {
            ...t,
            activeMode: ModeZeroThink,
            phase: PhaseRetryingZeroThink,
          }),
          llmSlot' = SlotFree,
          silentEmptyDrops' = silentEmptyDrops,
          cov' = { ...cov, emptyThinkingRecovered: true },
        }
      } else {
        // Bug 3a (`brain.py:571-599`): silently drops empty reply and snaps back to UiIdle with no response/error!
        all {
          turns' = turns.put(id, {
            ...t,
            phase: PhaseIdle,
            uiState: UiIdle,
          }),
          llmSlot' = SlotFree,
          silentEmptyDrops' = silentEmptyDrops + 1,
          cov' = cov,
        }
      },
      ollamaLoaded' = ollamaLoaded,
      llamaServerLoaded' = llamaServerLoaded,
      activeGlobalModel' = activeGlobalModel,
      warmupSeq' = warmupSeq,
      pendingWarmups' = pendingWarmups,
      history' = history,
      lastKvPromptTag' = lastKvPromptTag,
      kvModeInvalidations' = kvModeInvalidations,
      modeSwitchReloads' = modeSwitchReloads,
      staleWarmupEvictions' = staleWarmupEvictions,
      ollamaCgroupOom' = ollamaCgroupOom,
      dualBackendCollision' = dualBackendCollision,
      atemLeakedToTts' = atemLeakedToTts,
      activePoolCrossKilled' = activePoolCrossKilled,
      cfg' = cfg,
      cuda' = cuda,
    }
  }

  // Complete zero-think fallback retry after `to=self` hit `num_predict`: emits clean spoken reply or turn error.
  action llmZeroThinkRetryComplete(id: int, succeeded: bool): bool = {
    val t = turns.get(id)
    val b = modelBackend(t.selectedModel)
    all {
      t.phase == PhaseRetryingZeroThink,
      isSlotFree(llmSlot),
      if (succeeded) {
        all {
          turns' = turns.put(id, {
            ...t,
            phase: PhaseSpeakingUser,
          }),
          llmSlot' = SlotBusy({ clientId: id, gen: t.workerGen, backend: b, zombie: false }),
        }
      } else {
        // Explicit turn error + clean return to UiIdle, never orphan audio_start
        all {
          turns' = turns.put(id, {
            ...t,
            phase: PhaseIdle,
            uiState: UiIdle,
          }),
          llmSlot' = SlotFree,
        }
      },
      ollamaLoaded' = ollamaLoaded,
      llamaServerLoaded' = llamaServerLoaded,
      activeGlobalModel' = activeGlobalModel,
      warmupSeq' = warmupSeq,
      pendingWarmups' = pendingWarmups,
      history' = history,
      lastKvPromptTag' = lastKvPromptTag,
      kvModeInvalidations' = kvModeInvalidations,
      modeSwitchReloads' = modeSwitchReloads,
      staleWarmupEvictions' = staleWarmupEvictions,
      silentEmptyDrops' = silentEmptyDrops,
      ollamaCgroupOom' = ollamaCgroupOom,
      dualBackendCollision' = dualBackendCollision,
      atemLeakedToTts' = atemLeakedToTts,
      activePoolCrossKilled' = activePoolCrossKilled,
      cfg' = cfg,
      cuda' = cuda,
      cov' = cov,
    }
  }

  // LLM emits a visible `to=user` sentence into the bounded queue.
  action llmProduceSentence(id: int): bool = {
    val t = turns.get(id)
    all {
      t.phase == PhaseSpeakingUser,
      isSlotHeldBy(llmSlot, id, t.workerGen),
      turns' = turns.put(id, { ...t, llmProducedSentence: true }),
      llmSlot' = llmSlot,
      ollamaLoaded' = ollamaLoaded,
      llamaServerLoaded' = llamaServerLoaded,
      activeGlobalModel' = activeGlobalModel,
      warmupSeq' = warmupSeq,
      pendingWarmups' = pendingWarmups,
      history' = history,
      lastKvPromptTag' = lastKvPromptTag,
      kvModeInvalidations' = kvModeInvalidations,
      modeSwitchReloads' = modeSwitchReloads,
      staleWarmupEvictions' = staleWarmupEvictions,
      silentEmptyDrops' = silentEmptyDrops,
      ollamaCgroupOom' = ollamaCgroupOom,
      dualBackendCollision' = dualBackendCollision,
      atemLeakedToTts' = atemLeakedToTts,
      activePoolCrossKilled' = activePoolCrossKilled,
      cfg' = cfg,
      cuda' = cuda,
      cov' = cov,
    }
  }

  // LLM finishes streaming all tokens ahead of TTS (common with fast DFlash decode!).
  // Bug 4a (`brain.py:598`): if `commitHistoryAfterTts == false`, `_llm_producer` commits to `history`
  // BEFORE TTS finishes speaking, so a subsequent TTS failure or pre-speech barge-in poisons `history`!
  action llmFinishAheadOfTts(id: int): bool = {
    val t = turns.get(id)
    val shouldPrematureCommit =
      not(cfg.commitHistoryAfterTts) and
      t.llmProducedSentence and
      t.workerHistoryGen == t.historyGen and
      not(t.historyCommittedThisTurn)
    val entry: HistoryEntry = {
      clientId: id,
      gen: t.workerGen,
      historyGen: t.workerHistoryGen,
      spoken: t.ttsSpokenSentence,
      failed: false,
      cancelled: false,
    }
    val nextHistory = if (shouldPrematureCommit) history.append(entry) else history
    val nextIdx = if (shouldPrematureCommit) history.length() else t.pendingCommitIndex
    all {
      t.phase == PhaseSpeakingUser,
      t.llmProducedSentence,
      isSlotHeldBy(llmSlot, id, t.workerGen),
      turns' = turns.put(id, {
        ...t,
        phase: PhaseDrainingTts,
        historyCommittedThisTurn: t.historyCommittedThisTurn or shouldPrematureCommit,
        pendingCommitIndex: nextIdx,
      }),
      llmSlot' = SlotFree,
      history' = nextHistory,
      ollamaLoaded' = ollamaLoaded,
      llamaServerLoaded' = llamaServerLoaded,
      activeGlobalModel' = activeGlobalModel,
      warmupSeq' = warmupSeq,
      pendingWarmups' = pendingWarmups,
      lastKvPromptTag' = lastKvPromptTag,
      kvModeInvalidations' = kvModeInvalidations,
      modeSwitchReloads' = modeSwitchReloads,
      staleWarmupEvictions' = staleWarmupEvictions,
      silentEmptyDrops' = silentEmptyDrops,
      ollamaCgroupOom' = ollamaCgroupOom,
      dualBackendCollision' = dualBackendCollision,
      atemLeakedToTts' = atemLeakedToTts,
      activePoolCrossKilled' = activePoolCrossKilled,
      cfg' = cfg,
      cuda' = cuda,
      cov' = cov,
    }
  }

  // TTS synthesizes a sentence and emits deferred `audio_start` + PCM chunk.
  action ttsSpeakSentence(id: int): bool = {
    val t = turns.get(id)
    val updatedHistory =
      if (t.pendingCommitIndex >= 0 and t.pendingCommitIndex < history.length()) {
        val old = history[t.pendingCommitIndex]
        history.replaceAt(t.pendingCommitIndex, { ...old, spoken: true })
      } else history
    all {
      t.phase == PhaseSpeakingUser or t.phase == PhaseDrainingTts,
      t.llmProducedSentence,
      turns' = turns.put(id, {
        ...t,
        uiState: UiSpeaking,
        audioStarted: true,
        ttsSpokenSentence: true,
      }),
      history' = updatedHistory,
      llmSlot' = llmSlot,
      ollamaLoaded' = ollamaLoaded,
      llamaServerLoaded' = llamaServerLoaded,
      activeGlobalModel' = activeGlobalModel,
      warmupSeq' = warmupSeq,
      pendingWarmups' = pendingWarmups,
      lastKvPromptTag' = lastKvPromptTag,
      kvModeInvalidations' = kvModeInvalidations,
      modeSwitchReloads' = modeSwitchReloads,
      staleWarmupEvictions' = staleWarmupEvictions,
      silentEmptyDrops' = silentEmptyDrops,
      ollamaCgroupOom' = ollamaCgroupOom,
      dualBackendCollision' = dualBackendCollision,
      atemLeakedToTts' = atemLeakedToTts,
      activePoolCrossKilled' = activePoolCrossKilled,
      cfg' = cfg,
      cuda' = cuda,
      cov' = cov,
    }
  }

  // Pipeline completes TTS normally, commits spoken reply to history (if not reset), and returns to UiIdle.
  action pipelineCompleteClean(id: int): bool = {
    val t = turns.get(id)
    val shouldCommit =
      cfg.commitHistoryAfterTts and
      t.ttsSpokenSentence and
      t.workerHistoryGen == t.historyGen and
      not(t.cancelRequested) and
      not(t.historyCommittedThisTurn)
    val entry: HistoryEntry = {
      clientId: id,
      gen: t.workerGen,
      historyGen: t.workerHistoryGen,
      spoken: true,
      failed: false,
      cancelled: false,
    }
    all {
      t.phase == PhaseDrainingTts,
      t.ttsSpokenSentence,
      turns' = turns.put(id, {
        ...t,
        phase: PhaseIdle,
        uiState: UiIdle,
        audioStarted: false,
        historyCommittedThisTurn: t.historyCommittedThisTurn or shouldCommit,
        pendingCommitIndex: -1,
      }),
      history' = if (shouldCommit) history.append(entry) else history,
      llmSlot' = llmSlot,
      ollamaLoaded' = ollamaLoaded,
      llamaServerLoaded' = llamaServerLoaded,
      activeGlobalModel' = activeGlobalModel,
      warmupSeq' = warmupSeq,
      pendingWarmups' = pendingWarmups,
      lastKvPromptTag' = lastKvPromptTag,
      kvModeInvalidations' = kvModeInvalidations,
      modeSwitchReloads' = modeSwitchReloads,
      staleWarmupEvictions' = staleWarmupEvictions,
      silentEmptyDrops' = silentEmptyDrops,
      ollamaCgroupOom' = ollamaCgroupOom,
      dualBackendCollision' = dualBackendCollision,
      atemLeakedToTts' = atemLeakedToTts,
      activePoolCrossKilled' = activePoolCrossKilled,
      cfg' = cfg,
      cuda' = cuda,
      cov' = cov,
    }
  }

  // TTS fails with GPU error while LLM may STILL be streaming (`PhaseSpeakingUser`) or already finished (`PhaseDrainingTts`).
  // Bug 2b (`pipeline.py:285,307`): if `pipelineCancelsLlmOnFailure == false`, an in-flight LLM stream is NOT
  // aborted and becomes a zombie holding the single LLM slot!
  // Bug 4a (`brain.py:598`): if LLM already committed before TTS failed, `history` is poisoned with a failed turn!
  action ttsFailMidTurn(id: int): bool = {
    val t = turns.get(id)
    val llmStillActive = isSlotHeldBy(llmSlot, id, t.workerGen)
    val nextSlot =
      if (llmStillActive) {
        if (cfg.pipelineCancelsLlmOnFailure) SlotFree
        else match llmSlot {
          | SlotBusy(b) => SlotBusy({ ...b, zombie: true })
          | SlotFree => SlotFree
        }
      } else llmSlot
    val updatedHistory =
      if (t.pendingCommitIndex >= 0 and t.pendingCommitIndex < history.length()) {
        val old = history[t.pendingCommitIndex]
        history.replaceAt(t.pendingCommitIndex, { ...old, failed: true })
      } else history
    all {
      (t.phase == PhaseSpeakingUser or t.phase == PhaseDrainingTts),
      t.llmProducedSentence,
      turns' = turns.put(id, {
        ...t,
        phase: PhaseIdle,
        uiState: UiIdle,
        audioStarted: false,
        pendingCommitIndex: -1,
      }),
      llmSlot' = nextSlot,
      history' = updatedHistory,
      ollamaLoaded' = ollamaLoaded,
      llamaServerLoaded' = llamaServerLoaded,
      activeGlobalModel' = activeGlobalModel,
      warmupSeq' = warmupSeq,
      pendingWarmups' = pendingWarmups,
      lastKvPromptTag' = lastKvPromptTag,
      kvModeInvalidations' = kvModeInvalidations,
      modeSwitchReloads' = modeSwitchReloads,
      staleWarmupEvictions' = staleWarmupEvictions,
      silentEmptyDrops' = silentEmptyDrops,
      ollamaCgroupOom' = ollamaCgroupOom,
      dualBackendCollision' = dualBackendCollision,
      atemLeakedToTts' = atemLeakedToTts,
      activePoolCrossKilled' = activePoolCrossKilled,
      cfg' = cfg,
      cuda' = cuda,
      cov' = {
        ...cov,
        ttsFailureCancelsLlm: cov.ttsFailureCancelsLlm or (llmStillActive and cfg.pipelineCancelsLlmOnFailure),
      },
    }
  }

  // User presses Escape / interrupts during thinking (`to=self`), speaking, or TTS draining.
  action clientCancelEscape(id: int): bool = {
    val t = turns.get(id)
    val wasThinking = t.phase == PhaseThinkingSelf
    val nextSlot = match llmSlot {
      | SlotBusy(b) =>
          if (b.clientId == id and b.gen == t.workerGen) SlotFree
          else llmSlot
      | SlotFree => SlotFree
    }
    val updatedHistory =
      if (t.pendingCommitIndex >= 0 and t.pendingCommitIndex < history.length()) {
        val old = history[t.pendingCommitIndex]
        history.replaceAt(t.pendingCommitIndex, { ...old, cancelled: true })
      } else history
    all {
      t.phase == PhaseThinkingSelf or t.phase == PhaseRetryingZeroThink or t.phase == PhaseSpeakingUser or t.phase == PhaseDrainingTts,
      t.turnGen < MAX_GEN,
      turns' = turns.put(id, {
        ...t,
        phase: PhaseIdle,
        uiState: UiIdle,
        turnGen: t.turnGen + 1,
        cancelRequested: true,
        audioStarted: false,
        pendingCommitIndex: -1,
      }),
      llmSlot' = nextSlot,
      history' = updatedHistory,
      ollamaLoaded' = ollamaLoaded,
      llamaServerLoaded' = llamaServerLoaded,
      activeGlobalModel' = activeGlobalModel,
      warmupSeq' = warmupSeq,
      pendingWarmups' = pendingWarmups,
      lastKvPromptTag' = lastKvPromptTag,
      kvModeInvalidations' = kvModeInvalidations,
      modeSwitchReloads' = modeSwitchReloads,
      staleWarmupEvictions' = staleWarmupEvictions,
      silentEmptyDrops' = silentEmptyDrops,
      ollamaCgroupOom' = ollamaCgroupOom,
      dualBackendCollision' = dualBackendCollision,
      atemLeakedToTts' = atemLeakedToTts,
      activePoolCrossKilled' = activePoolCrossKilled,
      cfg' = cfg,
      cuda' = cuda,
      cov' = {
        ...cov,
        bargeInDuringThinking: cov.bargeInDuringThinking or wasThinking,
      },
    }
  }

  // User clicks Reset: clears history, bumps `historyGen`, aborts any in-flight stream cleanly.
  action clientReset(id: int): bool = {
    val t = turns.get(id)
    val wasActive = t.phase != PhaseIdle
    val nextSlot = match llmSlot {
      | SlotBusy(b) =>
          if (b.clientId == id and b.gen == t.workerGen) SlotFree
          else llmSlot
      | SlotFree => SlotFree
    }
    all {
      t.turnGen < MAX_GEN,
      t.historyGen < MAX_GEN,
      turns' = turns.put(id, {
        ...t,
        phase: PhaseIdle,
        uiState: UiIdle,
        turnGen: t.turnGen + 1,
        historyGen: t.historyGen + 1,
        cancelRequested: true,
        audioStarted: false,
        pendingCommitIndex: -1,
      }),
      llmSlot' = nextSlot,
      history' = List(),
      ollamaLoaded' = ollamaLoaded,
      llamaServerLoaded' = llamaServerLoaded,
      activeGlobalModel' = activeGlobalModel,
      warmupSeq' = warmupSeq,
      pendingWarmups' = pendingWarmups,
      lastKvPromptTag' = 0,
      kvModeInvalidations' = kvModeInvalidations,
      modeSwitchReloads' = modeSwitchReloads,
      staleWarmupEvictions' = staleWarmupEvictions,
      silentEmptyDrops' = silentEmptyDrops,
      ollamaCgroupOom' = ollamaCgroupOom,
      dualBackendCollision' = dualBackendCollision,
      atemLeakedToTts' = atemLeakedToTts,
      activePoolCrossKilled' = activePoolCrossKilled,
      cfg' = cfg,
      cuda' = cuda,
      cov' = { ...cov, resetMidTurn: cov.resetMidTurn or wasActive },
    }
  }

  // llama-server (or Ollama) returns HTTP 500 before streaming begins (or on zero-think retry):
  // slot stays free, UI returns cleanly to UiIdle with turn error, 0 orphan audio_start, clean history.
  action llmHttpError500(id: int): bool = {
    val t = turns.get(id)
    all {
      t.phase == PhasePreStream or t.phase == PhaseRetryingZeroThink,
      not(t.cancelRequested),
      isSlotFree(llmSlot),
      turns' = turns.put(id, {
        ...t,
        phase: PhaseIdle,
        uiState: UiIdle,
        audioStarted: false,
        pendingCommitIndex: -1,
      }),
      llmSlot' = SlotFree,
      history' = history,
      ollamaLoaded' = ollamaLoaded,
      llamaServerLoaded' = llamaServerLoaded,
      activeGlobalModel' = activeGlobalModel,
      warmupSeq' = warmupSeq,
      pendingWarmups' = pendingWarmups,
      lastKvPromptTag' = lastKvPromptTag,
      kvModeInvalidations' = kvModeInvalidations,
      modeSwitchReloads' = modeSwitchReloads,
      staleWarmupEvictions' = staleWarmupEvictions,
      silentEmptyDrops' = silentEmptyDrops,
      ollamaCgroupOom' = ollamaCgroupOom,
      dualBackendCollision' = dualBackendCollision,
      atemLeakedToTts' = atemLeakedToTts,
      activePoolCrossKilled' = activePoolCrossKilled,
      cfg' = cfg,
      cuda' = cuda,
      cov' = { ...cov, llmHttp500Recovered: true },
    }
  }

  // llama-server disconnects mid-stream (TCP drop / truncated SSE / missing [DONE]) during thinking
  // or speaking: releases slot, emits audio_end(failed=true) if audio started + error + UiIdle,
  // and never commits partial/failed turn to history.
  action llmDisconnectMidStream(id: int): bool = {
    val t = turns.get(id)
    all {
      t.phase == PhaseThinkingSelf or t.phase == PhaseSpeakingUser,
      isSlotHeldBy(llmSlot, id, t.workerGen),
      turns' = turns.put(id, {
        ...t,
        phase: PhaseIdle,
        uiState: UiIdle,
        audioStarted: false,
        pendingCommitIndex: -1,
      }),
      llmSlot' = SlotFree,
      history' = history,
      ollamaLoaded' = ollamaLoaded,
      llamaServerLoaded' = llamaServerLoaded,
      activeGlobalModel' = activeGlobalModel,
      warmupSeq' = warmupSeq,
      pendingWarmups' = pendingWarmups,
      lastKvPromptTag' = lastKvPromptTag,
      kvModeInvalidations' = kvModeInvalidations,
      modeSwitchReloads' = modeSwitchReloads,
      staleWarmupEvictions' = staleWarmupEvictions,
      silentEmptyDrops' = silentEmptyDrops,
      ollamaCgroupOom' = ollamaCgroupOom,
      dualBackendCollision' = dualBackendCollision,
      atemLeakedToTts' = atemLeakedToTts,
      activePoolCrossKilled' = activePoolCrossKilled,
      cfg' = cfg,
      cuda' = cuda,
      cov' = { ...cov, llmMidStreamDisconnectRecovered: true },
    }
  }

  // --- CUDA Graph Capture / Process-Global CUDAGeneratorImpl RNG Exclusivity Actions ---

  action unchangedNonCudaState: bool = all {
    turns' = turns,
    ollamaLoaded' = ollamaLoaded,
    llamaServerLoaded' = llamaServerLoaded,
    activeGlobalModel' = activeGlobalModel,
    llmSlot' = llmSlot,
    warmupSeq' = warmupSeq,
    pendingWarmups' = pendingWarmups,
    history' = history,
    lastKvPromptTag' = lastKvPromptTag,
    kvModeInvalidations' = kvModeInvalidations,
    modeSwitchReloads' = modeSwitchReloads,
    staleWarmupEvictions' = staleWarmupEvictions,
    silentEmptyDrops' = silentEmptyDrops,
    ollamaCgroupOom' = ollamaCgroupOom,
    dualBackendCollision' = dualBackendCollision,
    atemLeakedToTts' = atemLeakedToTts,
    activePoolCrossKilled' = activePoolCrossKilled,
    cfg' = cfg,
  }

  // Retired Turn 1 worker executes `pipeline.py:finally:` `_cancel_brain_inflight(self.brain)` AFTER
  // Turn 2 has already started and opened its own pooled HTTP client.
  // Without `scopedRetiredCancel` (`cancel_inflight(only_stopped=True)`), retired Turn 1 kills Turn 2's active pool!
  action retiredWorkerFinallyCleanup(id: int): bool = {
    val t = turns.get(id)
    val activeNewTurn = isSlotHeldBy(llmSlot, id, t.workerGen)
    val crossKilled = activeNewTurn and not(cfg.scopedRetiredCancel)
    all {
      t.retiredWorkerCleanupPending,
      turns' = turns.put(id, {
        ...t,
        retiredWorkerCleanupPending: false,
        phase: if (crossKilled) PhaseIdle else t.phase,
        uiState: if (crossKilled) UiIdle else t.uiState,
        audioStarted: if (crossKilled) false else t.audioStarted,
      }),
      llmSlot' = if (crossKilled) SlotFree else llmSlot,
      activePoolCrossKilled' = (activePoolCrossKilled or crossKilled),
      ollamaLoaded' = ollamaLoaded,
      llamaServerLoaded' = llamaServerLoaded,
      activeGlobalModel' = activeGlobalModel,
      warmupSeq' = warmupSeq,
      pendingWarmups' = pendingWarmups,
      history' = history,
      lastKvPromptTag' = lastKvPromptTag,
      kvModeInvalidations' = kvModeInvalidations,
      modeSwitchReloads' = modeSwitchReloads,
      staleWarmupEvictions' = staleWarmupEvictions,
      silentEmptyDrops' = silentEmptyDrops,
      ollamaCgroupOom' = ollamaCgroupOom,
      dualBackendCollision' = dualBackendCollision,
      atemLeakedToTts' = atemLeakedToTts,
      cfg' = cfg,
      cuda' = cuda,
      cov' = {
        ...cov,
        scopedRetiredCancelVerified: cov.scopedRetiredCancelVerified or (activeNewTurn and cfg.scopedRetiredCancel),
      },
    }
  }

  // Startup thread performs non-capture CUDA/RNG work (weight load, FP8 quant, prompt_cache, torch.randn).
  action cudaBeginStartupOp(actor: CudaActor): bool = {
    val isWhisper = actor == CudaActorWhisperInit
    val isVoxcpm = actor == CudaActorVoxcpmInit
    val orderOk =
      if (isWhisper) not(cuda.whisperBooted)
      else if (isVoxcpm) not(cuda.voxcpmBooted) and (not(cfg.enforceCudaCaptureExclusivity) or cuda.whisperBooted)
      else false
    all {
      isWhisper or isVoxcpm,
      orderOk,
      not(actor.in(cuda.runningOp)),
      not(actor.in(cuda.capturing)),
      not(cfg.enforceCudaCaptureExclusivity) or not(cuda.gpuLockHeld),
      cuda' = {
        ...cuda,
        gpuLockHeld: if (cfg.enforceCudaCaptureExclusivity) true else cuda.gpuLockHeld,
        runningOp: cuda.runningOp.union(Set(actor)),
        rngCorrupted: cuda.rngCorrupted or cuda.capturing.size() > 0,
      },
      cov' = cov,
      unchangedNonCudaState,
    }
  }

  action cudaEndStartupOp(actor: CudaActor): bool = all {
    actor == CudaActorWhisperInit or actor == CudaActorVoxcpmInit,
    actor.in(cuda.runningOp),
    cuda' = {
      ...cuda,
      gpuLockHeld: if (cfg.enforceCudaCaptureExclusivity) false else cuda.gpuLockHeld,
      runningOp: cuda.runningOp.exclude(Set(actor)),
    },
    cov' = cov,
    unchangedNonCudaState,
  }

  // Startup thread enters `with torch.cuda.graph(graph):` (locks global CUDAGeneratorImpl into capture mode).
  action cudaBeginStartupCapture(actor: CudaActor): bool = {
    val isWhisper = actor == CudaActorWhisperInit
    val isVoxcpm = actor == CudaActorVoxcpmInit
    val orderOk =
      if (isWhisper) not(cuda.whisperBooted)
      else if (isVoxcpm) not(cuda.voxcpmBooted) and (not(cfg.enforceCudaCaptureExclusivity) or cuda.whisperBooted)
      else false
    all {
      isWhisper or isVoxcpm,
      orderOk,
      not(actor.in(cuda.capturing)),
      not(actor.in(cuda.runningOp)),
      not(cfg.enforceCudaCaptureExclusivity) or not(cuda.gpuLockHeld),
      cuda' = {
        ...cuda,
        gpuLockHeld: if (cfg.enforceCudaCaptureExclusivity) true else cuda.gpuLockHeld,
        capturing: cuda.capturing.union(Set(actor)),
        rngCorrupted: cuda.rngCorrupted or cuda.capturing.size() > 0 or cuda.runningOp.size() > 0,
      },
      cov' = cov,
      unchangedNonCudaState,
    }
  }

  action cudaFinishStartupCapture(actor: CudaActor): bool = {
    val isWhisper = actor == CudaActorWhisperInit
    val isVoxcpm = actor == CudaActorVoxcpmInit
    all {
      isWhisper or isVoxcpm,
      actor.in(cuda.capturing),
      cuda' = {
        ...cuda,
        gpuLockHeld: if (cfg.enforceCudaCaptureExclusivity) false else cuda.gpuLockHeld,
        capturing: cuda.capturing.exclude(Set(actor)),
        whisperBooted: cuda.whisperBooted or isWhisper,
        voxcpmBooted: cuda.voxcpmBooted or isVoxcpm,
        capturedSteps: if (isVoxcpm) cuda.capturedSteps.union(Set(6)) else cuda.capturedSteps,
      },
      cov' = { ...cov, cudaGraphCaptureVerified: true },
      unchangedNonCudaState,
    }
  }

  // Runtime on-demand `_MegaStepRunner._get_or_capture_graph(steps, cfg)` when user switches `steps` (4/6/8/10).
  // Under `enforceCudaCaptureExclusivity = true`, `runtime.gpu_lock` excludes all STT/TTS CUDA ops.
  // Under `false`, `CUDA_GRAPH_LOCK` alone only serializes capture-vs-capture, NOT capture-vs-runtime CUDA/RNG ops!
  action cudaBeginMegaStepCapture(steps: int): bool = all {
    steps.in(Set(4, 6, 8, 10)),
    not(steps.in(cuda.capturedSteps)),
    cuda.whisperBooted and cuda.voxcpmBooted,
    not(CudaActorTtsMegaCapture.in(cuda.capturing)),
    if (cfg.enforceCudaCaptureExclusivity) not(cuda.gpuLockHeld) else cuda.capturing.size() == 0,
    cuda' = {
      ...cuda,
      gpuLockHeld: if (cfg.enforceCudaCaptureExclusivity) true else cuda.gpuLockHeld,
      capturing: cuda.capturing.union(Set(CudaActorTtsMegaCapture)),
      capturingSteps: steps,
      rngCorrupted: cuda.rngCorrupted or cuda.capturing.size() > 0 or cuda.runningOp.size() > 0,
    },
    cov' = cov,
    unchangedNonCudaState,
  }

  action cudaFinishMegaStepCapture: bool = all {
    CudaActorTtsMegaCapture.in(cuda.capturing),
    cuda' = {
      ...cuda,
      gpuLockHeld: if (cfg.enforceCudaCaptureExclusivity) false else cuda.gpuLockHeld,
      capturing: cuda.capturing.exclude(Set(CudaActorTtsMegaCapture)),
      capturedSteps: cuda.capturedSteps.union(Set(cuda.capturingSteps)),
      capturingSteps: 0,
    },
    cov' = {
      ...cov,
      cudaGraphCaptureVerified: true,
      nonDefaultStepsCaptured: cov.nonDefaultStepsCaptured or (cuda.capturingSteps != 6),
    },
    unchangedNonCudaState,
  }

  // Runtime non-capture CUDA/RNG operation (`WhisperSTT.transcribe`, `HalVoiceEngine.stream_speech` / `torch.randn`).
  action cudaBeginRuntimeOp(actor: CudaActor): bool = all {
    actor == CudaActorSttInfer or actor == CudaActorTtsStream,
    cuda.whisperBooted and cuda.voxcpmBooted,
    not(actor.in(cuda.runningOp)),
    not(cfg.enforceCudaCaptureExclusivity) or not(cuda.gpuLockHeld),
    cuda' = {
      ...cuda,
      gpuLockHeld: if (cfg.enforceCudaCaptureExclusivity) true else cuda.gpuLockHeld,
      runningOp: cuda.runningOp.union(Set(actor)),
      rngCorrupted: cuda.rngCorrupted or cuda.capturing.size() > 0,
    },
    cov' = cov,
    unchangedNonCudaState,
  }

  action cudaEndRuntimeOp(actor: CudaActor): bool = all {
    actor == CudaActorSttInfer or actor == CudaActorTtsStream,
    actor.in(cuda.runningOp),
    cuda' = {
      ...cuda,
      gpuLockHeld: if (cfg.enforceCudaCaptureExclusivity) false else cuda.gpuLockHeld,
      runningOp: cuda.runningOp.exclude(Set(actor)),
    },
    cov' = cov,
    unchangedNonCudaState,
  }

  action executeAnyOllamaWarmup: bool = {
    nondet req = pendingWarmups.oneOf()
    executeOllamaWarmup(req)
  }

  action step: bool = {
    nondet id = CLIENTS.oneOf()
    nondet m = selectableModels(cfg.enforceBackendExclusivity).oneOf()
    nondet mode = Set(ModeZeroThink, ModeLowThink).oneOf()
    nondet s = Set(4, 6, 8, 10).oneOf()
    any {
      clientSetModel(id, m),
      executeAnyOllamaWarmup,
      startTurn(id, mode),
      cancelInPreStream(id),
      llmOpenStream(id),
      llmHttpError500(id),
      llmDisconnectMidStream(id),
      llmThinkingToSpeaking(id, false),
      llmThinkingToSpeaking(id, true),
      llmExhaustedInThinking(id),
      llmZeroThinkRetryComplete(id, true),
      llmZeroThinkRetryComplete(id, false),
      llmProduceSentence(id),
      llmFinishAheadOfTts(id),
      ttsSpeakSentence(id),
      pipelineCompleteClean(id),
      ttsFailMidTurn(id),
      clientCancelEscape(id),
      clientReset(id),
      retiredWorkerFinallyCleanup(id),
      cudaBeginStartupOp(CudaActorWhisperInit),
      cudaEndStartupOp(CudaActorWhisperInit),
      cudaBeginStartupOp(CudaActorVoxcpmInit),
      cudaEndStartupOp(CudaActorVoxcpmInit),
      cudaBeginStartupCapture(CudaActorWhisperInit),
      cudaFinishStartupCapture(CudaActorWhisperInit),
      cudaBeginStartupCapture(CudaActorVoxcpmInit),
      cudaFinishStartupCapture(CudaActorVoxcpmInit),
      cudaBeginMegaStepCapture(s),
      cudaFinishMegaStepCapture,
      cudaBeginRuntimeOp(CudaActorSttInfer),
      cudaEndRuntimeOp(CudaActorSttInfer),
      cudaBeginRuntimeOp(CudaActorTtsStream),
      cudaEndRuntimeOp(CudaActorTtsStream),
    }
  }

  // --- Safety Invariants ---

  // 1a. Ollama cgroup cap (`20-memory-cap.conf`: MemoryMax=16G, MemorySwapMax=0) is never exceeded.
  val invOllamaCgroupSafe: bool =
    not(ollamaCgroupOom) and modelMemGb(ollamaLoaded) <= OLLAMA_CGROUP_MAX_GB

  // 1b. Strictly at most ONE LLM backend (Ollama OR direct llama-server) holds weights resident.
  val invSingleActiveLlmBackend: bool =
    not(dualBackendCollision) and (ollamaLoaded == ModelNone or llamaServerLoaded == ModelNone)

  // 1c. Total resident memory (Whisper + VoxCPM2 + active LLM) stays inside heavy.slice 64G high-water mark.
  val invHeavySliceMemoryBudget: bool =
    WHISPER_GB + VOXCPM_GB + modelMemGb(ollamaLoaded) + modelMemGb(llamaServerLoaded) <= HEAVY_SLICE_MAX_GB

  // 1d. Dual-mode routing NEVER thrashes models (evicting/reloading 15-19 GB on ZeroThink <-> LowThink).
  val invNoDualModeModelThrash: bool = modeSwitchReloads == 0

  // 1e. Stale async Ollama warmups NEVER evict a newer `set_model` selection.
  val invNoStaleWarmupEviction: bool = staleWarmupEvictions == 0

  // 2. Single LLM slot is NEVER occupied by a zombie cancelled/failed HTTP stream.
  val invNoZombieLlmSlot: bool = not(isZombieSlot(llmSlot))

  // 3a. Empty-speech exhaustion during `to=self` NEVER silently drops a turn without retry or error.
  val invNoSilentEmptyDrop: bool = silentEmptyDrops == 0

  // 3b. Raw ATEM `to=self` hidden thinking NEVER leaks into VoxCPM2 TTS.
  val invNoAtemLeakToTts: bool = not(atemLeakedToTts)

  // 4a. Conversation history ONLY contains turns that actually spoke cleanly (never failed or pre-speech cancelled).
  val invHistoryMatchesSpoken: bool =
    history.foldl(true, (acc, h) => acc and h.spoken and not(h.failed) and not(h.cancelled))

  // 4b. System prompt `messages[0]` remains invariant across ZeroThink <-> LowThink (preserving GPU KV prefix cache).
  val invKvSystemPrefixStable: bool = kvModeInvalidations == 0

  // 5. Empty or thinking-cancelled turns NEVER emit orphan `audio_start`.
  val invNoOrphanAudioStart: bool =
    CLIENTS.forall(i => not(turns.get(i).orphanAudioStart))

  // 6. Quiescent client UI is NEVER stuck in UiThinking or UiSpeaking when worker phase is PhaseIdle.
  val invNoStuckUiWhenIdle: bool =
    CLIENTS.forall(i => turns.get(i).phase == PhaseIdle implies turns.get(i).uiState == UiIdle)

  // 7. CUDA Graph Capture / RNG Exclusivity: while any thread is inside `torch.cuda.graph()`,
  //    NO other thread may execute a concurrent CUDA capture or CUDA/RNG op (preventing
  //    PyTorch `CUDAGeneratorImpl`: "Offset increment outside graph capture encountered unexpectedly").
  val invCudaGraphCaptureExclusive: bool = and {
    not(cuda.rngCorrupted),
    cuda.capturing.size() <= 1,
    cuda.capturing.size() > 0 implies cuda.runningOp.size() == 0,
  }

  // 8. Scoped retired-worker cleanup NEVER cancels a newer turn's active pooled HTTP client (Bug 2c).
  val invNoCrossTurnPoolCancel: bool = not(activePoolCrossKilled)

  val allInvariants: bool = and {
    invOllamaCgroupSafe,
    invSingleActiveLlmBackend,
    invHeavySliceMemoryBudget,
    invNoDualModeModelThrash,
    invNoStaleWarmupEviction,
    invNoZombieLlmSlot,
    invNoSilentEmptyDrop,
    invNoAtemLeakToTts,
    invHistoryMatchesSpoken,
    invKvSystemPrefixStable,
    invNoOrphanAudioStart,
    invNoStuckUiWhenIdle,
    invCudaGraphCaptureExclusive,
    invNoCrossTurnPoolCancel,
  }

  // --- Reachability Witnesses ---
  val wZeroThinkTurn: bool = cov.zeroThinkTurn
  val wLowThinkTurn: bool = cov.lowThinkTurn
  val wBargeInDuringThinking: bool = cov.bargeInDuringThinking
  val wEmptyThinkingRecovered: bool = cov.emptyThinkingRecovered
  val wBackendSwitchClean: bool = cov.backendSwitchClean
  val wTtsFailureCancelsLlm: bool = cov.ttsFailureCancelsLlm
  val wWarmupSerialized: bool = cov.warmupSerialized
  val wResetMidTurn: bool = cov.resetMidTurn
  val wKvCachePreservedAcrossModes: bool = cov.kvCachePreservedAcrossModes
  val wCudaGraphCaptureVerified: bool = cov.cudaGraphCaptureVerified
  val wNonDefaultStepsCaptured: bool = cov.nonDefaultStepsCaptured
  val wLlmHttp500Recovered: bool = cov.llmHttp500Recovered
  val wLlmMidStreamDisconnectRecovered: bool = cov.llmMidStreamDisconnectRecovered
  val wScopedRetiredCancelVerified: bool = cov.scopedRetiredCancelVerified
}
