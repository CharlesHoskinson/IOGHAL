// Quint counterexample proofs (reproducing each real bug in the current code / naive Glimmer
// design) AND verification tests for the fixed Muse Glimmer 30B runtime architecture.

module glimmer_runtime_test {
  import glimmer_runtime.* from "./glimmer_runtime"

  // ============================================================================
  // PART 1: CONCRETE COUNTEREXAMPLE PROOFS OF REAL BUGS
  // ============================================================================

  // Counterexample 1a: Loading stock 20 GB `muse-glimmer:30b-q4_K_M-dflash` on Ollama
  // violates `20-memory-cap.conf` (`MemoryMax=16G`) and triggers cgroup memory.max stall.
  run bug1a_ollamaCgroupOomCounterexampleTest =
    initWithConfig({ ...fixedConfig, enforceBackendExclusivity: false }, ModelQwen9bOllama)
      .then(clientSetModel(1, ModelGlimmerQ4Ollama))
      .then(executeOllamaWarmup({ seq: 1, model: ModelGlimmerQ4Ollama }))
      .expect(ollamaCgroupOom)
      .expect(not(invOllamaCgroupSafe))

  // Counterexample 1b: Switching from direct `llama-server` (19 GB) to Ollama (10 GB)
  // without explicitly unloading the outgoing backend leaves BOTH LLM servers resident!
  run bug1b_dualBackendCollisionCounterexampleTest =
    initWithConfig({ ...fixedConfig, enforceBackendExclusivity: false }, ModelGlimmerQ4Llama)
      .then(clientSetModel(1, ModelQwen9bOllama))
      .then(executeOllamaWarmup({ seq: 1, model: ModelQwen9bOllama }))
      .expect(dualBackendCollision)
      .expect(ollamaLoaded == ModelQwen9bOllama)
      .expect(llamaServerLoaded == ModelGlimmerQ4Llama)
      .expect(not(invSingleActiveLlmBackend))

  // Counterexample 1c: Naive dual-mode routing using two Ollama model names
  // (`hal-glimmer-fast` vs `hal-glimmer-think`) under OLLAMA_MAX_LOADED_MODELS=1
  // triggers a full 15 GB eviction + reload on every ZeroThink <-> LowThink switch!
  run bug1c_dualModeOllamaThrashCounterexampleTest =
    initWithConfig({ ...fixedConfig, singleModelDualMode: false }, ModelGlimmerQ3Ollama)
      .then(startTurn(1, ModeZeroThink))
      .then(llmOpenStream(1)) // evicts ModelGlimmerQ3Ollama -> loads ModelGlimmerFastOllama
      .expect(modeSwitchReloads == 1)
      .expect(not(invNoDualModeModelThrash))

  // Counterexample 1d (`server.py:219-256`): Unserialized async Ollama warmups allow a stale
  // warmup POST (seq=1) to land AFTER a newer `set_model` warmup (seq=2), evicting the active model!
  run bug1d_staleWarmupEvictsNewModelCounterexampleTest =
    initWithConfig({ ...fixedConfig, serializeWarmups: false }, ModelQwen9bOllama)
      .then(clientSetModel(1, ModelQwen9bOllama))   // schedules warmup seq=1
      .then(clientSetModel(1, ModelGlimmerQ3Ollama)) // schedules warmup seq=2
      .then(executeOllamaWarmup({ seq: 2, model: ModelGlimmerQ3Ollama })) // loads Glimmer Q3
      .then(executeOllamaWarmup({ seq: 1, model: ModelQwen9bOllama }))    // stale seq=1 EVICTS Glimmer Q3!
      .expect(ollamaLoaded == ModelQwen9bOllama)
      .expect(staleWarmupEvictions == 1)
      .expect(not(invNoStaleWarmupEviction))

  // Counterexample 2a (`brain.py:540-548`): Barge-in / Escape during PhasePreStream
  // (before `_active_clients.add(client)`) misses `cancel_inflight()`, so the cancelled turn
  // opens a ZOMBIE HTTP stream that occupies the single LLM inference slot!
  run bug2a_preStreamCancelZombieSlotCounterexampleTest =
    initWithConfig({ ...fixedConfig, checkCancelBeforeStream: false }, ModelGlimmerQ4Llama)
      .then(startTurn(1, ModeLowThink))
      .then(cancelInPreStream(1))
      .then(llmOpenStream(1)) // no pre-stream stop_check -> zombie occupies single LLM slot!
      .expect(isZombieSlot(llmSlot))
      .expect(not(invNoZombieLlmSlot))

  // Counterexample 2b (`pipeline.py:285,307` & `server.py:480`): TTS GPU failure mid-turn sets
  // `cancel_event` without calling `brain.cancel_inflight()`, leaving the active LLM HTTP stream
  // as a zombie holding the single LLM slot!
  run bug2b_ttsFailureLeaksLlmSlotCounterexampleTest =
    initWithConfig({ ...fixedConfig, pipelineCancelsLlmOnFailure: false }, ModelGlimmerQ4Llama)
      .then(startTurn(1, ModeLowThink))
      .then(llmOpenStream(1))
      .then(llmThinkingToSpeaking(1, false))
      .then(llmProduceSentence(1))
      .then(ttsFailMidTurn(1))
      .expect(isZombieSlot(llmSlot))
      .expect(not(invNoZombieLlmSlot))

  // Counterexample 3a (`brain.py:571-599`): Glimmer exhausts `num_predict` while still in
  // `to=self` hidden thinking (`done=true`, `reply=""`) -> silent drop with 0 speech & 0 error!
  run bug3a_emptyThinkingSilentDropCounterexampleTest =
    initWithConfig({ ...fixedConfig, handleEmptyThinkingReply: false }, ModelGlimmerQ4Llama)
      .then(startTurn(1, ModeLowThink))
      .then(llmOpenStream(1))
      .then(llmExhaustedInThinking(1))
      .expect(silentEmptyDrops == 1)
      .expect(not(invNoSilentEmptyDrop))

  // Counterexample 3b (`brain.py:123-193`): Raw ATEM `to=self` tokens are NOT stripped by
  // `_ThinkStripper`, leaking private reasoning aloud to VoxCPM2 TTS!
  run bug3b_atemLeakToTtsCounterexampleTest =
    initWithConfig({ ...fixedConfig, stripAtemSelfChannel: false }, ModelGlimmerQ4Llama)
      .then(startTurn(1, ModeLowThink))
      .then(llmOpenStream(1))
      .then(llmThinkingToSpeaking(1, true))
      .expect(atemLeakedToTts)
      .expect(not(invNoAtemLeakToTts))

  // Counterexample 4a (`brain.py:598` vs `pipeline.py:172-267`): Fast DFlash LLM finishes and
  // commits `full_reply` to `brain.history` BEFORE TTS speaks; subsequent TTS failure poisons history!
  run bug4a_prematureHistoryCommitPoisonedOnTtsFailCounterexampleTest =
    initWithConfig({ ...fixedConfig, commitHistoryAfterTts: false }, ModelGlimmerQ4Llama)
      .then(startTurn(1, ModeZeroThink))
      .then(llmOpenStream(1))
      .then(llmProduceSentence(1))
      .then(llmFinishAheadOfTts(1)) // commits to history before TTS speaks!
      .then(ttsFailMidTurn(1))      // TTS fails -> history now has unspoken failed reply!
      .expect(not(invHistoryMatchesSpoken))

  // Counterexample 4b (`brain.py:486-488`): Mutating `messages[0]` system prompt between
  // ZeroThink and LowThink invalidates the entire GPU KV prefix cache!
  run bug4b_systemPromptMutationBreaksKvCacheCounterexampleTest =
    initWithConfig({ ...fixedConfig, keepSystemPromptInvariant: false }, ModelGlimmerQ4Llama)
      .then(startTurn(1, ModeZeroThink))
      .then(llmOpenStream(1))
      .then(llmProduceSentence(1))
      .then(llmFinishAheadOfTts(1))
      .then(ttsSpeakSentence(1))
      .then(pipelineCompleteClean(1))
      .then(startTurn(1, ModeLowThink))
      .then(llmOpenStream(1))
      .expect(kvModeInvalidations == 1)
      .expect(not(invKvSystemPrefixStable))

  // Counterexample 5a (`server.py` old concurrent startup threads): Running WhisperSTT and
  // HalVoiceEngine initialization concurrently lets one thread run a CUDA/RNG op while the other
  // is inside `torch.cuda.graph()`, corrupting PyTorch's process-global `CUDAGeneratorImpl`
  // ("Offset increment outside graph capture encountered unexpectedly")!
  run bug5a_concurrentStartupCudaGraphRngRaceCounterexampleTest =
    initStartupWithConfig({ ...fixedConfig, enforceCudaCaptureExclusivity: false }, ModelGlimmerQ4Llama)
      .then(cudaBeginStartupCapture(CudaActorWhisperInit))
      .then(cudaBeginStartupOp(CudaActorVoxcpmInit))
      .expect(cuda.rngCorrupted)
      .expect(not(invCudaGraphCaptureExclusive))

  // Counterexample 5b (`tts.py` `_MegaStepRunner._get_or_capture_graph`): Capturing a CUDA graph
  // on-demand for a non-default `steps` value (e.g. steps=8) while protected ONLY by
  // `CUDA_GRAPH_LOCK` (without a shared `gpu_lock` across non-capture CUDA ops like STT) lets
  // another thread touch CUDA/RNG mid-capture and corrupt `CUDAGeneratorImpl`!
  run bug5b_nonDefaultStepsCaptureRaceWithoutGpuLockCounterexampleTest =
    initWithConfig({ ...fixedConfig, enforceCudaCaptureExclusivity: false }, ModelGlimmerQ4Llama)
      .then(cudaBeginMegaStepCapture(8))
      .then(cudaBeginRuntimeOp(CudaActorSttInfer))
      .expect(cuda.rngCorrupted)
      .expect(not(invCudaGraphCaptureExclusive))

  // ============================================================================
  // PART 2: FIXED ARCHITECTURE VERIFICATION TESTS
  // ============================================================================

  // Fixed 1: Dual-mode ZeroThink -> LowThink on direct llama-server Glimmer preserves KV prefix
  // cache, triggers 0 reloads, handles `to=self` -> `to=user`, and commits history only after TTS.
  run fixedDualModeKvAndHistoryTest =
    init
      .then(startTurn(1, ModeZeroThink))
      .then(llmOpenStream(1))
      .then(llmProduceSentence(1))
      .then(llmFinishAheadOfTts(1))
      .expect(history.length() == 0) // NOT committed before TTS speaks!
      .then(ttsSpeakSentence(1))
      .then(pipelineCompleteClean(1))
      .expect(history.length() == 1)
      .then(startTurn(1, ModeLowThink))
      .then(llmOpenStream(1))
      .then(llmThinkingToSpeaking(1, true)) // raw ATEM stripped cleanly
      .then(llmProduceSentence(1))
      .then(llmFinishAheadOfTts(1))
      .then(ttsSpeakSentence(1))
      .then(pipelineCompleteClean(1))
      .expect(history.length() == 2)
      .expect(cov.kvCachePreservedAcrossModes)
      .expect(allInvariants)

  // Fixed 2: Barge-in during hidden `to=self` thinking + pre-stream cancel cleanly releases slot
  // with 0 zombie streams, 0 orphan audio_start, and 0 history pollution.
  run fixedBargeInDuringThinkingAndPreStreamTest =
    init
      .then(startTurn(1, ModeLowThink))
      .then(cancelInPreStream(1))
      .then(llmOpenStream(1)) // atomic stop_check aborts cleanly before opening HTTP stream
      .expect(isSlotFree(llmSlot))
      .then(startTurn(1, ModeLowThink))
      .then(llmOpenStream(1))
      .expect(turns.get(1).phase == PhaseThinkingSelf)
      .then(clientCancelEscape(1))
      .expect(isSlotFree(llmSlot))
      .expect(turns.get(1).uiState == UiIdle)
      .expect(history.length() == 0)
      .expect(cov.bargeInDuringThinking)
      .expect(allInvariants)

  // Fixed 3: `num_predict` exhausted in `to=self` retries cleanly in ModeZeroThink (both success
  // and fallback error paths return cleanly to UiIdle without silent drop or orphan audio_start).
  run fixedEmptyThinkingRetryRecoveryTest =
    init
      .then(startTurn(1, ModeLowThink))
      .then(llmOpenStream(1))
      .then(llmExhaustedInThinking(1))
      .expect(turns.get(1).phase == PhaseRetryingZeroThink)
      .then(llmZeroThinkRetryComplete(1, true))
      .then(llmProduceSentence(1))
      .then(llmFinishAheadOfTts(1))
      .then(ttsSpeakSentence(1))
      .then(pipelineCompleteClean(1))
      .expect(turns.get(1).uiState == UiIdle)
      .expect(history.length() == 1)
      .expect(cov.emptyThinkingRecovered)
      .expect(allInvariants)

  // Fixed 4: Backend switch (llama-server -> Ollama -> llama-server) maintains strict residency
  // exclusivity, serialized warmups, and cgroup safety.
  run fixedBackendSwitchAndWarmupTest =
    init
      .then(clientSetModel(1, ModelQwen9bOllama))
      .expect(llamaServerLoaded == ModelNone)
      .then(executeOllamaWarmup({ seq: 1, model: ModelQwen9bOllama }))
      .expect(ollamaLoaded == ModelQwen9bOllama)
      .then(clientSetModel(1, ModelGlimmerQ4Llama))
      .expect(ollamaLoaded == ModelNone)
      .expect(llamaServerLoaded == ModelGlimmerQ4Llama)
      .expect(cov.backendSwitchClean)
      .expect(cov.warmupSerialized)
      .expect(allInvariants)

  // Fixed 5: TTS failure mid-turn immediately cancels in-flight LLM stream & never poisons history.
  run fixedTtsFailureCancelsLlmAndKeepsHistoryCleanTest =
    init
      .then(startTurn(1, ModeLowThink))
      .then(llmOpenStream(1))
      .then(llmThinkingToSpeaking(1, false))
      .then(llmProduceSentence(1))
      .then(ttsFailMidTurn(1))
      .expect(isSlotFree(llmSlot))
      .expect(turns.get(1).uiState == UiIdle)
      .expect(history.length() == 0)
      .expect(cov.ttsFailureCancelsLlm)
      .expect(allInvariants)

  // Fixed 6: Sequential WhisperSTT -> HalVoiceEngine startup AND gpu_lock-serialized on-demand
  // `_MegaStepRunner` capture for non-default `steps=8` maintain strict CUDA Graph / RNG exclusivity.
  run fixedSequentialStartupAndGpuLockedMegaCaptureTest =
    initStartup
      .then(cudaBeginStartupOp(CudaActorWhisperInit))
      .then(cudaEndStartupOp(CudaActorWhisperInit))
      .then(cudaBeginStartupCapture(CudaActorWhisperInit))
      .then(cudaFinishStartupCapture(CudaActorWhisperInit))
      .expect(cuda.whisperBooted)
      .expect(not(cuda.voxcpmBooted))
      .then(cudaBeginStartupOp(CudaActorVoxcpmInit))
      .then(cudaEndStartupOp(CudaActorVoxcpmInit))
      .then(cudaBeginStartupCapture(CudaActorVoxcpmInit))
      .then(cudaFinishStartupCapture(CudaActorVoxcpmInit))
      .expect(cuda.voxcpmBooted)
      .expect(cuda.capturedSteps == Set(6))
      .then(cudaBeginMegaStepCapture(8))
      .then(cudaFinishMegaStepCapture)
      .expect(cuda.capturedSteps == Set(6, 8))
      .then(cudaBeginRuntimeOp(CudaActorSttInfer))
      .then(cudaEndRuntimeOp(CudaActorSttInfer))
      .expect(cov.cudaGraphCaptureVerified)
      .expect(cov.nonDefaultStepsCaptured)
      .expect(not(cuda.rngCorrupted))
      .expect(invCudaGraphCaptureExclusive)
      .expect(allInvariants)

  // Fixed 7: llama-server HTTP 500 AND mid-stream disconnect (after sentence 1 spoke) both release
  // the single LLM slot, return UI to UiIdle, emit 0 orphan audio_start, and never poison history.
  run fixedLlmHttp500AndMidStreamDisconnectTest =
    init
      .then(startTurn(1, ModeZeroThink))
      .then(llmHttpError500(1))
      .expect(isSlotFree(llmSlot))
      .expect(turns.get(1).uiState == UiIdle)
      .expect(history.length() == 0)
      .expect(cov.llmHttp500Recovered)
      .then(startTurn(1, ModeLowThink))
      .then(llmOpenStream(1))
      .then(llmThinkingToSpeaking(1, false))
      .then(llmProduceSentence(1))
      .then(ttsSpeakSentence(1))
      .then(llmDisconnectMidStream(1))
      .expect(isSlotFree(llmSlot))
      .expect(turns.get(1).uiState == UiIdle)
      .expect(history.length() == 0)
      .expect(cov.llmMidStreamDisconnectRecovered)
      .expect(allInvariants)

  // Counterexample 2c (`pipeline.py:351` + pooled HTTP client): Retired Turn 1's `finally:` runs
  // unscoped `brain.cancel_inflight()` AFTER Turn 2 already acquired its pooled HTTP client,
  // cross-cancelling Turn 2 mid-stream!
  run bug2c_retiredWorkerKillsNewTurnPoolCounterexampleTest =
    initWithConfig({ ...fixedConfig, scopedRetiredCancel: false }, ModelGlimmerQ4Llama)
      .then(startTurn(1, ModeLowThink))
      .then(llmOpenStream(1))
      .then(startTurn(1, ModeZeroThink))       // barge-in starts Turn 2 while Turn 1 unwinds
      .then(llmOpenStream(1))                  // Turn 2 acquires fresh pooled HTTP client
      .then(retiredWorkerFinallyCleanup(1))    // unscoped cancel kills Turn 2's live client!
      .expect(activePoolCrossKilled)
      .expect(not(invNoCrossTurnPoolCancel))

  // Fixed 8: Scoped `cancel_inflight(only_stopped=True)` in retired Turn 1 cleanup preserves
  // Turn 2's live pooled HTTP client so Turn 2 completes cleanly with all invariants satisfied.
  run fixedScopedCancelPreservesNewTurnPoolTest =
    init
      .then(startTurn(1, ModeLowThink))
      .then(llmOpenStream(1))
      .then(startTurn(1, ModeZeroThink))
      .then(llmOpenStream(1))
      .then(retiredWorkerFinallyCleanup(1))
      .expect(not(activePoolCrossKilled))
      .expect(isSlotHeldBy(llmSlot, 1, 2))
      .expect(cov.scopedRetiredCancelVerified)
      .then(llmProduceSentence(1))
      .then(llmFinishAheadOfTts(1))
      .then(ttsSpeakSentence(1))
      .then(pipelineCompleteClean(1))
      .expect(history.length() == 1)
      .expect(allInvariants)
}
