module ioghal_protocol_test {
  import ioghal_protocol.* from "./ioghal_protocol"

  // 1. Reset followed immediately by a text turn echoes wire turn_id unchanged
  run resetThenTextTurnTest =
    init
      .then(clientReset(1))
      .then(serverProcessC2S(1))
      .then(clientReceiveS2C(1))
      .then(clientSubmitText(1))
      .then(serverProcessC2S(1))
      .then(workerSearchComplete(1, false, false))
      .then(workerLlmProduceSentence(1))
      .then(workerTtsPopAndAcquireGpu(1))
      .then(workerTtsEmitChunk(1))
      .then(clientReceiveS2C(1)) // EvAudioStart
      .then(clientReceiveS2C(1)) // EvPcmChunk
      .expect(clients.get(1).playingTurnId == clients.get(1).activeTurnId)
      .expect(clients.get(1).uiState == UiSpeaking)
      .expect(allInvariants)

  // 2. PipeWire 29.5s capture_limit auto-stops recording and submits audio
  run captureLimitAutoFinishesTest =
    init
      .then(clientBeginPtt(1, SpaceOwner))
      .then(serverProcessC2S(1))
      .then(clientReceiveS2C(1))
      .then(serverMicHitLimit(1))
      .then(clientReceiveS2C(1)) // EvCaptureLimit triggers auto MsgPttStop
      .then(serverProcessC2S(1)) // MsgPttStop closes mic & launches STT
      .expect(not(clients.get(1).pttActive))
      .expect(not(servers.get(1).micRecording))
      .expect(hw.micOwner == 0)
      .expect(servers.get(1).workerStage == WorkerStt)
      .expect(allInvariants)

  // 3. Browser-mic barge-in does NOT snap UI back to Idle
  run browserBargeInStaysListeningTest =
    init
      .then(clientToggleMicMode(1))
      .then(clientSubmitText(1))
      .then(serverProcessC2S(1))
      .then(workerSearchComplete(1, false, false))
      .then(workerLlmProduceSentence(1))
      .then(workerTtsPopAndAcquireGpu(1))
      .then(workerTtsEmitChunk(1))
      .then(clientReceiveS2C(1))
      .then(clientReceiveS2C(1))
      .then(clientBeginPtt(1, SpaceOwner))
      .then(serverProcessC2S(1)) // silent interrupt
      .expect(clients.get(1).pttActive)
      .expect(clients.get(1).uiState == UiListening)
      .expect(clients.get(1).playingTurnId == 0)
      .expect(allInvariants)

  // 4. Multi-tab mic contention cleanly rejects tab 2 without disrupting tab 1
  run multiTabMicContentionTest =
    init
      .then(clientBeginPtt(1, SpaceOwner))
      .then(serverProcessC2S(1))
      .then(clientBeginPtt(2, SpaceOwner))
      .then(serverProcessC2S(2))
      .then(clientReceiveS2C(2)) // EvCaptureFailed
      .expect(hw.micOwner == 1)
      .expect(servers.get(1).micRecording)
      .expect(not(clients.get(2).pttActive))
      .expect(clients.get(2).uiState == UiIdle)
      .expect(allInvariants)

  // 5. Concurrent LLM producer queues sentence 2 while TTS holds GPU for sentence 1,
  //    then finishes cleanly and unlocks Replay
  run concurrentPipelineOverlapAndReplayTest =
    init
      .then(clientSubmitText(1))
      .then(serverProcessC2S(1))
      .then(workerSearchComplete(1, true, false))
      .then(clientReceiveS2C(1)) // EvState(UiThinking) from web search
      .then(workerLlmProduceSentence(1))
      .then(workerTtsPopAndAcquireGpu(1))
      .then(workerLlmProduceSentence(1)) // overlap: LLM queues sentence 2 while TTS holds GPU!
      .then(workerLlmFinish(1))
      .then(workerTtsEmitChunk(1))
      .then(workerTtsFinishSentence(1))
      .then(workerTtsPopAndAcquireGpu(1))
      .then(workerTtsEmitChunk(1))
      .then(workerTtsFinishSentence(1))
      .then(workerPipelineFinish(1))
      .then(clientReceiveS2C(1)) // EvAudioStart
      .then(clientReceiveS2C(1)) // EvPcmChunk
      .then(clientReceiveS2C(1)) // EvPcmChunk
      .then(clientReceiveS2C(1)) // EvAudioEnd
      .then(clientPlaybackDrain(1))
      .expect(cov.pipelineOverlap)
      .expect(cov.webSearchTurn)
      .expect(clients.get(1).hasReplay)
      .expect(clients.get(1).uiState == UiIdle)
      .then(clientStartReplay(1))
      .expect(clients.get(1).replaying)
      .expect(clients.get(1).uiState == UiSpeaking)
      .then(clientPlaybackDrain(1))
      .expect(not(clients.get(1).replaying))
      .expect(clients.get(1).uiState == UiIdle)
      .expect(allInvariants)

  // 6. Barge-in while worker holds GPU retires old worker cleanly; new turn waits until GPU released
  run retiredWorkerGpuHandoffTest =
    init
      .then(clientSubmitText(1))
      .then(serverProcessC2S(1))
      .then(workerSearchComplete(1, false, false))
      .then(workerLlmProduceSentence(1))
      .then(workerTtsPopAndAcquireGpu(1))
      .then(clientSubmitText(1)) // barge-in while Turn 1 still holds GPU!
      .then(serverProcessC2S(1)) // Turn 2 starts; Turn 1 is now retired GPU holder
      .then(retiredWorkerReleaseGpu)
      .expect(cov.retiredWorkerGpuHandoff)
      .expect(hw.gpuHolder == GpuFree)
      .then(workerSearchComplete(1, false, false))
      .then(workerLlmProduceSentence(1))
      .then(workerTtsPopAndAcquireGpu(1))
      .expect(servers.get(1).ttsHoldingGpu)
      .expect(allInvariants)

  // 7. Invalid audio_b64 upload emits turn-scoped error + idle state and never sticks client in UiThinking
  run invalidAudioUploadRecoversToIdleTest =
    init
      .then(clientToggleMicMode(1))
      .then(clientBeginPtt(1, SpaceOwner))
      .then(serverProcessC2S(1))
      .then(clientFinishPtt(1, false)) // invalid base64 payload
      .then(serverProcessC2S(1))
      .then(clientReceiveS2C(1)) // EvError(tid)
      .then(clientReceiveS2C(1)) // EvState(UiIdle)
      .expect(clients.get(1).uiState == UiIdle)
      .expect(not(clients.get(1).pendingTurn))
      .expect(cov.invalidAudioRecovery)
      .expect(allInvariants)

  // 8. Model switch (valid & invalid) mid-turn does not clobber active thinking state
  run setModelMidTurnDoesNotClobberThinkingTest =
    init
      .then(clientSubmitText(1))
      .then(clientSetModel(1, true))
      .then(clientSetModel(1, false))
      .then(serverProcessC2S(1)) // MsgText
      .then(serverProcessC2S(1)) // MsgSetModel(valid) -> EvSystemStatus
      .then(serverProcessC2S(1)) // MsgSetModel(invalid) -> EvError(tid=0)
      .then(clientReceiveS2C(1)) // EvSystemStatus ignored while pendingTurn
      .then(clientReceiveS2C(1)) // EvError(tid=0) ignored for active turn state
      .expect(clients.get(1).uiState == UiThinking)
      .expect(clients.get(1).pendingTurn)
      .expect(cov.setModelMidTurn)
      .expect(allInvariants)

  // 9. Mid-turn GPU failure after audio_start emits audio_end(failed=True) before state(idle)
  //    so client never sticks in UiSpeaking (regression test for Quint-found Bug #6)
  run midTurnGpuFailureAfterAudioStartReturnsToIdleTest =
    init
      .then(clientSubmitText(1))
      .then(serverProcessC2S(1))
      .then(workerSearchComplete(1, false, false))
      .then(workerLlmProduceSentence(1))
      .then(workerTtsPopAndAcquireGpu(1))
      .then(workerTtsEmitChunk(1))
      .then(clientReceiveS2C(1)) // EvAudioStart
      .then(clientReceiveS2C(1)) // EvPcmChunk -> UiSpeaking
      .expect(clients.get(1).uiState == UiSpeaking)
      .then(workerGpuFail(1))
      .then(clientReceiveS2C(1)) // EvAudioEnd(failed=true) closes open stream cleanly
      .then(clientReceiveS2C(1)) // EvError logs error without violating audio framing
      .then(clientReceiveS2C(1)) // EvState(UiIdle) sets error status detail
      .expect(clients.get(1).uiState == UiIdle)
      .expect(clients.get(1).playingTurnId == 0)
      .expect(not(clients.get(1).pendingTurn))
      .expect(hw.gpuHolder == GpuFree)
      .expect(allInvariants)

  // 10. Barge-in / Escape during Glimmer `to=self` hidden thinking aborts cleanly before
  //     audio_start, frees the LLM stream, and returns client to UiIdle without orphan framing.
  run bargeInDuringHiddenThinkingReturnsIdleTest =
    init
      .then(clientSubmitText(1))
      .then(serverProcessC2S(1))
      .then(workerSearchComplete(1, false, false))
      .then(workerLlmEnterThinking(1))
      .expect(servers.get(1).llmThinkingSelf)
      .expect(not(servers.get(1).workerAudioStarted))
      .then(clientCancelEscape(1))
      .then(serverProcessC2S(1)) // MsgInterrupt aborts hidden thinking & closes LLM stream
      .then(clientReceiveS2C(1)) // EvState(UiIdle)
      .expect(clients.get(1).uiState == UiIdle)
      .expect(not(servers.get(1).llmStreamActive))
      .expect(not(servers.get(1).llmThinkingSelf))
      .expect(cov.thinkingSelfBargeIn)
      .expect(allInvariants)

  // 11. Glimmer hits `num_predict` while still inside `to=self` thinking (0 visible tokens):
  //     returns cleanly to UiIdle with ZERO orphan `audio_start` (both clean-idle and error modes).
  run emptyThinkingReplyCleanReturnToIdleTest =
    init
      .then(clientSubmitText(1))
      .then(serverProcessC2S(1))
      .then(workerSearchComplete(1, false, false))
      .then(workerLlmEnterThinking(1))
      .then(workerLlmExhaustedInThinking(1, false))
      .then(workerPipelineFinish(1))
      .then(clientReceiveS2C(1)) // EvState(UiIdle) — no audio_start ever emitted
      .expect(clients.get(1).uiState == UiIdle)
      .expect(not(clients.get(1).pendingTurn))
      .expect(clients.get(1).audioOpenTid == 0)
      .expect(cov.emptyThinkingRecovery)
      .expect(allInvariants)

  // 12. Glimmer hits `num_predict` in `to=self` surfaced as explicit turn error + idle state.
  run emptyThinkingReplyErrorReturnToIdleTest =
    init
      .then(clientSubmitText(1))
      .then(serverProcessC2S(1))
      .then(workerSearchComplete(1, false, false))
      .then(workerLlmEnterThinking(1))
      .then(workerLlmExhaustedInThinking(1, true))
      .then(workerPipelineFinish(1))
      .then(clientReceiveS2C(1)) // EvError(tid)
      .then(clientReceiveS2C(1)) // EvState(UiIdle)
      .expect(clients.get(1).uiState == UiIdle)
      .expect(not(clients.get(1).pendingTurn))
      .expect(not(servers.get(1).llmStreamActive))
      .expect(allInvariants)

  // 13. AudioContext suspended prevents Replay from entering stuck UiSpeaking state; resuming unlocks it.
  run suspendedAudioGuardsReplayTest =
    init
      .then(clientSubmitText(1))
      .then(serverProcessC2S(1))
      .then(workerSearchComplete(1, false, false))
      .then(workerLlmProduceSentence(1))
      .then(workerLlmFinish(1))
      .then(workerTtsPopAndAcquireGpu(1))
      .then(workerTtsEmitChunk(1))
      .then(workerTtsFinishSentence(1))
      .then(workerPipelineFinish(1))
      .then(clientReceiveS2C(1)) // EvAudioStart
      .then(clientReceiveS2C(1)) // EvPcmChunk
      .then(clientReceiveS2C(1)) // EvAudioEnd
      .then(clientPlaybackDrain(1))
      .expect(clients.get(1).hasReplay)
      .then(clientSetAudioSuspended(1, true))
      .expect(cov.replaySuspendedGuard)
      .then(clientSetAudioSuspended(1, false))
      .then(clientStartReplay(1))
      .expect(clients.get(1).replaying)
      .expect(clients.get(1).uiState == UiSpeaking)
      .then(clientPlaybackDrain(1))
      .expect(clients.get(1).uiState == UiIdle)
      .expect(allInvariants)

  // 14. llama-server HTTP 500 before audio emits turn-scoped error + idle state with 0 orphan audio_start.
  run llmHttp500BeforeAudioRecoversToIdleTest =
    init
      .then(clientSubmitText(1))
      .then(serverProcessC2S(1))
      .then(workerSearchComplete(1, false, false))
      .then(workerLlmError(1))
      .then(workerPipelineFinish(1))
      .then(clientReceiveS2C(1)) // EvError(tid)
      .then(clientReceiveS2C(1)) // EvState(UiIdle)
      .expect(clients.get(1).uiState == UiIdle)
      .expect(not(clients.get(1).pendingTurn))
      .expect(clients.get(1).audioOpenTid == 0)
      .expect(not(servers.get(1).llmStreamActive))
      .expect(allInvariants)

  // 15. llama-server mid-stream disconnect AFTER sentence 1 started audio closes open stream with
  //     EvAudioEnd(failed=true) -> EvError -> EvState(UiIdle) without framing violation or stuck UI.
  run llmDisconnectMidStreamAfterAudioStartRecoversCleanlyTest =
    init
      .then(clientSubmitText(1))
      .then(serverProcessC2S(1))
      .then(workerSearchComplete(1, false, false))
      .then(workerLlmProduceSentence(1))
      .then(workerTtsPopAndAcquireGpu(1))
      .then(workerTtsEmitChunk(1))
      .then(workerLlmError(1)) // mid-stream disconnect while TTS speaks sentence 1
      .then(workerTtsFinishSentence(1))
      .then(workerPipelineFinish(1))
      .then(clientReceiveS2C(1)) // EvAudioStart
      .then(clientReceiveS2C(1)) // EvPcmChunk -> UiSpeaking
      .expect(clients.get(1).uiState == UiSpeaking)
      .then(clientReceiveS2C(1)) // EvAudioEnd(failed=true)
      .then(clientReceiveS2C(1)) // EvError
      .then(clientReceiveS2C(1)) // EvState(UiIdle)
      .expect(clients.get(1).uiState == UiIdle)
      .expect(clients.get(1).playingTurnId == 0)
      .expect(not(clients.get(1).pendingTurn))
      .expect(not(clients.get(1).protocolViolation))
      .expect(allInvariants)

  // Connected-session step omitting random disconnects so multi-turn replay & LLM/TTS overlap
  // are deeply stress-tested across thousands of random traces.
  action stepConnected: bool = {
    nondet id = CLIENTS.oneOf()
    any {
      runtimeWarmupAcquire,
      runtimeWarmupRelease,
      retiredWorkerReleaseGpu,
      clientBeginPtt(id, SpaceOwner),
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
}
