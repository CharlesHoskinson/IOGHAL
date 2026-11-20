module ioghal_protocol_test {
  import ioghal_protocol.* from "./ioghal_protocol"

  // Deterministic scenario test: Reset followed immediately by a text turn echoes wire turn_id unchanged
  run resetThenTextTurnTest =
    init
      .then(clientReset(1))
      .then(serverProcessC2S(1))
      .then(clientReceiveS2C(1))
      .then(clientSubmitText(1))
      .then(serverProcessC2S(1))
      .then(workerLlmToTts(1))
      .then(workerEmitTtsChunk(1))
      .then(clientReceiveS2C(1)) // EvAudioStart
      .then(clientReceiveS2C(1)) // EvPcmChunk
      .expect(clients.get(1).playingTurnId == clients.get(1).activeTurnId)
      .expect(clients.get(1).uiState == UiSpeaking)
      .expect(allInvariants)

  // Deterministic scenario test: PipeWire 29.5s capture_limit auto-stops recording and submits audio
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
      .expect(micOwner == 0)
      .expect(servers.get(1).workerStage == WorkerStt)
      .expect(allInvariants)

  // Deterministic scenario test: Browser-mic barge-in does NOT snap UI back to Idle
  run browserBargeInStaysListeningTest =
    init
      .then(clientToggleMicMode(1))
      .then(clientSubmitText(1))
      .then(serverProcessC2S(1))
      .then(workerLlmToTts(1))
      .then(workerEmitTtsChunk(1))
      .then(clientReceiveS2C(1))
      .then(clientReceiveS2C(1))
      .then(clientBeginPtt(1, SpaceOwner))
      .then(serverProcessC2S(1)) // silent interrupt
      .expect(clients.get(1).pttActive)
      .expect(clients.get(1).uiState == UiListening)
      .expect(clients.get(1).playingTurnId == 0)
      .expect(allInvariants)

  // Deterministic scenario test: Multi-tab mic contention cleanly rejects tab 2 without disrupting tab 1
  run multiTabMicContentionTest =
    init
      .then(clientBeginPtt(1, SpaceOwner))
      .then(serverProcessC2S(1))
      .then(clientBeginPtt(2, SpaceOwner))
      .then(serverProcessC2S(2))
      .then(clientReceiveS2C(2)) // EvCaptureFailed
      .expect(micOwner == 1)
      .expect(servers.get(1).micRecording)
      .expect(not(clients.get(2).pttActive))
      .expect(clients.get(2).uiState == UiIdle)
      .expect(allInvariants)
}
