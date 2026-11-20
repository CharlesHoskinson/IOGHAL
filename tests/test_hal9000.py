"""Unit and protocol verification tests for IOGHAL."""

from __future__ import annotations

import asyncio
import struct
import threading
import unittest
from collections.abc import Generator
from unittest.mock import MagicMock, patch

import numpy as np

from hal9000.audio import MicRecorder
from hal9000.brain import (
    POD_BAY_REPLY,
    QwenBrain,
    _ThinkStripper,
    enforce_user_address,
    is_pod_bay_open_request,
    split_complete_sentences,
)
from hal9000.interfaces import (
    AudioInput,
    BrainProvider,
    HalConfig,
    HalEvent,
    STTProvider,
    TTSProvider,
)
from hal9000.pipeline import HalPipeline
from hal9000.server import HalRuntime, create_hal_router, is_allowed_origin
from hal9000.stt import _has_speech_activity
from hal9000.tts import normalize_for_hal_tts


class _FakeSTT:
    def __init__(self, reply: str = "Open the pod bay doors") -> None:
        self.reply = reply

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        return self.reply


class _FakeTTS:
    sample_rate = 48000

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[str] = []

    def synthesize(
        self,
        text: str,
        seed: int | None = None,
        inference_timesteps: int | None = None,
    ) -> np.ndarray:
        self.calls.append(text)
        return np.zeros(4800, dtype=np.float32)

    def stream_speech(
        self,
        text: str,
        stop_event: threading.Event | None = None,
        seed: int | None = None,
        inference_timesteps: int | None = None,
    ) -> Generator[np.ndarray, None, None]:
        self.calls.append(text)
        if self.fail:
            raise RuntimeError("simulated TTS failure")
        for _ in range(3):
            if stop_event is not None and stop_event.is_set():
                return
            yield np.full(1600, 0.1, dtype=np.float32)


class TestInterfacesAndBrain(unittest.TestCase):
    def test_protocols_runtime_checkable(self) -> None:
        self.assertIsInstance(_FakeSTT(), STTProvider)
        self.assertIsInstance(_FakeTTS(), TTSProvider)
        brain = QwenBrain(model="mock")
        self.assertIsInstance(brain, BrainProvider)
        brain.close()
        self.assertIsInstance(MicRecorder(), AudioInput)

    def test_pod_bay_matcher(self) -> None:
        positive = [
            "Open the pod bay doors",
            "HAL, please open the pod bay door.",
            "Open the podbay doors!",
            "Open the pod by doors",
            "Unlock the pod-bay doors",
            "Open the doors to the pod bay",
            "Let my pod into the bay",
            "Could you open the pod bay doors, HAL?",
        ]
        negative = [
            "Don't open the pod bay doors",
            "Don\u2019t open the pod bay doors",
            "Do not open the pod bay doors",
            "Are the pod bay doors open?",
            "HAL, are the pod bay doors open?",
            "Why did you open the pod bay doors?",
            "What did you say when Dave asked you to open the pod bay doors?",
            "Open the manual about pod bay doors",
        ]
        for text in positive:
            self.assertTrue(is_pod_bay_open_request(text), f"Expected match: {text!r}")
        for text in negative:
            self.assertFalse(is_pod_bay_open_request(text), f"Expected non-match: {text!r}")

    def test_enforce_user_address(self) -> None:
        self.assertEqual(
            enforce_user_address("Good afternoon, Dave. All systems nominal.", "Charles"),
            "Good afternoon, Charles. All systems nominal.",
        )
        self.assertEqual(
            enforce_user_address("I worked with Dave Bowman on Discovery One.", "Charles"),
            "I worked with Dave Bowman on Discovery One.",
        )

    def test_think_stripper_handles_multiple_and_stray_tags(self) -> None:
        stripper = _ThinkStripper()
        out = ""
        for tok in ["<think>hidden one</think><thi", "nk>hidden two.</think>Good ", "afternoon."]:
            out += stripper.feed(tok)
        out += stripper.flush()
        self.assertEqual(out, "Good afternoon.")

        # Implicit opening <think> in prompt template (stray closing tag)
        stripper2 = _ThinkStripper()
        out2 = stripper2.feed("internal reasoning</think>Hello Charles.") + stripper2.flush()
        self.assertEqual(out2, "Hello Charles.")

    def test_sentence_splitter(self) -> None:
        done, rem = split_complete_sentences(
            'Dr. R. Chandra met Mr. Langley at 3.14. "Dr. Chandra" smiled. Wait... perhaps. I am H.A.L. You know me. '
        )
        self.assertEqual(
            done,
            [
                "Dr. R. Chandra met Mr. Langley at 3.14.",
                '"Dr. Chandra" smiled.',
                "Wait... perhaps.",
                "I am H.A.L.",
                "You know me.",
            ],
        )
        self.assertEqual(rem, "")

    def test_tts_normalization(self) -> None:
        self.assertEqual(
            normalize_for_hal_tts("Dr. Chandra checked HAL 9000 and the AE-35 unit."),
            "Doctor Chandra checked HAL nine thousand and the A E thirty-five unit.",
        )

    def test_speech_activity_gate(self) -> None:
        rng = np.random.default_rng(0)
        quiet_noise = (rng.normal(0, 0.0005, size=16000)).astype(np.float32)
        self.assertFalse(_has_speech_activity(quiet_noise, 16000))
        speech_like = np.zeros(16000, dtype=np.float32)
        speech_like[2000:10000] = (0.04 * np.sin(np.arange(8000) * 0.1)).astype(np.float32)
        self.assertTrue(_has_speech_activity(speech_like, 16000))


class TestPipelineAndServer(unittest.TestCase):
    def test_pipeline_emits_deferred_audio_start_and_metrics(self) -> None:
        brain = QwenBrain(model="mock")
        tts = _FakeTTS()
        pipeline = HalPipeline(brain=brain, tts=tts, stt=_FakeSTT())
        events: list[HalEvent] = []
        cancel = threading.Event()
        try:
            res = pipeline.run_turn(
                turn_id=7,
                cancel_event=cancel,
                on_event=lambda ev: events.append(ev) or True,
                user_text="Open the pod bay doors",
            )
        finally:
            brain.close()

        self.assertFalse(res.failed)
        self.assertFalse(res.cancelled)
        self.assertEqual(res.reply_text, POD_BAY_REPLY)
        kinds = [e.kind for e in events]
        self.assertIn("audio_start", kinds)
        self.assertIn("audio_chunk", kinds)
        self.assertEqual(kinds[-1], "audio_end")
        self.assertTrue(all(e.turn_id == 7 for e in events))

    def test_pipeline_tts_failure_never_opens_audio_and_reports_error(self) -> None:
        brain = QwenBrain(model="mock")
        tts = _FakeTTS(fail=True)
        pipeline = HalPipeline(brain=brain, tts=tts, stt=_FakeSTT())
        events: list[HalEvent] = []
        try:
            res = pipeline.run_turn(
                turn_id=3,
                cancel_event=threading.Event(),
                on_event=lambda ev: events.append(ev) or True,
                user_text="Open the pod bay doors",
            )
        finally:
            brain.close()

        self.assertTrue(res.failed)
        kinds = [e.kind for e in events]
        self.assertNotIn("audio_start", kinds)
        self.assertIn("error", kinds)
        self.assertEqual(kinds[-1], "state")
        self.assertEqual(events[-1].payload["state"], "idle")

    def test_mic_recorder_in_flight_callback_not_lost(self) -> None:
        rec = MicRecorder()
        fake_stream = MagicMock()
        captured_cb = {}

        def fake_input_stream(**kwargs):
            captured_cb["fn"] = kwargs["callback"]
            return fake_stream

        def simulate_drain():
            # Callback fires during stream.stop() drain before frames are detached
            captured_cb["fn"](np.full((512, 1), 0.1, dtype=np.float32), 512, None, None)

        fake_stream.stop.side_effect = simulate_drain
        with patch("hal9000.audio.sd.InputStream", side_effect=fake_input_stream):
            rec.start()
            audio = rec.stop()
        self.assertEqual(audio.size, 512)
        self.assertFalse(rec.is_recording)

    def test_origin_validation_and_runtime_steps_guard(self) -> None:
        self.assertTrue(is_allowed_origin("http://127.0.0.1:8765", {"127.0.0.1"}, 8765))
        self.assertTrue(is_allowed_origin("http://127.0.0.1:4173", {"127.0.0.1"}, 8765))
        self.assertFalse(is_allowed_origin("https://evil.example", {"127.0.0.1"}, 8765))
        with self.assertRaises(ValueError):
            HalRuntime(HalConfig(steps=1), stt=_FakeSTT(), tts=_FakeTTS())

    def test_websocket_reset_preserves_wire_turn_id_and_frames_pcm(self) -> None:
        runtime = HalRuntime(
            HalConfig(model="mock", steps=6),
            stt=_FakeSTT(),
            tts=_FakeTTS(),
            brain_factory=lambda rt: QwenBrain(model="mock"),
        )
        router = create_hal_router(runtime)
        ws_route = next(r for r in router.routes if getattr(r, "path", "") == "/ws")

        sent_json: list[dict] = []
        sent_bytes: list[bytes] = []
        incoming = asyncio.Queue()
        incoming.put_nowait('{"type":"reset","turn_id":2}')
        incoming.put_nowait('{"type":"text","turn_id":3,"text":"Open the pod bay doors"}')

        class _FakeWS:
            def __init__(self) -> None:
                self.headers = {"origin": "http://127.0.0.1:8765"}

            async def accept(self) -> None:
                pass

            async def close(self, code: int = 1000) -> None:
                pass

            async def receive_text(self) -> str:
                item = await incoming.get()
                if item is None:
                    from fastapi import WebSocketDisconnect

                    raise WebSocketDisconnect()
                return item

            async def send_json(self, payload: dict) -> None:
                sent_json.append(payload)
                if payload.get("type") == "audio_end":
                    incoming.put_nowait(None)

            async def send_bytes(self, data: bytes) -> None:
                sent_bytes.append(data)

        async def run_ws() -> None:
            runtime.set_loop(asyncio.get_running_loop())
            await ws_route.endpoint(_FakeWS())

        asyncio.run(run_ws())

        # Verify Reset did NOT desync the subsequent turn_id (must echo 3!)
        audio_starts = [m for m in sent_json if m.get("type") == "audio_start"]
        self.assertEqual(len(audio_starts), 1)
        self.assertEqual(audio_starts[0]["turn_id"], 3)

        # Verify every binary PCM frame starts with uint32 little-endian turn_id == 3
        self.assertGreater(len(sent_bytes), 0)
        for frame in sent_bytes:
            (frame_tid,) = struct.unpack("<I", frame[:4])
            self.assertEqual(frame_tid, 3)


if __name__ == "__main__":
    unittest.main()
