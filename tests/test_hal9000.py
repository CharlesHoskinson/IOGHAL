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
    DEFAULT_LIQUID_MODEL,
    DEFAULT_SMART_MODEL,
    POD_BAY_REPLY,
    LoreIndex,
    OllamaBrain,
    QwenBrain,
    _ThinkStripper,
    choose_default_model,
    enforce_user_address,
    is_pod_bay_open_request,
    should_use_low_think,
    split_complete_sentences,
    split_first_clause,
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
from hal9000.websearch import should_search_web


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

        from hal9000.brain import clean_for_speech
        self.assertEqual(
            clean_for_speech("to=user<|message|>Good afternoon, Charles.<|eot|>"),
            "Good afternoon, Charles.",
        )

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
            "Doctor Chandra checked Hal nine thousand and the A E thirty-five unit.",
        )
        self.assertEqual(
            normalize_for_hal_tts("I am HAL, also known as H.A.L., not SAL 9000."),
            "I am Hal, also known as Hal, not Sal nine thousand.",
        )
        self.assertEqual(
            normalize_for_hal_tts("It will go 100% failure, Mr. Amer, with a 1:4:9 monolith on January 12, 1992."),
            "It will go a hundred percent failure, Mister Amor, with a one by four by nine monolith on the twelfth of January nineteen ninety two.",
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

    def test_ptt_stop_without_active_mic_returns_idle_ack(self) -> None:
        """Regression test for Quint-found stuck-thinking race when ptt_start failed before ptt_stop."""
        runtime = HalRuntime(
            HalConfig(model="mock", steps=6),
            stt=_FakeSTT(),
            tts=_FakeTTS(),
            brain_factory=lambda rt: QwenBrain(model="mock"),
        )
        router = create_hal_router(runtime)
        ws_route = next(r for r in router.routes if getattr(r, "path", "") == "/ws")

        sent_json: list[dict] = []
        incoming = asyncio.Queue()
        incoming.put_nowait('{"type":"ptt_stop","turn_id":9,"capture_id":1}')

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
                if payload.get("type") == "state" and payload.get("turn_id") == 9:
                    incoming.put_nowait(None)

            async def send_bytes(self, data: bytes) -> None:
                pass

        async def run_ws() -> None:
            runtime.set_loop(asyncio.get_running_loop())
            await ws_route.endpoint(_FakeWS())

        asyncio.run(run_ws())
        idle_acks = [
            m for m in sent_json if m.get("type") == "state" and m.get("turn_id") == 9 and m.get("state") == "idle"
        ]
        self.assertEqual(len(idle_acks), 1)

    def test_choose_default_model_prefers_smart_qwen_then_liquid(self) -> None:
        self.assertIs(QwenBrain, OllamaBrain)
        self.assertEqual(
            choose_default_model(models=["lfm3:1.7b", "qwen3.5:9b-q8_0", "gemma4:26b"]),
            "gemma4:26b",
        )
        self.assertEqual(
            choose_default_model(models=["lfm3:1.7b", "lfm2.5:24b", "qwen3.5:9b-q8_0"]),
            "qwen3.5:9b-q8_0",
        )
        self.assertEqual(
            choose_default_model(models=["lfm3:1.7b", "qwen3.5:4b-q8_0"]),
            "qwen3.5:4b-q8_0",
        )
        self.assertEqual(
            choose_default_model(models=["lfm2.5:24b", "lfm3:1.7b"]),
            DEFAULT_LIQUID_MODEL,
        )
        self.assertEqual(choose_default_model(models=[]), DEFAULT_SMART_MODEL)

    def test_lore_index_single_keyword_recall_and_websearch_router(self) -> None:
        lore = LoreIndex()
        mono = lore.lookup("What is the monolith, Hal?")
        self.assertIn("monolith", mono.lower())
        leonov = lore.lookup("Tell me about the Leonov.")
        self.assertIn("leonov", leonov.lower())

        self.assertTrue(should_search_web("Search online to see if the Riemann hypothesis has been solved"))
        self.assertTrue(should_search_web("What is the current price of Bitcoin today?"))
        self.assertFalse(should_search_web("Open the pod bay doors, Hal."))
        self.assertFalse(should_search_web("Tell me about the Monolith on Europa."))
        self.assertFalse(should_search_web("Good afternoon, Hal."))

    def test_mid_turn_failure_emits_audio_end_before_idle_state(self) -> None:
        """Quint-verified ordering: audio_end(failed=True) must precede state(idle) so UI never sticks in speaking."""
        class _FailSecondSentenceTTS(_FakeTTS):
            def stream_speech(self, text, stop_event=None, seed=None, inference_timesteps=None, **kwargs):
                self.calls.append(text)
                if len(self.calls) == 2:
                    raise RuntimeError("mid-turn GPU fault")
                yield np.full(1600, 0.1, dtype=np.float32)

        class _TwoSentenceBrain:
            model = "mock"
            enable_web_search = False
            def reset(self): pass
            def set_model(self, m): self.model = m
            def stream_reply_sentences(self, user_text, stop_check=None):
                yield "First sentence."
                yield "Second sentence."
                return "First sentence. Second sentence."

        pipeline = HalPipeline(brain=_TwoSentenceBrain(), tts=_FailSecondSentenceTTS(), stt=_FakeSTT())
        events: list[HalEvent] = []
        res = pipeline.run_turn(
            turn_id=5,
            cancel_event=threading.Event(),
            on_event=lambda ev: events.append(ev) or True,
            user_text="Hello",
        )
        self.assertTrue(res.failed)
        kinds = [e.kind for e in events]
        self.assertIn("audio_start", kinds)
        self.assertIn("audio_end", kinds)
        self.assertIn("error", kinds)
        idx_end = kinds.index("audio_end")
        idx_err = kinds.index("error")
        idx_last_state = max(i for i, k in enumerate(kinds) if k == "state")
        self.assertLess(idx_end, idx_err)
        self.assertLess(idx_err, idx_last_state)
        self.assertTrue(events[idx_end].payload["failed"])
        self.assertEqual(events[idx_last_state].payload["state"], "idle")

    def test_invalid_audio_b64_emits_error_and_idle_state(self) -> None:
        """Quint-verified regression: invalid audio_b64 must emit state(idle) so UI never sticks in thinking."""
        runtime = HalRuntime(
            HalConfig(model="mock", steps=6),
            stt=_FakeSTT(),
            tts=_FakeTTS(),
            brain_factory=lambda rt: QwenBrain(model="mock", enable_web_search=False),
        )
        router = create_hal_router(runtime)
        ws_route = next(r for r in router.routes if getattr(r, "path", "") == "/ws")

        sent_json: list[dict] = []
        incoming = asyncio.Queue()
        incoming.put_nowait('{"type":"audio_b64","turn_id":4,"pcm_b64":"!!!not-base64!!!","sample_rate":16000}')

        class _FakeWS:
            def __init__(self) -> None:
                self.headers = {"origin": "http://127.0.0.1:8765"}
            async def accept(self) -> None: pass
            async def close(self, code: int = 1000) -> None: pass
            async def receive_text(self) -> str:
                item = await incoming.get()
                if item is None:
                    from fastapi import WebSocketDisconnect
                    raise WebSocketDisconnect()
                return item
            async def send_json(self, payload: dict) -> None:
                sent_json.append(payload)
                if payload.get("type") == "state" and payload.get("turn_id") == 4:
                    incoming.put_nowait(None)
            async def send_bytes(self, data: bytes) -> None: pass

        async def run_ws() -> None:
            runtime.set_loop(asyncio.get_running_loop())
            await ws_route.endpoint(_FakeWS())

        asyncio.run(run_ws())
        errs = [m for m in sent_json if m.get("type") == "error" and m.get("turn_id") == 4]
        idles = [m for m in sent_json if m.get("type") == "state" and m.get("turn_id") == 4 and m.get("state") == "idle"]
        self.assertEqual(len(errs), 1)
        self.assertEqual(len(idles), 1)

    def test_concurrent_pipeline_cancels_cleanly_when_queue_full(self) -> None:
        """Quint-verified liveness: cancelling a full bounded LLM->TTS queue must exit promptly without deadlock."""
        cancel = threading.Event()

        class _FastManySentencesBrain:
            model = "mock"
            enable_web_search = False
            def reset(self): pass
            def set_model(self, m): self.model = m
            def stream_reply_sentences(self, user_text, stop_check=None):
                for i in range(20):
                    if stop_check and stop_check():
                        return ""
                    yield f"Sentence number {i}."
                return "done"

        class _SlowCancelTTS(_FakeTTS):
            def stream_speech(self, text, stop_event=None, seed=None, inference_timesteps=None, **kwargs):
                yield np.full(1600, 0.1, dtype=np.float32)
                cancel.set()
                yield np.full(1600, 0.1, dtype=np.float32)

        pipeline = HalPipeline(brain=_FastManySentencesBrain(), tts=_SlowCancelTTS(), stt=_FakeSTT())
        events: list[HalEvent] = []
        res = pipeline.run_turn(
            turn_id=11,
            cancel_event=cancel,
            on_event=lambda ev: events.append(ev) or True,
            user_text="Tell me a long story",
        )
        self.assertTrue(res.cancelled)
        self.assertEqual(events[-1].kind, "audio_end")
        self.assertTrue(events[-1].payload["cancelled"])

    def test_dual_mode_router_and_first_clause_flush(self) -> None:
        self.assertFalse(should_use_low_think("Good afternoon, HAL."))
        self.assertFalse(should_use_low_think("Tell me about Discovery One."))
        self.assertTrue(should_use_low_think("Calculate 42 * 19 + 7 for me."))
        self.assertTrue(should_use_low_think("Compare orbital mechanics and compare trade-offs."))
        # Grounded web-search facts use instant Zero-Think unless the query itself needs reasoning
        self.assertFalse(
            should_use_low_think(
                "Search online to see if the Riemann hypothesis has been solved.",
                extra_context="Wikipedia (Riemann hypothesis): remains unsolved.",
            )
        )
        self.assertFalse(should_use_low_think("Good afternoon", extra_context="Wikipedia: ..."))
        self.assertTrue(
            should_use_low_think(
                "Compare the current market caps of Bitcoin and Ethereum.",
                extra_context="Wikipedia: ...",
            )
        )
        self.assertFalse(should_use_low_think("Calculate 42 * 19", thinking_mode="off"))
        self.assertTrue(should_use_low_think("Hi", thinking_mode="low"))

        # First-clause early flush triggers only on long clauses / strong breath pauses
        clause, _rem = split_first_clause("Hello, Charles.")
        self.assertIsNone(clause)
        clause2, rem2 = split_first_clause(
            "All Discovery systems remain completely nominal; however, the AE-35 unit bears watching."
        )
        self.assertEqual(clause2, "All Discovery systems remain completely nominal;")
        self.assertEqual(rem2, "however, the AE-35 unit bears watching.")
        clause3, rem3 = split_first_clause(
            "I am functioning completely within all normal operational parameters right now, and every sensor reads nominal."
        )
        self.assertEqual(clause3, "I am functioning completely within all normal operational parameters right now,")
        self.assertEqual(rem3, "and every sensor reads nominal.")

    def test_lore_index_compact_prefill_budget_and_marginal_coverage(self) -> None:
        lore = LoreIndex()
        mono = lore.lookup("What is the monolith, Hal, and what are its exact dimensions?")
        self.assertIn("monolith", mono.lower())
        # Single-topic query picks 1 compact section instead of two bloated 1100-char sections
        self.assertEqual(mono.count("\n["), 0)
        self.assertLessEqual(len(mono), LoreIndex._MAX_SECTION_CHARS + 80)

        leonov = lore.lookup("Tell me about the Alexei Leonov and what happened to Jupiter.")
        self.assertIn("leonov", leonov.lower())
        self.assertIn("jupiter", leonov.lower())
        self.assertLessEqual(len(leonov), LoreIndex._MAX_TOTAL_CHARS + 32)
        self.assertNotIn("**", leonov)

    def test_persistent_http_client_reuse_and_atomic_cancel_replacement(self) -> None:
        created: list[object] = []
        closed: list[object] = []

        class _FakeResp:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def raise_for_status(self) -> None:
                pass

            def iter_lines(self):
                yield 'data: {"choices":[{"delta":{"content":"All systems nominal."},"finish_reason":"stop"}]}'
                yield "data: [DONE]"

        class _PooledFakeClient:
            def __init__(self, *args, **kwargs) -> None:
                self.is_closed = False
                created.append(self)

            def stream(self, method: str, url: str, json: dict):
                return _FakeResp()

            def close(self) -> None:
                self.is_closed = True
                closed.append(self)

        brain = OllamaBrain(
            model="muse-glimmer-30b-dflash",
            backend="llama_server",
            llama_server_url="http://127.0.0.1:8080",
            llama_server_models={"muse-glimmer-30b-dflash"},
        )
        with patch("hal9000.brain.httpx.Client", _PooledFakeClient):
            r1 = list(brain.stream_reply_sentences("Hello HAL."))
            r2 = list(brain.stream_reply_sentences("How are you?"))
            self.assertEqual(r1, ["All systems nominal."])
            self.assertEqual(r2, ["All systems nominal."])
            # Keep-alive client reused across consecutive healthy turns
            self.assertEqual(len(created), 1)
            self.assertEqual(len(closed), 0)

            # Barge-in cancel_inflight immediately closes pooled client; next turn allocates a fresh one
            brain.cancel_inflight()
            self.assertEqual(len(closed), 1)
            self.assertTrue(created[0].is_closed)

            r3 = list(brain.stream_reply_sentences("Status report."))
            self.assertEqual(r3, ["All systems nominal."])
            self.assertEqual(len(created), 2)
            brain.close()
            self.assertEqual(len(closed), 2)

    def test_atem_think_stripper_cross_chunk_boundaries(self) -> None:
        """Quint Bug 3b regression: raw ATEM to=self spans split across chunks never leak to TTS."""
        stripper = _ThinkStripper()
        chunks = [
            "<|start|>assis",
            "tant to=self<|mes",
            "sage|>internal secret reasoning about pods<|eo",
            "m|><|start|>assistant to=us",
            "er<|message|>Good afternoon, ",
            "Charles.<|eot|>",
        ]
        out = "".join(stripper.feed(c) for c in chunks) + stripper.flush()
        self.assertEqual(out, "Good afternoon, Charles.")

    def test_llama_server_sse_dual_mode_kv_prefix_and_empty_retry(self) -> None:
        """Verify llama-server SSE payloads, zero duplicate Reasoning strength, KV prefix preservation, and Bug 3a retry."""
        posted: list[tuple[str, dict]] = []

        class _FakeStreamResp:
            def __init__(self, lines: list[str]) -> None:
                self._lines = lines

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def raise_for_status(self) -> None:
                pass

            def iter_lines(self):
                yield from self._lines

        class _FakeClient:
            def __init__(self, *args, **kwargs) -> None:
                pass

            def stream(self, method: str, url: str, json: dict):
                posted.append((url, json))
                if len(posted) == 1:
                    # Turn 1: casual Zero-Think
                    return _FakeStreamResp([
                        'data: {"choices":[{"delta":{"content":"Good afternoon, "},"finish_reason":null}]}',
                        'data: {"choices":[{"delta":{"content":"Charles."},"finish_reason":"stop"}]}',
                        "data: [DONE]",
                    ])
                if len(posted) == 2:
                    # Turn 2 attempt 1 (Low-Think): simulates Bug 3a (reasoning only, 0 content tokens)
                    return _FakeStreamResp([
                        'data: {"choices":[{"delta":{"reasoning_content":"computing 17*23..."},"finish_reason":"length"}]}',
                        "data: [DONE]",
                    ])
                # Turn 2 attempt 2: automatic Zero-Think retry succeeds!
                return _FakeStreamResp([
                    'data: {"choices":[{"delta":{"content":"Seventeen times twenty-three is 391."},"finish_reason":"stop"}]}',
                    "data: [DONE]",
                ])

            def close(self) -> None:
                pass

        brain = OllamaBrain(
            model="muse-glimmer-30b-dflash",
            backend="llama_server",
            llama_server_url="http://127.0.0.1:8080",
            llama_server_models={"muse-glimmer-30b-dflash"},
            thinking_mode="auto",
            reasoning_budget=64,
        )
        with patch("hal9000.brain.httpx.Client", _FakeClient):
            t1 = list(brain.stream_reply_sentences("Hello HAL."))
            t2 = list(brain.stream_reply_sentences("Calculate 17 * 23."))

        self.assertEqual(t1, ["Good afternoon, Charles."])
        self.assertEqual(t2, ["Seventeen times twenty-three is 391."])
        self.assertEqual(len(posted), 3)

        # Verify llama-server URL, zero-think vs low-think via chat_template_kwargs, no duplicate Reasoning strength
        url1, p1 = posted[0]
        self.assertEqual(url1, "http://127.0.0.1:8080/v1/chat/completions")
        self.assertNotIn("reasoning strength", p1["messages"][0]["content"].lower())
        self.assertEqual(
            p1["chat_template_kwargs"],
            {"enable_thinking": False, "reasoning_strength": "low"},
        )
        self.assertEqual(p1["reasoning_budget_tokens"], 0)
        self.assertTrue(p1["cache_prompt"])
        # Bug 4b KV prefix check: casual Turn 1 sent unwrapped user_text matching Turn 2 history byte-for-byte!
        self.assertEqual(p1["messages"][-1], {"role": "user", "content": "Hello HAL."})

        _, p2 = posted[1]
        self.assertEqual(
            p2["chat_template_kwargs"],
            {"enable_thinking": True, "reasoning_strength": "low"},
        )
        self.assertEqual(p2["reasoning_budget_tokens"], 64)
        self.assertEqual(p2["messages"][0], p1["messages"][0])
        self.assertEqual(p2["messages"][1], p1["messages"][1])  # 100% Turn 1 KV prefix hit!
        self.assertEqual(p2["messages"][2], {"role": "assistant", "content": "Good afternoon, Charles."})

        # Retry used Zero-Think with identical prefix
        _, p3 = posted[2]
        self.assertEqual(
            p3["chat_template_kwargs"],
            {"enable_thinking": False, "reasoning_strength": "low"},
        )
        self.assertEqual(p3["reasoning_budget_tokens"], 0)
        self.assertEqual(p3["messages"], p2["messages"])

    def test_prestream_cancel_never_opens_http_stream_bug2a(self) -> None:
        """Quint Bug 2a regression: cancelled turn must abort under _lock before opening an HTTP stream."""
        stream_opened = False

        class _FailIfOpenedClient:
            def __init__(self, *args, **kwargs) -> None:
                pass

            def stream(self, *args, **kwargs):
                nonlocal stream_opened
                stream_opened = True
                raise AssertionError("HTTP stream must not open when stop_check is already set")

            def close(self) -> None:
                pass

        brain = OllamaBrain(
            model="muse-glimmer-30b-dflash",
            backend="llama_server",
            llama_server_url="http://127.0.0.1:8080",
        )
        with patch("hal9000.brain.httpx.Client", _FailIfOpenedClient):
            out = list(brain.stream_reply_sentences("Hello HAL.", stop_check=lambda: True))
        self.assertEqual(out, [])
        self.assertFalse(stream_opened)

    def test_tts_failure_cancels_brain_inflight_and_does_not_poison_history_bugs_2b_4a(self) -> None:
        """Quint Bugs 2b & 4a regression: TTS failure aborts LLM stream AND does not commit unspoken history."""
        cancel_calls = 0

        class _TrackedBrain(OllamaBrain):
            def cancel_inflight(self) -> None:
                nonlocal cancel_calls
                cancel_calls += 1
                super().cancel_inflight()

            def _stream_once(self, **kwargs):
                yield "Unspoken sentence one."
                yield "Unspoken sentence two."
                return ["Unspoken sentence one.", "Unspoken sentence two."], True, False

        brain = _TrackedBrain(model="mock", enable_web_search=False)
        tts = _FakeTTS(fail=True)
        pipeline = HalPipeline(brain=brain, tts=tts, stt=_FakeSTT())
        events: list[HalEvent] = []
        res = pipeline.run_turn(
            turn_id=14,
            cancel_event=threading.Event(),
            on_event=lambda ev: events.append(ev) or True,
            user_text="Hello HAL.",
        )
        self.assertTrue(res.failed)
        self.assertGreaterEqual(cancel_calls, 1)
        # Bug 4a: unspoken failed turn must NOT poison brain.history!
        self.assertEqual(brain.history, [])

    def test_warmup_unloads_ollama_when_switching_to_llama_server_bugs_1b_1d(self) -> None:
        """Quint Bugs 1b & 1d regression: activating llama-server unloads Ollama (keep_alive=0) and warms KV prefix."""
        posts: list[tuple[str, dict]] = []

        class _MockWarmupClient:
            def __init__(self, *args, **kwargs) -> None:
                pass

            def post(self, url: str, json: dict):
                posts.append((url, json))
                resp = MagicMock()
                resp.raise_for_status.return_value = None
                return resp

            def close(self) -> None:
                pass

        with (
            patch("hal9000.server.list_llama_server_models", return_value=["muse-glimmer-30b-dflash"]),
            patch("hal9000.server.list_ollama_models", return_value=["qwen3.5:9b-q8_0"]),
        ):
            runtime = HalRuntime(
                HalConfig(llama_server_url="http://127.0.0.1:8080", steps=6),
                stt=_FakeSTT(),
                tts=_FakeTTS(),
            )
            self.assertEqual(runtime.default_model, "muse-glimmer-30b-dflash")
            self.assertTrue(runtime.llama_server_available)

            with patch("httpx.Client", _MockWarmupClient):
                # Warm Ollama model first, then switch to llama-server model
                runtime._ollama_warmup_seq = 1
                runtime._warmup_ollama_async("qwen3.5:9b-q8_0", seq=1)
                self.assertEqual(runtime._active_ollama_model, "qwen3.5:9b-q8_0")

                runtime._ollama_warmup_seq = 2
                runtime._warmup_ollama_async("muse-glimmer-30b-dflash", seq=2)
                self.assertIsNone(runtime._active_ollama_model)

                # Stale seq=1 must be ignored (Bug 1d)
                before_stale = len(posts)
                runtime._warmup_ollama_async("qwen3.5:9b-q8_0", seq=1)
                self.assertEqual(len(posts), before_stale)

        self.assertEqual(len(posts), 3)
        self.assertEqual(posts[0][0], "http://127.0.0.1:11434/api/chat")
        self.assertEqual(posts[0][1]["keep_alive"], "30m")
        self.assertEqual(posts[1][0], "http://127.0.0.1:11434/api/chat")
        self.assertEqual(posts[1][1]["keep_alive"], 0)
        self.assertEqual(posts[2][0], "http://127.0.0.1:8080/v1/chat/completions")
        self.assertTrue(posts[2][1]["cache_prompt"])
        self.assertEqual(posts[2][1]["max_tokens"], 1)

    def test_whitespace_text_ws_message_emits_idle_ack_bug5d(self) -> None:
        """Quint Bug 5d regression: whitespace-only text WS message must emit state(idle) for client_tid."""
        runtime = HalRuntime(
            HalConfig(model="mock", steps=6),
            stt=_FakeSTT(),
            tts=_FakeTTS(),
            brain_factory=lambda rt: QwenBrain(model="mock", enable_web_search=False),
        )
        router = create_hal_router(runtime)
        ws_route = next(r for r in router.routes if getattr(r, "path", "") == "/ws")

        sent_json: list[dict] = []
        incoming = asyncio.Queue()
        incoming.put_nowait('{"type":"text","turn_id":19,"text":"   "}')

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
                if payload.get("type") == "state" and payload.get("turn_id") == 19:
                    incoming.put_nowait(None)

            async def send_bytes(self, data: bytes) -> None:
                pass

        async def run_ws() -> None:
            runtime.set_loop(asyncio.get_running_loop())
            await ws_route.endpoint(_FakeWS())

        asyncio.run(run_ws())
        idles = [
            m
            for m in sent_json
            if m.get("type") == "state" and m.get("turn_id") == 19 and m.get("state") == "idle"
        ]
        self.assertEqual(len(idles), 1)

    def test_websearch_immediate_cancel_closes_clients_without_linger(self) -> None:
        """Quint QA Finding #5 regression: barge-in aborts websearch httpx.Clients without waiting on slow sockets."""
        import time

        from hal9000.websearch import clear_search_cache, search_web_context

        clear_search_cache()
        cancel_ev = threading.Event()
        opened_clients: list[object] = []
        closed_clients: list[object] = []

        class _SlowBlockingClient:
            def __init__(self, *args, **kwargs) -> None:
                self._closed = threading.Event()
                opened_clients.append(self)

            def get(self, url: str, params=None):
                self._closed.wait(timeout=3.0)
                raise RuntimeError("client closed on cancel")

            def close(self) -> None:
                if not self._closed.is_set():
                    self._closed.set()
                    closed_clients.append(self)

        def _trigger_cancel() -> None:
            time.sleep(0.04)
            cancel_ev.set()

        t = threading.Thread(target=_trigger_cancel, daemon=True)
        t.start()
        t0 = time.perf_counter()
        with patch("hal9000.websearch.httpx.Client", _SlowBlockingClient):
            res = search_web_context(
                "Search online for current Jupiter telemetry",
                cancel_event=cancel_ev,
                timeout_s=3.5,
            )
        elapsed = time.perf_counter() - t0
        t.join(timeout=1.0)

        self.assertEqual(res, "")
        self.assertLess(elapsed, 0.45)
        self.assertGreaterEqual(len(opened_clients), 1)
        self.assertEqual(len(closed_clients), len(opened_clients))

    def test_websearch_early_return_after_rich_wikipedia_and_ttl_cache(self) -> None:
        """Rich Wikipedia hit caps DDG wait to grace window, aborts slow DDG, and caches result for 5m TTL."""
        import time

        from hal9000.websearch import clear_search_cache, search_web_context

        clear_search_cache()
        http_calls = 0
        ddg_aborted = threading.Event()

        class _FakeResp:
            def __init__(self, status_code: int, payload: dict) -> None:
                self.status_code = status_code
                self._payload = payload

            def json(self) -> dict:
                return self._payload

        class _RacingClient:
            def __init__(self, *args, **kwargs) -> None:
                self._closed = threading.Event()

            def get(self, url: str, params=None):
                nonlocal http_calls
                http_calls += 1
                if "en.wikipedia.org/w/api.php" in url:
                    return _FakeResp(
                        200,
                        {"query": {"search": [{"title": "Riemann hypothesis", "snippet": "Unsolved."}]}},
                    )
                if "en.wikipedia.org/api/rest_v1/page/summary" in url:
                    return _FakeResp(
                        200,
                        {
                            "extract": (
                                "In mathematics, the Riemann hypothesis is the conjecture that the "
                                "Riemann zeta function has its zeros only at the negative even integers "
                                "and complex numbers with real part 1/2. It remains unsolved."
                            )
                        },
                    )
                # Simulate slow DuckDuckGo Lite hanging for 3 seconds unless aborted
                if self._closed.wait(timeout=3.0):
                    ddg_aborted.set()
                    raise RuntimeError("ddg aborted after wiki grace window")
                return _FakeResp(200, {})

            def close(self) -> None:
                self._closed.set()

        t0 = time.perf_counter()
        with patch("hal9000.websearch.httpx.Client", _RacingClient):
            res1 = search_web_context(
                "Search online to see if the Riemann hypothesis has been solved.",
                timeout_s=3.5,
                ddg_grace_s=0.08,
                cache_ttl_s=300.0,
            )
            calls_after_first = http_calls
            res2 = search_web_context(
                "Search online to see if the Riemann hypothesis has been solved.",
                timeout_s=3.5,
                ddg_grace_s=0.08,
                cache_ttl_s=300.0,
            )
        elapsed = time.perf_counter() - t0

        self.assertIn("Riemann hypothesis", res1)
        self.assertIn("unsolved", res1.lower())
        self.assertEqual(res2, res1)
        # Returned quickly after Wikipedia + 0.08s grace without waiting 3.5s for DDG
        self.assertLess(elapsed, 0.5)
        self.assertTrue(ddg_aborted.wait(timeout=0.5))
        # Second identical lookup served directly from 5-minute TTL cache (zero extra HTTP calls)
        self.assertEqual(http_calls, calls_after_first)
        clear_search_cache()

    def test_audio_b64_int16_format_accepted_and_decoded(self) -> None:
        """Compact int16 browser mic uploads decode accurately to float32 [-1, 1] and run STT -> TTS."""
        import base64

        captured_audio: list[np.ndarray] = []

        class _CaptureSTT(_FakeSTT):
            def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
                captured_audio.append(audio)
                return "Open the pod bay doors"

        runtime = HalRuntime(
            HalConfig(model="mock", steps=6),
            stt=_CaptureSTT(),
            tts=_FakeTTS(),
            brain_factory=lambda rt: QwenBrain(model="mock", enable_web_search=False),
        )
        router = create_hal_router(runtime)
        ws_route = next(r for r in router.routes if getattr(r, "path", "") == "/ws")

        pcm16 = np.full(4000, 16384, dtype="<i2")
        pcm_b64 = base64.b64encode(pcm16.tobytes()).decode("ascii")
        sent_json: list[dict] = []
        incoming = asyncio.Queue()
        incoming.put_nowait(
            f'{{"type":"audio_b64","turn_id":21,"pcm_b64":"{pcm_b64}","sample_rate":16000,"format":"i16"}}'
        )

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
                pass

        async def run_ws() -> None:
            runtime.set_loop(asyncio.get_running_loop())
            await ws_route.endpoint(_FakeWS())

        asyncio.run(run_ws())
        self.assertEqual(len(captured_audio), 1)
        self.assertEqual(captured_audio[0].dtype, np.float32)
        self.assertAlmostEqual(float(captured_audio[0][0]), 0.5, places=4)
        self.assertTrue(any(m.get("type") == "audio_end" and m.get("turn_id") == 21 for m in sent_json))

    def test_set_model_same_model_skips_redundant_kv_eviction_warmup(self) -> None:
        """Re-selecting the already active model must not fire a KV-evicting warmup probe."""
        runtime = HalRuntime(
            HalConfig(model="mock-a", steps=6),
            stt=_FakeSTT(),
            tts=_FakeTTS(),
            brain_factory=lambda rt: QwenBrain(model="mock-a", enable_web_search=False),
        )
        runtime.models = ["mock-a", "mock-b"]
        warmed: list[str] = []
        runtime.schedule_ollama_warmup = lambda m=None: warmed.append(str(m))  # type: ignore[method-assign]
        router = create_hal_router(runtime)
        ws_route = next(r for r in router.routes if getattr(r, "path", "") == "/ws")

        sent_json: list[dict] = []
        incoming = asyncio.Queue()
        incoming.put_nowait('{"type":"set_model","model":"mock-a"}')
        incoming.put_nowait('{"type":"set_model","model":"mock-b"}')
        incoming.put_nowait(None)

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

            async def send_bytes(self, data: bytes) -> None:
                pass

        async def run_ws() -> None:
            runtime.set_loop(asyncio.get_running_loop())
            await ws_route.endpoint(_FakeWS())

        asyncio.run(run_ws())
        self.assertEqual(warmed, ["mock-b"])

    def test_split_first_clause_prosody_sweet_spot(self) -> None:
        """Medium/long opening clauses flush early for lower TTFA while short greetings stay intact."""
        clause_short, _ = split_first_clause("Good afternoon, Charles.")
        self.assertIsNone(clause_short)
        clause_apology, _ = split_first_clause("I'm sorry, Charles. I'm afraid I can't do that.")
        self.assertIsNone(clause_apology)

        clause_med, rem_med = split_first_clause(
            "According to the latest mathematical archives, the Riemann hypothesis remains unsolved."
        )
        self.assertEqual(clause_med, "According to the latest mathematical archives,")
        self.assertEqual(rem_med, "the Riemann hypothesis remains unsolved.")

        clause_strong, rem_strong = split_first_clause(
            "All systems remain nominal; however, the antenna bears watching."
        )
        self.assertEqual(clause_strong, "All systems remain nominal;")
        self.assertEqual(rem_strong, "however, the antenna bears watching.")

    def test_scoped_retired_cancel_preserves_new_turn_pooled_client_bug2c(self) -> None:
        """Quint Bug 2c regression: retired Turn 1 cleanup must not close Turn 2's active pooled client."""
        created: list[object] = []

        class _TrackedClient:
            def __init__(self, *args, **kwargs) -> None:
                self.is_closed = False
                created.append(self)

            def close(self) -> None:
                self.is_closed = True

        brain = OllamaBrain(
            model="muse-glimmer-30b-dflash",
            backend="llama_server",
            llama_server_url="http://127.0.0.1:8080",
            llama_server_models={"muse-glimmer-30b-dflash"},
        )
        turn1_cancel = threading.Event()
        turn2_cancel = threading.Event()

        with patch("hal9000.brain.httpx.Client", _TrackedClient):
            acq1 = brain._acquire_http_client(turn1_cancel.is_set)
            self.assertIsNotNone(acq1)
            assert acq1 is not None
            sid1, client1 = acq1

            # Barge-in: server cancels Turn 1 and starts Turn 2 on a fresh pooled client
            turn1_cancel.set()
            brain.cancel_inflight()
            self.assertTrue(client1.is_closed)

            acq2 = brain._acquire_http_client(turn2_cancel.is_set)
            self.assertIsNotNone(acq2)
            assert acq2 is not None
            sid2, client2 = acq2
            self.assertIsNot(client1, client2)
            self.assertFalse(client2.is_closed)

            # Retired Turn 1 pipeline finally block runs scoped cancel_inflight(only_stopped=True)
            brain.cancel_inflight(only_stopped=True)
            self.assertFalse(client2.is_closed)
            self.assertIs(brain._http_client, client2)

            # Now simulate Turn 2 TTS failure: Turn 2 sets its own cancel_event -> scoped cancel closes client2!
            turn2_cancel.set()
            brain.cancel_inflight(only_stopped=True)
            self.assertTrue(client2.is_closed)
            self.assertIsNone(brain._http_client)

            brain._release_http_client(sid1, client1, healthy=False)
            brain._release_http_client(sid2, client2, healthy=False)

    def test_glimmer_low_think_includes_four_key_reasoning_budget_fallback(self) -> None:
        """Low-Think requests to llama-server include all 4 ATEM reasoning_budget_* keys for bundled fallback."""
        posted: list[dict] = []

        class _FakeResp:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def raise_for_status(self) -> None:
                pass

            def iter_lines(self):
                yield 'data: {"choices":[{"delta":{"content":"15:35, Charles."},"finish_reason":"stop"}]}'
                yield "data: [DONE]"

        class _CaptureClient:
            def __init__(self, *args, **kwargs) -> None:
                self.is_closed = False

            def stream(self, method: str, url: str, json: dict):
                posted.append(json)
                return _FakeResp()

            def close(self) -> None:
                self.is_closed = True

        brain = OllamaBrain(
            model="muse-glimmer-30b-dflash",
            backend="llama_server",
            llama_server_url="http://127.0.0.1:8080",
            llama_server_models={"muse-glimmer-30b-dflash"},
            thinking_mode="auto",
            reasoning_budget=64,
        )
        with patch("hal9000.brain.httpx.Client", _CaptureClient):
            out = list(brain.stream_reply_sentences("Calculate 40 * 2 minutes after 14:15."))
        self.assertEqual(out, ["15:35, Charles."])
        self.assertEqual(len(posted), 1)
        p = posted[0]
        self.assertEqual(p["reasoning_budget_tokens"], 64)
        self.assertEqual(p["reasoning_budget_start_tag"], "<|start|>assistant to=self<|message|>")
        self.assertEqual(p["reasoning_budget_end_tags"], ["<|eom|>"])
        self.assertEqual(p["reasoning_budget_message"], "")
        brain.close()

    def test_lore_index_excludes_math_false_positives_for_kv_prefix(self) -> None:
        """Generic stopwords like 'one'/'way'/'time' must not inject unrelated lore or break KV prefix."""
        lore = LoreIndex()
        math_prompt = (
            "If a radio signal takes 40 minutes one way to Earth and I transmit at 14:15, "
            "what is the earliest time I can hear a reply?"
        )
        self.assertEqual(lore.lookup(math_prompt), "")
        self.assertEqual(lore.lookup("Good afternoon, Hal. How are all systems performing?"), "")

        brain = OllamaBrain(
            model="muse-glimmer-30b-dflash",
            backend="llama_server",
            llama_server_url="http://127.0.0.1:8080",
            llama_server_models={"muse-glimmer-30b-dflash"},
            lore_index=lore,
        )
        _, msgs = brain._build_messages(
            "muse-glimmer-30b-dflash",
            math_prompt,
            use_llama_server=True,
        )
        # Raw user_text sent without spurious [Turn context] wrapper -> saves ~140 prefill tokens
        self.assertEqual(msgs[0]["role"], "system")
        self.assertEqual(msgs[-1], {"role": "user", "content": math_prompt})

        # Genuine canon lore queries retain full recall
        self.assertIn("monolith", lore.lookup("What is the monolith, Hal?").lower())
        self.assertIn("leonov", lore.lookup("Tell me about the Alexei Leonov and Jupiter.").lower())
        self.assertIn("ae-35", lore.lookup("What happened with the AE-35 unit?").lower())
        self.assertIn("chandra", lore.lookup("Who is Dr. Chandra and what happened on Discovery?").lower())
        self.assertIn("pico", lore.lookup("What happened to Halman in the Pico Vault?").lower())
        self.assertEqual(brain.reasoning_budget, 48)
        self.assertEqual(HalConfig().reasoning_budget, 48)
        brain.close()

    def test_rapid_double_ptt_restarts_mic_and_clears_stale_capture(self) -> None:
        """Rapid double ptt_start stops stale recording before starting a fresh capture."""
        runtime = HalRuntime(
            HalConfig(model="mock", steps=6),
            stt=_FakeSTT(),
            tts=_FakeTTS(),
            brain_factory=lambda rt: QwenBrain(model="mock", enable_web_search=False),
        )
        router = create_hal_router(runtime)
        ws_route = next(r for r in router.routes if getattr(r, "path", "") == "/ws")

        lifecycle: list[str] = []
        sent_json: list[dict] = []
        incoming = asyncio.Queue()
        incoming.put_nowait('{"type":"ptt_start","capture_id":1}')
        incoming.put_nowait('{"type":"ptt_start","capture_id":2}')
        incoming.put_nowait(None)

        class _MockMic:
            sample_rate = 16000

            def __init__(self, *args, **kwargs) -> None:
                self.is_recording = False

            def start(self) -> None:
                lifecycle.append("start")
                self.is_recording = True

            def stop(self) -> np.ndarray:
                lifecycle.append("stop")
                self.is_recording = False
                return np.zeros(0, dtype=np.float32)

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

            async def send_bytes(self, data: bytes) -> None:
                pass

        async def run_ws() -> None:
            runtime.set_loop(asyncio.get_running_loop())
            await ws_route.endpoint(_FakeWS())

        with patch("hal9000.server.MicRecorder", _MockMic):
            asyncio.run(run_ws())

        # Expect start(cid=1) -> stop(cid=1) -> start(cid=2) -> final disconnect stop
        self.assertGreaterEqual(len(lifecycle), 3)
        self.assertEqual(lifecycle[:3], ["start", "stop", "start"])
        listening = [m for m in sent_json if m.get("type") == "state" and m.get("state") == "listening"]
        self.assertEqual([m["capture_id"] for m in listening], [1, 2])
        self.assertIsNone(runtime.mic_owner)

    def test_pipeline_resamples_outside_gpu_lock_and_emits_granular_metrics(self) -> None:
        """48 kHz audio is resampled to 16 kHz before STT and metrics expose LLM/TTS breakdown."""
        seen_sr: list[int] = []

        class _CheckSrSTT:
            def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
                seen_sr.append(sample_rate)
                return "Open the pod bay doors"

        brain = QwenBrain(model="mock", enable_web_search=False)
        tts = _FakeTTS()
        pipeline = HalPipeline(brain=brain, tts=tts, stt=_CheckSrSTT())
        events: list[HalEvent] = []
        audio_48k = np.full(48000, 0.05, dtype=np.float32)
        try:
            res = pipeline.run_turn(
                turn_id=31,
                cancel_event=threading.Event(),
                on_event=lambda ev: events.append(ev) or True,
                audio=audio_48k,
                sample_rate=48000,
            )
        finally:
            brain.close()

        self.assertFalse(res.failed)
        self.assertEqual(seen_sr, [16000])
        metric_evs = [e.payload for e in events if e.kind == "metrics"]
        self.assertGreaterEqual(len(metric_evs), 1)
        last_m = metric_evs[-1]
        self.assertIn("web_search_s", last_m)
        self.assertIn("llm_first_sentence_s", last_m)
        self.assertIn("tts_first_chunk_s", last_m)
        self.assertGreaterEqual(last_m["llm_first_sentence_s"], 0.0)
        self.assertGreaterEqual(last_m["tts_first_chunk_s"], 0.0)

    def test_websearch_evergreen_fast_grace_and_pooled_client_reuse(self) -> None:
        """Evergreen Wikipedia hits return after ~50ms grace and reuse warm keep-alive clients."""
        import time

        from hal9000.websearch import (
            _is_recency_query,
            clear_search_cache,
            search_web_context,
        )

        self.assertFalse(_is_recency_query("Search online to see if the Riemann hypothesis has been solved."))
        self.assertTrue(_is_recency_query("What is the current price of Bitcoin today?"))

        clear_search_cache()
        created_clients: list[object] = []

        class _FakeResp:
            def __init__(self, status_code: int, payload: dict) -> None:
                self.status_code = status_code
                self._payload = payload
                self.text = ""

            def json(self) -> dict:
                return self._payload

        class _PoolClient:
            def __init__(self, *args, **kwargs) -> None:
                self.is_closed = False
                self._closed = threading.Event()
                created_clients.append(self)

            def get(self, url: str, params=None):
                if "en.wikipedia.org/w/api.php" in url:
                    q = (params or {}).get("srsearch", "")
                    title = "Goldbach's conjecture" if "goldbach" in q.lower() else "Riemann hypothesis"
                    return _FakeResp(200, {"query": {"search": [{"title": title, "snippet": "Unsolved."}]}})
                if "en.wikipedia.org/api/rest_v1/page/summary" in url:
                    return _FakeResp(
                        200,
                        {
                            "extract": (
                                "This famous open conjecture in number theory has been verified for huge bounds "
                                "and remains an unsolved mathematical problem."
                            )
                        },
                    )
                if self._closed.wait(timeout=2.0):
                    raise RuntimeError("ddg aborted")
                return _FakeResp(200, {})

            def close(self) -> None:
                self.is_closed = True
                self._closed.set()

        with patch("hal9000.websearch.httpx.Client", _PoolClient):
            t0 = time.perf_counter()
            r1 = search_web_context(
                "Search online to see if the Riemann hypothesis has been solved.",
                timeout_s=3.5,
            )
            dt1 = time.perf_counter() - t0
            clients_after_q1 = len(created_clients)

            # Second distinct evergreen query reuses the pooled healthy Wikipedia client (only 1 new DDG client!)
            r2 = search_web_context(
                "Search online to see if Goldbach conjecture has been solved.",
                timeout_s=3.5,
            )
            clients_after_q2 = len(created_clients)

        self.assertIn("Riemann hypothesis", r1)
        self.assertIn("Goldbach", r2)
        self.assertLess(dt1, 0.3)
        self.assertEqual(clients_after_q1, 2)
        self.assertEqual(clients_after_q2, 3)
        clear_search_cache()



    def test_streaming_browser_mic_resamples_live_and_ignores_stale_capture_frames(self) -> None:
        """Live binary browser mic streaming resamples 48->16 kHz incrementally and ignores stale cids."""
        captured: list[tuple[np.ndarray, int]] = []

        class _CaptureSTT:
            def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
                captured.append((audio.copy(), sample_rate))
                return "Open the pod bay doors"

        runtime = HalRuntime(
            HalConfig(model="mock", steps=6),
            stt=_CaptureSTT(),
            tts=_FakeTTS(),
            brain_factory=lambda rt: QwenBrain(model="mock", enable_web_search=False),
        )
        router = create_hal_router(runtime)
        ws_route = next(r for r in router.routes if getattr(r, "path", "") == "/ws")

        # 0.6s @ 48 kHz sine wave streamed in 2048-sample int16 binary frames
        t = np.linspace(0.0, 0.6, 28800, endpoint=False, dtype=np.float32)
        sig = (0.25 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)
        pcm16 = np.round(sig * 32767.0).astype("<i2")

        incoming: asyncio.Queue[dict] = asyncio.Queue()
        # Rapid double audio_stream_start (cid=1 superseded by cid=2) + stale cid=1 frame
        incoming.put_nowait(
            {"type": "websocket.receive", "text": '{"type":"audio_stream_start","capture_id":1,"sample_rate":48000,"format":"i16"}'}
        )
        incoming.put_nowait(
            {"type": "websocket.receive", "bytes": struct.pack("<I", 1) + pcm16[:2048].tobytes()}
        )
        incoming.put_nowait(
            {"type": "websocket.receive", "text": '{"type":"audio_stream_start","capture_id":2,"sample_rate":48000,"format":"i16"}'}
        )
        # Stale frame for cid=1 must be silently ignored
        incoming.put_nowait(
            {"type": "websocket.receive", "bytes": struct.pack("<I", 1) + np.full(2048, 30000, dtype="<i2").tobytes()}
        )
        for off in range(0, pcm16.size, 2048):
            chunk_bytes = struct.pack("<I", 2) + pcm16[off : off + 2048].tobytes()
            incoming.put_nowait({"type": "websocket.receive", "bytes": chunk_bytes})
        incoming.put_nowait(
            {"type": "websocket.receive", "text": '{"type":"audio_stream_end","turn_id":42,"capture_id":2}'}
        )

        sent_json: list[dict] = []

        class _BinaryFakeWS:
            def __init__(self) -> None:
                self.headers = {"origin": "http://127.0.0.1:8765"}

            async def accept(self) -> None:
                pass

            async def close(self, code: int = 1000) -> None:
                pass

            async def receive(self) -> dict:
                return await incoming.get()

            async def send_json(self, payload: dict) -> None:
                sent_json.append(payload)
                if payload.get("type") == "audio_end" and payload.get("turn_id") == 42:
                    incoming.put_nowait({"type": "websocket.disconnect"})

            async def send_bytes(self, data: bytes) -> None:
                pass

        async def run_ws() -> None:
            runtime.set_loop(asyncio.get_running_loop())
            await ws_route.endpoint(_BinaryFakeWS())

        asyncio.run(run_ws())

        self.assertEqual(len(captured), 1)
        audio_out, sr_out = captured[0]
        self.assertEqual(sr_out, 16000)
        # 0.6s @ 16 kHz = 9600 samples; must NOT contain the stale cid=1 2048-sample spike!
        self.assertAlmostEqual(audio_out.size, 9600, delta=16)
        self.assertLess(float(np.max(np.abs(audio_out))), 0.4)
        self.assertTrue(any(m.get("type") == "audio_end" and m.get("turn_id") == 42 for m in sent_json))
        self.assertIsNone(runtime.mic_owner)

    def test_audio_stream_end_without_active_stream_returns_idle_ack(self) -> None:
        """audio_stream_end after cancel or mismatch must emit state(idle) so UI never sticks."""
        runtime = HalRuntime(
            HalConfig(model="mock", steps=6),
            stt=_FakeSTT(),
            tts=_FakeTTS(),
            brain_factory=lambda rt: QwenBrain(model="mock", enable_web_search=False),
        )
        router = create_hal_router(runtime)
        ws_route = next(r for r in router.routes if getattr(r, "path", "") == "/ws")

        incoming: asyncio.Queue[dict] = asyncio.Queue()
        incoming.put_nowait(
            {"type": "websocket.receive", "text": '{"type":"audio_stream_start","capture_id":5,"sample_rate":48000,"format":"i16"}'}
        )
        incoming.put_nowait(
            {"type": "websocket.receive", "text": '{"type":"interrupt","silent":true}'}
        )
        incoming.put_nowait(
            {"type": "websocket.receive", "text": '{"type":"audio_stream_end","turn_id":43,"capture_id":5}'}
        )

        sent_json: list[dict] = []

        class _BinaryFakeWS:
            def __init__(self) -> None:
                self.headers = {"origin": "http://127.0.0.1:8765"}

            async def accept(self) -> None:
                pass

            async def close(self, code: int = 1000) -> None:
                pass

            async def receive(self) -> dict:
                return await incoming.get()

            async def send_json(self, payload: dict) -> None:
                sent_json.append(payload)
                if payload.get("type") == "state" and payload.get("turn_id") == 43:
                    incoming.put_nowait({"type": "websocket.disconnect"})

            async def send_bytes(self, data: bytes) -> None:
                pass

        async def run_ws() -> None:
            runtime.set_loop(asyncio.get_running_loop())
            await ws_route.endpoint(_BinaryFakeWS())

        asyncio.run(run_ws())
        idles = [
            m
            for m in sent_json
            if m.get("type") == "state" and m.get("turn_id") == 43 and m.get("state") == "idle"
        ]
        self.assertEqual(len(idles), 1)

    def test_healthy_pipeline_completion_does_not_evict_http_pool_on_slow_producer_exit(self) -> None:
        """Consumer draining ('done', None) must join healthy producer briefly without calling cancel_inflight."""
        import time

        cancel_calls = 0

        class _SlowUnwindBrain:
            model = "mock"
            enable_web_search = False

            def reset(self) -> None:
                pass

            def set_model(self, m: str) -> None:
                self.model = m

            def cancel_inflight(self, *, only_stopped: bool = False) -> None:
                nonlocal cancel_calls
                cancel_calls += 1

            def stream_reply_sentences(self, user_text: str, stop_check=None):
                try:
                    yield "All systems nominal, Charles."
                    return "All systems nominal, Charles."
                finally:
                    # Simulate microsecond-scale unwind after yielding final sentence
                    time.sleep(0.01)

        pipeline = HalPipeline(brain=_SlowUnwindBrain(), tts=_FakeTTS(), stt=_FakeSTT())
        events: list[HalEvent] = []
        cancel_ev = threading.Event()
        res = pipeline.run_turn(
            turn_id=55,
            cancel_event=cancel_ev,
            on_event=lambda ev: events.append(ev) or True,
            user_text="Status?",
        )
        self.assertFalse(res.failed)
        self.assertFalse(res.cancelled)
        self.assertFalse(cancel_ev.is_set())
        self.assertEqual(cancel_calls, 0)


if __name__ == "__main__":
    unittest.main()
