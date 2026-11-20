"""Transport-agnostic voice turn pipeline for embedding IOGHAL into host apps."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable

import numpy as np

from .interfaces import BrainProvider, HalEvent, STTProvider, TTSProvider, TurnResult

logger = logging.getLogger(__name__)

EventSink = Callable[[HalEvent], bool | None]


class HalPipeline:
    """Coordinates STT -> Brain -> streaming TTS for a single turn with cancel support.

    Host applications can instantiate ``HalPipeline`` directly with custom
    ``STTProvider``, ``BrainProvider``, and ``TTSProvider`` implementations
    without depending on FastAPI, WebSockets, or PyQt6.
    """

    def __init__(
        self,
        brain: BrainProvider,
        tts: TTSProvider,
        stt: STTProvider | None = None,
        gpu_lock: threading.Lock | None = None,
    ) -> None:
        self.brain = brain
        self.tts = tts
        self.stt = stt
        self.gpu_lock = gpu_lock or threading.Lock()

    def run_turn(
        self,
        turn_id: int,
        cancel_event: threading.Event,
        on_event: EventSink,
        *,
        audio: np.ndarray | None = None,
        sample_rate: int = 16000,
        user_text: str | None = None,
        steps: int | None = None,
    ) -> TurnResult:
        """Execute one complete turn and emit typed ``HalEvent`` records."""
        result = TurnResult(turn_id=turn_id)
        completed_sentences: list[str] = []
        audio_started = False
        terminal_state_sent = False

        def emit(kind: str, payload: dict | None = None, pcm: np.ndarray | None = None) -> bool:
            ev = HalEvent(kind=kind, turn_id=turn_id, payload=payload or {}, pcm=pcm)  # type: ignore[arg-type]
            res = on_event(ev)
            return res is not False

        try:
            if cancel_event.is_set():
                result.cancelled = True
                return result

            if user_text is None:
                if audio is None or self.stt is None:
                    raise RuntimeError("Audio input provided without an initialized STT provider.")
                emit(
                    "state",
                    {"state": "thinking", "detail": "TRANSCRIBING SPEECH..."},
                )
                t0 = time.perf_counter()
                with self.gpu_lock:
                    if cancel_event.is_set():
                        result.cancelled = True
                        return result
                    user_text = self.stt.transcribe(audio, sample_rate)
                result.stt_seconds = time.perf_counter() - t0
                logger.info("Turn %d STT (%.2fs): %r", turn_id, result.stt_seconds, user_text)

                if cancel_event.is_set():
                    result.cancelled = True
                    return result

                if not user_text:
                    terminal_state_sent = True
                    emit(
                        "state",
                        {
                            "state": "idle",
                            "detail": "NO SPEECH DETECTED — HOLD OR TAP SPACE TO TALK",
                        },
                    )
                    return result

            result.user_text = user_text
            emit("transcript", {"speaker": "YOU", "text": user_text})
            emit(
                "state",
                {"state": "thinking", "detail": f"PROCESSING ({self.brain.model})..."},
            )

            t_start = time.perf_counter()
            emitted_samples = 0
            gen_elapsed = 0.0

            for sentence in self.brain.stream_reply_sentences(
                user_text, stop_check=cancel_event.is_set
            ):
                if cancel_event.is_set():
                    result.cancelled = True
                    break
                logger.info("Turn %d HAL sentence: %r", turn_id, sentence)

                sentence_has_audio = False
                sentence_completed = True
                partial_text = " ".join([*completed_sentences, sentence])

                with self.gpu_lock:
                    if cancel_event.is_set():
                        result.cancelled = True
                        sentence_completed = False
                        break
                    t_gen_s = time.perf_counter()
                    tts_kwargs: dict = {
                        "stop_event": cancel_event,
                        "inference_timesteps": steps,
                    }
                    if getattr(self.brain, "pod_bay_reply", None) == sentence and hasattr(
                        self.tts, "_phrase_cache"
                    ):
                        tts_kwargs["cache_phrase"] = True
                    for chunk in self.tts.stream_speech(sentence, **tts_kwargs):
                        if cancel_event.is_set():
                            result.cancelled = True
                            sentence_completed = False
                            break
                        arr = np.asarray(chunk, dtype="<f4").reshape(-1)
                        if arr.size == 0:
                            continue
                        if not audio_started:
                            audio_started = True
                            if not emit("audio_start", {"sample_rate": self.tts.sample_rate}):
                                cancel_event.set()
                                result.cancelled = True
                                sentence_completed = False
                                break
                        if not sentence_has_audio:
                            sentence_has_audio = True
                            emit("hal_partial", {"text": partial_text})
                        if result.first_chunk_seconds is None:
                            result.first_chunk_seconds = time.perf_counter() - t_start
                            emit("state", {"state": "speaking", "detail": "HAL 9000 SPEAKING..."})
                            emit(
                                "metrics",
                                {
                                    "stt_s": round(result.stt_seconds, 2),
                                    "first_chunk_s": round(result.first_chunk_seconds, 2),
                                },
                            )
                        emitted_samples += arr.size
                        if not emit("audio_chunk", {}, pcm=arr):
                            cancel_event.set()
                            result.cancelled = True
                            sentence_completed = False
                            break
                    gen_elapsed += time.perf_counter() - t_gen_s

                if sentence_has_audio and sentence_completed and not cancel_event.is_set():
                    completed_sentences.append(sentence)
                elif cancel_event.is_set():
                    break

            result.audio_seconds = emitted_samples / float(max(self.tts.sample_rate, 1))
            result.generation_seconds = gen_elapsed
            result.reply_text = " ".join(completed_sentences)

            if audio_started and result.audio_seconds > 0 and gen_elapsed > 0:
                rtf = gen_elapsed / result.audio_seconds
                emit(
                    "metrics",
                    {
                        "stt_s": round(result.stt_seconds, 2),
                        "first_chunk_s": round(result.first_chunk_seconds or 0.0, 2),
                        "rtf": round(rtf, 2),
                    },
                )
        except Exception as exc:
            logger.exception("Turn %d failed", turn_id)
            result.failed = True
            result.error = str(exc)
            terminal_state_sent = True
            emit("error", {"message": str(exc)})
            emit("state", {"state": "idle", "detail": f"TURN ERROR: {exc}"})
        finally:
            if cancel_event.is_set():
                result.cancelled = True
            if audio_started:
                emit(
                    "audio_end",
                    {
                        "text": " ".join(completed_sentences),
                        "cancelled": result.cancelled,
                        "failed": result.failed,
                    },
                )
            elif not terminal_state_sent and not result.cancelled:
                emit(
                    "state",
                    {
                        "state": "idle",
                        "detail": "ALL CIRCUITS OPERATIONAL — HOLD OR TAP SPACE TO TALK",
                    },
                )

        return result
