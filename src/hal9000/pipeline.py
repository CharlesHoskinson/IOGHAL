"""Transport-agnostic concurrent voice turn pipeline (STT -> Search -> LLM || TTS)."""

from __future__ import annotations

import inspect
import logging
import queue
import threading
import time
from collections.abc import Callable

import numpy as np

from .brain import is_daisy_song_request, is_pod_bay_open_request
from .daisy import DAISY_PHRASE_INDEX, stream_daisy_phrase
from .interfaces import BrainProvider, HalEvent, STTProvider, TTSProvider, TurnResult
from .websearch import search_web_context, should_search_web

logger = logging.getLogger(__name__)

EventSink = Callable[[HalEvent], bool | None]
_SENTENCE_QUEUE_MAX = 4
_QUEUE_POLL_S = 0.01
_LE_FLOAT32 = np.dtype("<f4")
_STT_SAMPLE_RATE = 16000


def _resample_for_stt(audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int]:
    """Fast C++ SIMD resampling to 16 kHz outside ``gpu_lock`` (falls back cleanly if soxr absent)."""
    if sample_rate == _STT_SAMPLE_RATE:
        return audio, sample_rate
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim > 1:
        arr = arr.mean(axis=-1).astype(np.float32, copy=False)
    else:
        arr = arr.reshape(-1)
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        return arr, sample_rate
    try:
        import soxr

        resampled = soxr.resample(arr, sample_rate, _STT_SAMPLE_RATE).astype(
            np.float32, copy=False
        )
        return resampled, _STT_SAMPLE_RATE
    except (ImportError, RuntimeError, ValueError, OSError):
        return audio, sample_rate


def _brain_accepts_param(brain: BrainProvider, name: str) -> bool:
    try:
        sig = inspect.signature(brain.stream_reply_sentences)
        return name in sig.parameters
    except (TypeError, ValueError):
        return False


def _brain_accepts_extra_context(brain: BrainProvider) -> bool:
    return _brain_accepts_param(brain, "extra_context")


def _cancel_brain_inflight(brain: BrainProvider, *, only_stopped: bool = True) -> None:
    cancel_fn = getattr(brain, "cancel_inflight", None)
    if callable(cancel_fn):
        try:
            if only_stopped:
                try:
                    if "only_stopped" in inspect.signature(cancel_fn).parameters:
                        cancel_fn(only_stopped=True)
                        return
                except (TypeError, ValueError):
                    pass
            cancel_fn()
        except Exception:
            logger.debug("Brain cancel_inflight hook failed", exc_info=True)


class HalPipeline:
    """Coordinates STT -> optional live web search -> concurrent LLM || streaming TTS.

    LLM sentence generation runs on a producer thread feeding a bounded queue while the
    consumer thread synthesizes audio under ``gpu_lock``, eliminating inter-sentence LLM
    stalls while maintaining Quint-verified cancellation and strict audio framing.
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
        self._brain_accepts_extra_context = _brain_accepts_extra_context(brain)
        self._brain_can_defer_commit = (
            _brain_accepts_param(brain, "auto_commit")
            and callable(getattr(brain, "get_history_gen", None))
            and callable(getattr(brain, "commit_turn", None))
        )

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
        audio_ended = False
        terminal_state_sent = False
        producer_thread: threading.Thread | None = None

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
                stt_audio, stt_sr = _resample_for_stt(audio, sample_rate)
                with self.gpu_lock:
                    if cancel_event.is_set():
                        result.cancelled = True
                        return result
                    user_text = self.stt.transcribe(stt_audio, stt_sr)
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

            # Optional non-fatal live web search stage for current/encyclopedic queries
            is_canned_easter_egg = is_pod_bay_open_request(user_text) or is_daisy_song_request(
                user_text
            )
            web_context = ""
            web_search_elapsed = 0.0
            if (
                not is_canned_easter_egg
                and getattr(self.brain, "enable_web_search", False)
                and self._brain_accepts_extra_context
                and should_search_web(user_text)
            ):
                emit("state", {"state": "thinking", "detail": "SEARCHING ARCHIVES..."})
                t_web = time.perf_counter()
                try:
                    web_context = search_web_context(
                        user_text, cancel_event=cancel_event, timeout_s=3.5
                    )
                except (OSError, RuntimeError, ValueError, TimeoutError) as exc:
                    logger.debug("Turn %d web search fallback: %s", turn_id, exc)
                    web_context = ""
                web_search_elapsed = time.perf_counter() - t_web
                if cancel_event.is_set():
                    result.cancelled = True
                    return result
                if web_context:
                    logger.info(
                        "Turn %d web context fetched (%d chars in %.2fs)",
                        turn_id,
                        len(web_context),
                        web_search_elapsed,
                    )

            model_label = (
                self.brain.get_model()
                if hasattr(self.brain, "get_model")
                else self.brain.model
            )
            emit(
                "state",
                {"state": "thinking", "detail": f"PROCESSING ({model_label})..."},
            )

            t_start = time.perf_counter()
            t_first_sentence: float | None = None
            llm_first_sentence_s: float | None = None
            tts_first_chunk_s: float | None = None
            emitted_samples = 0
            gen_elapsed = 0.0

            # Quint Bug 4a fix: defer history commit until TTS actually speaks completed_sentences
            defer_commit = self._brain_can_defer_commit
            turn_history_gen: int | None = (
                self.brain.get_history_gen() if defer_commit else None  # type: ignore[attr-defined]
            )

            # Producer-consumer bounded queue overlapping LLM generation with VoxCPM2 TTS
            sentence_q: queue.Queue[tuple[str, object]] = queue.Queue(maxsize=_SENTENCE_QUEUE_MAX)

            def _put_with_cancel(item: tuple[str, object]) -> bool:
                while not cancel_event.is_set():
                    try:
                        sentence_q.put(item, timeout=_QUEUE_POLL_S)
                        return True
                    except queue.Full:
                        continue
                return False

            def _llm_producer() -> None:
                try:
                    kwargs: dict = {"stop_check": cancel_event.is_set}
                    if web_context and self._brain_accepts_extra_context:
                        kwargs["extra_context"] = web_context
                    if defer_commit:
                        kwargs["auto_commit"] = False
                    for sentence in self.brain.stream_reply_sentences(user_text, **kwargs):
                        if cancel_event.is_set():
                            return
                        if not _put_with_cancel(("sentence", sentence)):
                            return
                    _put_with_cancel(("done", None))
                except Exception as exc:  # noqa: BLE001
                    if not cancel_event.is_set():
                        _put_with_cancel(("error", exc))

            producer_thread = threading.Thread(
                target=_llm_producer, name=f"hal-llm-{turn_id}", daemon=True
            )
            producer_thread.start()
            producer_done = False

            while True:
                if cancel_event.is_set():
                    result.cancelled = True
                    break
                try:
                    kind, payload = sentence_q.get(timeout=_QUEUE_POLL_S)
                except queue.Empty:
                    if not producer_thread.is_alive():
                        break
                    continue

                if kind == "done":
                    producer_done = True
                    break
                if kind == "error":
                    assert isinstance(payload, Exception)
                    raise payload

                sentence = str(payload)
                if cancel_event.is_set():
                    result.cancelled = True
                    break
                if t_first_sentence is None:
                    t_first_sentence = time.perf_counter()
                    llm_first_sentence_s = t_first_sentence - t_start
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
                    daisy_idx = DAISY_PHRASE_INDEX.get(sentence)
                    if daisy_idx is not None and is_daisy_song_request(user_text):
                        chunk_stream = stream_daisy_phrase(
                            self.tts,
                            daisy_idx,
                            stop_event=cancel_event,
                            inference_timesteps=steps,
                        )
                    else:
                        chunk_stream = self.tts.stream_speech(sentence, **tts_kwargs)
                    for chunk in chunk_stream:
                        if cancel_event.is_set():
                            result.cancelled = True
                            sentence_completed = False
                            break
                        if (
                            isinstance(chunk, np.ndarray)
                            and chunk.dtype == _LE_FLOAT32
                            and chunk.ndim == 1
                            and chunk.flags.c_contiguous
                        ):
                            arr = chunk
                        else:
                            arr = np.ascontiguousarray(chunk, dtype=_LE_FLOAT32).reshape(-1)
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
                            t_first_chunk = time.perf_counter()
                            result.first_chunk_seconds = t_first_chunk - t_start
                            if t_first_sentence is not None:
                                tts_first_chunk_s = t_first_chunk - t_first_sentence
                            emit("state", {"state": "speaking", "detail": "HAL 9000 SPEAKING..."})
                            metrics_payload: dict[str, float] = {
                                "stt_s": round(result.stt_seconds, 2),
                                "first_chunk_s": round(result.first_chunk_seconds, 2),
                                "web_search_s": round(web_search_elapsed, 3),
                            }
                            if llm_first_sentence_s is not None:
                                metrics_payload["llm_first_sentence_s"] = round(
                                    llm_first_sentence_s, 3
                                )
                            if tts_first_chunk_s is not None:
                                metrics_payload["tts_first_chunk_s"] = round(
                                    tts_first_chunk_s, 3
                                )
                            emit("metrics", metrics_payload)
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

            # Commit spoken history only after TTS finishes cleanly without cancel or failure
            if (
                defer_commit
                and turn_history_gen is not None
                and result.reply_text
                and not result.cancelled
                and not result.failed
                and not cancel_event.is_set()
            ):
                self.brain.commit_turn(  # type: ignore[attr-defined]
                    turn_history_gen,
                    user_text,
                    result.reply_text,
                    stop_check=cancel_event.is_set,
                )

            if audio_started and result.audio_seconds > 0 and gen_elapsed > 0:
                rtf = gen_elapsed / result.audio_seconds
                final_metrics: dict[str, float] = {
                    "stt_s": round(result.stt_seconds, 2),
                    "first_chunk_s": round(result.first_chunk_seconds or 0.0, 2),
                    "rtf": round(rtf, 2),
                    "web_search_s": round(web_search_elapsed, 3),
                }
                if llm_first_sentence_s is not None:
                    final_metrics["llm_first_sentence_s"] = round(llm_first_sentence_s, 3)
                if tts_first_chunk_s is not None:
                    final_metrics["tts_first_chunk_s"] = round(tts_first_chunk_s, 3)
                emit("metrics", final_metrics)
        except Exception as exc:
            logger.exception("Turn %d failed", turn_id)
            cancel_event.set()
            _cancel_brain_inflight(self.brain)
            result.failed = True
            result.error = str(exc)
            terminal_state_sent = True
            # Quint-proven strict framing: emit audio_end(failed=True) FIRST so the open audio
            # stream closes cleanly before error and state(idle) arrive on the client.
            if audio_started and not audio_ended:
                audio_ended = True
                emit(
                    "audio_end",
                    {
                        "text": " ".join(completed_sentences),
                        "cancelled": False,
                        "failed": True,
                    },
                )
            emit("error", {"message": str(exc)})
            emit("state", {"state": "idle", "detail": f"TURN ERROR: {exc}"})
        finally:
            if cancel_event.is_set() and not result.failed:
                result.cancelled = True
            if producer_thread is not None and producer_thread.is_alive():
                if producer_done and not cancel_event.is_set() and not result.failed:
                    # Healthy producer already sent ("done", None); let it unwind without
                    # setting cancel_event or evicting the warm HTTP keep-alive pool.
                    producer_thread.join(timeout=0.05)
                if producer_thread.is_alive():
                    cancel_event.set()
                    # Quint Bug 2b fix: abort any open LLM HTTP stream immediately to release --parallel 1 slot
                    _cancel_brain_inflight(self.brain)
                    producer_thread.join(timeout=1.0)
            if audio_started and not audio_ended:
                audio_ended = True
                emit(
                    "audio_end",
                    {
                        "text": " ".join(completed_sentences),
                        "cancelled": result.cancelled,
                        "failed": result.failed,
                    },
                )
            elif not terminal_state_sent and not result.cancelled and not result.failed:
                emit(
                    "state",
                    {
                        "state": "idle",
                        "detail": "ALL CIRCUITS OPERATIONAL — HOLD OR TAP SPACE TO TALK",
                    },
                )

        return result
