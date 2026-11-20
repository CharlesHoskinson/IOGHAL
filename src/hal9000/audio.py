"""Audio capture (microphone) and streaming playback via PipeWire/PortAudio."""

from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Callable

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)

MAX_RECORDING_SECONDS = 29.5


class _CaptureBuffer:
    """Per-stream capture state so an in-flight callback never mutates a newer session."""

    def __init__(self, max_samples: int) -> None:
        self.max_samples = max_samples
        self.frames: list[np.ndarray] = []
        self.captured_samples: int = 0
        self.hit_limit: bool = False
        self.active: bool = True


class MicRecorder:
    """Push-to-talk 16 kHz mono microphone recorder with transactional lifecycle."""

    def __init__(
        self,
        sample_rate: int = 16000,
        device: int | str | None = None,
        on_level: Callable[[float], None] | None = None,
        max_duration_s: float = MAX_RECORDING_SECONDS,
        on_max_duration: Callable[[], None] | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.device = device
        self.on_level = on_level
        self.max_duration_s = max_duration_s
        self.on_max_duration = on_max_duration
        self._max_samples = int(sample_rate * max_duration_s)
        self._stream: sd.InputStream | None = None
        self._buf: _CaptureBuffer | None = None
        self._lock = threading.Lock()
        self._recording = False
        self._hit_limit = False

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def hit_duration_limit(self) -> bool:
        return self._hit_limit

    def start(self) -> None:
        with self._lock:
            if self._recording:
                return
            buf = _CaptureBuffer(self._max_samples)
            self._hit_limit = False
            stream: sd.InputStream | None = None
            try:
                stream = sd.InputStream(
                    samplerate=self.sample_rate,
                    channels=1,
                    dtype="float32",
                    device=self.device,
                    blocksize=512,
                    callback=lambda indata, frames, time_info, status: self._callback(
                        buf, indata, frames, time_info, status
                    ),
                )
                self._buf = buf
                self._stream = stream
                self._recording = True
                stream.start()
            except Exception:
                buf.active = False
                self._recording = False
                self._stream = None
                self._buf = None
                if stream is not None:
                    try:
                        stream.close()
                    except sd.PortAudioError:
                        pass
                raise

    def _callback(
        self,
        buf: _CaptureBuffer,
        indata: np.ndarray,
        frames: int,
        time_info,
        status,
    ) -> None:
        if status:
            logger.debug("InputStream status: %s", status)
        if not buf.active:
            return
        chunk = indata[:, 0].copy()
        remaining = buf.max_samples - buf.captured_samples
        if remaining <= 0:
            return
        if chunk.size > remaining:
            chunk = chunk[:remaining]
        buf.frames.append(chunk)
        buf.captured_samples += chunk.size

        if self.on_level is not None:
            rms = float(np.sqrt(np.mean(chunk * chunk)))
            try:
                self.on_level(min(1.0, rms * 6.0))
            except Exception:
                logger.debug("on_level callback failed", exc_info=True)

        if buf.captured_samples >= buf.max_samples and not buf.hit_limit:
            buf.hit_limit = True
            self._hit_limit = True
            if self.on_max_duration is not None:
                try:
                    self.on_max_duration()
                except Exception:
                    logger.debug("on_max_duration callback failed", exc_info=True)

    def stop(self) -> np.ndarray:
        with self._lock:
            if not self._recording and self._stream is None:
                return np.zeros(0, dtype=np.float32)
            self._recording = False
            stream = self._stream
            buf = self._buf
            self._stream = None
            self._buf = None

        # Stop & drain stream BEFORE detaching buf.frames so an in-flight callback cannot lose a block
        if stream is not None:
            try:
                stream.stop()
            except sd.PortAudioError as exc:
                logger.debug("InputStream stop error: %s", exc)
            finally:
                try:
                    stream.close()
                except sd.PortAudioError as exc:
                    logger.debug("InputStream close error: %s", exc)

        if buf is not None:
            buf.active = False
            frames = buf.frames
            buf.frames = []
        else:
            frames = []

        if self.on_level is not None:
            try:
                self.on_level(0.0)
            except Exception:
                logger.debug("on_level reset failed", exc_info=True)

        if not frames:
            return np.zeros(0, dtype=np.float32)

        audio = np.concatenate(frames).astype(np.float32, copy=False)
        audio = audio - float(np.mean(audio))
        # Trim initial ~20ms only if an obvious hardware stream-open pop is present
        trim = int(self.sample_rate * 0.02)
        if audio.size > trim + int(self.sample_rate * 0.25):
            head_peak = float(np.max(np.abs(audio[:trim])))
            next_peak = float(np.max(np.abs(audio[trim : trim * 4]))) + 1e-6
            if head_peak > 0.35 and head_peak > next_peak * 6.0:
                audio = audio[trim:]
                audio = audio - float(np.mean(audio))
        return audio


class StreamingAudioPlayer:
    """Low-latency streaming audio player for 48 kHz mono float32 TTS chunks."""

    def __init__(
        self,
        sample_rate: int = 48000,
        device: int | str | None = None,
        on_level: Callable[[float], None] | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.device = device
        self.on_level = on_level
        self._lock = threading.Lock()
        self._queue: queue.Queue[np.ndarray | None] | None = None
        self._stop_event: threading.Event | None = None
        self._thread: threading.Thread | None = None
        self._session_id = 0
        self._playing = threading.Event()
        self._last_error: Exception | None = None

    @property
    def is_playing(self) -> bool:
        return self._playing.is_set()

    @property
    def last_error(self) -> Exception | None:
        return self._last_error

    def start_session(self) -> None:
        """Prepare a fresh isolated player session (queue + stop_event + thread)."""
        self.stop()
        with self._lock:
            self._session_id += 1
            sid = self._session_id
            q: queue.Queue[np.ndarray | None] = queue.Queue()
            stop_ev = threading.Event()
            self._queue = q
            self._stop_event = stop_ev
            self._last_error = None
            self._playing.set()
            t = threading.Thread(target=self._run, args=(sid, q, stop_ev), daemon=True)
            self._thread = t
            t.start()

    def enqueue(self, chunk: np.ndarray) -> None:
        with self._lock:
            q = self._queue
            stop_ev = self._stop_event
        if q is None or stop_ev is None or stop_ev.is_set():
            return
        arr = np.asarray(chunk, dtype=np.float32).reshape(-1)
        if arr.size > 0:
            q.put(arr)

    def finish(self, timeout: float = 30.0) -> None:
        """Signal no more chunks will arrive and wait for playback to drain."""
        with self._lock:
            q = self._queue
            t = self._thread
        if q is not None:
            q.put(None)
        if t is not None:
            t.join(timeout=timeout)

    def stop(self) -> None:
        """Immediately interrupt playback and wait cleanly for the output thread."""
        with self._lock:
            q = self._queue
            stop_ev = self._stop_event
            t = self._thread
            self._session_id += 1
        if stop_ev is not None:
            stop_ev.set()
        if q is not None:
            q.put(None)
        if t is not None and t.is_alive():
            t.join(timeout=2.0)
            if not t.is_alive():
                with self._lock:
                    if self._thread is t:
                        self._thread = None
        else:
            with self._lock:
                if self._thread is t:
                    self._thread = None
        self._playing.clear()
        if self.on_level is not None:
            try:
                self.on_level(0.0)
            except Exception:
                logger.debug("on_level reset failed", exc_info=True)

    def _run(
        self,
        sid: int,
        q: queue.Queue[np.ndarray | None],
        stop_ev: threading.Event,
    ) -> None:
        blocksize = 1024
        done_event = threading.Event()
        pending: list[np.ndarray] = []
        eof = False

        def pull_samples_nowait(n: int) -> tuple[np.ndarray, bool]:
            nonlocal eof
            out = np.zeros(n, dtype=np.float32)
            filled = 0
            while filled < n:
                if stop_ev.is_set():
                    return out, True
                if not pending:
                    if eof:
                        return out, True
                    try:
                        item = q.get_nowait()
                    except queue.Empty:
                        return out, False
                    if item is None:
                        eof = True
                        if filled == 0:
                            return out, True
                        return out, False
                    pending.append(item)
                head = pending[0]
                take = min(n - filled, head.size)
                out[filled : filled + take] = head[:take]
                filled += take
                if take == head.size:
                    pending.pop(0)
                else:
                    pending[0] = head[take:]
            return out, False

        def callback(outdata: np.ndarray, frames: int, time_info, status) -> None:
            samples, finished = pull_samples_nowait(frames)
            outdata[:, 0] = samples
            if self.on_level is not None:
                rms = float(np.sqrt(np.mean(samples * samples)))
                try:
                    self.on_level(min(1.0, rms * 4.5))
                except Exception:
                    logger.debug("playback on_level callback failed", exc_info=True)
            if finished:
                done_event.set()
                raise sd.CallbackStop

        try:
            with sd.OutputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                device=self.device,
                blocksize=blocksize,
                callback=callback,
            ):
                while not done_event.is_set() and not stop_ev.is_set():
                    done_event.wait(timeout=0.05)
        except Exception as exc:
            logger.exception("Audio playback failed")
            with self._lock:
                if self._session_id == sid:
                    self._last_error = exc
        finally:
            with self._lock:
                is_current = self._session_id == sid
            if is_current:
                self._playing.clear()
                if self.on_level is not None:
                    try:
                        self.on_level(0.0)
                    except Exception:
                        logger.debug("playback on_level reset failed", exc_info=True)
