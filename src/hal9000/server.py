"""FastAPI + WebSocket server and embeddable router for IOGHAL."""

from __future__ import annotations

import argparse
import asyncio
import base64
import binascii
import contextlib
import json
import logging
import os
import struct
import threading
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

import httpx
import numpy as np
import sounddevice as sd
import uvicorn
from fastapi import APIRouter, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .audio import MAX_RECORDING_SECONDS, MicRecorder
from .brain import (
    DEFAULT_OLLAMA_NUM_CTX,
    HAL_SYSTEM_PROMPT,
    LoreIndex,
    OllamaBrain,
    choose_default_model,
    is_glimmer_model,
    list_llama_server_models,
    list_ollama_models,
)
from .interfaces import BrainProvider, HalConfig, HalEvent, STTProvider, TTSProvider
from .pipeline import HalPipeline
from .stt import WhisperSTT
from .tts import (
    DEFAULT_PROJECT_DIR,
    MIN_DIFFUSION_STEPS,
    HalVoiceEngine,
    resolve_reference_wav,
)

logger = logging.getLogger(__name__)

WEB_DIST_DIR = DEFAULT_PROJECT_DIR / "web" / "dist"
MAX_TEXT_CHARS = 4000
MAX_AUDIO_B64_CHARS = 16 * 1024 * 1024
MIN_SAMPLE_RATE = 8000
MAX_SAMPLE_RATE = 96000
MAX_UPLOAD_DURATION_S = 30.0
MIN_AUDIO_DURATION_S = 0.2
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


def is_allowed_origin(
    origin: str | None,
    allowed_hosts: set[str],
    port: int,
    extra_origins: tuple[str, ...] = (),
) -> bool:
    """Validate browser Origin header before accepting a WebSocket controlling local audio."""
    if not origin:
        return False
    if origin in extra_origins:
        return True
    parsed = urlparse(origin)
    if parsed.scheme not in ("http", "https"):
        return False
    host = (parsed.hostname or "").lower()
    if host not in allowed_hosts:
        return False
    origin_port = parsed.port or (443 if parsed.scheme == "https" else 80)
    return origin_port in {port, 5173, 4173}


def _parse_positive_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if value > 0 else None


class HalRuntime:
    """Shared GPU runtime (STT + TTS + lore index) with per-connection conversation isolation.

    Host applications may inject custom ``stt``, ``tts``, or ``brain_factory`` implementations
    via constructor arguments to plug IOGHAL into a larger system.
    """

    def __init__(
        self,
        config: HalConfig | None = None,
        *,
        model: str | None = None,
        ref_wav: Path | None = None,
        ollama_url: str | None = None,
        llama_server_url: str | None = None,
        llm_backend: Literal["auto", "ollama", "llama_server"] | None = None,
        thinking_mode: Literal["auto", "off", "low"] | None = None,
        reasoning_budget: int | None = None,
        compile_model: bool | None = None,
        steps: int | None = None,
        host: str | None = None,
        port: int | None = None,
        stt: STTProvider | None = None,
        tts: TTSProvider | None = None,
        brain_factory: Callable[[HalRuntime], BrainProvider] | None = None,
    ) -> None:
        cfg = config or HalConfig()
        env_llama_url = os.environ.get("HAL_LLAMA_SERVER_URL") or None
        env_backend = os.environ.get("HAL_LLM_BACKEND") or None
        env_thinking = os.environ.get("HAL_THINKING_MODE") or None
        env_budget_raw = os.environ.get("HAL_REASONING_BUDGET")
        env_budget = int(env_budget_raw) if env_budget_raw and env_budget_raw.isdigit() else None

        resolved_llama_url = (
            llama_server_url
            if llama_server_url is not None
            else (cfg.llama_server_url if cfg.llama_server_url is not None else env_llama_url)
        )
        resolved_backend: Literal["auto", "ollama", "llama_server"] = (
            llm_backend
            if llm_backend is not None
            else (
                env_backend  # type: ignore[assignment]
                if cfg.llm_backend == "auto" and env_backend in ("auto", "ollama", "llama_server")
                else cfg.llm_backend
            )
        )
        resolved_thinking: Literal["auto", "off", "low"] = (
            thinking_mode
            if thinking_mode is not None
            else (
                env_thinking  # type: ignore[assignment]
                if cfg.thinking_mode == "auto" and env_thinking in ("auto", "off", "low")
                else cfg.thinking_mode
            )
        )
        resolved_budget = (
            reasoning_budget
            if reasoning_budget is not None
            else (env_budget if env_budget is not None else cfg.reasoning_budget)
        )

        self.config = HalConfig(
            model=model if model is not None else cfg.model,
            ollama_url=ollama_url if ollama_url is not None else cfg.ollama_url,
            llm_backend=resolved_backend,
            llama_server_url=resolved_llama_url,
            thinking_mode=resolved_thinking,
            reasoning_budget=max(0, int(resolved_budget)),
            ref_wav=resolve_reference_wav(ref_wav if ref_wav is not None else cfg.ref_wav),
            ref_text=cfg.ref_text,
            steps=steps if steps is not None else cfg.steps,
            cfg_value=cfg.cfg_value,
            compile_model=compile_model if compile_model is not None else cfg.compile_model,
            whisper_model_id=cfg.whisper_model_id,
            voxcpm_model_id=cfg.voxcpm_model_id,
            lore_dir=cfg.lore_dir,
            user_name=cfg.user_name,
            host=host if host is not None else cfg.host,
            port=port if port is not None else cfg.port,
            allowed_origins=cfg.allowed_origins,
        )
        if self.config.steps < MIN_DIFFUSION_STEPS:
            raise ValueError(
                f"steps must be >= {MIN_DIFFUSION_STEPS}, got {self.config.steps}"
            )

        self.ref_wav = self.config.ref_wav
        self.ollama_url = self.config.ollama_url
        self.llama_server_url = (
            self.config.llama_server_url.rstrip("/") if self.config.llama_server_url else None
        )
        self.compile_model = self.config.compile_model
        self.steps = self.config.steps
        self.host = self.config.host
        self.port = self.config.port
        self.is_loopback = self.host.lower() in LOOPBACK_HOSTS
        self.allowed_hosts = set(LOOPBACK_HOSTS) | {self.host.lower()}
        self._brain_factory = brain_factory

        # Discover llama-server and Ollama models once cleanly
        llama_discovered = (
            list_llama_server_models(self.llama_server_url)
            if self.llama_server_url and self.config.llm_backend != "ollama"
            else []
        )
        ollama_discovered = (
            list_ollama_models(self.ollama_url)
            if self.config.llm_backend != "llama_server"
            else []
        )
        self.llama_server_models: set[str] = set(llama_discovered)
        self.llama_server_available: bool = len(llama_discovered) > 0
        self.ollama_available: bool = len(ollama_discovered) > 0
        discovered = [
            *llama_discovered,
            *(m for m in ollama_discovered if m not in self.llama_server_models),
        ]
        self.default_model: str = self.config.model or (
            llama_discovered[0]
            if llama_discovered
            else choose_default_model(self.ollama_url, models=discovered)
        )
        self.models: list[str] = (
            list(discovered)
            if self.default_model in discovered
            else [self.default_model, *discovered]
        )
        self.lore_index = LoreIndex(self.config.lore_dir)

        self.stt: STTProvider | None = stt
        self.tts: TTSProvider | None = tts
        self.sample_rate: int = tts.sample_rate if tts is not None else 48000

        self.ready: bool = stt is not None and tts is not None
        self.error: str | None = None
        self.stage: str = (
            "ALL CIRCUITS OPERATIONAL — HOLD OR TAP SPACE TO TALK"
            if self.ready
            else "INITIALIZING NEURAL SUBSYSTEMS..."
        )
        self.gpu_lock = threading.Lock()
        self.mic_lock = threading.Lock()
        self.mic_owner: WebSocket | None = None
        self._shutdown_event = threading.Event()
        self._loader_thread: threading.Thread | None = None
        self._ollama_warmup_lock = threading.Lock()
        self._warmup_http_lock = threading.Lock()
        self._ollama_warmup_seq = 0
        self._active_warmup_clients: set[Any] = set()
        self._active_ollama_model: str | None = None

        # Per-connection single-writer callbacks (never call ws.send_* directly from broadcast)
        self._listeners: dict[WebSocket, Callable[[dict[str, Any]], None]] = {}
        self._active_closers: dict[WebSocket, Callable[[], None]] = {}
        self._loop: asyncio.AbstractEventLoop | None = None

    def _is_llama_server_model(self, model: str) -> bool:
        if self.config.llm_backend == "llama_server":
            return True
        if self.config.llm_backend == "ollama":
            return False
        return bool(self.llama_server_url) and model in self.llama_server_models

    def create_brain(self) -> BrainProvider:
        if self._brain_factory is not None:
            return self._brain_factory(self)
        return OllamaBrain(
            model=self.default_model,
            base_url=self.ollama_url,
            user_name=self.config.user_name,
            lore_index=self.lore_index,
            backend=self.config.llm_backend,
            llama_server_url=self.llama_server_url,
            llama_server_models=self.llama_server_models,
            thinking_mode=self.config.thinking_mode,
            reasoning_budget=self.config.reasoning_budget,
        )

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def _broadcast_from_thread(self, payload: dict[str, Any]) -> None:
        if self._loop is None or self._loop.is_closed():
            return
        self._loop.call_soon_threadsafe(self._dispatch_broadcast_on_loop, payload)

    def _dispatch_broadcast_on_loop(self, payload: dict[str, Any]) -> None:
        for send_fn in list(self._listeners.values()):
            try:
                send_fn(payload)
            except Exception:
                logger.debug("Listener broadcast failed", exc_info=True)

    def start_background_load(self) -> None:
        if self.ready:
            return
        if self._loader_thread is not None and self._loader_thread.is_alive():
            return
        self._loader_thread = threading.Thread(target=self._load_models, daemon=True)
        self._loader_thread.start()

    def shutdown(self) -> None:
        """Cancel in-flight work and notify all active connections to close cleanly."""
        self._shutdown_event.set()
        self.cancel_inflight_warmups()
        for closer in list(self._active_closers.values()):
            try:
                closer()
            except Exception:
                logger.debug("Connection shutdown hook failed", exc_info=True)
        if self._loader_thread is not None and self._loader_thread.is_alive():
            self._loader_thread.join(timeout=2.0)

    def cancel_inflight_warmups(self) -> None:
        """Abort any active warmup HTTP probes so a newer set_model or live turn is never stalled."""
        with self._ollama_warmup_lock:
            self._ollama_warmup_seq += 1
            clients = list(self._active_warmup_clients)
            self._active_warmup_clients.clear()
        for client in clients:
            with contextlib.suppress(httpx.HTTPError, OSError, RuntimeError):
                client.close()

    def _run_cancellable_post(
        self, url: str, payload: dict[str, Any], seq: int | None
    ) -> bool:
        client = httpx.Client(
            timeout=httpx.Timeout(connect=5.0, read=120.0, write=10.0, pool=5.0)
        )
        with self._ollama_warmup_lock:
            if self._shutdown_event.is_set() or (
                seq is not None and seq != self._ollama_warmup_seq
            ):
                client.close()
                return False
            self._active_warmup_clients.add(client)
        try:
            client.post(url, json=payload).raise_for_status()
            return True
        finally:
            with self._ollama_warmup_lock:
                self._active_warmup_clients.discard(client)
            with contextlib.suppress(httpx.HTTPError, OSError, RuntimeError):
                client.close()

    def _warmup_ollama_async(self, model: str | None = None, seq: int | None = None) -> None:
        if self._shutdown_event.is_set():
            return
        target_model = model or self.default_model
        use_llama_server = self._is_llama_server_model(target_model)

        # Quint Bug 1d fix: serialize all warmup/unload HTTP calls under _warmup_http_lock
        # and re-check seq atomically before opening each tracked client so stale warmups can never evict.
        with self._warmup_http_lock:
            with self._ollama_warmup_lock:
                if self._shutdown_event.is_set() or (
                    seq is not None and seq != self._ollama_warmup_seq
                ):
                    return
                prev_ollama = self._active_ollama_model

            try:
                if use_llama_server:
                    # Quint Bug 1b fix: unload any previously resident Ollama model before using llama-server
                    if prev_ollama and self.ollama_available:
                        try:
                            if self._run_cancellable_post(
                                f"{self.ollama_url.rstrip('/')}/api/chat",
                                {"model": prev_ollama, "messages": [], "keep_alive": 0},
                                seq,
                            ):
                                with self._ollama_warmup_lock:
                                    if self._active_ollama_model == prev_ollama:
                                        self._active_ollama_model = None
                                logger.info(
                                    "Unloaded Ollama model %s before activating llama-server %s.",
                                    prev_ollama,
                                    target_model,
                                )
                        except (httpx.HTTPError, OSError) as exc:
                            logger.debug("Ollama unload skipped for %s: %s", prev_ollama, exc)

                    if not self.llama_server_url:
                        return
                    sys_prompt = HAL_SYSTEM_PROMPT.format(user_name=self.config.user_name)
                    warm_payload: dict[str, Any] = {
                        "model": target_model,
                        "messages": [
                            {"role": "system", "content": sys_prompt},
                            {"role": "user", "content": "Ready."},
                        ],
                        "stream": False,
                        "cache_prompt": True,
                        "max_tokens": 1,
                        "temperature": 0.0,
                        "reasoning_budget_tokens": 0,
                        "chat_template_kwargs": (
                            {"enable_thinking": False, "reasoning_strength": "low"}
                            if is_glimmer_model(target_model)
                            else {"enable_thinking": False}
                        ),
                    }
                    if is_glimmer_model(target_model):
                        warm_payload["reasoning_format"] = "deepseek"
                    if self._run_cancellable_post(
                        f"{self.llama_server_url}/v1/chat/completions",
                        warm_payload,
                        seq,
                    ):
                        logger.info("Warmed llama-server KV prefix for %s.", target_model)
                    return

                if not self.ollama_available:
                    return
                if self._run_cancellable_post(
                    f"{self.ollama_url.rstrip('/')}/api/chat",
                    {
                        "model": target_model,
                        "messages": [],
                        "keep_alive": "30m",
                        "options": {"num_ctx": DEFAULT_OLLAMA_NUM_CTX},
                    },
                    seq,
                ):
                    with self._ollama_warmup_lock:
                        if seq is None or seq == self._ollama_warmup_seq:
                            self._active_ollama_model = target_model
                    logger.info("Preloaded Ollama model %s into memory.", target_model)
            except (httpx.HTTPError, OSError) as exc:
                logger.warning("LLM preload skipped for %s: %s", target_model, exc)

    def schedule_ollama_warmup(self, model: str | None = None) -> None:
        target_model = model or self.default_model
        use_llama_server = self._is_llama_server_model(target_model)
        if self._shutdown_event.is_set():
            return
        if use_llama_server and not self.llama_server_url:
            return
        if not use_llama_server and not self.ollama_available:
            return
        with self._ollama_warmup_lock:
            self._ollama_warmup_seq += 1
            seq = self._ollama_warmup_seq
            if not use_llama_server:
                self._active_ollama_model = target_model
            stale_clients = list(self._active_warmup_clients)
            self._active_warmup_clients.clear()
        for client in stale_clients:
            with contextlib.suppress(httpx.HTTPError, OSError, RuntimeError):
                client.close()
        threading.Thread(
            target=self._warmup_ollama_async,
            args=(target_model, seq),
            daemon=True,
        ).start()

    def _warmup_tts_async(self) -> None:
        if self._shutdown_event.is_set() or self.tts is None:
            return
        if not self.gpu_lock.acquire(blocking=False):
            return
        try:
            if isinstance(self.tts, HalVoiceEngine):
                self.tts.warmup_streaming(seed=42, max_chunks=1)
            else:
                _ = self.tts.synthesize("Yes.", seed=42)
            if not self._shutdown_event.is_set() and isinstance(self.stt, WhisperSTT):
                self.stt.warmup()
        except (RuntimeError, ValueError, OSError) as exc:
            logger.debug("Background GPU warmup skipped: %s", exc)
        finally:
            self.gpu_lock.release()

    def _load_models(self) -> None:
        try:
            t_boot = time.perf_counter()
            self.stage = "LOADING WHISPER LARGE-V3 + VOXCPM2..."
            self._broadcast_from_thread(self.status_payload(include_model=False))

            # Load WhisperSTT and HalVoiceEngine sequentially so CUDA graph captures & RNG state never race across threads.
            if self.stt is None:
                self.stt = WhisperSTT(model_id=self.config.whisper_model_id)

            if self.tts is None:
                self.tts = HalVoiceEngine(
                    model_id=self.config.voxcpm_model_id,
                    reference_wav=self.ref_wav,
                    reference_text=self.config.ref_text,
                    optimize=self.compile_model,
                    cfg_value=self.config.cfg_value,
                    inference_timesteps=self.steps,
                )
            self.sample_rate = self.tts.sample_rate

            if self._shutdown_event.is_set():
                return

            self.ready = True
            self.stage = "ALL CIRCUITS OPERATIONAL — HOLD OR TAP SPACE TO TALK"
            self._broadcast_from_thread(self.status_payload(include_model=False))
            logger.info("IOGHAL runtime operational in %.2fs.", time.perf_counter() - t_boot)

            # Warm CUDA streaming kernels and preload Ollama after weights are resident
            threading.Thread(target=self._warmup_tts_async, daemon=True).start()
            self.schedule_ollama_warmup(self.default_model)
        except Exception as exc:
            logger.exception("Failed to initialize IOGHAL runtime")
            self.error = str(exc)
            self.stage = f"INITIALIZATION ERROR: {exc}"
            self._broadcast_from_thread(self.status_payload(include_model=False))

    def status_payload(
        self,
        model: str | None = None,
        steps: int | None = None,
        include_model: bool = True,
    ) -> dict[str, Any]:
        state = "error" if self.error else ("idle" if self.ready else "loading")
        payload: dict[str, Any] = {
            "type": "system_status",
            "ready": self.ready,
            "error": self.error,
            "stage": self.stage,
            "state": state,
            "models": self.models,
            "ollama_available": self.ollama_available,
            "llama_server_available": self.llama_server_available,
            "local_mic_allowed": self.is_loopback,
            "sample_rate": self.sample_rate,
            "max_duration_s": MAX_RECORDING_SECONDS,
            "steps": steps if steps is not None else self.steps,
        }
        if include_model:
            payload["model"] = model or self.default_model
        return payload


def create_hal_router(runtime: HalRuntime, prefix: str = "") -> APIRouter:
    """Create a mountable FastAPI ``APIRouter`` exposing ``/api/health`` and ``/ws``.

    Host applications can mount this router at any prefix::

        app.include_router(create_hal_router(runtime, prefix="/hal"))
    """
    router = APIRouter(prefix=prefix)

    @router.get("/api/health")
    async def health() -> JSONResponse:
        return JSONResponse(runtime.status_payload())

    @router.websocket("/ws")
    async def websocket_endpoint(ws: WebSocket) -> None:
        origin = ws.headers.get("origin")
        if not is_allowed_origin(
            origin,
            runtime.allowed_hosts,
            runtime.port,
            extra_origins=runtime.config.allowed_origins,
        ):
            logger.warning("Rejected WebSocket from disallowed origin: %r", origin)
            await ws.close(code=1008)
            return

        await ws.accept()
        loop = asyncio.get_running_loop()

        brain = runtime.create_brain()
        conn_steps = runtime.steps
        ws_closed = threading.Event()
        state_lock = threading.Lock()
        turn_gen = 0
        active_cancel = threading.Event()
        active_capture_id: int | None = None
        last_level_ts = 0.0

        # Queue holds (kind, payload, gen_or_none) so retired audio is discarded before transmission
        out_queue: asyncio.Queue[tuple[str, Any, int | None] | None] = asyncio.Queue(maxsize=512)

        def current_gen() -> int:
            with state_lock:
                return turn_gen

        def force_close_on_loop() -> None:
            if not ws_closed.is_set():
                ws_closed.set()
            try:
                out_queue.put_nowait(None)
            except asyncio.QueueFull:
                pass
            if not loop.is_closed():
                asyncio.ensure_future(_safe_close_ws())

        async def _safe_close_ws() -> None:
            try:
                await ws.close(code=1011)
            except (RuntimeError, WebSocketDisconnect, OSError) as exc:
                logger.debug("WebSocket forced close ignored: %s", exc)

        async def writer_loop() -> None:
            try:
                while True:
                    item = await out_queue.get()
                    if item is None:
                        break
                    kind, payload, item_gen = item
                    if item_gen is not None and item_gen != current_gen():
                        continue
                    if kind == "json":
                        await ws.send_json(payload)
                    elif kind == "bytes":
                        await ws.send_bytes(payload)
            except (RuntimeError, WebSocketDisconnect, OSError):
                ws_closed.set()
            finally:
                ws_closed.set()

        writer_task = asyncio.create_task(writer_loop())

        def enqueue_on_loop(
            kind: str,
            payload: Any,
            item_gen: int | None = None,
            drop_if_full: bool = False,
        ) -> bool:
            if ws_closed.is_set() or loop.is_closed():
                return False
            try:
                out_queue.put_nowait((kind, payload, item_gen))
                return True
            except asyncio.QueueFull:
                if drop_if_full:
                    return False
                logger.warning("WebSocket outbound queue saturated; closing slow client")
                ws_closed.set()
                force_close_on_loop()
                return False

        def send_json_threadsafe(
            payload: dict[str, Any],
            item_gen: int | None = None,
            *,
            drop_if_full: bool = False,
        ) -> bool:
            if ws_closed.is_set() or loop.is_closed():
                return False
            loop.call_soon_threadsafe(
                enqueue_on_loop, "json", payload, item_gen, drop_if_full
            )
            return True

        def send_pcm_nonblocking(turn_id: int, pcm: np.ndarray, item_gen: int) -> bool:
            """Non-blocking PCM handoff so a slow client never stalls runtime.gpu_lock."""
            if ws_closed.is_set() or loop.is_closed():
                return False
            framed = struct.pack("<I", turn_id) + np.asarray(pcm, dtype="<f4").tobytes()
            loop.call_soon_threadsafe(enqueue_on_loop, "bytes", framed, item_gen)
            return not ws_closed.is_set()

        def begin_turn(client_tid: int | None) -> tuple[int, int, threading.Event]:
            """Retire any prior worker via internal ``turn_gen`` while echoing client's wire ``turn_id``."""
            nonlocal turn_gen, active_cancel
            with state_lock:
                active_cancel.set()
                turn_gen += 1
                active_cancel = threading.Event()
                wire_tid = client_tid if client_tid is not None else turn_gen
                cancel_ev = active_cancel
                gen_now = turn_gen
            runtime.cancel_inflight_warmups()
            cancel_fn = getattr(brain, "cancel_inflight", None)
            if callable(cancel_fn):
                cancel_fn()
            return wire_tid, gen_now, cancel_ev

        def cancel_active_turn() -> None:
            nonlocal turn_gen
            with state_lock:
                active_cancel.set()
                turn_gen += 1
            cancel_fn = getattr(brain, "cancel_inflight", None)
            if callable(cancel_fn):
                cancel_fn()

        def reject_turn(tid: int | None, message: str, detail: str | None = None) -> None:
            """Quint-verified rejection: always pair turn-scoped error with idle state."""
            cancel_active_turn()
            enqueue_on_loop("json", {"type": "error", "turn_id": tid, "message": message})
            enqueue_on_loop(
                "json",
                {
                    "type": "state",
                    "turn_id": tid,
                    "state": "idle",
                    "detail": detail or "ALL CIRCUITS OPERATIONAL — HOLD OR TAP SPACE TO TALK",
                },
            )

        def on_mic_level(lvl: float) -> None:
            nonlocal last_level_ts
            now = time.monotonic()
            if now - last_level_ts >= 0.033:
                last_level_ts = now
                with state_lock:
                    cid = active_capture_id
                send_json_threadsafe(
                    {"type": "mic_level", "capture_id": cid, "level": round(lvl, 3)},
                    drop_if_full=True,
                )

        def on_mic_max_duration() -> None:
            with state_lock:
                cid = active_capture_id
            send_json_threadsafe(
                {
                    "type": "capture_limit",
                    "capture_id": cid,
                    "message": f"Reached {int(MAX_RECORDING_SECONDS)}s recording limit.",
                }
            )

        mic = MicRecorder(on_level=on_mic_level, on_max_duration=on_mic_max_duration)

        async def stop_local_mic_async() -> np.ndarray:
            nonlocal active_capture_id
            with state_lock:
                active_capture_id = None
            audio = await asyncio.to_thread(mic.stop)
            with runtime.mic_lock:
                if runtime.mic_owner is ws:
                    runtime.mic_owner = None
            return audio

        def run_turn_worker(
            wire_tid: int,
            my_gen: int,
            turn_cancel: threading.Event,
            *,
            audio: np.ndarray | None = None,
            sample_rate: int = 16000,
            user_text: str | None = None,
            steps: int | None = None,
        ) -> None:
            if turn_cancel.is_set() or ws_closed.is_set():
                return

            if not runtime.ready or runtime.tts is None or runtime.stt is None:
                send_json_threadsafe(
                    {
                        "type": "error",
                        "turn_id": wire_tid,
                        "message": "IOGHAL is still initializing.",
                    },
                    my_gen,
                )
                send_json_threadsafe(
                    {
                        "type": "state",
                        "turn_id": wire_tid,
                        "state": "idle",
                        "detail": "INITIALIZING NEURAL SUBSYSTEMS...",
                    },
                    my_gen,
                )
                return

            if not runtime._is_llama_server_model(brain.model) and runtime.ollama_available:
                with runtime._ollama_warmup_lock:
                    runtime._active_ollama_model = brain.model

            pipeline = HalPipeline(
                brain=brain,
                tts=runtime.tts,
                stt=runtime.stt,
                gpu_lock=runtime.gpu_lock,
            )

            def handle_event(ev: HalEvent) -> bool:
                if ws_closed.is_set() or my_gen != current_gen():
                    return False
                if ev.kind == "audio_chunk":
                    assert ev.pcm is not None
                    return send_pcm_nonblocking(wire_tid, ev.pcm, my_gen)
                if ev.kind == "state" and ev.payload.get("state") == "idle" and mic.is_recording:
                    return True
                msg = {"type": ev.kind, "turn_id": wire_tid, **ev.payload}
                return send_json_threadsafe(msg, my_gen)

            pipeline.run_turn(
                wire_tid,
                turn_cancel,
                handle_event,
                audio=audio,
                sample_rate=sample_rate,
                user_text=user_text,
                steps=steps,
            )

        runtime._listeners[ws] = lambda payload: enqueue_on_loop("json", payload)
        runtime._active_closers[ws] = lambda: loop.call_soon_threadsafe(force_close_on_loop)

        try:
            enqueue_on_loop(
                "json", runtime.status_payload(model=brain.model, steps=conn_steps)
            )
            while not ws_closed.is_set():
                try:
                    raw = await ws.receive_text()
                except KeyError:
                    continue
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    enqueue_on_loop("json", {"type": "error", "message": "Invalid JSON."})
                    continue
                if not isinstance(msg, dict):
                    enqueue_on_loop("json", {"type": "error", "message": "Expected JSON object."})
                    continue

                mtype = msg.get("type")
                client_tid = _parse_positive_int(msg.get("turn_id"))
                capture_id = _parse_positive_int(msg.get("capture_id"))

                if mtype == "ptt_start":
                    if not runtime.ready:
                        enqueue_on_loop(
                            "json",
                            {
                                "type": "capture_failed",
                                "capture_id": capture_id,
                                "message": "IOGHAL is still initializing.",
                            },
                        )
                        continue
                    if not runtime.is_loopback:
                        enqueue_on_loop(
                            "json",
                            {
                                "type": "capture_failed",
                                "capture_id": capture_id,
                                "message": "Local host microphone is disabled on non-loopback bindings. Use MIC: BROWSER.",
                            },
                        )
                        continue
                    cancel_active_turn()
                    acquired = False
                    with runtime.mic_lock:
                        if runtime.mic_owner is None or runtime.mic_owner is ws:
                            runtime.mic_owner = ws
                            acquired = True
                    if not acquired:
                        enqueue_on_loop(
                            "json",
                            {
                                "type": "capture_failed",
                                "capture_id": capture_id,
                                "message": "Local microphone is active in another tab.",
                            },
                        )
                        continue
                    try:
                        with state_lock:
                            active_capture_id = capture_id
                        if not mic.is_recording:
                            await asyncio.to_thread(mic.start)
                        logger.info("PTT started (PipeWire local mic, capture_id=%s)", capture_id)
                        enqueue_on_loop(
                            "json",
                            {
                                "type": "state",
                                "capture_id": capture_id,
                                "state": "listening",
                                "detail": "LISTENING — RELEASE (OR PRESS SPACE) TO SEND",
                            },
                        )
                    except (sd.PortAudioError, OSError, RuntimeError) as exc:
                        with state_lock:
                            active_capture_id = None
                        with runtime.mic_lock:
                            if runtime.mic_owner is ws:
                                runtime.mic_owner = None
                        logger.exception("Microphone start failed")
                        enqueue_on_loop(
                            "json",
                            {
                                "type": "capture_failed",
                                "capture_id": capture_id,
                                "message": f"Microphone error: {exc}",
                            },
                        )

                elif mtype == "ptt_stop":
                    if not mic.is_recording:
                        enqueue_on_loop(
                            "json",
                            {
                                "type": "state",
                                "turn_id": client_tid,
                                "state": "idle",
                                "detail": "ALL CIRCUITS OPERATIONAL — HOLD OR TAP SPACE TO TALK",
                            },
                        )
                        continue
                    audio = await stop_local_mic_async()
                    dur_s = audio.size / float(mic.sample_rate)
                    logger.info("PTT stopped: %.2fs (%d samples)", dur_s, audio.size)
                    if dur_s < MIN_AUDIO_DURATION_S:
                        enqueue_on_loop(
                            "json",
                            {
                                "type": "state",
                                "turn_id": client_tid,
                                "state": "idle",
                                "detail": "AUDIO TOO SHORT — HOLD OR TAP SPACE TO TALK",
                            },
                        )
                    else:
                        wire_tid, my_gen, turn_cancel = begin_turn(client_tid)
                        threading.Thread(
                            target=run_turn_worker,
                            args=(wire_tid, my_gen, turn_cancel),
                            kwargs={
                                "audio": audio,
                                "sample_rate": mic.sample_rate,
                                "steps": conn_steps,
                            },
                            daemon=True,
                        ).start()

                elif mtype == "audio_b64":
                    if not runtime.ready:
                        reject_turn(client_tid, "IOGHAL is still initializing.", "INITIALIZING NEURAL SUBSYSTEMS...")
                        continue
                    if mic.is_recording:
                        await stop_local_mic_async()
                    pcm_b64 = msg.get("pcm_b64")
                    sr = msg.get("sample_rate", 16000)
                    if (
                        not isinstance(pcm_b64, str)
                        or not pcm_b64
                        or len(pcm_b64) > MAX_AUDIO_B64_CHARS
                    ):
                        reject_turn(client_tid, "Invalid audio payload.")
                        continue
                    if (
                        isinstance(sr, bool)
                        or not isinstance(sr, int)
                        or not (MIN_SAMPLE_RATE <= sr <= MAX_SAMPLE_RATE)
                    ):
                        reject_turn(client_tid, "Invalid sample rate.")
                        continue
                    try:
                        pcm_bytes = base64.b64decode(pcm_b64, validate=True)
                    except (binascii.Error, ValueError):
                        reject_turn(client_tid, "Malformed base64 audio.")
                        continue
                    if len(pcm_bytes) % 4 != 0:
                        reject_turn(client_tid, "Unaligned float32 audio.")
                        continue
                    audio = np.frombuffer(pcm_bytes, dtype="<f4").copy()
                    dur_s = audio.size / float(sr)
                    if dur_s < MIN_AUDIO_DURATION_S:
                        enqueue_on_loop(
                            "json",
                            {
                                "type": "state",
                                "turn_id": client_tid,
                                "state": "idle",
                                "detail": "AUDIO TOO SHORT — HOLD OR TAP SPACE TO TALK",
                            },
                        )
                        continue
                    if dur_s > MAX_UPLOAD_DURATION_S:
                        reject_turn(client_tid, f"Audio exceeds {int(MAX_UPLOAD_DURATION_S)}s limit.")
                        continue
                    if not np.all(np.isfinite(audio)):
                        reject_turn(client_tid, "Non-finite audio samples.")
                        continue

                    logger.info("Browser audio received: %.2fs @ %d Hz", dur_s, sr)
                    wire_tid, my_gen, turn_cancel = begin_turn(client_tid)
                    threading.Thread(
                        target=run_turn_worker,
                        args=(wire_tid, my_gen, turn_cancel),
                        kwargs={"audio": audio, "sample_rate": sr, "steps": conn_steps},
                        daemon=True,
                    ).start()

                elif mtype == "text":
                    raw_text = msg.get("text")
                    if not isinstance(raw_text, str):
                        enqueue_on_loop(
                            "json",
                            {
                                "type": "state",
                                "turn_id": client_tid,
                                "state": "idle",
                                "detail": "ALL CIRCUITS OPERATIONAL — HOLD OR TAP SPACE TO TALK",
                            },
                        )
                        continue
                    text = raw_text.strip()
                    if not text:
                        # Quint Bug 5d fix: whitespace-only text must ack idle so wire protocol is total
                        enqueue_on_loop(
                            "json",
                            {
                                "type": "state",
                                "turn_id": client_tid,
                                "state": "idle",
                                "detail": "ALL CIRCUITS OPERATIONAL — HOLD OR TAP SPACE TO TALK",
                            },
                        )
                        continue
                    if not runtime.ready:
                        reject_turn(client_tid, "IOGHAL is still initializing.", "INITIALIZING NEURAL SUBSYSTEMS...")
                        continue
                    if len(text) > MAX_TEXT_CHARS:
                        reject_turn(
                            client_tid,
                            f"Message exceeds {MAX_TEXT_CHARS} characters.",
                            f"MESSAGE TOO LONG (MAX {MAX_TEXT_CHARS} CHARS)",
                        )
                        continue
                    if mic.is_recording:
                        await stop_local_mic_async()
                    logger.info("Text turn received: %r", text)
                    wire_tid, my_gen, turn_cancel = begin_turn(client_tid)
                    threading.Thread(
                        target=run_turn_worker,
                        args=(wire_tid, my_gen, turn_cancel),
                        kwargs={"user_text": text, "steps": conn_steps},
                        daemon=True,
                    ).start()

                elif mtype == "interrupt":
                    cancel_active_turn()
                    if mic.is_recording:
                        await stop_local_mic_async()
                    if not msg.get("silent"):
                        enqueue_on_loop(
                            "json",
                            {
                                "type": "state",
                                "turn_id": client_tid,
                                "state": "idle",
                                "detail": "ALL CIRCUITS OPERATIONAL — HOLD OR TAP SPACE TO TALK",
                            },
                        )

                elif mtype == "set_model":
                    new_model = msg.get("model")
                    if isinstance(new_model, str) and new_model in runtime.models:
                        brain.set_model(new_model)
                        runtime.schedule_ollama_warmup(new_model)
                        enqueue_on_loop(
                            "json", runtime.status_payload(model=brain.model, steps=conn_steps)
                        )
                    else:
                        enqueue_on_loop(
                            "json",
                            {"type": "error", "message": "Requested model is not installed."},
                        )

                elif mtype == "set_steps":
                    req_steps = _parse_positive_int(msg.get("steps"))
                    if req_steps is not None and MIN_DIFFUSION_STEPS <= req_steps <= 20:
                        conn_steps = req_steps
                        enqueue_on_loop(
                            "json", runtime.status_payload(model=brain.model, steps=conn_steps)
                        )
                    else:
                        enqueue_on_loop(
                            "json",
                            {
                                "type": "error",
                                "message": f"Steps must be between {MIN_DIFFUSION_STEPS} and 20.",
                            },
                        )

                elif mtype == "reset":
                    cancel_active_turn()
                    if mic.is_recording:
                        await stop_local_mic_async()
                    brain.reset()
                    enqueue_on_loop("json", {"type": "cleared", "turn_id": client_tid})

        except WebSocketDisconnect:
            pass
        finally:
            ws_closed.set()
            cancel_active_turn()
            runtime._listeners.pop(ws, None)
            runtime._active_closers.pop(ws, None)
            try:
                await stop_local_mic_async()
            except (sd.PortAudioError, OSError, RuntimeError) as exc:
                logger.debug("Mic cleanup on disconnect failed: %s", exc)
            close_fn = getattr(brain, "close", None)
            if callable(close_fn):
                close_fn()
            writer_task.cancel()
            await asyncio.gather(writer_task, return_exceptions=True)

    return router


def create_app(runtime: HalRuntime, serve_frontend: bool = True) -> FastAPI:
    """Create a standalone FastAPI app serving the IOGHAL API, WebSocket, and React UI."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        runtime.set_loop(asyncio.get_running_loop())
        runtime.start_background_load()
        try:
            yield
        finally:
            runtime.shutdown()

    app = FastAPI(title="IOGHAL — Pluggable Local GPU Voice Runtime", lifespan=lifespan)
    app.include_router(create_hal_router(runtime))

    if serve_frontend:
        dist_root = WEB_DIST_DIR.resolve()
        index_file = dist_root / "index.html"
        if dist_root.is_dir() and index_file.is_file():
            assets_dir = dist_root / "assets"
            if assets_dir.is_dir():
                app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

            @app.get("/{full_path:path}")
            async def serve_spa(full_path: str):
                if full_path == "api" or full_path.startswith("api/"):
                    raise HTTPException(status_code=404)
                if not full_path:
                    return FileResponse(str(index_file))
                candidate = (dist_root / full_path).resolve()
                if dist_root not in candidate.parents:
                    raise HTTPException(status_code=404)
                if candidate.is_file():
                    return FileResponse(str(candidate))
                if "." in Path(full_path).name:
                    raise HTTPException(status_code=404)
                return FileResponse(str(index_file))

    return app


def main() -> None:
    parser = argparse.ArgumentParser(
        description="IOGHAL Web Server (FastAPI + VoxCPM2 + Liquid + Whisper)"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--model", default=None)
    parser.add_argument("--ref-wav", type=Path, default=None)
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--llama-server-url", default=os.environ.get("HAL_LLAMA_SERVER_URL"))
    parser.add_argument(
        "--llm-backend",
        choices=("auto", "ollama", "llama_server"),
        default=os.environ.get("HAL_LLM_BACKEND", "auto"),
    )
    parser.add_argument(
        "--thinking-mode",
        choices=("auto", "off", "low"),
        default=os.environ.get("HAL_THINKING_MODE", "auto"),
    )
    env_budget_raw = os.environ.get("HAL_REASONING_BUDGET", "")
    default_reasoning_budget = int(env_budget_raw) if env_budget_raw.isdigit() else 64
    parser.add_argument(
        "--reasoning-budget",
        type=int,
        default=default_reasoning_budget,
    )
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--user-name", default="Charles")
    args = parser.parse_args()

    if args.steps < MIN_DIFFUSION_STEPS:
        parser.error(f"--steps must be >= {MIN_DIFFUSION_STEPS}")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    cfg = HalConfig(
        model=args.model,
        ref_wav=args.ref_wav,
        ollama_url=args.ollama_url,
        llm_backend=args.llm_backend,
        llama_server_url=args.llama_server_url,
        thinking_mode=args.thinking_mode,
        reasoning_budget=args.reasoning_budget,
        compile_model=args.compile,
        steps=args.steps,
        user_name=args.user_name,
        host=args.host,
        port=args.port,
    )
    runtime = HalRuntime(cfg)
    app = create_app(runtime)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
