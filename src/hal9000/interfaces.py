"""Pluggable protocol interfaces and typed configuration for IOGHAL.

Host applications can swap any subsystem (STT, LLM brain, TTS, or audio input)
by supplying an object implementing the corresponding ``Protocol``.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Generator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class STTProvider(Protocol):
    """Speech-to-text provider converting mono float32 audio into text."""

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str: ...


@runtime_checkable
class BrainProvider(Protocol):
    """Stateful conversational LLM streaming complete spoken sentences."""

    model: str

    def reset(self) -> None: ...

    def set_model(self, model: str) -> None: ...

    def stream_reply_sentences(
        self,
        user_text: str,
        stop_check: Callable[[], bool] | None = None,
    ) -> Generator[str, None, str]: ...


@runtime_checkable
class TTSProvider(Protocol):
    """Voice synthesis provider emitting mono float32 audio at ``sample_rate`` Hz."""

    sample_rate: int

    def synthesize(
        self,
        text: str,
        seed: int | None = None,
        inference_timesteps: int | None = None,
    ) -> np.ndarray: ...

    def stream_speech(
        self,
        text: str,
        stop_event: threading.Event | None = None,
        seed: int | None = None,
        inference_timesteps: int | None = None,
    ) -> Generator[np.ndarray, None, None]: ...


@runtime_checkable
class AudioInput(Protocol):
    """Push-to-talk audio capture source returning mono float32 PCM on stop."""

    sample_rate: int

    @property
    def is_recording(self) -> bool: ...

    def start(self) -> None: ...

    def stop(self) -> np.ndarray: ...


HalEventKind = Literal[
    "state",
    "transcript",
    "audio_start",
    "hal_partial",
    "audio_chunk",
    "metrics",
    "audio_end",
    "error",
]


@dataclass(slots=True)
class HalEvent:
    """Typed event emitted by ``HalPipeline.run_turn`` to any UI or transport sink."""

    kind: HalEventKind
    turn_id: int
    payload: dict[str, Any] = field(default_factory=dict)
    pcm: np.ndarray | None = None


@dataclass(slots=True)
class TurnResult:
    """Summary returned when a single conversational turn finishes."""

    turn_id: int
    user_text: str = ""
    reply_text: str = ""
    stt_seconds: float = 0.0
    first_chunk_seconds: float | None = None
    audio_seconds: float = 0.0
    generation_seconds: float = 0.0
    cancelled: bool = False
    failed: bool = False
    error: str | None = None


@dataclass(slots=True)
class HalConfig:
    """Configuration for constructing ``HalRuntime`` or embedding IOGHAL."""

    model: str | None = None
    ollama_url: str = "http://127.0.0.1:11434"
    llm_backend: Literal["auto", "ollama", "llama_server"] = "auto"
    llama_server_url: str | None = None
    thinking_mode: Literal["auto", "off", "low"] = "auto"
    reasoning_budget: int = 48
    ref_wav: Path | None = None
    ref_text: str | None = None
    steps: int = 6
    cfg_value: float = 2.0
    compile_model: bool = False
    whisper_model_id: str = "openai/whisper-large-v3"
    voxcpm_model_id: str = "openbmb/VoxCPM2"
    lore_dir: Path | None = None
    user_name: str = "Charles"
    host: str = "127.0.0.1"
    port: int = 8765
    allowed_origins: tuple[str, ...] = ()
