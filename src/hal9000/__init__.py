"""IOGHAL — Pluggable local GPU Push-to-Talk voice runtime (VoxCPM2 + Liquid + Whisper)."""

from .audio import MAX_RECORDING_SECONDS, MicRecorder, StreamingAudioPlayer
from .brain import (
    HAL_SYSTEM_PROMPT,
    POD_BAY_REPLY,
    LoreIndex,
    OllamaBrain,
    QwenBrain,
    choose_default_model,
    choose_default_qwen_model,
    is_daisy_song_request,
    is_inkling_model,
    is_pod_bay_open_request,
    list_ollama_models,
)
from .daisy import DAISY_LYRIC_PHRASES, stream_daisy_phrase
from .interfaces import (
    AudioInput,
    BrainProvider,
    HalConfig,
    HalEvent,
    STTProvider,
    TTSProvider,
    TurnResult,
)
from .pipeline import HalPipeline
from .server import HalRuntime, create_app, create_hal_router
from .stt import WhisperSTT
from .tts import HalVoiceEngine, normalize_for_hal_tts, resolve_reference_wav

__all__ = [
    "DAISY_LYRIC_PHRASES",
    "HAL_SYSTEM_PROMPT",
    "MAX_RECORDING_SECONDS",
    "POD_BAY_REPLY",
    "AudioInput",
    "BrainProvider",
    "HalConfig",
    "HalEvent",
    "HalPipeline",
    "HalRuntime",
    "HalVoiceEngine",
    "LoreIndex",
    "MicRecorder",
    "OllamaBrain",
    "QwenBrain",
    "STTProvider",
    "StreamingAudioPlayer",
    "TTSProvider",
    "TurnResult",
    "WhisperSTT",
    "choose_default_model",
    "choose_default_qwen_model",
    "create_app",
    "create_hal_router",
    "is_daisy_song_request",
    "is_inkling_model",
    "is_pod_bay_open_request",
    "list_ollama_models",
    "normalize_for_hal_tts",
    "resolve_reference_wav",
    "stream_daisy_phrase",
]
