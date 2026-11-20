"""VoxCPM2 HAL 9000 voice-cloning TTS engine with pre-built prompt caching."""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from collections.abc import Generator
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from huggingface_hub.errors import LocalEntryNotFoundError
from voxcpm import VoxCPM

if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")

logger = logging.getLogger(__name__)

DEFAULT_PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_REF_WAV = DEFAULT_PROJECT_DIR / "voices" / "hal9000_ref.wav"
DEFAULT_REF_TXT = DEFAULT_PROJECT_DIR / "voices" / "hal9000_ref.txt"
MIN_DIFFUSION_STEPS = 2

_PRONUNCIATION_SUBS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bHAL[\s-]*9000\b", re.IGNORECASE), "HAL nine thousand"),
    (re.compile(r"\bSAL[\s-]*9000\b", re.IGNORECASE), "SAL nine thousand"),
    (re.compile(r"\bAE[\s-]*35\b", re.IGNORECASE), "A E thirty-five"),
    (re.compile(r"\bTMA[\s-]*1\b", re.IGNORECASE), "T M A one"),
    (re.compile(r"\bNSC[\s-]*342/23\b", re.IGNORECASE), "N S C three forty-two twenty-three"),
    (re.compile(r"\bDrs\.\s*"), "Doctors "),
    (re.compile(r"\bDr\.\s*"), "Doctor "),
    (re.compile(r"\bMr\.\s*"), "Mister "),
    (re.compile(r"\bMrs\.\s*"), "Missus "),
    (re.compile(r"\bProf\.\s*"), "Professor "),
)


def normalize_for_hal_tts(text: str) -> str:
    """Expand space-odyssey abbreviations so VoxCPM2 pronounces them naturally."""
    cleaned = text.strip()
    if not cleaned:
        return ""
    for pat, repl in _PRONUNCIATION_SUBS:
        cleaned = pat.sub(repl, cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def resolve_reference_wav(ref_wav: Path | str | None = None) -> Path:
    """Resolve the voice reference WAV from argument, HAL_REF_WAV env var, or default path."""
    if ref_wav is not None:
        return Path(ref_wav)
    env_wav = os.environ.get("HAL_REF_WAV")
    if env_wav:
        return Path(env_wav)
    return DEFAULT_REF_WAV


class HalVoiceEngine:
    """Wraps OpenBMB VoxCPM2 with a pre-cached voice prompt implementing ``TTSProvider``."""

    def __init__(
        self,
        model_id: str = "openbmb/VoxCPM2",
        reference_wav: Path | str | None = None,
        reference_text: str | None = None,
        device: str | None = None,
        optimize: bool = False,
        cfg_value: float = 2.0,
        inference_timesteps: int = 6,
    ) -> None:
        if inference_timesteps < MIN_DIFFUSION_STEPS:
            raise ValueError(
                f"inference_timesteps must be >= {MIN_DIFFUSION_STEPS} "
                f"(1 skips the learned diffusion update), got {inference_timesteps}"
            )

        self.model_id = model_id
        self.reference_wav = resolve_reference_wav(reference_wav)
        if not self.reference_wav.exists():
            raise FileNotFoundError(
                f"Voice reference audio not found: {self.reference_wav}. "
                "Place a 48 kHz mono reference WAV at voices/hal9000_ref.wav or pass --ref-wav / HAL_REF_WAV."
            )

        # Pair bundled transcript only with bundled WAV; otherwise use sibling .txt or reference-only mode
        if reference_text is None:
            txt_path = self.reference_wav.with_suffix(".txt")
            if txt_path.exists():
                reference_text = txt_path.read_text(encoding="utf-8").strip()
            elif (
                self.reference_wav.resolve() == DEFAULT_REF_WAV.resolve()
                and DEFAULT_REF_TXT.exists()
            ):
                reference_text = DEFAULT_REF_TXT.read_text(encoding="utf-8").strip()

        # Trailing space is required for continuation concatenation in VoxCPM2Model
        self.reference_text = (reference_text.strip() + " ") if reference_text else None
        self.cfg_value = cfg_value
        self.inference_timesteps = inference_timesteps
        self._phrase_cache: dict[tuple[str, int], list[np.ndarray]] = {}
        self._cache_lock = threading.Lock()

        t0 = time.perf_counter()
        try:
            self.model = VoxCPM.from_pretrained(
                model_id,
                load_denoiser=False,
                local_files_only=True,
                optimize=optimize,
                device=device,
            )
        except (LocalEntryNotFoundError, FileNotFoundError):
            logger.info("VoxCPM2 cache incomplete for %s; downloading from HuggingFace...", model_id)
            self.model = VoxCPM.from_pretrained(
                model_id,
                load_denoiser=False,
                local_files_only=False,
                optimize=optimize,
                device=device,
            )
        t_load = time.perf_counter() - t0
        self.sample_rate: int = int(self.model.tts_model.sample_rate)

        t_prompt = time.perf_counter()
        self.prompt_cache = self.model.tts_model.build_prompt_cache(
            prompt_text=self.reference_text,
            prompt_wav_path=str(self.reference_wav) if self.reference_text else None,
            reference_wav_path=str(self.reference_wav),
            trim_silence_vad=True,
        )
        logger.info(
            "Loaded VoxCPM2 (%s) @ %d Hz (%s mode) — weights %.2fs, prompt cache %.2fs",
            model_id,
            self.sample_rate,
            self.prompt_cache.get("mode", "unknown"),
            t_load,
            time.perf_counter() - t_prompt,
        )

    def _effective_steps(self, inference_timesteps: int | None) -> int:
        steps = self.inference_timesteps if inference_timesteps is None else inference_timesteps
        if steps < MIN_DIFFUSION_STEPS:
            raise ValueError(f"inference_timesteps must be >= {MIN_DIFFUSION_STEPS}, got {steps}")
        return steps

    def warmup_streaming(self, seed: int = 42, max_chunks: int = 4) -> float:
        """Warm up the streaming decoder path actually used by live turns."""
        t0 = time.perf_counter()
        stop_ev = threading.Event()
        gen = self.stream_speech("Affirmative.", stop_event=stop_ev, seed=seed)
        try:
            for idx, _ in enumerate(gen, start=1):
                if idx >= max_chunks:
                    stop_ev.set()
                    break
        finally:
            gen.close()
        elapsed = time.perf_counter() - t0
        logger.info("Warmed up VoxCPM2 streaming path in %.2fs", elapsed)
        return elapsed

    def synthesize(
        self,
        text: str,
        seed: int | None = None,
        inference_timesteps: int | None = None,
    ) -> np.ndarray:
        """Generate a complete 48 kHz mono float32 waveform in the cloned voice."""
        spoken = normalize_for_hal_tts(text)
        if not spoken:
            return np.zeros(0, dtype=np.float32)
        steps = self._effective_steps(inference_timesteps)
        wav, _, _ = self.model.tts_model.generate_with_prompt_cache(
            target_text=spoken,
            prompt_cache=self.prompt_cache,
            cfg_value=self.cfg_value,
            inference_timesteps=steps,
            retry_badcase=True,
            seed=seed,
        )
        return wav.squeeze(0).cpu().numpy().astype(np.float32, copy=False)

    def synthesize_to_file(
        self,
        text: str,
        output_path: Path | str,
        seed: int | None = None,
        inference_timesteps: int | None = None,
    ) -> Path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        wav = self.synthesize(text, seed=seed, inference_timesteps=inference_timesteps)
        sf.write(str(out), wav, self.sample_rate)
        return out

    def stream_speech(
        self,
        text: str,
        stop_event: threading.Event | None = None,
        seed: int | None = None,
        inference_timesteps: int | None = None,
        cache_phrase: bool = False,
    ) -> Generator[np.ndarray, None, None]:
        """Yield 48 kHz mono float32 audio chunks in real time."""
        if stop_event is not None and stop_event.is_set():
            return
        spoken = normalize_for_hal_tts(text)
        if not spoken:
            return
        steps = self._effective_steps(inference_timesteps)
        cache_key = (spoken, steps)

        if cache_phrase:
            with self._cache_lock:
                cached = self._phrase_cache.get(cache_key)
            if cached:
                for chunk in cached:
                    if stop_event is not None and stop_event.is_set():
                        return
                    yield chunk
                return

        gen = self.model.tts_model.generate_with_prompt_cache_streaming(
            target_text=spoken,
            prompt_cache=self.prompt_cache,
            cfg_value=self.cfg_value,
            inference_timesteps=steps,
            retry_badcase=False,
            retry_badcase_ratio_threshold=3.5,
            seed=seed,
        )
        collected: list[np.ndarray] | None = [] if cache_phrase else None
        completed_cleanly = False
        try:
            for wav_chunk, _, _ in gen:
                if stop_event is not None and stop_event.is_set():
                    break
                arr = wav_chunk.squeeze(0).cpu().numpy().astype(np.float32, copy=False)
                if collected is not None:
                    collected.append(arr.copy())
                yield arr
                if stop_event is not None and stop_event.is_set():
                    break
            else:
                completed_cleanly = True
        finally:
            gen.close()

        if completed_cleanly and collected:
            with self._cache_lock:
                self._phrase_cache[cache_key] = collected
