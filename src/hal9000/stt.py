"""Local GPU Speech-to-Text using cached OpenAI Whisper Large v3 via Transformers."""

from __future__ import annotations

import logging
import time

import numpy as np
import torch
from huggingface_hub.errors import LocalEntryNotFoundError
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

logger = logging.getLogger(__name__)

HAL_WHISPER_PROMPT = (
    "HAL 9000, Charles, Dave Bowman, Frank Poole, Dr. Chandra, Discovery One, "
    "AE-35, TMA-1, Europa, Monolith, pod bay doors."
)


def _has_speech_activity(audio: np.ndarray, sample_rate: int = 16000) -> bool:
    """Lightweight frame-level energy check rejecting silence and flat low-level room hiss."""
    if audio.size < int(sample_rate * 0.2):
        return False
    rms = float(np.sqrt(np.mean(audio * audio)))
    peak = float(np.max(np.abs(audio)))
    if rms < 4e-4 or peak < 3e-3:
        return False

    frame_len = max(1, int(sample_rate * 0.02))  # 20ms frames
    n_frames = audio.size // frame_len
    if n_frames < 5:
        return rms >= 1.5e-3

    framed = audio[: n_frames * frame_len].reshape(n_frames, frame_len)
    frame_rms = np.sqrt(np.mean(framed * framed, axis=1))
    noise_floor = float(np.percentile(frame_rms, 20))
    # Require either clearly audible absolute energy or frames rising above room noise floor
    threshold = max(0.0025, noise_floor * 2.2)
    voiced_frames = int(np.sum(frame_rms >= threshold))
    if voiced_frames >= 3:
        return True
    # Allow sustained loud continuous speech where 20th percentile is already speech
    return noise_floor >= 0.008 and peak >= 0.02


class WhisperSTT:
    """Fast local-first CUDA Whisper transcriber implementing ``STTProvider``."""

    def __init__(
        self,
        model_id: str = "openai/whisper-large-v3",
        device: str | None = None,
        local_files_only: bool | None = None,
        domain_prompt: str | None = HAL_WHISPER_PROMPT,
    ) -> None:
        self.model_id = model_id
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        if self.device.startswith("cuda"):
            self.dtype = (
                torch.bfloat16
                if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
                else torch.float16
            )
        else:
            self.dtype = torch.float32

        t0 = time.perf_counter()
        self.processor, self.model = self._load_local_first(model_id, local_files_only)
        self.model = self.model.to(self.device)
        self.model.eval()
        if getattr(self.model.generation_config, "forced_decoder_ids", None) is not None:
            self.model.generation_config.forced_decoder_ids = None

        self._prompt_ids: torch.Tensor | None = None
        if domain_prompt:
            try:
                ids = self.processor.get_prompt_ids(domain_prompt, return_tensors="pt")
                self._prompt_ids = ids.to(self.device)
            except Exception:
                logger.debug("Could not build Whisper domain prompt IDs", exc_info=True)

        logger.info(
            "Loaded Whisper STT (%s) on %s (%s) in %.2fs",
            model_id,
            self.device,
            self.dtype,
            time.perf_counter() - t0,
        )

    def _load_local_first(
        self, model_id: str, local_files_only: bool | None
    ) -> tuple[AutoProcessor, AutoModelForSpeechSeq2Seq]:
        if local_files_only is True:
            proc = AutoProcessor.from_pretrained(model_id, local_files_only=True)
            mdl = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id, dtype=self.dtype, local_files_only=True
            )
            return proc, mdl

        if local_files_only is None:
            try:
                proc = AutoProcessor.from_pretrained(model_id, local_files_only=True)
                mdl = AutoModelForSpeechSeq2Seq.from_pretrained(
                    model_id, dtype=self.dtype, local_files_only=True
                )
                return proc, mdl
            except (LocalEntryNotFoundError, FileNotFoundError, OSError):
                logger.info("Whisper cache incomplete for %s; downloading from HuggingFace...", model_id)

        proc = AutoProcessor.from_pretrained(model_id, local_files_only=False)
        mdl = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, dtype=self.dtype, local_files_only=False
        )
        return proc, mdl

    @torch.inference_mode()
    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """Transcribe a mono/multi-channel float waveform (auto-resamples to 16 kHz)."""
        arr = np.asarray(audio, dtype=np.float32)
        if arr.ndim > 1:
            arr = arr.mean(axis=-1).astype(np.float32, copy=False)
        else:
            arr = arr.reshape(-1)
        if arr.size == 0 or not np.all(np.isfinite(arr)):
            return ""

        if sample_rate != 16000:
            import librosa

            arr = librosa.resample(arr, orig_sr=sample_rate, target_sr=16000).astype(
                np.float32, copy=False
            )
            sample_rate = 16000

        arr = arr - float(np.mean(arr))
        if not _has_speech_activity(arr, sample_rate):
            return ""

        inputs = self.processor(
            arr,
            sampling_rate=sample_rate,
            return_tensors="pt",
            return_attention_mask=True,
        )
        input_features = inputs.input_features.to(device=self.device, dtype=self.dtype)
        attention_mask = inputs.attention_mask.to(device=self.device)

        gen_kwargs: dict = {
            "attention_mask": attention_mask,
            "language": "en",
            "task": "transcribe",
            "max_length": 256,
            "temperature": 0.0,
            "no_speech_threshold": 0.6,
            "logprob_threshold": -1.0,
            "compression_ratio_threshold": 1.35,
        }
        if self._prompt_ids is not None:
            gen_kwargs["prompt_ids"] = self._prompt_ids

        predicted_ids = self.model.generate(input_features, **gen_kwargs)
        text = self.processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return text.strip()
