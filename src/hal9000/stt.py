"""Local GPU Speech-to-Text using cached OpenAI Whisper Large v3 with native FP8 acceleration."""

from __future__ import annotations

import logging
import time

import numpy as np
import torch
from huggingface_hub.errors import LocalEntryNotFoundError
from torch import nn
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from transformers.modeling_outputs import BaseModelOutput

from ._cuda_ops import FP8Linear, capture_cuda_graph, probe_fp8_supported

logger = logging.getLogger(__name__)

HAL_WHISPER_PROMPT = (
    "HAL 9000, Charles, Dave Bowman, Frank Poole, Dr. Chandra, Discovery One, "
    "Alexei Leonov, AE-35, TMA-1, Europa, Monolith, pod bay doors."
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


def _quantize_whisper_fp8(model: nn.Module) -> int:
    """Convert Whisper encoder/decoder projection linears to native CUDA FP8."""
    count = 0

    def _walk(root: nn.Module) -> None:
        nonlocal count
        for name, child in list(root.named_children()):
            if name == "proj_out":
                continue
            if (
                isinstance(child, nn.Linear)
                and child.in_features % 16 == 0
                and child.out_features % 16 == 0
                and child.in_features >= 256
                and child.out_features >= 256
            ):
                setattr(root, name, FP8Linear(child.weight, child.bias))
                count += 1
            else:
                _walk(child)

    _walk(model)
    return count


class WhisperSTT:
    """Fast local-first CUDA Whisper transcriber with native FP8 & greedy decoding."""

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

        # Clean generation_config to eliminate deprecation & max_length/max_new_tokens warnings
        gen_cfg = self.model.generation_config
        if getattr(gen_cfg, "forced_decoder_ids", None) is not None:
            gen_cfg.forced_decoder_ids = None
        if hasattr(gen_cfg, "suppress_tokens"):
            gen_cfg.suppress_tokens = None
        if hasattr(gen_cfg, "begin_suppress_tokens"):
            gen_cfg.begin_suppress_tokens = None
        gen_cfg.max_length = 160

        self.fp8_enabled = False
        self.encoder_graphed = False
        if self.dtype == torch.bfloat16 and probe_fp8_supported(self.device):
            try:
                inner = getattr(self.model, "model", self.model)
                n_quant = _quantize_whisper_fp8(inner)
                torch.cuda.empty_cache()
                self.fp8_enabled = n_quant > 0
                # Capture fixed-shape (1, 128, 3000) Whisper encoder in a static CUDA graph
                enc = getattr(inner, "encoder", None)
                if enc is not None and hasattr(enc, "config"):
                    n_mels = int(getattr(enc.config, "num_mel_bins", 128))
                    seq_len = int(enc.config.max_source_positions) * enc.conv1.stride[0] * enc.conv2.stride[0]
                    raw_enc_fwd = enc.forward
                    sample_feats = torch.zeros((1, n_mels, seq_len), device=self.device, dtype=self.dtype)
                    with torch.inference_mode():
                        replay_enc = capture_cuda_graph(
                            lambda feats: raw_enc_fwd(feats).last_hidden_state,
                            (sample_feats,),
                        )

                    def _fast_enc_forward(
                        input_features: torch.Tensor,
                        attention_mask=None,
                        **kwargs,
                    ) -> BaseModelOutput:
                        if (
                            input_features.shape == (1, n_mels, seq_len)
                            and input_features.dtype == self.dtype
                            and not kwargs.get("output_attentions")
                            and not kwargs.get("output_hidden_states")
                        ):
                            return BaseModelOutput(last_hidden_state=replay_enc(input_features))
                        return raw_enc_fwd(input_features, attention_mask=attention_mask, **kwargs)

                    enc.forward = _fast_enc_forward  # type: ignore[method-assign]
                    self.encoder_graphed = True
            except (RuntimeError, ValueError, TypeError, AttributeError, OSError) as exc:
                logger.debug("Whisper FP8/CUDA-graph acceleration skipped: %s", exc)

        self._prompt_ids: torch.Tensor | None = None
        if domain_prompt:
            try:
                ids = self.processor.get_prompt_ids(domain_prompt, return_tensors="pt")
                self._prompt_ids = ids.to(self.device)
            except Exception:
                logger.debug("Could not build Whisper domain prompt IDs", exc_info=True)

        logger.info(
            "Loaded Whisper STT (%s) on %s (%s%s) in %.2fs",
            model_id,
            self.device,
            self.dtype,
            " + FP8" if self.fp8_enabled else "",
            time.perf_counter() - t0,
        )

    def _load_local_first(
        self, model_id: str, local_files_only: bool | None
    ) -> tuple[AutoProcessor, AutoModelForSpeechSeq2Seq]:
        load_kwargs: dict = {"dtype": self.dtype}
        if self.device.startswith("cuda"):
            load_kwargs["attn_implementation"] = "sdpa"

        if local_files_only is True:
            proc = AutoProcessor.from_pretrained(model_id, local_files_only=True)
            mdl = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id, local_files_only=True, **load_kwargs
            )
            return proc, mdl

        if local_files_only is None:
            try:
                proc = AutoProcessor.from_pretrained(model_id, local_files_only=True)
                mdl = AutoModelForSpeechSeq2Seq.from_pretrained(
                    model_id, local_files_only=True, **load_kwargs
                )
                return proc, mdl
            except (LocalEntryNotFoundError, FileNotFoundError, OSError):
                logger.info("Whisper cache incomplete for %s; downloading from HuggingFace...", model_id)

        proc = AutoProcessor.from_pretrained(model_id, local_files_only=False)
        mdl = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, local_files_only=False, **load_kwargs
        )
        return proc, mdl

    @torch.inference_mode()
    def warmup(self) -> float:
        """Warm Whisper FP8/CUDA-graph encoder and decoder kernels on a 1s synthetic probe."""
        t0 = time.perf_counter()
        t = np.linspace(0.0, 1.0, 16000, endpoint=False, dtype=np.float32)
        probe = (0.04 * np.sin(2.0 * np.pi * 220.0 * t)).astype(np.float32)
        inputs = self.processor(
            probe,
            sampling_rate=16000,
            return_tensors="pt",
            return_attention_mask=True,
        )
        input_features = inputs.input_features.to(device=self.device, dtype=self.dtype)
        attention_mask = inputs.attention_mask.to(device=self.device)
        gen_kwargs: dict = {
            "attention_mask": attention_mask,
            "language": "en",
            "task": "transcribe",
            "max_length": 64,
            "do_sample": False,
            "num_beams": 1,
            "use_cache": True,
        }
        if self._prompt_ids is not None:
            gen_kwargs["prompt_ids"] = self._prompt_ids
        _ = self.model.generate(input_features, **gen_kwargs)
        elapsed = time.perf_counter() - t0
        logger.info("Warmed up Whisper STT (%s) in %.2fs", self.model_id, elapsed)
        return elapsed

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

        # Direct greedy decode avoids HuggingFace's slow multi-temperature fallback loop (~10x faster)
        gen_kwargs: dict = {
            "attention_mask": attention_mask,
            "language": "en",
            "task": "transcribe",
            "max_length": 160,
            "do_sample": False,
            "num_beams": 1,
            "use_cache": True,
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
