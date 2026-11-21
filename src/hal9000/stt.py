"""Local GPU Speech-to-Text using cached OpenAI Whisper Large v3 with native FP8 acceleration."""

from __future__ import annotations

import copy
import logging
import time
import types

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub.errors import LocalEntryNotFoundError
from torch import nn
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from transformers.cache_utils import EncoderDecoderCache
from transformers.generation import GenerationConfig, GenerationMixin
from transformers.modeling_outputs import BaseModelOutput

from ._cuda_ops import (
    CUDA_GRAPH_LOCK,
    FP8Linear,
    capture_cuda_graph,
    probe_fp8_supported,
)

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


def _fused_self_attn_forward(
    self: nn.Module,
    hidden_states: torch.Tensor,
    key_value_states: torch.Tensor | None = None,
    past_key_values=None,
    attention_mask: torch.Tensor | None = None,
    output_attentions: bool = False,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    bsz, q_len, _ = hidden_states.shape
    qkv = self.qkv_proj(hidden_states)
    q, k, v = qkv.chunk(3, dim=-1)
    q = q.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    k = k.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    v = v.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

    if past_key_values is not None:
        cache = (
            past_key_values.self_attention_cache
            if isinstance(past_key_values, EncoderDecoderCache)
            else past_key_values
        )
        k, v = cache.update(k, v, self.layer_idx)

    is_causal = q_len > 1 and attention_mask is None and self.is_causal
    attn_output = F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=attention_mask,
        dropout_p=0.0,
        is_causal=is_causal,
        scale=1.0,
    )
    attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, self.embed_dim)
    return self.out_proj(attn_output), None


def _fused_cross_attn_forward(
    self: nn.Module,
    hidden_states: torch.Tensor,
    key_value_states: torch.Tensor | None = None,
    past_key_values=None,
    attention_mask: torch.Tensor | None = None,
    output_attentions: bool = False,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    bsz, q_len, _ = hidden_states.shape
    q = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

    is_updated = False
    cross_cache = None
    if past_key_values is not None and isinstance(past_key_values, EncoderDecoderCache):
        is_updated = bool(past_key_values.is_updated.get(self.layer_idx))
        past_key_values.is_updated[self.layer_idx] = True
        cross_cache = past_key_values.cross_attention_cache

    if cross_cache is not None and is_updated:
        k = cross_cache.layers[self.layer_idx].keys
        v = cross_cache.layers[self.layer_idx].values
    else:
        src_states = key_value_states if key_value_states is not None else hidden_states
        src_len = src_states.shape[1]
        kv = self.kv_proj(src_states)
        k, v = kv.chunk(2, dim=-1)
        k = k.view(bsz, src_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        v = v.view(bsz, src_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        if cross_cache is not None:
            k, v = cross_cache.update(k, v, self.layer_idx)

    attn_output = F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=attention_mask,
        dropout_p=0.0,
        is_causal=False,
        scale=1.0,
    )
    attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, self.embed_dim)
    return self.out_proj(attn_output), None


def _fuse_self_attn_fp8(attn: nn.Module) -> int:
    """Fuse Whisper self-attention Q/K/V into one FP8Linear (folding q_scaling, zero bias for K)."""
    if not (
        isinstance(getattr(attn, "q_proj", None), nn.Linear)
        and isinstance(getattr(attn, "k_proj", None), nn.Linear)
        and isinstance(getattr(attn, "v_proj", None), nn.Linear)
        and isinstance(getattr(attn, "out_proj", None), nn.Linear)
    ):
        return 0
    embed_dim = attn.q_proj.in_features
    if (
        embed_dim < 256
        or embed_dim % 16 != 0
        or attn.q_proj.out_features != embed_dim
        or attn.k_proj.out_features != embed_dim
        or attn.v_proj.out_features != embed_dim
    ):
        return 0

    scaling = float(attn.scaling)
    q_w = (attn.q_proj.weight.detach().float() * scaling).to(attn.q_proj.weight.dtype)
    k_w = attn.k_proj.weight.detach()
    v_w = attn.v_proj.weight.detach()
    qkv_w = torch.cat([q_w, k_w, v_w], dim=0).contiguous()

    has_bias = (
        attn.q_proj.bias is not None
        or attn.k_proj.bias is not None
        or attn.v_proj.bias is not None
    )
    if has_bias:
        zero_b = torch.zeros(embed_dim, device=q_w.device, dtype=q_w.dtype)
        q_b = (
            (attn.q_proj.bias.detach().float() * scaling).to(q_w.dtype)
            if attn.q_proj.bias is not None
            else zero_b
        )
        k_b = attn.k_proj.bias.detach() if attn.k_proj.bias is not None else zero_b
        v_b = attn.v_proj.bias.detach() if attn.v_proj.bias is not None else zero_b
        qkv_b: torch.Tensor | None = torch.cat([q_b, k_b, v_b], dim=0).contiguous()
    else:
        qkv_b = None

    attn.qkv_proj = FP8Linear(qkv_w, qkv_b)
    attn.out_proj = FP8Linear(attn.out_proj.weight, attn.out_proj.bias)
    attn._q_scaled = True
    del attn.q_proj, attn.k_proj, attn.v_proj
    attn.forward = types.MethodType(_fused_self_attn_forward, attn)
    return 4


def _fuse_cross_attn_fp8(attn: nn.Module) -> int:
    """Fuse Whisper cross-attention K/V into one FP8Linear and fold q_scaling into q_proj."""
    if not (
        isinstance(getattr(attn, "q_proj", None), nn.Linear)
        and isinstance(getattr(attn, "k_proj", None), nn.Linear)
        and isinstance(getattr(attn, "v_proj", None), nn.Linear)
        and isinstance(getattr(attn, "out_proj", None), nn.Linear)
    ):
        return 0
    embed_dim = attn.q_proj.in_features
    if (
        embed_dim < 256
        or embed_dim % 16 != 0
        or attn.q_proj.out_features != embed_dim
        or attn.k_proj.out_features != embed_dim
        or attn.v_proj.out_features != embed_dim
    ):
        return 0

    scaling = float(attn.scaling)
    q_w = (attn.q_proj.weight.detach().float() * scaling).to(attn.q_proj.weight.dtype)
    q_b = (
        (attn.q_proj.bias.detach().float() * scaling).to(q_w.dtype)
        if attn.q_proj.bias is not None
        else None
    )
    attn.q_proj = FP8Linear(q_w, q_b)
    attn._q_scaled = True

    k_w = attn.k_proj.weight.detach()
    v_w = attn.v_proj.weight.detach()
    kv_w = torch.cat([k_w, v_w], dim=0).contiguous()
    if attn.k_proj.bias is not None or attn.v_proj.bias is not None:
        zero_b = torch.zeros(embed_dim, device=k_w.device, dtype=k_w.dtype)
        k_b = attn.k_proj.bias.detach() if attn.k_proj.bias is not None else zero_b
        v_b = attn.v_proj.bias.detach() if attn.v_proj.bias is not None else zero_b
        kv_b: torch.Tensor | None = torch.cat([k_b, v_b], dim=0).contiguous()
    else:
        kv_b = None

    attn.kv_proj = FP8Linear(kv_w, kv_b)
    attn.out_proj = FP8Linear(attn.out_proj.weight, attn.out_proj.bias)
    del attn.k_proj, attn.v_proj
    attn.forward = types.MethodType(_fused_cross_attn_forward, attn)
    return 4


def _quantize_whisper_fp8(model: nn.Module) -> int:
    """Fuse Whisper QKV/KV attention projections and convert eligible linears to native CUDA FP8."""
    count = 0

    enc = getattr(model, "encoder", None)
    if enc is not None and hasattr(enc, "layers"):
        for layer in enc.layers:
            if hasattr(layer, "self_attn"):
                count += _fuse_self_attn_fp8(layer.self_attn)

    dec = getattr(model, "decoder", None)
    if dec is not None and hasattr(dec, "layers"):
        for layer in dec.layers:
            if hasattr(layer, "self_attn"):
                count += _fuse_self_attn_fp8(layer.self_attn)
            if hasattr(layer, "encoder_attn"):
                count += _fuse_cross_attn_fp8(layer.encoder_attn)

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


class _FastWhisperDecoder:
    """Static-KV-cache + CUDA-graphed greedy Whisper decoder (cross-KV+prefill graph + 1-token step graph)."""

    def __init__(
        self,
        model: nn.Module,
        prefix_ids: torch.Tensor,
        *,
        device: str,
        dtype: torch.dtype,
        max_new_tokens: int = 160,
        suppress_tokens: list[int] | tuple[int, ...] | None = None,
        begin_suppress_tokens: list[int] | tuple[int, ...] | None = None,
    ) -> None:
        inner = getattr(model, "model", model)
        self.decoder = inner.decoder
        self.proj_out = model.proj_out
        cfg = model.config
        gen_cfg = model.generation_config

        self.device = device
        self.dtype = dtype
        self.num_layers = len(self.decoder.layers)
        self.num_heads = int(cfg.decoder_attention_heads)
        self.embed_dim = int(cfg.d_model)
        self.head_dim = self.embed_dim // self.num_heads
        self.src_len = int(cfg.max_source_positions)
        self.vocab_size = int(cfg.vocab_size)
        self.max_target_positions = int(cfg.max_target_positions)

        eos_raw = getattr(gen_cfg, "eos_token_id", getattr(cfg, "eos_token_id", 50257))
        if isinstance(eos_raw, (list, tuple)):
            eos_raw = eos_raw[0]
        self.eos_token_id = int(eos_raw)

        self._static_prefix_ids = prefix_ids.to(device=device, dtype=torch.long).contiguous()
        self.prefix_len = int(self._static_prefix_ids.shape[1])
        self._static_prefix_pos = (
            self.decoder.embed_positions.weight[: self.prefix_len]
            .unsqueeze(0)
            .to(device=device, dtype=dtype)
            .contiguous()
        )

        # Align static self-attention KV cache length to 64 (within max_target_positions)
        aligned_len = ((self.prefix_len + int(max_new_tokens) + 63) // 64) * 64
        self.max_cache_len = min(self.max_target_positions, max(self.prefix_len + 1, aligned_len))

        self._suppress_mask = self._build_token_mask(suppress_tokens)
        self._begin_suppress_mask = self._build_token_mask(begin_suppress_tokens)

        # Pre-allocated static buffers owned for CUDA graph lifetime
        self._static_enc_hidden = torch.zeros(
            (1, self.src_len, self.embed_dim), device=device, dtype=dtype
        )
        self._self_k = [
            torch.zeros((1, self.num_heads, self.max_cache_len, self.head_dim), device=device, dtype=dtype)
            for _ in range(self.num_layers)
        ]
        self._self_v = [
            torch.zeros((1, self.num_heads, self.max_cache_len, self.head_dim), device=device, dtype=dtype)
            for _ in range(self.num_layers)
        ]
        self._cross_k = [
            torch.zeros((1, self.num_heads, self.src_len, self.head_dim), device=device, dtype=dtype)
            for _ in range(self.num_layers)
        ]
        self._cross_v = [
            torch.zeros((1, self.num_heads, self.src_len, self.head_dim), device=device, dtype=dtype)
            for _ in range(self.num_layers)
        ]
        self._arange = torch.arange(self.max_cache_len, device=device, dtype=torch.long)
        self._cur_token = torch.zeros((1, 1), device=device, dtype=torch.long)
        self._pos_id = torch.zeros((1,), device=device, dtype=torch.long)
        self._out_token = torch.zeros((), device=device, dtype=torch.long)

        self._prefill_graph: torch.cuda.CUDAGraph | None = None
        self._step_graph: torch.cuda.CUDAGraph | None = None
        self._capture_graphs()

    def _build_token_mask(self, token_ids: list[int] | tuple[int, ...] | None) -> torch.Tensor | None:
        if not token_ids:
            return None
        mask = torch.zeros(self.vocab_size, device=self.device, dtype=torch.bool)
        valid = [int(t) for t in token_ids if 0 <= int(t) < self.vocab_size]
        if not valid:
            return None
        mask[torch.tensor(valid, device=self.device, dtype=torch.long)] = True
        return mask

    def _project_self_qkv(
        self, attn: nn.Module, h: torch.Tensor, seq_len: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if hasattr(attn, "qkv_proj"):
            q, k, v = attn.qkv_proj(h).chunk(3, dim=-1)
            if not getattr(attn, "_q_scaled", False):
                q = q * attn.scaling
        else:
            q = attn.q_proj(h)
            if not getattr(attn, "_q_scaled", False):
                q = q * attn.scaling
            k = attn.k_proj(h)
            v = attn.v_proj(h)
        q = q.view(1, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(1, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(1, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        return q, k, v

    def _project_cross_kv(
        self, enc_attn: nn.Module, enc_hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if hasattr(enc_attn, "kv_proj"):
            k, v = enc_attn.kv_proj(enc_hidden).chunk(2, dim=-1)
        else:
            k = enc_attn.k_proj(enc_hidden)
            v = enc_attn.v_proj(enc_hidden)
        k = k.view(1, self.src_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(1, self.src_len, self.num_heads, self.head_dim).transpose(1, 2)
        return k, v

    def _project_cross_q(self, enc_attn: nn.Module, h: torch.Tensor, seq_len: int) -> torch.Tensor:
        q = enc_attn.q_proj(h)
        if not getattr(enc_attn, "_q_scaled", False):
            q = q * enc_attn.scaling
        return q.view(1, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def _prefill_fn(self) -> None:
        h = self.decoder.embed_tokens(self._static_prefix_ids) + self._static_prefix_pos
        p_len = self.prefix_len
        for idx, layer in enumerate(self.decoder.layers):
            residual = h
            normed = layer.self_attn_layer_norm(h)
            q, k, v = self._project_self_qkv(layer.self_attn, normed, p_len)
            self._self_k[idx][:, :, :p_len, :].copy_(k)
            self._self_v[idx][:, :, :p_len, :].copy_(v)
            attn_out = F.scaled_dot_product_attention(
                q, k, v, dropout_p=0.0, is_causal=True, scale=1.0
            )
            attn_out = attn_out.transpose(1, 2).contiguous().view(1, p_len, self.embed_dim)
            h = residual + layer.self_attn.out_proj(attn_out)

            residual = h
            normed = layer.encoder_attn_layer_norm(h)
            ck, cv = self._project_cross_kv(layer.encoder_attn, self._static_enc_hidden)
            self._cross_k[idx].copy_(ck)
            self._cross_v[idx].copy_(cv)
            cq = self._project_cross_q(layer.encoder_attn, normed, p_len)
            cross_out = F.scaled_dot_product_attention(
                cq,
                self._cross_k[idx],
                self._cross_v[idx],
                dropout_p=0.0,
                is_causal=False,
                scale=1.0,
            )
            cross_out = cross_out.transpose(1, 2).contiguous().view(1, p_len, self.embed_dim)
            h = residual + layer.encoder_attn.out_proj(cross_out)

            residual = h
            normed = layer.final_layer_norm(h)
            h = residual + layer.fc2(layer.activation_fn(layer.fc1(normed)))

        last_h = self.decoder.layer_norm(h[:, -1:, :])
        logits = self.proj_out(last_h).view(-1)
        if self._suppress_mask is not None:
            logits = logits.masked_fill(self._suppress_mask, float("-inf"))
        if self._begin_suppress_mask is not None:
            logits = logits.masked_fill(self._begin_suppress_mask, float("-inf"))
        first_tok = logits.argmax(dim=-1)
        self._cur_token.copy_(first_tok.view(1, 1))
        self._out_token.copy_(first_tok)
        self._pos_id.fill_(p_len)

    def _step_fn(self) -> None:
        pos_embed = F.embedding(self._pos_id, self.decoder.embed_positions.weight).unsqueeze(0)
        h = self.decoder.embed_tokens(self._cur_token) + pos_embed
        attn_mask = (self._arange <= self._pos_id).view(1, 1, 1, -1)

        for idx, layer in enumerate(self.decoder.layers):
            residual = h
            normed = layer.self_attn_layer_norm(h)
            q, k, v = self._project_self_qkv(layer.self_attn, normed, 1)
            self._self_k[idx].index_copy_(2, self._pos_id, k)
            self._self_v[idx].index_copy_(2, self._pos_id, v)
            attn_out = F.scaled_dot_product_attention(
                q,
                self._self_k[idx],
                self._self_v[idx],
                attn_mask=attn_mask,
                dropout_p=0.0,
                is_causal=False,
                scale=1.0,
            )
            attn_out = attn_out.transpose(1, 2).reshape(1, 1, self.embed_dim)
            h = residual + layer.self_attn.out_proj(attn_out)

            residual = h
            normed = layer.encoder_attn_layer_norm(h)
            cq = self._project_cross_q(layer.encoder_attn, normed, 1)
            cross_out = F.scaled_dot_product_attention(
                cq,
                self._cross_k[idx],
                self._cross_v[idx],
                dropout_p=0.0,
                is_causal=False,
                scale=1.0,
            )
            cross_out = cross_out.transpose(1, 2).reshape(1, 1, self.embed_dim)
            h = residual + layer.encoder_attn.out_proj(cross_out)

            residual = h
            normed = layer.final_layer_norm(h)
            h = residual + layer.fc2(layer.activation_fn(layer.fc1(normed)))

        h = self.decoder.layer_norm(h)
        logits = self.proj_out(h).view(-1)
        if self._suppress_mask is not None:
            logits = logits.masked_fill(self._suppress_mask, float("-inf"))
        next_tok = logits.argmax(dim=-1)
        self._cur_token.copy_(next_tok.view(1, 1))
        self._out_token.copy_(next_tok)
        self._pos_id.add_(1)

    def _capture_graphs(self) -> None:
        with CUDA_GRAPH_LOCK:
            torch.cuda.synchronize()
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.inference_mode(), torch.cuda.stream(stream):
                for _ in range(3):
                    self._prefill_fn()
                for _ in range(3):
                    self._pos_id.fill_(self.prefix_len)
                    self._step_fn()
            torch.cuda.current_stream().wait_stream(stream)

            prefill_graph = torch.cuda.CUDAGraph()
            with torch.inference_mode(), torch.cuda.graph(prefill_graph):
                self._prefill_fn()

            self._pos_id.fill_(self.prefix_len)
            step_graph = torch.cuda.CUDAGraph()
            with torch.inference_mode(), torch.cuda.graph(step_graph):
                self._step_fn()

            torch.cuda.synchronize()

        self._prefill_graph = prefill_graph
        self._step_graph = step_graph
        for k_buf, v_buf in zip(self._self_k, self._self_v, strict=True):
            k_buf.zero_()
            v_buf.zero_()

    @torch.inference_mode()
    def decode(self, encoder_hidden_states: torch.Tensor, max_new_tokens: int = 160) -> list[int]:
        assert self._prefill_graph is not None and self._step_graph is not None
        max_steps = min(int(max_new_tokens), self.max_cache_len - self.prefix_len)
        if max_steps <= 0:
            return []

        self._static_enc_hidden.copy_(encoder_hidden_states)
        self._prefill_graph.replay()
        tok = int(self._out_token.item())
        if tok == self.eos_token_id:
            return []

        tokens = [tok]
        for _ in range(max_steps - 1):
            self._step_graph.replay()
            tok = int(self._out_token.item())
            if tok == self.eos_token_id:
                break
            tokens.append(tok)
        return tokens


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

        # Pre-allocate exact PyTorch GPU log-mel filterbank matching WhisperFeatureExtractor
        fe = getattr(self.processor, "feature_extractor", None)
        self._mel_n_fft = int(getattr(fe, "n_fft", 400))
        self._mel_hop_length = int(getattr(fe, "hop_length", 160))
        self._mel_n_samples = int(getattr(fe, "n_samples", 480000))
        self._mel_dither = float(getattr(fe, "dither", 0.0))
        self._mel_window: torch.Tensor | None = None
        self._mel_filters_t: torch.Tensor | None = None
        self._waveform_buf: torch.Tensor | None = None
        mel_filters = getattr(fe, "mel_filters", None)
        if mel_filters is not None:
            self._mel_window = torch.hann_window(
                self._mel_n_fft, device=self.device, dtype=torch.float32
            )
            self._mel_filters_t = (
                torch.as_tensor(mel_filters, dtype=torch.float32, device=self.device)
                .T.contiguous()
            )
            self._waveform_buf = torch.zeros(
                (1, self._mel_n_samples), device=self.device, dtype=torch.float32
            )

        self._prompt_ids: torch.Tensor | None = None
        if domain_prompt:
            try:
                ids = self.processor.get_prompt_ids(domain_prompt, return_tensors="pt")
                self._prompt_ids = ids.to(self.device)
            except Exception:
                logger.debug("Could not build Whisper domain prompt IDs", exc_info=True)
        self._decoder_prefix_ids = self._build_decoder_prefix_ids()

        self.fp8_enabled = False
        self.encoder_graphed = False
        self.decoder_graphed = False
        self._fast_decoder: _FastWhisperDecoder | None = None
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

        if self.device.startswith("cuda") and torch.cuda.is_available():
            try:
                self._fast_decoder = _FastWhisperDecoder(
                    self.model,
                    self._decoder_prefix_ids,
                    device=self.device,
                    dtype=self.dtype,
                    max_new_tokens=160,
                    suppress_tokens=getattr(gen_cfg, "suppress_tokens", None),
                    begin_suppress_tokens=getattr(gen_cfg, "begin_suppress_tokens", None),
                )
                self.decoder_graphed = True
            except (RuntimeError, ValueError, TypeError, AttributeError, OSError) as exc:
                logger.debug("Whisper CUDA-graph decoder skipped: %s", exc)
                self._fast_decoder = None

        logger.info(
            "Loaded Whisper STT (%s) on %s (%s%s%s) in %.2fs",
            model_id,
            self.device,
            self.dtype,
            " + FP8" if self.fp8_enabled else "",
            " + CG-Decoder" if self.decoder_graphed else "",
            time.perf_counter() - t0,
        )

    def _build_decoder_prefix_ids(self) -> torch.Tensor:
        """Build static [prompt_ids?, <|startoftranscript|>, <|en|>, <|transcribe|>, <|notimestamps|>] prefix."""
        gen_cfg = self.model.generation_config
        init_tokens = [int(gen_cfg.decoder_start_token_id)]
        if getattr(gen_cfg, "is_multilingual", True):
            lang_to_id = getattr(gen_cfg, "lang_to_id", None)
            if isinstance(lang_to_id, dict) and "<|en|>" in lang_to_id:
                init_tokens.append(int(lang_to_id["<|en|>"]))
            task_to_id = getattr(gen_cfg, "task_to_id", None)
            if isinstance(task_to_id, dict) and "transcribe" in task_to_id:
                init_tokens.append(int(task_to_id["transcribe"]))
        no_ts_id = getattr(gen_cfg, "no_timestamps_token_id", None)
        if no_ts_id is not None:
            init_tokens.append(int(no_ts_id))

        init_t = torch.tensor(init_tokens, device=self.device, dtype=torch.long)
        if self._prompt_ids is not None:
            prompt_1d = self._prompt_ids.to(device=self.device, dtype=torch.long).reshape(-1)
            prefix = torch.cat([prompt_1d, init_t], dim=0)
        else:
            prefix = init_t
        return prefix.unsqueeze(0).contiguous()

    def _make_greedy_gen_config(self, max_new_tokens: int) -> GenerationConfig:
        """Build a clean GenerationConfig that avoids WhisperGenerationMixin deprecation warnings."""
        cfg = copy.deepcopy(self.model.generation_config)
        max_pos = int(getattr(self.model.config, "max_target_positions", 448))
        cfg.do_sample = False
        cfg.num_beams = 1
        cfg.use_cache = True
        cfg.max_new_tokens = None
        cfg.max_length = min(max_pos, int(self._decoder_prefix_ids.shape[1]) + int(max_new_tokens))
        cfg.forced_decoder_ids = None
        cfg.suppress_tokens = None
        cfg.begin_suppress_tokens = None
        return cfg

    def _extract_input_features(self, audio_16k: np.ndarray) -> torch.Tensor:
        """Compute exact Whisper log-mel spectrogram directly on target device (zero CPU STFT)."""
        if (
            self._mel_window is None
            or self._mel_filters_t is None
            or self._waveform_buf is None
        ):
            inputs = self.processor(
                audio_16k,
                sampling_rate=16000,
                return_tensors="pt",
                return_attention_mask=False,
            )
            return inputs.input_features.to(device=self.device, dtype=self.dtype)

        n = min(int(audio_16k.size), self._mel_n_samples)
        wav = self._waveform_buf
        wav.zero_()
        if n > 0:
            src = torch.from_numpy(np.ascontiguousarray(audio_16k[:n], dtype=np.float32))
            wav[0, :n].copy_(src)
        if self._mel_dither != 0.0:
            wav = wav + self._mel_dither * torch.randn_like(wav)

        stft = torch.stft(
            wav,
            n_fft=self._mel_n_fft,
            hop_length=self._mel_hop_length,
            window=self._mel_window,
            return_complex=True,
        )
        stft_trim = stft[..., :-1]
        magnitudes = stft_trim.real.square() + stft_trim.imag.square()
        mel_spec = torch.matmul(self._mel_filters_t, magnitudes)
        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        log_spec = torch.maximum(log_spec, log_spec.amax() - 8.0)
        log_spec = (log_spec + 4.0) * 0.25
        return log_spec.to(dtype=self.dtype)

    def _generate_greedy(self, input_features: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """Run fast CUDA-graphed greedy decoding with clean GenerationMixin fallback."""
        if self._fast_decoder is not None:
            inner = getattr(self.model, "model", self.model)
            enc_hidden = inner.encoder(input_features).last_hidden_state
            tokens = self._fast_decoder.decode(enc_hidden, max_new_tokens=max_new_tokens)
            return torch.tensor([tokens], dtype=torch.long)

        prefix_ids = self._decoder_prefix_ids
        gen_cfg = self._make_greedy_gen_config(max_new_tokens)
        sequences = GenerationMixin.generate(
            self.model,
            inputs=input_features,
            generation_config=gen_cfg,
            decoder_input_ids=prefix_ids,
        )
        return sequences[:, prefix_ids.shape[1] :]

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
        input_features = self._extract_input_features(probe)
        _ = self._generate_greedy(input_features, max_new_tokens=64)
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

        input_features = self._extract_input_features(arr)
        predicted_ids = self._generate_greedy(input_features, max_new_tokens=160)
        text = self.processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return text.strip()
