"""VoxCPM2 HAL 9000 voice-cloning TTS engine with Blackwell Mega-Graph, INT4/FP8, and stream-overlapped AudioVAE."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
import types
from collections.abc import Generator
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from huggingface_hub.errors import LocalEntryNotFoundError
from torch import nn
from torch.nn.utils import remove_weight_norm
from voxcpm import VoxCPM

from ._cuda_ops import (
    CUDA_GRAPH_LOCK,
    TRITON_AVAILABLE,
    CUDAGraphModule,
    FP8Linear,
    Int4Linear,
    capture_cuda_graph,
    probe_fp8_supported,
)

if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

logger = logging.getLogger(__name__)

DEFAULT_PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_REF_WAV = DEFAULT_PROJECT_DIR / "voices" / "hal9000_ref.wav"
DEFAULT_REF_TXT = DEFAULT_PROJECT_DIR / "voices" / "hal9000_ref.txt"
DEFAULT_LORA_DIR = DEFAULT_PROJECT_DIR / "voices" / "hal9000_lora"
MIN_DIFFUSION_STEPS = 2
_FAST_KV_MAX_LEN = 1024

# Preserve backward compatibility for scripts importing _FP8Linear from hal9000.tts
_FP8Linear = FP8Linear

_PRONUNCIATION_SUBS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bHAL[\s-]*9000\b", re.IGNORECASE), "Hal nine thousand"),
    (re.compile(r"\bSAL[\s-]*9000\b", re.IGNORECASE), "Sal nine thousand"),
    (re.compile(r"\bH\.A\.L\.?(?=\s|$|[,.;:!?])", re.IGNORECASE), "Hal"),
    (re.compile(r"\bS\.A\.L\.?(?=\s|$|[,.;:!?])", re.IGNORECASE), "Sal"),
    (re.compile(r"\bHAL\b"), "Hal"),
    (re.compile(r"\bSAL\b"), "Sal"),
    (re.compile(r"\b9000\b"), "nine thousand"),
    (re.compile(r"\b2001\b"), "two thousand and one"),
    (re.compile(r"\b2010\b"), "two thousand and ten"),
    (re.compile(r"\b2061\b"), "twenty sixty-one"),
    (re.compile(r"\b3001\b"), "three thousand and one"),
    (re.compile(r"\bAE[\s-]*35\b", re.IGNORECASE), "A E thirty-five"),
    (re.compile(r"\bTMA[\s-]*1\b", re.IGNORECASE), "T M A one"),
    (re.compile(r"\bNSC[\s-]*342/23\b", re.IGNORECASE), "N S C three forty-two twenty-three"),
    (re.compile(r"\bEVA\b"), "E V A"),
    (re.compile(r"\bBBC\b"), "B B C"),
    (re.compile(r"\b1:4:9\b"), "one by four by nine"),
    (re.compile(r"\b100%"), "a hundred percent"),
    (re.compile(r"(\d+)%"), r"\1 percent"),
    (re.compile(r"\bJanuary\s+12,?\s+1992\b", re.IGNORECASE), "the twelfth of January nineteen ninety two"),
    (re.compile(r"\bMr\.\s*Amer\b", re.IGNORECASE), "Mister Amor"),
    (re.compile(r"\bDrs\.\s*"), "Doctors "),
    (re.compile(r"\bDr\.\s*"), "Doctor "),
    (re.compile(r"\bMr\.\s*"), "Mister "),
    (re.compile(r"\bMrs\.\s*"), "Missus "),
    (re.compile(r"\bMs\.\s*"), "Mizz "),
    (re.compile(r"\bProf\.\s*"), "Professor "),
)


def normalize_for_hal_tts(text: str) -> str:
    """Expand Space Odyssey names, abbreviations, and symbols to match Douglas Rain's diction."""
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


def _resolve_lora_dir() -> Path | None:
    env_path = os.environ.get("HAL_LORA_DIR")
    if env_path:
        p = Path(env_path)
        return p if p.is_dir() else None
    if (DEFAULT_LORA_DIR / "lora_weights.safetensors").is_file() or (
        DEFAULT_LORA_DIR / "lora_weights.ckpt"
    ).is_file():
        return DEFAULT_LORA_DIR
    return None


def _merge_lora_into_state_dict(
    model_sd: dict[str, torch.Tensor], lora_dir: Path, device: str, lm_dtype: torch.dtype
) -> int:
    """Permanently merge trained VoxCPM2 LoRA weights into base weights (0.0ms inference overhead)."""
    from safetensors.torch import load_file

    sf_path = lora_dir / "lora_weights.safetensors"
    ckpt_path = lora_dir / "lora_weights.ckpt"
    cfg_path = lora_dir / "lora_config.json"

    if sf_path.is_file():
        lora_sd = load_file(str(sf_path), device=device)
    elif ckpt_path.is_file():
        raw = torch.load(str(ckpt_path), map_location=device, weights_only=True)
        lora_sd = raw.get("state_dict", raw)
    else:
        return 0

    alpha = 32.0
    r = 32
    if cfg_path.is_file():
        try:
            meta = json.loads(cfg_path.read_text(encoding="utf-8"))
            lora_cfg = meta.get("lora_config", meta)
            alpha = float(lora_cfg.get("alpha", alpha))
            r = int(lora_cfg.get("r", r))
        except (OSError, ValueError, TypeError, KeyError) as exc:
            logger.debug("Using default LoRA rank/alpha for %s: %s", cfg_path, exc)
    scaling = alpha / max(r, 1)

    merged = 0
    for key in list(lora_sd.keys()):
        if not key.endswith(".lora_A"):
            continue
        prefix = key[: -len(".lora_A")]
        key_b = f"{prefix}.lora_B"
        weight_key = f"{prefix}.weight"
        if key_b not in lora_sd or weight_key not in model_sd:
            continue
        lora_a = lora_sd[key].float()
        lora_b = lora_sd[key_b].float()
        delta = (lora_b @ lora_a) * scaling
        base = model_sd[weight_key].float()
        if delta.shape == base.shape:
            model_sd[weight_key] = (base + delta).to(lm_dtype)
            merged += 1
    if merged:
        logger.info("Merged %d HAL 9000 LoRA matrices from %s (scale=%.2f)", merged, lora_dir, scaling)
    return merged


def _prompt_cache_file(model_id: str, ref_wav: Path, ref_text: str | None) -> Path:
    cache_root = Path(os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache")) / "ioghal"
    st = ref_wav.stat()
    key_material = json.dumps(
        {
            "v": 1,
            "model_id": model_id,
            "ref_wav": str(ref_wav.resolve()),
            "size": st.st_size,
            "mtime_ns": st.st_mtime_ns,
            "ref_text": ref_text or "",
            "trim_silence_vad": True,
        },
        sort_keys=True,
    ).encode("utf-8")
    digest = hashlib.sha256(key_material).hexdigest()[:16]
    return cache_root / f"prompt_cache_{digest}.pt"


def _fuse_minicpm_model(
    model: nn.Module,
    *,
    use_int4: bool = False,
    arange_buf: torch.Tensor | None = None,
) -> None:
    """Fuse QKV and Gate/Up projections across a MiniCPMModel using hybrid INT4/FP8."""
    from voxcpm.modules.minicpm4.model import apply_rotary_pos_emb

    def _make_linear(w: torch.Tensor, allow_int4: bool) -> nn.Module:
        if allow_int4 and use_int4 and TRITON_AVAILABLE and w.shape[1] % 128 == 0:
            return Int4Linear(w, group_size=128)
        return FP8Linear(w)

    for layer in model.layers:
        attn = layer.self_attn
        qkv_w = torch.cat(
            [attn.q_proj.weight, attn.k_proj.weight, attn.v_proj.weight], dim=0
        ).contiguous()
        attn.qkv_proj = _make_linear(qkv_w, allow_int4=True)
        attn.o_proj = _make_linear(attn.o_proj.weight, allow_int4=True)
        del attn.q_proj, attn.k_proj, attn.v_proj
        attn._q_dim = attn.num_heads * attn.head_dim
        attn._kv_dim = attn.num_key_value_heads * attn.head_dim

        def fused_forward(self, hidden_states, position_emb, is_causal, norm_weight=None, norm_eps=1e-6):
            bsz, q_len, _ = hidden_states.size()
            if norm_weight is not None:
                qkv = self.qkv_proj.forward_rmsnorm(hidden_states, norm_weight, norm_eps)
            else:
                qkv = self.qkv_proj(hidden_states)
            q, k, v = qkv.split([self._q_dim, self._kv_dim, self._kv_dim], dim=-1)
            q = q.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            k = k.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            v = v.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            if position_emb is not None:
                cos, sin = position_emb
                q, k = apply_rotary_pos_emb(q, k, cos, sin)
            attn_out = F.scaled_dot_product_attention(
                q, k, v, is_causal=is_causal, enable_gqa=True
            )
            attn_out = attn_out.transpose(1, 2).contiguous().reshape(bsz, q_len, self._q_dim)
            return self.o_proj(attn_out), (k, v)

        attn.forward = types.MethodType(fused_forward, attn)

        if arange_buf is not None:
            def fused_step(
                self,
                hidden_states,
                position_emb,
                position_id,
                kv_cache,
                norm_weight=None,
                norm_eps=1e-6,
                _arange=arange_buf,
            ):
                bsz, _ = hidden_states.size()
                if norm_weight is not None:
                    qkv = self.qkv_proj.forward_rmsnorm(hidden_states, norm_weight, norm_eps)
                else:
                    qkv = self.qkv_proj(hidden_states)
                q, k, v = qkv.split([self._q_dim, self._kv_dim, self._kv_dim], dim=-1)
                q = q.view(bsz, 1, self.num_heads, self.head_dim).transpose(1, 2)
                k = k.view(bsz, 1, self.num_key_value_heads, self.head_dim).transpose(1, 2)
                v = v.view(bsz, 1, self.num_key_value_heads, self.head_dim).transpose(1, 2)
                if position_emb is not None:
                    cos, sin = position_emb
                    q, k = apply_rotary_pos_emb(q, k, cos, sin)
                key_cache, value_cache = kv_cache
                key_cache[:, :, position_id, :] = k
                value_cache[:, :, position_id, :] = v
                attn_mask = (_arange <= position_id).view(1, 1, 1, -1)
                attn_out = F.scaled_dot_product_attention(
                    q, key_cache, value_cache, attn_mask=attn_mask, enable_gqa=True
                )
                attn_out = attn_out.transpose(1, 2).reshape(bsz, self._q_dim)
                return self.o_proj(attn_out)

            attn.forward_step = types.MethodType(fused_step, attn)

        mlp = layer.mlp
        gu_w = torch.cat([mlp.gate_proj.weight, mlp.up_proj.weight], dim=0).contiguous()
        # Gate+Up (12288x2048) is 2.4x faster in INT4; Down (2048x6144) is faster & sharper in FP8
        mlp.gate_up_proj = _make_linear(gu_w, allow_int4=True)
        mlp.down_proj = FP8Linear(mlp.down_proj.weight)
        del mlp.gate_proj, mlp.up_proj

        qkv_is_fp8 = isinstance(attn.qkv_proj, FP8Linear)
        gu_is_fp8 = isinstance(mlp.gate_up_proj, FP8Linear)
        down_is_fp8 = isinstance(mlp.down_proj, FP8Linear)

        if down_is_fp8:
            def fast_mlp(self, x):
                return self.down_proj.forward_silu_mul(self.gate_up_proj(x))
        else:
            def fast_mlp(self, x):
                g, u = self.gate_up_proj(x).chunk(2, dim=-1)
                return self.down_proj(self.act_fn(g) * u)

        mlp.forward = types.MethodType(fast_mlp, mlp)

        if gu_is_fp8 and down_is_fp8:
            def fast_mlp_rmsnorm(self, x, norm_weight, norm_eps):
                return self.down_proj.forward_silu_mul(
                    self.gate_up_proj.forward_rmsnorm(x, norm_weight, norm_eps)
                )

            mlp.forward_rmsnorm = types.MethodType(fast_mlp_rmsnorm, mlp)

        if qkv_is_fp8 or (gu_is_fp8 and down_is_fp8):
            mup_scale = (
                layer.scale_depth / (layer.num_hidden_layers**0.5)
                if layer.use_mup
                else None
            )
            in_eps = float(layer.input_layernorm.variance_epsilon)
            post_eps = float(layer.post_attention_layernorm.variance_epsilon)
            fuse_in_norm = qkv_is_fp8
            fuse_post_norm = gu_is_fp8 and down_is_fp8

            def fused_layer_forward(
                self,
                hidden_states,
                position_emb,
                is_causal,
                _mup=mup_scale,
                _in_eps=in_eps,
                _post_eps=post_eps,
                _fuse_in=fuse_in_norm,
                _fuse_post=fuse_post_norm,
            ):
                residual = hidden_states
                if _fuse_in:
                    attn_out, present_kv = self.self_attn(
                        hidden_states,
                        position_emb,
                        is_causal,
                        norm_weight=self.input_layernorm.weight,
                        norm_eps=_in_eps,
                    )
                else:
                    attn_out, present_kv = self.self_attn(
                        self.input_layernorm(hidden_states),
                        position_emb,
                        is_causal,
                    )
                hidden_states = residual + (attn_out * _mup if _mup is not None else attn_out)

                residual = hidden_states
                if _fuse_post:
                    mlp_out = self.mlp.forward_rmsnorm(
                        hidden_states, self.post_attention_layernorm.weight, _post_eps
                    )
                else:
                    mlp_out = self.mlp(self.post_attention_layernorm(hidden_states))
                hidden_states = residual + (mlp_out * _mup if _mup is not None else mlp_out)
                return hidden_states, present_kv

            layer.forward = types.MethodType(fused_layer_forward, layer)

            if arange_buf is not None:
                def fused_layer_step(
                    self,
                    hidden_states,
                    position_emb,
                    position_id,
                    kv_cache,
                    _mup=mup_scale,
                    _in_eps=in_eps,
                    _post_eps=post_eps,
                    _fuse_in=fuse_in_norm,
                    _fuse_post=fuse_post_norm,
                ):
                    residual = hidden_states
                    if _fuse_in:
                        attn_out = self.self_attn.forward_step(
                            hidden_states,
                            position_emb,
                            position_id,
                            kv_cache,
                            norm_weight=self.input_layernorm.weight,
                            norm_eps=_in_eps,
                        )
                    else:
                        attn_out = self.self_attn.forward_step(
                            self.input_layernorm(hidden_states),
                            position_emb,
                            position_id,
                            kv_cache,
                        )
                    hidden_states = residual + (attn_out * _mup if _mup is not None else attn_out)

                    residual = hidden_states
                    if _fuse_post:
                        mlp_out = self.mlp.forward_rmsnorm(
                            hidden_states, self.post_attention_layernorm.weight, _post_eps
                        )
                    else:
                        mlp_out = self.mlp(self.post_attention_layernorm(hidden_states))
                    hidden_states = residual + (mlp_out * _mup if _mup is not None else mlp_out)
                    return hidden_states

                layer.forward_step = types.MethodType(fused_layer_step, layer)


class _FastAudioVAEGraph:
    """Lossless FP32 static rolling-buffer CUDA Graph for AudioVAEV2 streaming decode_chunk."""

    def __init__(self, audio_vae, patch_size: int = 4) -> None:
        self.vae = audio_vae
        self.patch_size = patch_size
        self.device = next(audio_vae.decoder.parameters()).device
        self.dtype = next(audio_vae.decoder.parameters()).dtype
        self._state_bufs: list[torch.Tensor] = []
        self._keepalive: list[torch.Tensor] = []
        self._originals: list[tuple[nn.Module, object]] = []
        self._sr_cond = torch.tensor([audio_vae.out_sample_rate], device=self.device, dtype=torch.int32)
        self._static_z = torch.zeros((1, audio_vae.latent_dim, patch_size), device=self.device, dtype=self.dtype)
        self._graph: torch.cuda.CUDAGraph | None = None
        self._static_out: torch.Tensor | None = None

        self._build()

    def _build(self) -> None:
        from voxcpm.modules.audiovae.audio_vae_v2 import (
            CausalConv1d,
            CausalTransposeConv1d,
            SampleRateConditionLayer,
            Snake1d,
        )

        from ._cuda_ops import snake1d_fp32

        sr_idx = (
            self.vae.decoder.get_sr_idx(self._sr_cond)
            if getattr(self.vae.decoder, "sr_bin_boundaries", None) is not None
            else None
        )

        for mod in self.vae.decoder.modules():
            if isinstance(mod, CausalConv1d):
                pad = mod._CausalConv1d__padding * 2 - mod._CausalConv1d__output_padding
                if pad > 0:
                    orig = mod.forward
                    buf = torch.zeros((1, mod.in_channels, pad), device=self.device, dtype=self.dtype)
                    self._state_bufs.append(buf)

                    def _make_conv(_m, _b, _p):
                        def fwd(x):
                            x_pad = torch.cat([_b, x], dim=-1)
                            if x.shape[-1] >= _p:
                                _b.copy_(x[:, :, -_p:])
                            else:
                                _b.copy_(x_pad[:, :, -_p:])
                            return nn.Conv1d.forward(_m, x_pad)
                        return fwd

                    mod.forward = _make_conv(mod, buf, pad)
                    self._originals.append((mod, orig))
            elif isinstance(mod, CausalTransposeConv1d):
                trim = mod._CausalTransposeConv1d__padding * 2 - mod._CausalTransposeConv1d__output_padding
                ctx = (mod.kernel_size[0] - 1) // mod.stride[0]
                if ctx > 0:
                    orig = mod.forward
                    buf = torch.zeros((1, mod.in_channels, ctx), device=self.device, dtype=self.dtype)
                    self._state_bufs.append(buf)

                    def _make_tconv(_m, _b, _c, _t):
                        def fwd(x):
                            x_full = torch.cat([_b, x], dim=-1)
                            _b.copy_(x[:, :, -_c:])
                            out = nn.ConvTranspose1d.forward(_m, x_full)
                            left = _c * _m.stride[0]
                            return out[..., left:-_t] if _t > 0 else out[..., left:]
                        return fwd

                    mod.forward = _make_tconv(mod, buf, ctx, trim)
                    self._originals.append((mod, orig))
            elif isinstance(mod, Snake1d) and TRITON_AVAILABLE and self.dtype == torch.float32:
                orig = mod.forward
                alpha_1d = mod.alpha.detach().reshape(-1).to(device=self.device, dtype=torch.float32).contiguous()
                inv_alpha_1d = (alpha_1d + 1e-9).reciprocal().contiguous()
                self._keepalive.extend((alpha_1d, inv_alpha_1d))

                def _make_snake(_a, _ia):
                    def fwd(x):
                        return snake1d_fp32(x, _a, _ia)
                    return fwd

                mod.forward = _make_snake(alpha_1d, inv_alpha_1d)
                self._originals.append((mod, orig))
            elif (
                isinstance(mod, SampleRateConditionLayer)
                and sr_idx is not None
                and mod.cond_type in ("scale_bias", "scale_bias_init")
                and isinstance(mod.out_layer, nn.Identity)
            ):
                orig = mod.forward
                with torch.no_grad():
                    scale = mod.scale_embed(sr_idx).unsqueeze(-1).contiguous()
                    bias = mod.bias_embed(sr_idx).unsqueeze(-1).contiguous()
                self._keepalive.extend((scale, bias))

                def _make_sr_cond(_s, _b):
                    def fwd(x, _sr_cond=None):
                        return x * _s + _b
                    return fwd

                mod.forward = _make_sr_cond(scale, bias)
                self._originals.append((mod, orig))

        with CUDA_GRAPH_LOCK:
            torch.cuda.synchronize()
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.inference_mode(), torch.cuda.stream(stream):
                for _ in range(3):
                    _ = self.vae.decoder(self._static_z, self._sr_cond)
            torch.cuda.current_stream().wait_stream(stream)
            self.reset()

            graph = torch.cuda.CUDAGraph()
            with torch.inference_mode(), torch.cuda.graph(graph):
                self._static_out = self.vae.decoder(self._static_z, self._sr_cond)
            torch.cuda.synchronize()
        self._graph = graph
        self.reset()

        # Restore original forwards so non-streaming .decode() still works cleanly!
        for mod, orig in self._originals:
            mod.forward = orig
        self._originals.clear()

    def reset(self) -> None:
        for b in self._state_bufs:
            b.zero_()

    def decode_chunk(self, z_chunk: torch.Tensor) -> torch.Tensor:
        assert self._graph is not None and self._static_out is not None
        self._static_z.copy_(z_chunk)
        self._graph.replay()
        return self._static_out


class _MegaStepRunner:
    """Captures the ENTIRE VoxCPM2 autoregressive step (DiT Euler loop + feat_enc + stop + base_lm + res_lm) into 1 CUDA Graph per (steps, cfg)."""

    def __init__(self, tts_model) -> None:
        self.m = tts_model
        self.device = tts_model.device
        from voxcpm.model.utils import get_dtype
        self.dtype = get_dtype(tts_model.config.dtype)
        self._graphs: dict[tuple[int, float], dict] = {}
        self._prefix_kv_cache: dict[int, dict] = {}
        self._vae_graph = _FastAudioVAEGraph(tts_model.audio_vae, patch_size=tts_model.patch_size)
        self._vae_stream = torch.cuda.Stream()
        # Double-buffered pinned CPU staging for non-blocking D2H audio transfer
        chunk_samples = tts_model.patch_size * tts_model._decode_chunk_size
        self._pinned_bufs = [
            torch.empty(chunk_samples, dtype=torch.float32, pin_memory=True),
            torch.empty(chunk_samples, dtype=torch.float32, pin_memory=True),
        ]
        self._latent_staging = [
            torch.empty((1, tts_model.feat_dim, tts_model.patch_size), device=self.device, dtype=torch.float32),
            torch.empty((1, tts_model.feat_dim, tts_model.patch_size), device=self.device, dtype=torch.float32),
        ]
        self._stop_staging = [
            torch.empty((), dtype=torch.int64, pin_memory=True),
            torch.empty((), dtype=torch.int64, pin_memory=True),
        ]
        self._latent_ready_events = [torch.cuda.Event(), torch.cuda.Event()]
        self._vae_done_events = [torch.cuda.Event(), torch.cuda.Event()]

    def _get_or_capture_graph(self, n_timesteps: int, cfg_value: float) -> dict:
        key = (int(n_timesteps), round(float(cfg_value), 3))
        if key in self._graphs:
            return self._graphs[key]

        m = self.m
        device = self.device
        dtype = self.dtype
        h_lm = int(m.config.lm_config.hidden_size)
        h_dit = int(m.config.dit_config.hidden_dim)
        p = m.patch_size
        d = m.feat_dim

        # Static inputs/state updated in-place across autoregressive steps (shared buffer for batched DiT proj)
        lm_res_hidden = torch.zeros((2, 1, h_lm), device=device, dtype=dtype)
        lm_hidden = lm_res_hidden[0]
        res_hidden = lm_res_hidden[1]
        prefix_cond = torch.zeros((1, p, d), device=device, dtype=dtype)
        noise_z = torch.zeros((1, d, p), device=device, dtype=dtype)
        pos_id = torch.zeros((1,), device=device, dtype=torch.long)

        # Precompute fixed Euler schedule & CFG zero buffers
        t_span = torch.linspace(1, 0, n_timesteps + 1, device=device, dtype=dtype)
        t_span = t_span + 1.0 * (torch.cos(torch.pi / 2 * t_span) - 1 + t_span)
        zero_init_steps = max(1, int(len(t_span) * 0.04))
        cfm = m.feat_decoder
        estimator = getattr(m, "_dit_estimator_raw", cfm.estimator)
        feat_enc = getattr(m, "_feat_encoder_raw", m.feat_encoder)
        base_step = getattr(m.base_lm, "_forward_step_raw", m.base_lm.forward_step)
        res_step = getattr(m.residual_lm, "_forward_step_raw", m.residual_lm.forward_step)

        # Pre-pack lm_to_dit_proj + res_to_dit_proj into 1 exact pure-BF16 batched GEMM writing directly into mu_in[0]
        fuse_dit_proj = isinstance(m.lm_to_dit_proj, nn.Linear) and isinstance(m.res_to_dit_proj, nn.Linear)
        if fuse_dit_proj:
            dit_proj_w = torch.stack(
                (m.lm_to_dit_proj.weight.t(), m.res_to_dit_proj.weight.t()), dim=0
            ).contiguous()
            has_lm_b = m.lm_to_dit_proj.bias is not None
            has_res_b = m.res_to_dit_proj.bias is not None
            if has_lm_b or has_res_b:
                zero_b = torch.zeros((h_dit,), device=device, dtype=dtype)
                dit_proj_b = torch.stack(
                    (
                        m.lm_to_dit_proj.bias if has_lm_b else zero_b,
                        m.res_to_dit_proj.bias if has_res_b else zero_b,
                    ),
                    dim=0,
                ).unsqueeze(1).contiguous()
            else:
                dit_proj_b = None
        else:
            dit_proj_w = None
            dit_proj_b = None
        mu_in = torch.zeros((2, h_dit * 2), device=device, dtype=dtype)
        mu_pos_view = mu_in[0].view(2, 1, h_dit)
        dt_zeros = torch.zeros((2,), device=device, dtype=dtype)
        schedule_steps: list[tuple[torch.Tensor, torch.Tensor, bool]] = []
        t_cur = t_span[0]
        dt_cur = t_span[0] - t_span[1]
        for s_idx in range(1, len(t_span)):
            is_zero = s_idx <= zero_init_steps
            schedule_steps.append((t_cur.clone(), dt_cur.clone(), is_zero))
            t_cur = t_cur - dt_cur
            if s_idx < len(t_span) - 1:
                dt_cur = t_cur - t_span[s_idx + 1]

        def step_fn():
            if fuse_dit_proj:
                if dit_proj_b is not None:
                    torch.baddbmm(dit_proj_b, lm_res_hidden, dit_proj_w, out=mu_pos_view)
                else:
                    torch.bmm(lm_res_hidden, dit_proj_w, out=mu_pos_view)
            else:
                dit_h = torch.cat((m.lm_to_dit_proj(lm_hidden), m.res_to_dit_proj(res_hidden)), dim=-1)
                mu_in[0:1].copy_(dit_h)
            cond_t = prefix_cond.transpose(1, 2).contiguous()
            x = noise_z
            cond_in = torch.cat((cond_t, cond_t), dim=0)

            for t_val, dt_val, is_zero in schedule_steps:
                if not is_zero:
                    x_in = torch.cat((x, x), dim=0)
                    t_in = t_val.expand(2)
                    dt_in = dt_val.expand(2) if cfm.mean_mode else dt_zeros
                    est = estimator(x_in, mu_in, t_in, cond_in, dt_in)
                    pos_v, neg_v = est[0:1], est[1:2]
                    pos_flat = pos_v.reshape(1, -1)
                    neg_flat = neg_v.reshape(1, -1)
                    st_star = (
                        torch.sum(pos_flat * neg_flat, dim=1, keepdim=True)
                        / (torch.sum(neg_flat * neg_flat, dim=1, keepdim=True) + 1e-8)
                    ).view(1, 1, 1)
                    dphi = neg_v * st_star + cfg_value * (pos_v - neg_v * st_star)
                    x = x - dt_val * dphi

            # x is [1, d, p] -> pred_feat is [1, p, d]
            pred_feat = x.transpose(1, 2).contiguous()
            curr_embed = m.enc_to_lm_proj(feat_enc(pred_feat.unsqueeze(1)))[:, 0, :]

            stop_flag = m.stop_head(m.stop_actn(m.stop_proj(lm_hidden))).argmax(dim=-1)[0]

            next_lm = base_step(curr_embed, pos_id)
            next_lm = m.fsq_layer(next_lm)
            curr_res_in = m.fusion_concat_proj(torch.cat((next_lm, curr_embed), dim=-1))
            next_res = res_step(curr_res_in, pos_id)

            # Update rolling state in-place for the next step
            lm_hidden.copy_(next_lm)
            res_hidden.copy_(next_res)
            prefix_cond.copy_(pred_feat)
            pos_id.add_(1)
            return x.to(torch.float32), stop_flag

        # Save KV cache positions before warmup/capture
        with CUDA_GRAPH_LOCK:
            torch.cuda.synchronize()
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.inference_mode(), torch.cuda.stream(stream):
                for _ in range(3):
                    pos_id.zero_()
                    _ = step_fn()
            torch.cuda.current_stream().wait_stream(stream)

            pos_id.zero_()
            graph = torch.cuda.CUDAGraph()
            with torch.inference_mode(), torch.cuda.graph(graph):
                out_latent_fp32, out_stop = step_fn()
            torch.cuda.synchronize()

        entry = {
            "graph": graph,
            "lm_hidden": lm_hidden,
            "res_hidden": res_hidden,
            "prefix_cond": prefix_cond,
            "noise_z": noise_z,
            "pos_id": pos_id,
            "out_latent": out_latent_fp32,
            "out_stop": out_stop,
            # Keep every captured constant buffer alive so PyTorch's caching allocator never reuses them!
            "_keepalive": (
                lm_res_hidden,
                dit_proj_w,
                dit_proj_b,
                mu_in,
                mu_pos_view,
                dt_zeros,
                schedule_steps,
                t_span,
            ),
        }
        self._graphs[key] = entry
        return entry

    @staticmethod
    def _forward_lm_with_prefix(
        lm: nn.Module,
        suffix_embeds: torch.Tensor,
        prefix_kv: list[tuple[torch.Tensor, torch.Tensor]],
        prefix_len: int,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """Exact causal incremental prefill reusing a precomputed KV prefix snapshot."""
        from voxcpm.modules.minicpm4.model import apply_rotary_pos_emb

        bsz, suffix_len, _ = suffix_embeds.size()
        device = suffix_embeds.device
        total_len = prefix_len + suffix_len
        pos_ids = torch.arange(prefix_len, total_len, dtype=torch.long, device=device)
        position_emb = lm.rope_emb(pos_ids) if lm.rope_emb is not None else None

        # Causal mask: suffix query i (global pos prefix_len + i) attends to all keys j <= prefix_len + i
        q_pos = pos_ids.view(suffix_len, 1)
        k_pos = torch.arange(total_len, dtype=torch.long, device=device).view(1, total_len)
        attn_mask = (k_pos <= q_pos).view(1, 1, suffix_len, total_len)

        hidden_states = suffix_embeds
        next_cache: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer, (pre_k, pre_v) in zip(lm.layers, prefix_kv):
            residual = hidden_states
            attn = layer.self_attn
            if isinstance(attn.qkv_proj, FP8Linear):
                qkv = attn.qkv_proj.forward_rmsnorm(
                    hidden_states,
                    layer.input_layernorm.weight,
                    float(layer.input_layernorm.variance_epsilon),
                )
            else:
                qkv = attn.qkv_proj(layer.input_layernorm(hidden_states))
            q, k, v = qkv.split([attn._q_dim, attn._kv_dim, attn._kv_dim], dim=-1)
            q = q.view(bsz, suffix_len, attn.num_heads, attn.head_dim).transpose(1, 2)
            k = k.view(bsz, suffix_len, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)
            v = v.view(bsz, suffix_len, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)
            if position_emb is not None:
                cos, sin = position_emb
                q, k = apply_rotary_pos_emb(q, k, cos, sin)
            k_full = torch.cat((pre_k, k), dim=2)
            v_full = torch.cat((pre_v, v), dim=2)
            attn_out = F.scaled_dot_product_attention(
                q, k_full, v_full, attn_mask=attn_mask, enable_gqa=True
            )
            attn_out = attn_out.transpose(1, 2).contiguous().reshape(bsz, suffix_len, attn._q_dim)
            attn_out = attn.o_proj(attn_out)
            if layer.use_mup:
                hidden_states = residual + attn_out * (layer.scale_depth / (layer.num_hidden_layers**0.5))
            else:
                hidden_states = residual + attn_out

            residual = hidden_states
            if hasattr(layer.mlp, "forward_rmsnorm"):
                h_mlp = layer.mlp.forward_rmsnorm(
                    hidden_states,
                    layer.post_attention_layernorm.weight,
                    float(layer.post_attention_layernorm.variance_epsilon),
                )
            else:
                h_mlp = layer.mlp(layer.post_attention_layernorm(hidden_states))
            if layer.use_mup:
                hidden_states = residual + h_mlp * (layer.scale_depth / (layer.num_hidden_layers**0.5))
            else:
                hidden_states = residual + h_mlp
            next_cache.append((k_full, v_full))

        hidden_states = lm.norm(hidden_states)
        return hidden_states, next_cache

    def _get_or_build_ref_continuation_prefix(self, prompt_cache: dict) -> dict | None:
        """Precompute static reference feat_encoder embeddings and causal KV prefix once per prompt_cache."""
        cache_key = id(prompt_cache)
        if cache_key in self._prefix_kv_cache:
            return self._prefix_kv_cache[cache_key]

        m = self.m
        device = self.device
        dtype = self.dtype
        prompt_text = prompt_cache.get("prompt_text", "")
        # Trim trailing space so SentencePiece/BPE boundary matches tokenizer(prompt_text + target_text)
        stable_prompt_text = prompt_text.rstrip()
        stable_text_tokens = torch.LongTensor(m.text_tokenizer(stable_prompt_text))
        n_stable_text = int(stable_text_tokens.shape[0])

        ref_audio_feat = prompt_cache["ref_audio_feat"]
        prompt_audio_feat = prompt_cache["audio_feat"]
        ref_tokens, ref_feats, ref_t_mask, ref_a_mask = m._make_ref_prefix(ref_audio_feat, stable_text_tokens.device)

        prefill_enc = getattr(m, "_feat_encoder_raw", m.feat_encoder)
        scale_emb = m.config.lm_config.scale_emb if getattr(m.config.lm_config, "use_mup", False) else 1.0

        # Precompute patch-independent feat_encoder embeddings for both static audio segments ONCE
        ref_feats_gpu = ref_feats.unsqueeze(0).to(device=device, dtype=dtype)
        prompt_audio_gpu = prompt_audio_feat.unsqueeze(0).to(device=device, dtype=dtype)
        ref_feat_embed = m.enc_to_lm_proj(prefill_enc(ref_feats_gpu))
        prompt_audio_embed = m.enc_to_lm_proj(prefill_enc(prompt_audio_gpu))

        # Causal static prefix: [ref_tokens, stable_text_tokens]
        prefix_tokens = torch.cat([ref_tokens, stable_text_tokens]).unsqueeze(0).to(device)
        prefix_t_mask = torch.cat([ref_t_mask, torch.ones(n_stable_text, dtype=torch.int32)]).unsqueeze(0).to(device)
        prefix_a_mask = torch.cat([ref_a_mask, torch.zeros(n_stable_text, dtype=torch.int32)]).unsqueeze(0).to(device)
        prefix_len = int(prefix_tokens.shape[1])

        prefix_text_embed = m.base_lm.embed_tokens(prefix_tokens) * scale_emb
        zeros_text_feat = torch.zeros((1, n_stable_text, ref_feat_embed.shape[-1]), device=device, dtype=dtype)
        prefix_feat_embed = torch.cat((ref_feat_embed, zeros_text_feat), dim=1)
        prefix_combined = prefix_t_mask.unsqueeze(-1) * prefix_text_embed + prefix_a_mask.unsqueeze(-1) * prefix_feat_embed

        enc_pre, base_pre_kv = m.base_lm(inputs_embeds=prefix_combined, is_causal=True)
        enc_pre = m.fsq_layer(enc_pre) * prefix_a_mask.unsqueeze(-1) + enc_pre * prefix_t_mask.unsqueeze(-1)
        res_pre_in = m.fusion_concat_proj(torch.cat((enc_pre, prefix_a_mask.unsqueeze(-1) * prefix_feat_embed), dim=-1))
        _, res_pre_kv = m.residual_lm(inputs_embeds=res_pre_in, is_causal=True)

        snap = {
            "stable_text_tokens": stable_text_tokens,
            "n_stable_text": n_stable_text,
            "prefix_len": prefix_len,
            "base_pre_kv": [(k.contiguous(), v.contiguous()) for k, v in base_pre_kv],
            "res_pre_kv": [(k.contiguous(), v.contiguous()) for k, v in res_pre_kv],
            "prompt_audio_embed": prompt_audio_embed.contiguous(),
            "last_audio_feat": prompt_audio_gpu[:, -1, ...].contiguous(),
        }
        self._prefix_kv_cache[cache_key] = snap
        return snap

    @torch.inference_mode()
    def stream_with_prompt_cache(
        self,
        target_text: str,
        prompt_cache: dict,
        *,
        cfg_value: float = 2.0,
        inference_timesteps: int = 6,
        min_len: int = 2,
        max_len: int = 2000,
        retry_badcase_ratio_threshold: float = 3.5,
        seed: int | None = None,
        stop_event: threading.Event | None = None,
    ) -> Generator[np.ndarray, None, None]:
        from voxcpm.model.utils import (
            apply_generation_seed,
            materialize_generation_seed,
        )

        m = self.m
        device = self.device
        dtype = self.dtype

        mode = prompt_cache.get("mode", "continuation") if prompt_cache else "zero_shot"
        if prompt_cache and mode in ("continuation", "ref_continuation"):
            text = prompt_cache.get("prompt_text", "") + target_text
        else:
            text = target_text

        full_text_tokens = torch.LongTensor(m.text_tokenizer(text))
        target_text_length = len(m.text_tokenizer(target_text))
        eff_max_len = min(int(target_text_length * retry_badcase_ratio_threshold + 10), max_len)

        current_seed = materialize_generation_seed(seed)
        apply_generation_seed(current_seed)
        m.last_successful_seed = current_seed

        # Capture/fetch Mega-Step CUDA graph BEFORE fill_caches so warmup never overwrites live KV!
        g = self._get_or_capture_graph(inference_timesteps, cfg_value)

        # Fast incremental prefill path for ref_continuation (bit-identical causal KV prefix reuse;
        # skips building unused full CPU/GPU prompt tensors when the static prefix snapshot hits)
        used_prefix_cache = False
        if mode == "ref_continuation" and prompt_cache:
            snap = self._get_or_build_ref_continuation_prefix(prompt_cache)
            n_stable = snap["n_stable_text"] if snap else 0
            if (
                snap is not None
                and full_text_tokens.shape[0] >= n_stable
                and torch.equal(full_text_tokens[:n_stable], snap["stable_text_tokens"])
            ):
                rem_text_tokens = torch.cat(
                    [
                        full_text_tokens[n_stable:],
                        torch.tensor([m.audio_start_token], dtype=torch.long),
                    ],
                    dim=-1,
                )
                n_rem_text = int(rem_text_tokens.shape[0])
                prompt_audio_embed = snap["prompt_audio_embed"]

                scale_emb = m.config.lm_config.scale_emb if getattr(m.config.lm_config, "use_mup", False) else 1.0
                rem_text_embed = m.base_lm.embed_tokens(rem_text_tokens.unsqueeze(0).to(device))
                if scale_emb != 1.0:
                    rem_text_embed = rem_text_embed * scale_emb
                suffix_combined = torch.cat((rem_text_embed, prompt_audio_embed), dim=1)

                enc_suf, kv_cache_tuple = self._forward_lm_with_prefix(
                    m.base_lm, suffix_combined, snap["base_pre_kv"], snap["prefix_len"]
                )
                m.base_lm.kv_cache.fill_caches(kv_cache_tuple)
                if prompt_audio_embed.shape[1] > 0:
                    enc_suf = torch.cat(
                        (enc_suf[:, :n_rem_text, :], m.fsq_layer(enc_suf[:, n_rem_text:, :])),
                        dim=1,
                    )
                init_lm = enc_suf[:, -1, :]

                zeros_rem_feat = torch.zeros(
                    (1, n_rem_text, prompt_audio_embed.shape[-1]), device=device, dtype=dtype
                )
                rem_feat_embed = torch.cat((zeros_rem_feat, prompt_audio_embed), dim=1)
                res_suf_in = m.fusion_concat_proj(torch.cat((enc_suf, rem_feat_embed), dim=-1))
                res_suf_out, res_kv = self._forward_lm_with_prefix(
                    m.residual_lm, res_suf_in, snap["res_pre_kv"], snap["prefix_len"]
                )
                m.residual_lm.kv_cache.fill_caches(res_kv)
                init_res = res_suf_out[:, -1, :]
                last_audio_cond = snap["last_audio_feat"]
                used_prefix_cache = True

        if not used_prefix_cache:
            # Standard full prefill fallback (only build full CPU/GPU tensors when prefix cache misses)
            text_token = torch.cat(
                [full_text_tokens, torch.tensor([m.audio_start_token], dtype=torch.int32)], dim=-1
            )
            text_length = text_token.shape[0]

            if mode in ("zero_shot", "continuation"):
                prompt_audio_feat = (
                    prompt_cache["audio_feat"]
                    if prompt_cache
                    else torch.empty((0, m.patch_size, m.audio_vae.latent_dim), dtype=torch.float32)
                )
                audio_length = prompt_audio_feat.size(0)
                text_token = torch.cat([text_token, torch.zeros(audio_length, dtype=torch.int32)])
                audio_feat = torch.cat(
                    [
                        torch.zeros((text_length, m.patch_size, m.audio_vae.latent_dim), dtype=torch.float32),
                        prompt_audio_feat,
                    ],
                    dim=0,
                )
                text_mask = torch.cat(
                    [torch.ones(text_length, dtype=torch.int32), torch.zeros(audio_length, dtype=torch.int32)]
                )
                audio_mask = torch.cat(
                    [torch.zeros(text_length, dtype=torch.int32), torch.ones(audio_length, dtype=torch.int32)]
                )
            elif mode == "reference":
                ref_tokens, ref_feats, ref_t_mask, ref_a_mask = m._make_ref_prefix(
                    prompt_cache["ref_audio_feat"], text_token.device
                )
                text_token = torch.cat([ref_tokens, text_token])
                audio_feat = torch.cat(
                    [ref_feats, torch.zeros((text_length, m.patch_size, m.audio_vae.latent_dim), dtype=torch.float32)],
                    dim=0,
                )
                text_mask = torch.cat([ref_t_mask, torch.ones(text_length, dtype=torch.int32)])
                audio_mask = torch.cat([ref_a_mask, torch.zeros(text_length, dtype=torch.int32)])
            else:
                ref_audio_feat = prompt_cache["ref_audio_feat"]
                prompt_audio_feat = prompt_cache["audio_feat"]
                prompt_audio_length = prompt_audio_feat.size(0)
                ref_tokens, ref_feats, ref_t_mask, ref_a_mask = m._make_ref_prefix(ref_audio_feat, text_token.device)
                text_token = torch.cat([ref_tokens, text_token, torch.zeros(prompt_audio_length, dtype=torch.int32)])
                audio_feat = torch.cat(
                    [
                        ref_feats,
                        torch.zeros((text_length, m.patch_size, m.audio_vae.latent_dim), dtype=torch.float32),
                        prompt_audio_feat,
                    ],
                    dim=0,
                )
                text_mask = torch.cat(
                    [ref_t_mask, torch.ones(text_length, dtype=torch.int32), torch.zeros(prompt_audio_length, dtype=torch.int32)]
                )
                audio_mask = torch.cat(
                    [ref_a_mask, torch.zeros(text_length, dtype=torch.int32), torch.ones(prompt_audio_length, dtype=torch.int32)]
                )

            text_token = text_token.unsqueeze(0).to(device)
            text_mask = text_mask.unsqueeze(0).to(device)
            audio_feat = audio_feat.unsqueeze(0).to(device=device, dtype=dtype)
            audio_mask = audio_mask.unsqueeze(0).to(device)

            prefill_enc = getattr(m, "_feat_encoder_raw", m.feat_encoder)
            feat_embed = m.enc_to_lm_proj(prefill_enc(audio_feat))
            scale_emb = m.config.lm_config.scale_emb if getattr(m.config.lm_config, "use_mup", False) else 1.0
            text_embed = m.base_lm.embed_tokens(text_token) * scale_emb
            combined_embed = text_mask.unsqueeze(-1) * text_embed + audio_mask.unsqueeze(-1) * feat_embed

            enc_outputs, kv_cache_tuple = m.base_lm(inputs_embeds=combined_embed, is_causal=True)
            m.base_lm.kv_cache.fill_caches(kv_cache_tuple)
            enc_outputs = m.fsq_layer(enc_outputs) * audio_mask.unsqueeze(-1) + enc_outputs * text_mask.unsqueeze(-1)
            init_lm = enc_outputs[:, -1, :]

            res_in = m.fusion_concat_proj(torch.cat((enc_outputs, audio_mask.unsqueeze(-1) * feat_embed), dim=-1))
            res_out, res_kv = m.residual_lm(inputs_embeds=res_in, is_causal=True)
            m.residual_lm.kv_cache.fill_caches(res_kv)
            init_res = res_out[:, -1, :]
            last_audio_cond = audio_feat[:, -1, ...]

        # Ensure step budget fits static KV cache
        prefill_len = int(m.base_lm.kv_cache.current_length)
        kv_room = max(0, int(m.base_lm.kv_cache.max_length) - prefill_len - 1)
        eff_max_len = min(eff_max_len, kv_room)
        if eff_max_len <= 0:
            return

        g["lm_hidden"].copy_(init_lm)
        g["res_hidden"].copy_(init_res)
        g["prefix_cond"].copy_(last_audio_cond)
        g["pos_id"].fill_(prefill_len)
        self._vae_graph.reset()

        # Pre-sample all diffusion noise for this sentence in 1 kernel call (reproducible with seed!)
        all_noise = torch.randn((eff_max_len, 1, m.feat_dim, m.patch_size), device=device, dtype=dtype)
        latent_staging = self._latent_staging
        stop_staging = self._stop_staging
        latent_ready_events = self._latent_ready_events
        vae_done_events = self._vae_done_events

        main_stream = torch.cuda.current_stream()
        vae_stream = self._vae_stream
        vae_slot_active = [False, False]
        pending_slot: int | None = None
        pending_step: int = -1

        try:
            for i in range(eff_max_len):
                if stop_event is not None and stop_event.is_set():
                    break

                slot = i & 1
                # Ensure previous use of this slot's staging buffer finished before overwriting
                if vae_slot_active[slot]:
                    vae_done_events[slot].synchronize()
                    vae_slot_active[slot] = False
                    if pending_slot == slot:
                        yield self._pinned_bufs[slot].numpy().copy()
                        pending_slot = None

                # Replay the ENTIRE autoregressive step on main_stream
                g["noise_z"].copy_(all_noise[i])
                g["graph"].replay()
                latent_staging[slot].copy_(g["out_latent"])
                stop_staging[slot].copy_(g["out_stop"], non_blocking=True)

                latent_ready_events[slot].record(main_stream)

                # Launch AudioVAE FP32 CUDA Graph + non-blocking D2H copy on vae_stream!
                with torch.cuda.stream(vae_stream):
                    vae_stream.wait_event(latent_ready_events[slot])
                    wav_gpu = self._vae_graph.decode_chunk(latent_staging[slot])
                    self._pinned_bufs[slot].copy_(wav_gpu.reshape(-1), non_blocking=True)
                    vae_done_events[slot].record(vae_stream)
                    vae_slot_active[slot] = True

                # Yield the PREVIOUS step's audio while current step's VAE decodes in parallel!
                prev_slot = 1 - slot
                if pending_slot == prev_slot and vae_slot_active[prev_slot]:
                    vae_done_events[prev_slot].synchronize()
                    vae_slot_active[prev_slot] = False
                    yield self._pinned_bufs[prev_slot].numpy().copy()
                    # Check previous step's stop flag (already ready with zero extra stall!)
                    if pending_step > min_len and int(stop_staging[prev_slot].item()) == 1:
                        vae_done_events[slot].synchronize()
                        vae_slot_active[slot] = False
                        pending_slot = None
                        break
                    pending_slot = None

                pending_slot = slot
                pending_step = i

                # Check stop flag immediately if this is the last allowed step
                if i == eff_max_len - 1:
                    vae_done_events[slot].synchronize()
                    vae_slot_active[slot] = False
                    yield self._pinned_bufs[slot].numpy().copy()
                    pending_slot = None

            # Flush final buffered chunk if loop exited without hitting max_len or stop_event
            if pending_slot is not None:
                if vae_slot_active[pending_slot]:
                    vae_done_events[pending_slot].synchronize()
                    vae_slot_active[pending_slot] = False
                if stop_event is None or not stop_event.is_set():
                    yield self._pinned_bufs[pending_slot].numpy().copy()
        finally:
            # Quint-audited safety: always drain async AudioVAE graph & D2H copy on early
            # GeneratorExit (barge-in or warmup_streaming) before releasing gpu_lock.
            vae_stream.synchronize()


def _accelerate_voxcpm2_cuda(tts_model, *, use_int4_lm: bool = False) -> None:
    """Apply inference-only Blackwell/CUDA speedups: fold weight_norm, hybrid INT4/FP8 fused QKV+GU, Mega-Step Graph, and overlapped AudioVAE Graph."""
    if not str(tts_model.device).startswith("cuda") or not torch.cuda.is_available():
        return

    t0 = time.perf_counter()
    # 1. Permanently fold AudioVAE weight_norm (mathematically identical, zero runtime overhead)
    for mod in tts_model.audio_vae.modules():
        if hasattr(mod, "weight_g") and hasattr(mod, "weight_v"):
            try:
                remove_weight_norm(mod)
            except ValueError:
                pass

    if str(tts_model.config.dtype) != "bfloat16" or not probe_fp8_supported(tts_model.device):
        return

    try:
        from voxcpm.model.utils import get_dtype

        lm_dtype = get_dtype(tts_model.config.dtype)
        max_kv = min(int(tts_model.config.max_length), _FAST_KV_MAX_LEN)
        tts_model.config.max_length = max_kv
        tts_model.base_lm.setup_cache(1, max_kv, tts_model.device, lm_dtype)
        tts_model.residual_lm.setup_cache(1, max_kv, tts_model.device, lm_dtype)
        arange_buf = torch.arange(max_kv, device=tts_model.device)

        # 2. Fuse QKV + Gate/Up on base_lm & residual_lm (FP8 backbone, pure BF16 continuous acoustic DiT/encoder)
        _fuse_minicpm_model(tts_model.base_lm, use_int4=use_int4_lm, arange_buf=arange_buf)
        _fuse_minicpm_model(tts_model.residual_lm, use_int4=use_int4_lm, arange_buf=arange_buf)
        # Keep feat_decoder (DiT flow-matching denoiser), feat_encoder, and acoustic projections in
        # pure BF16 inside the Mega-Step CUDA Graph for maximum voice timbre fidelity.
        torch.cuda.empty_cache()

        # Keep raw references for the Mega-Step Graph runner before wrapping individual paths
        tts_model.base_lm._forward_step_raw = tts_model.base_lm.forward_step
        tts_model.residual_lm._forward_step_raw = tts_model.residual_lm.forward_step
        tts_model._feat_encoder_raw = tts_model.feat_encoder
        tts_model._dit_estimator_raw = tts_model.feat_decoder.estimator

        # 4. Build Mega-Step + Stream-Overlapped AudioVAE runner (and pre-capture default step graph)
        tts_model._mega_runner = _MegaStepRunner(tts_model)

        # Also capture individual CUDA graphs for non-streaming synthesize() fallback
        h_lm = int(tts_model.config.lm_config.hidden_size)
        h_dit = int(tts_model.config.dit_config.hidden_dim) * 2
        with torch.inference_mode():
            tts_model.base_lm.forward_step = capture_cuda_graph(
                tts_model.base_lm._forward_step_raw,
                (
                    torch.zeros((1, h_lm), device=tts_model.device, dtype=lm_dtype),
                    torch.tensor([0], device=tts_model.device, dtype=torch.long),
                ),
            )
            tts_model.residual_lm.forward_step = capture_cuda_graph(
                tts_model.residual_lm._forward_step_raw,
                (
                    torch.zeros((1, h_lm), device=tts_model.device, dtype=lm_dtype),
                    torch.tensor([0], device=tts_model.device, dtype=torch.long),
                ),
            )
            tts_model.feat_encoder = CUDAGraphModule(
                tts_model._feat_encoder_raw,
                (
                    torch.zeros(
                        (1, 1, tts_model.patch_size, tts_model.feat_dim),
                        device=tts_model.device,
                        dtype=lm_dtype,
                    ),
                ),
            )
            tts_model.feat_decoder.estimator = CUDAGraphModule(
                tts_model._dit_estimator_raw,
                (
                    torch.zeros((2, tts_model.feat_dim, tts_model.patch_size), device=tts_model.device, dtype=lm_dtype),
                    torch.zeros((2, h_dit), device=tts_model.device, dtype=lm_dtype),
                    torch.zeros((2,), device=tts_model.device, dtype=lm_dtype),
                    torch.zeros((2, tts_model.feat_dim, tts_model.patch_size), device=tts_model.device, dtype=lm_dtype),
                    torch.zeros((2,), device=tts_model.device, dtype=lm_dtype),
                ),
            )

        logger.info(
            "Enabled VoxCPM2 %s + Mega-Step Graph + Stream-Overlapped AudioVAE (KV=%d) in %.2fs",
            "INT4/FP8" if (use_int4_lm and TRITON_AVAILABLE) else "FP8",
            max_kv,
            time.perf_counter() - t0,
        )
    except Exception as exc:
        logger.warning("VoxCPM2 CUDA acceleration skipped: %s", exc, exc_info=True)


def _load_voxcpm2_fast(
    model_id: str,
    *,
    local_files_only: bool,
    optimize: bool,
    device: str | None,
    use_int4_lm: bool = False,
) -> VoxCPM | None:
    """Fast meta-device VoxCPM2 loader avoiding slow random CPU weight initialization."""
    from safetensors.torch import load_file
    from transformers import LlamaTokenizerFast
    from voxcpm.core import resolve_model_path
    from voxcpm.model.utils import get_dtype, pick_runtime_dtype, resolve_runtime_device
    from voxcpm.model.voxcpm2 import VoxCPM2Model, VoxCPMConfig
    from voxcpm.modules.audiovae import AudioVAEV2
    from voxcpm.modules.minicpm4.model import MiniCPMLongRoPE

    local_path = resolve_model_path(model_id, local_files_only=local_files_only)
    cfg_path = os.path.join(local_path, "config.json")
    safetensors_path = os.path.join(local_path, "model.safetensors")
    if not (os.path.isfile(cfg_path) and os.path.isfile(safetensors_path)):
        return None

    with open(cfg_path, encoding="utf-8") as f:
        raw_cfg = f.read()
    if json.loads(raw_cfg).get("architecture", "").lower() != "voxcpm2":
        return None

    config = VoxCPMConfig.model_validate_json(raw_cfg)
    target_device = resolve_runtime_device(device, config.device)
    resolved_dtype = pick_runtime_dtype(target_device, config.dtype)
    config.device = target_device
    config.dtype = resolved_dtype
    lm_dtype = get_dtype(resolved_dtype)
    load_device = "cuda:0" if target_device == "cuda" else target_device

    tokenizer = LlamaTokenizerFast.from_pretrained(local_path)

    audiovae_sf = os.path.join(local_path, "audiovae.safetensors")
    audiovae_pth = os.path.join(local_path, "audiovae.pth")
    if os.path.isfile(audiovae_sf):
        vae_sd = load_file(audiovae_sf, device=load_device)
    elif os.path.isfile(audiovae_pth):
        ckpt = torch.load(audiovae_pth, map_location=load_device, weights_only=True)
        vae_sd = ckpt.get("state_dict", ckpt)
    else:
        return None

    model_sd = load_file(safetensors_path, device=load_device)

    for k, v in list(model_sd.items()):
        if v.is_floating_point() and v.dtype != lm_dtype:
            model_sd[k] = v.to(lm_dtype)

    # Merge optional HAL 9000 LoRA weights directly into base weights before quantization
    lora_dir = _resolve_lora_dir()
    if lora_dir is not None:
        _merge_lora_into_state_dict(model_sd, lora_dir, load_device, lm_dtype)

    for kw, val in vae_sd.items():
        if val.is_floating_point() and val.dtype != torch.float32:
            val = val.to(torch.float32)
        model_sd[f"audio_vae.{kw}"] = val

    audio_vae_config = getattr(config, "audio_vae_config", None)
    with torch.device("meta"):
        audio_vae = AudioVAEV2(config=audio_vae_config) if audio_vae_config else AudioVAEV2()
        tts_model = VoxCPM2Model(config, tokenizer, audio_vae, None, device=target_device)

    tts_model.load_state_dict(model_sd, strict=False, assign=True)
    del model_sd, vae_sd

    for module in tts_model.modules():
        if isinstance(module, MiniCPMLongRoPE):
            inv_freq = 1.0 / (
                module.base ** (torch.arange(0, module.dim, 2, device=target_device, dtype=torch.float32) / module.dim)
            )
            module.register_buffer("inv_freq", inv_freq, persistent=False)
            module._set_cos_sin_cache(
                seq_len=module.max_position_embeddings,
                device=target_device,
                dtype=torch.float32,
            )
        for hook in list(module._forward_pre_hooks.values()):
            if type(hook).__name__ == "WeightNorm":
                hook(module, None)

    for lm in (tts_model.base_lm, tts_model.residual_lm):
        lm.setup_cache(1, config.max_length, target_device, lm_dtype)

    if any(p.is_meta for p in tts_model.parameters()) or any(b.is_meta for b in tts_model.buffers()):
        raise RuntimeError("Fast VoxCPM2 load left unmaterialized meta tensors")

    tts_model = tts_model.eval()
    if optimize:
        tts_model = tts_model.optimize(disable=False)
    else:
        _accelerate_voxcpm2_cuda(tts_model, use_int4_lm=use_int4_lm)
    wrapper = object.__new__(VoxCPM)
    wrapper.tts_model = tts_model
    wrapper.text_normalizer = None
    wrapper.denoiser = None
    if optimize:
        tts_model.generate(target_text="Yes.", max_len=4)
    return wrapper


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
        use_int4_lm: bool = False,
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

        if reference_text is None:
            txt_path = self.reference_wav.with_suffix(".txt")
            if txt_path.exists():
                reference_text = txt_path.read_text(encoding="utf-8").strip()
            elif (
                self.reference_wav.resolve() == DEFAULT_REF_WAV.resolve()
                and DEFAULT_REF_TXT.exists()
            ):
                reference_text = DEFAULT_REF_TXT.read_text(encoding="utf-8").strip()

        self.reference_text = (reference_text.strip() + " ") if reference_text else None
        self.cfg_value = cfg_value
        self.inference_timesteps = inference_timesteps
        self.use_int4_lm = use_int4_lm
        self._phrase_cache: dict[tuple[str, int], list[np.ndarray]] = {}
        self._cache_lock = threading.Lock()

        t0 = time.perf_counter()
        self.model = self._load_model(
            model_id=model_id, optimize=optimize, device=device, use_int4_lm=use_int4_lm
        )
        t_load = time.perf_counter() - t0
        self.sample_rate: int = int(self.model.tts_model.sample_rate)

        t_prompt = time.perf_counter()
        self.prompt_cache, cache_hit = self._load_or_build_prompt_cache(model_id)
        logger.info(
            "Loaded VoxCPM2 (%s) @ %d Hz (%s mode) — weights %.2fs, prompt cache %.2fs (%s)",
            model_id,
            self.sample_rate,
            self.prompt_cache.get("mode", "unknown"),
            t_load,
            time.perf_counter() - t_prompt,
            "disk cache" if cache_hit else "computed",
        )

    def _load_model(
        self, *, model_id: str, optimize: bool, device: str | None, use_int4_lm: bool
    ) -> VoxCPM:
        for local_only in (True, False):
            try:
                fast = _load_voxcpm2_fast(
                    model_id,
                    local_files_only=local_only,
                    optimize=optimize,
                    device=device,
                    use_int4_lm=use_int4_lm,
                )
                if fast is not None:
                    return fast
                return VoxCPM.from_pretrained(
                    model_id,
                    load_denoiser=False,
                    local_files_only=local_only,
                    optimize=optimize,
                    device=device,
                )
            except (LocalEntryNotFoundError, FileNotFoundError):
                if not local_only:
                    raise
                logger.info("VoxCPM2 cache incomplete for %s; downloading from HuggingFace...", model_id)
            except (RuntimeError, ValueError, TypeError, KeyError, AttributeError, OSError) as exc:
                logger.warning("Fast VoxCPM2 load fell back to standard loader: %s", exc)
                return VoxCPM.from_pretrained(
                    model_id,
                    load_denoiser=False,
                    local_files_only=local_only,
                    optimize=optimize,
                    device=device,
                )
        raise RuntimeError(f"Unable to load VoxCPM model {model_id!r}")

    def _load_or_build_prompt_cache(self, model_id: str) -> tuple[dict, bool]:
        cache_path = _prompt_cache_file(model_id, self.reference_wav, self.reference_text)
        if cache_path.is_file():
            try:
                cached = torch.load(cache_path, map_location="cpu", weights_only=True)
                if isinstance(cached, dict) and "mode" in cached:
                    return cached, True
            except (OSError, RuntimeError, ValueError, EOFError) as exc:
                logger.debug("Ignoring stale prompt cache %s: %s", cache_path, exc)

        prompt_cache = self.model.tts_model.build_prompt_cache(
            prompt_text=self.reference_text,
            prompt_wav_path=str(self.reference_wav) if self.reference_text else None,
            reference_wav_path=str(self.reference_wav),
            trim_silence_vad=True,
        )
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = cache_path.with_suffix(f".tmp.{os.getpid()}")
            torch.save(prompt_cache, tmp_path)
            os.replace(tmp_path, cache_path)
        except OSError as exc:
            logger.debug("Could not persist prompt cache %s: %s", cache_path, exc)
        return prompt_cache, False

    def _effective_steps(self, inference_timesteps: int | None) -> int:
        steps = self.inference_timesteps if inference_timesteps is None else inference_timesteps
        if steps < MIN_DIFFUSION_STEPS:
            raise ValueError(f"inference_timesteps must be >= {MIN_DIFFUSION_STEPS}, got {steps}")
        return steps

    def warmup_streaming(
        self,
        seed: int = 42,
        max_chunks: int = 1,
        inference_timesteps: int | None = None,
    ) -> float:
        """Warm up the streaming decoder path (and capture the Mega-Step CUDA graph for default steps)."""
        t0 = time.perf_counter()
        stop_ev = threading.Event()
        steps = self._effective_steps(inference_timesteps)
        gen = self.stream_speech(
            "Yes.",
            stop_event=stop_ev,
            seed=seed,
            inference_timesteps=steps,
        )
        try:
            for idx, _ in enumerate(gen, start=1):
                if idx >= max_chunks:
                    stop_ev.set()
                    break
        finally:
            gen.close()
        elapsed = time.perf_counter() - t0
        logger.info("Warmed up VoxCPM2 streaming path (steps=%d) in %.2fs", steps, elapsed)
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

        mega_runner: _MegaStepRunner | None = getattr(self.model.tts_model, "_mega_runner", None)
        collected: list[np.ndarray] | None = [] if cache_phrase else None
        completed_cleanly = False

        if mega_runner is not None:
            gen = mega_runner.stream_with_prompt_cache(
                target_text=spoken,
                prompt_cache=self.prompt_cache,
                cfg_value=self.cfg_value,
                inference_timesteps=steps,
                retry_badcase_ratio_threshold=3.5,
                seed=seed,
                stop_event=stop_event,
            )
            try:
                for arr in gen:
                    if stop_event is not None and stop_event.is_set():
                        break
                    if collected is not None:
                        collected.append(arr.copy())
                    yield arr
                    if stop_event is not None and stop_event.is_set():
                        break
                else:
                    completed_cleanly = True
            finally:
                gen.close()
        else:
            gen = self.model.tts_model.generate_with_prompt_cache_streaming(
                target_text=spoken,
                prompt_cache=self.prompt_cache,
                cfg_value=self.cfg_value,
                inference_timesteps=steps,
                retry_badcase=False,
                retry_badcase_ratio_threshold=3.5,
                seed=seed,
            )
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
