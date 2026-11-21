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
    quantize_rowwise_fp8,
)

logger = logging.getLogger(__name__)

HAL_WHISPER_PROMPT = (
    "HAL 9000, Charles, Dave Bowman, Frank Poole, Dr. Chandra, Discovery One, "
    "Alexei Leonov, AE-35, TMA-1, Europa, Monolith, pod bay doors."
)

try:
    import triton
    import triton.language as tl

    @triton.jit
    def _layernorm_fp8_quant_kernel(
        x_ptr,
        w_ptr,
        b_ptr,
        y_ptr,
        s_ptr,
        n_cols,
        inv_n_cols,
        stride_x,
        stride_y,
        eps,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        x = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
        mean = tl.sum(x, axis=0) * inv_n_cols
        diff = tl.where(mask, x - mean, 0.0)
        var = tl.sum(diff * diff, axis=0) * inv_n_cols
        rstd = tl.math.rsqrt(var + eps)
        w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = tl.where(mask, diff * rstd * w + b, 0.0)
        amax = tl.max(tl.abs(y), axis=0)
        scale = tl.maximum(amax, 1e-12) / 448.0
        tl.store(s_ptr + row, scale)
        q = tl.clamp(y / scale, -448.0, 448.0).to(tl.float8e4nv)
        tl.store(y_ptr + row * stride_y + cols, q, mask=mask)

    @triton.jit
    def _residual_add_layernorm_fp8_quant_kernel(
        r_ptr,
        d_ptr,
        w_ptr,
        b_ptr,
        out_h_ptr,
        y_ptr,
        s_ptr,
        n_cols,
        inv_n_cols,
        stride_r,
        stride_d,
        stride_h,
        stride_y,
        eps,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        r_raw = tl.load(r_ptr + row * stride_r + cols, mask=mask, other=0.0)
        d_raw = tl.load(d_ptr + row * stride_d + cols, mask=mask, other=0.0)
        # Round residual sum to r_raw.dtype before LN for bit-exact equivalence with unfused BF16 add
        h_out = (r_raw.to(tl.float32) + d_raw.to(tl.float32)).to(r_raw.dtype)
        tl.store(out_h_ptr + row * stride_h + cols, h_out, mask=mask)
        h = h_out.to(tl.float32)
        mean = tl.sum(h, axis=0) * inv_n_cols
        diff = tl.where(mask, h - mean, 0.0)
        var = tl.sum(diff * diff, axis=0) * inv_n_cols
        rstd = tl.math.rsqrt(var + eps)
        w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = tl.where(mask, diff * rstd * w + b, 0.0)
        amax = tl.max(tl.abs(y), axis=0)
        scale = tl.maximum(amax, 1e-12) / 448.0
        tl.store(s_ptr + row, scale)
        q = tl.clamp(y / scale, -448.0, 448.0).to(tl.float8e4nv)
        tl.store(y_ptr + row * stride_y + cols, q, mask=mask)

    @triton.jit
    def _residual_add_layernorm_fp8_only_kernel(
        r_ptr,
        d_ptr,
        w_ptr,
        b_ptr,
        y_ptr,
        s_ptr,
        n_cols,
        inv_n_cols,
        stride_r,
        stride_d,
        stride_y,
        eps,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        r_raw = tl.load(r_ptr + row * stride_r + cols, mask=mask, other=0.0)
        d_raw = tl.load(d_ptr + row * stride_d + cols, mask=mask, other=0.0)
        h_out = (r_raw.to(tl.float32) + d_raw.to(tl.float32)).to(r_raw.dtype)
        h = h_out.to(tl.float32)
        mean = tl.sum(h, axis=0) * inv_n_cols
        diff = tl.where(mask, h - mean, 0.0)
        var = tl.sum(diff * diff, axis=0) * inv_n_cols
        rstd = tl.math.rsqrt(var + eps)
        w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = tl.where(mask, diff * rstd * w + b, 0.0)
        amax = tl.max(tl.abs(y), axis=0)
        scale = tl.maximum(amax, 1e-12) / 448.0
        tl.store(s_ptr + row, scale)
        q = tl.clamp(y / scale, -448.0, 448.0).to(tl.float8e4nv)
        tl.store(y_ptr + row * stride_y + cols, q, mask=mask)

    @triton.jit
    def _residual_add_layernorm_bf16_kernel(
        r_ptr,
        d_ptr,
        w_ptr,
        b_ptr,
        out_ptr,
        n_cols,
        inv_n_cols,
        stride_r,
        stride_d,
        stride_o,
        eps,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        r_raw = tl.load(r_ptr + row * stride_r + cols, mask=mask, other=0.0)
        d_raw = tl.load(d_ptr + row * stride_d + cols, mask=mask, other=0.0)
        h_out = (r_raw.to(tl.float32) + d_raw.to(tl.float32)).to(r_raw.dtype)
        h = h_out.to(tl.float32)
        mean = tl.sum(h, axis=0) * inv_n_cols
        diff = tl.where(mask, h - mean, 0.0)
        var = tl.sum(diff * diff, axis=0) * inv_n_cols
        rstd = tl.math.rsqrt(var + eps)
        w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = diff * rstd * w + b
        tl.store(out_ptr + row * stride_o + cols, y.to(r_raw.dtype), mask=mask)

    @triton.jit
    def _gelu_fp8_quant_kernel(
        x_ptr,
        y_ptr,
        s_ptr,
        n_cols,
        stride_x,
        stride_y,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        x = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
        y = x * 0.5 * (1.0 + tl.math.erf(x * 0.7071067811865476))
        y = tl.where(mask, y, 0.0)
        amax = tl.max(tl.abs(y), axis=0)
        scale = tl.maximum(amax, 1e-12) / 448.0
        tl.store(s_ptr + row, scale)
        q = tl.clamp(y / scale, -448.0, 448.0).to(tl.float8e4nv)
        tl.store(y_ptr + row * stride_y + cols, q, mask=mask)

    @triton.jit
    def _sdpa_transpose_fp8_quant_kernel(
        x_ptr,
        y_ptr,
        s_ptr,
        n_cols,
        stride_h,
        stride_t,
        stride_y,
        HEAD_DIM: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        t = tl.program_id(0)
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        head = cols // HEAD_DIM
        dim = cols % HEAD_DIM
        x = tl.load(x_ptr + head * stride_h + t * stride_t + dim, mask=mask, other=0.0).to(
            tl.float32
        )
        amax = tl.max(tl.abs(x), axis=0)
        scale = tl.maximum(amax, 1e-12) / 448.0
        tl.store(s_ptr + t, scale)
        q = tl.clamp(x / scale, -448.0, 448.0).to(tl.float8e4nv)
        tl.store(y_ptr + t * stride_y + cols, q, mask=mask)

    @triton.jit
    def _split_kv_transpose_cache_kernel(
        kv_ptr,
        k_out_ptr,
        v_out_ptr,
        src_len,
        stride_kv,
        stride_kh,
        stride_kt,
        stride_vh,
        stride_vt,
        EMBED_DIM: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        HEAD_BLOCK: tl.constexpr,
        BLOCK_T: tl.constexpr,
    ):
        pid_t = tl.program_id(0)
        pid_h = tl.program_id(1)
        ts = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
        ds = tl.arange(0, HEAD_BLOCK)
        mask = (ts[:, None] < src_len) & (ds[None, :] < HEAD_DIM)
        h_off = pid_h * HEAD_DIM
        kv_base = ts[:, None] * stride_kv + h_off + ds[None, :]
        k_tile = tl.load(kv_ptr + kv_base, mask=mask, other=0.0)
        v_tile = tl.load(kv_ptr + kv_base + EMBED_DIM, mask=mask, other=0.0)
        tl.store(
            k_out_ptr + pid_h * stride_kh + ts[:, None] * stride_kt + ds[None, :],
            k_tile,
            mask=mask,
        )
        tl.store(
            v_out_ptr + pid_h * stride_vh + ts[:, None] * stride_vt + ds[None, :],
            v_tile,
            mask=mask,
        )

    @triton.jit
    def _whisper_qkv_scatter_step_kernel(
        qkv_ptr,
        pos_id_ptr,
        q_out_ptr,
        k_cache_ptr,
        v_cache_ptr,
        stride_kc_h,
        stride_kc_t,
        stride_vc_h,
        stride_vc_t,
        NUM_HEADS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        HEAD_BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = tl.arange(0, HEAD_BLOCK)
        mask = offs < HEAD_DIM
        embed_dim = NUM_HEADS * HEAD_DIM

        if pid < NUM_HEADS:
            base = pid * HEAD_DIM
            q_val = tl.load(qkv_ptr + base + offs, mask=mask, other=0.0)
            tl.store(q_out_ptr + base + offs, q_val, mask=mask)
        elif pid < 2 * NUM_HEADS:
            kh = pid - NUM_HEADS
            base = embed_dim + kh * HEAD_DIM
            k_val = tl.load(qkv_ptr + base + offs, mask=mask, other=0.0)
            pos = tl.load(pos_id_ptr).to(tl.int64)
            dst = k_cache_ptr + kh * stride_kc_h + pos * stride_kc_t
            tl.store(dst + offs, k_val, mask=mask)
        else:
            vh = pid - 2 * NUM_HEADS
            base = 2 * embed_dim + vh * HEAD_DIM
            v_val = tl.load(qkv_ptr + base + offs, mask=mask, other=0.0)
            pos = tl.load(pos_id_ptr).to(tl.int64)
            dst = v_cache_ptr + vh * stride_vc_h + pos * stride_vc_t
            tl.store(dst + offs, v_val, mask=mask)

    @triton.jit
    def _whisper_cross_attn_splitkv_stage1_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        part_acc_ptr,
        part_m_ptr,
        part_l_ptr,
        src_len,
        stride_kh,
        stride_kt,
        stride_vh,
        stride_vt,
        stride_ah,
        stride_as,
        HEAD_DIM: tl.constexpr,
        HEAD_BLOCK: tl.constexpr,
        BLOCK_T: tl.constexpr,
        NUM_SPLITS: tl.constexpr,
    ):
        pid_h = tl.program_id(0)
        pid_s = tl.program_id(1)
        offs_d = tl.arange(0, HEAD_BLOCK)
        mask_d = offs_d < HEAD_DIM
        offs_t = tl.arange(0, BLOCK_T)

        q = tl.load(q_ptr + pid_h * HEAD_DIM + offs_d, mask=mask_d, other=0.0).to(tl.float32)
        k_head_ptr = k_ptr + pid_h * stride_kh
        v_head_ptr = v_ptr + pid_h * stride_vh

        split_size = tl.cdiv(src_len, NUM_SPLITS)
        t_start = pid_s * split_size
        t_end = tl.minimum(t_start + split_size, src_len)

        m_i = -1.0e38
        l_i = 0.0
        acc = tl.zeros((HEAD_BLOCK,), dtype=tl.float32)

        for t0 in range(t_start, t_end, BLOCK_T):
            t_idx = t0 + offs_t
            mask_t = t_idx < t_end
            mask_td = mask_t[:, None] & mask_d[None, :]
            t_i64 = t_idx.to(tl.int64)
            k_tile = tl.load(
                k_head_ptr + t_i64[:, None] * stride_kt + offs_d[None, :],
                mask=mask_td,
                other=0.0,
            ).to(tl.float32)
            v_tile = tl.load(
                v_head_ptr + t_i64[:, None] * stride_vt + offs_d[None, :],
                mask=mask_td,
                other=0.0,
            ).to(tl.float32)
            scores = tl.sum(k_tile * q[None, :], axis=1)
            scores = tl.where(mask_t, scores, -1.0e38)
            block_max = tl.max(scores, axis=0)
            m_new = tl.maximum(m_i, block_max)
            alpha = tl.exp(m_i - m_new)
            p = tl.where(mask_t, tl.exp(scores - m_new), 0.0)
            l_i = l_i * alpha + tl.sum(p, axis=0)
            acc = acc * alpha + tl.sum(p[:, None] * v_tile, axis=0)
            m_i = m_new

        ms_off = pid_h * NUM_SPLITS + pid_s
        tl.store(part_m_ptr + ms_off, m_i)
        tl.store(part_l_ptr + ms_off, l_i)
        tl.store(
            part_acc_ptr + pid_h * stride_ah + pid_s * stride_as + offs_d,
            acc,
            mask=mask_d,
        )

    @triton.jit
    def _whisper_cross_attn_splitkv_stage2_fp8_kernel(
        part_acc_ptr,
        part_m_ptr,
        part_l_ptr,
        y_fp8_ptr,
        s_out_ptr,
        embed_dim,
        stride_ah,
        stride_as,
        HEAD_DIM: tl.constexpr,
        NUM_SPLITS: tl.constexpr,
        BLOCK_COLS: tl.constexpr,
    ):
        cols = tl.arange(0, BLOCK_COLS)
        mask_c = cols < embed_dim
        head = cols // HEAD_DIM
        dim = cols % HEAD_DIM

        m_max = tl.full((BLOCK_COLS,), -1.0e38, dtype=tl.float32)
        for s in tl.static_range(0, NUM_SPLITS):
            m_s = tl.load(part_m_ptr + head * NUM_SPLITS + s, mask=mask_c, other=-1.0e38)
            m_max = tl.maximum(m_max, m_s)

        l_sum = tl.zeros((BLOCK_COLS,), dtype=tl.float32)
        acc_sum = tl.zeros((BLOCK_COLS,), dtype=tl.float32)
        for s in tl.static_range(0, NUM_SPLITS):
            m_s = tl.load(part_m_ptr + head * NUM_SPLITS + s, mask=mask_c, other=-1.0e38)
            l_s = tl.load(part_l_ptr + head * NUM_SPLITS + s, mask=mask_c, other=0.0)
            a_s = tl.load(
                part_acc_ptr + head * stride_ah + s * stride_as + dim,
                mask=mask_c,
                other=0.0,
            )
            w_s = tl.exp(m_s - m_max)
            l_sum += l_s * w_s
            acc_sum += a_s * w_s

        y = tl.where(mask_c, acc_sum / tl.maximum(l_sum, 1e-12), 0.0)
        amax = tl.max(tl.abs(y), axis=0)
        scale = tl.maximum(amax, 1e-12) / 448.0
        tl.store(s_out_ptr, scale)
        q = tl.clamp(y / scale, -448.0, 448.0).to(tl.float8e4nv)
        tl.store(y_fp8_ptr + cols, q, mask=mask_c)

    _STT_TRITON_AVAILABLE = True
except (ImportError, RuntimeError, OSError):
    _STT_TRITON_AVAILABLE = False


def _fp8_mm_prequant(
    x_fp8: torch.Tensor,
    x_scale: torch.Tensor,
    lin: FP8Linear,
    orig_shape: torch.Size,
) -> torch.Tensor:
    """Execute FP8Linear matmul from pre-quantized rowwise FP8 activations."""
    out = torch._scaled_mm(
        x_fp8,
        lin.w_t,
        scale_a=x_scale,
        scale_b=lin.w_scale,
        bias=lin.bias,
        out_dtype=torch.bfloat16,
    )
    return out.view(*orig_shape[:-1], lin.out_features)


def _layernorm_fp8_quant(x: torch.Tensor, ln: nn.LayerNorm) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused Triton LayerNorm + rowwise FP8 quant (zero intermediate BF16 round-trip)."""
    n_cols = x.shape[-1]
    if (
        _STT_TRITON_AVAILABLE
        and x.is_cuda
        and ln.weight is not None
        and ln.bias is not None
    ):
        block = triton.next_power_of_2(n_cols)
        if block <= 8192:
            x2d = x.reshape(-1, n_cols).contiguous()
            m_rows = x2d.shape[0]
            y_fp8 = torch.empty((m_rows, n_cols), device=x2d.device, dtype=torch.float8_e4m3fn)
            s = torch.empty((m_rows, 1), device=x2d.device, dtype=torch.float32)
            _layernorm_fp8_quant_kernel[(m_rows,)](
                x2d,
                ln.weight,
                ln.bias,
                y_fp8,
                s,
                n_cols,
                1.0 / float(n_cols),
                x2d.stride(0),
                y_fp8.stride(0),
                float(ln.eps),
                BLOCK_SIZE=block,
                num_warps=4 if block <= 2048 else 8,
            )
            return y_fp8, s
    return quantize_rowwise_fp8(ln(x).reshape(-1, n_cols))


def _residual_add_layernorm_fp8_quant(
    residual: torch.Tensor,
    delta: torch.Tensor,
    ln: nn.LayerNorm,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused Triton ResidualAdd + LayerNorm + rowwise FP8 quant (writes updated BF16 residual + FP8)."""
    orig_shape = residual.shape
    n_cols = orig_shape[-1]
    if (
        _STT_TRITON_AVAILABLE
        and residual.is_cuda
        and ln.weight is not None
        and ln.bias is not None
    ):
        block = triton.next_power_of_2(n_cols)
        if block <= 8192:
            r2d = residual.reshape(-1, n_cols).contiguous()
            d2d = delta.reshape(-1, n_cols).contiguous()
            m_rows = r2d.shape[0]
            out_h = torch.empty_like(r2d)
            y_fp8 = torch.empty((m_rows, n_cols), device=r2d.device, dtype=torch.float8_e4m3fn)
            s = torch.empty((m_rows, 1), device=r2d.device, dtype=torch.float32)
            _residual_add_layernorm_fp8_quant_kernel[(m_rows,)](
                r2d,
                d2d,
                ln.weight,
                ln.bias,
                out_h,
                y_fp8,
                s,
                n_cols,
                1.0 / float(n_cols),
                r2d.stride(0),
                d2d.stride(0),
                out_h.stride(0),
                y_fp8.stride(0),
                float(ln.eps),
                BLOCK_SIZE=block,
                num_warps=4 if block <= 2048 else 8,
            )
            return out_h.view(orig_shape), y_fp8, s
    out_h = residual + delta
    y_fp8, s = _layernorm_fp8_quant(out_h, ln)
    return out_h, y_fp8, s


def _residual_add_layernorm_fp8_only(
    residual: torch.Tensor,
    delta: torch.Tensor,
    ln: nn.LayerNorm,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused Triton ResidualAdd + LayerNorm + rowwise FP8 quant without storing BF16 residual."""
    n_cols = residual.shape[-1]
    if (
        _STT_TRITON_AVAILABLE
        and residual.is_cuda
        and ln.weight is not None
        and ln.bias is not None
    ):
        block = triton.next_power_of_2(n_cols)
        if block <= 8192:
            r2d = residual.reshape(-1, n_cols).contiguous()
            d2d = delta.reshape(-1, n_cols).contiguous()
            m_rows = r2d.shape[0]
            y_fp8 = torch.empty((m_rows, n_cols), device=r2d.device, dtype=torch.float8_e4m3fn)
            s = torch.empty((m_rows, 1), device=r2d.device, dtype=torch.float32)
            _residual_add_layernorm_fp8_only_kernel[(m_rows,)](
                r2d,
                d2d,
                ln.weight,
                ln.bias,
                y_fp8,
                s,
                n_cols,
                1.0 / float(n_cols),
                r2d.stride(0),
                d2d.stride(0),
                y_fp8.stride(0),
                float(ln.eps),
                BLOCK_SIZE=block,
                num_warps=4 if block <= 2048 else 8,
            )
            return y_fp8, s
    return _layernorm_fp8_quant(residual + delta, ln)


def _residual_add_layernorm_bf16(
    residual: torch.Tensor,
    delta: torch.Tensor,
    ln: nn.LayerNorm,
) -> torch.Tensor:
    """Fused Triton ResidualAdd + LayerNorm returning BF16 output."""
    orig_shape = residual.shape
    n_cols = orig_shape[-1]
    if (
        _STT_TRITON_AVAILABLE
        and residual.is_cuda
        and ln.weight is not None
        and ln.bias is not None
    ):
        block = triton.next_power_of_2(n_cols)
        if block <= 8192:
            r2d = residual.reshape(-1, n_cols).contiguous()
            d2d = delta.reshape(-1, n_cols).contiguous()
            m_rows = r2d.shape[0]
            out = torch.empty_like(r2d)
            _residual_add_layernorm_bf16_kernel[(m_rows,)](
                r2d,
                d2d,
                ln.weight,
                ln.bias,
                out,
                n_cols,
                1.0 / float(n_cols),
                r2d.stride(0),
                d2d.stride(0),
                out.stride(0),
                float(ln.eps),
                BLOCK_SIZE=block,
                num_warps=4 if block <= 2048 else 8,
            )
            return out.view(orig_shape)
    return ln(residual + delta)


def _layernorm_fp8_linear(x: torch.Tensor, ln: nn.LayerNorm, lin: nn.Module) -> torch.Tensor:
    """Fused Triton LayerNorm + rowwise FP8 quant + FP8Linear (zero intermediate BF16 round-trip)."""
    if (
        _STT_TRITON_AVAILABLE
        and isinstance(lin, FP8Linear)
        and x.is_cuda
        and ln.weight is not None
        and ln.bias is not None
    ):
        y_fp8, s = _layernorm_fp8_quant(x, ln)
        return _fp8_mm_prequant(y_fp8, s, lin, x.shape)
    return lin(ln(x))


def _gelu_fp8_linear(x: torch.Tensor, act_fn, lin: nn.Module) -> torch.Tensor:
    """Fused Triton exact-erf GELU + rowwise FP8 quant + FP8Linear (zero intermediate BF16 round-trip)."""
    if (
        _STT_TRITON_AVAILABLE
        and isinstance(lin, FP8Linear)
        and x.is_cuda
        and (act_fn is F.gelu or type(act_fn).__name__ == "GELUActivation")
    ):
        n_cols = lin.in_features
        block = triton.next_power_of_2(n_cols)
        if block <= 8192:
            x2d = x.reshape(-1, n_cols).contiguous()
            m_rows = x2d.shape[0]
            y_fp8 = torch.empty((m_rows, n_cols), device=x2d.device, dtype=torch.float8_e4m3fn)
            s = torch.empty((m_rows, 1), device=x2d.device, dtype=torch.float32)
            _gelu_fp8_quant_kernel[(m_rows,)](
                x2d,
                y_fp8,
                s,
                n_cols,
                x2d.stride(0),
                y_fp8.stride(0),
                BLOCK_SIZE=block,
                num_warps=4 if block <= 2048 else 8,
            )
            return _fp8_mm_prequant(y_fp8, s, lin, x.shape)
    return lin(act_fn(x))


def _sdpa_out_fp8_linear(
    attn_out: torch.Tensor,
    out_proj: nn.Module,
    bsz: int,
    seq_len: int,
    embed_dim: int,
) -> torch.Tensor:
    """Fuse SDPA (1, H, T, D) -> (T, H*D) transpose + rowwise FP8 quant directly into out_proj."""
    if (
        _STT_TRITON_AVAILABLE
        and bsz == 1
        and seq_len > 1
        and isinstance(out_proj, FP8Linear)
        and attn_out.is_cuda
        and attn_out.stride(-1) == 1
    ):
        block = triton.next_power_of_2(embed_dim)
        if block <= 8192:
            head_dim = int(attn_out.shape[-1])
            y_fp8 = torch.empty((seq_len, embed_dim), device=attn_out.device, dtype=torch.float8_e4m3fn)
            s = torch.empty((seq_len, 1), device=attn_out.device, dtype=torch.float32)
            _sdpa_transpose_fp8_quant_kernel[(seq_len,)](
                attn_out,
                y_fp8,
                s,
                embed_dim,
                attn_out.stride(1),
                attn_out.stride(2),
                y_fp8.stride(0),
                HEAD_DIM=head_dim,
                BLOCK_SIZE=block,
                num_warps=4 if block <= 2048 else 8,
            )
            return _fp8_mm_prequant(y_fp8, s, out_proj, torch.Size((1, seq_len, embed_dim)))
    flat = attn_out.transpose(1, 2).contiguous().view(bsz, seq_len, embed_dim)
    return out_proj(flat)


def _split_kv_transpose_into_cache(
    kv: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    num_heads: int,
    head_dim: int,
) -> None:
    """Write fused KV (1, src_len, 2*D) directly into static (1, H, src_len, head_dim) caches."""
    embed_dim = num_heads * head_dim
    src_len = int(k_cache.shape[2])
    if (
        _STT_TRITON_AVAILABLE
        and kv.is_cuda
        and k_cache.stride(-1) == 1
        and v_cache.stride(-1) == 1
    ):
        kv2d = kv.reshape(src_len, 2 * embed_dim).contiguous()
        head_block = triton.next_power_of_2(head_dim)
        block_t = 32
        grid = (triton.cdiv(src_len, block_t), num_heads)
        _split_kv_transpose_cache_kernel[grid](
            kv2d,
            k_cache,
            v_cache,
            src_len,
            kv2d.stride(0),
            k_cache.stride(1),
            k_cache.stride(2),
            v_cache.stride(1),
            v_cache.stride(2),
            EMBED_DIM=embed_dim,
            HEAD_DIM=head_dim,
            HEAD_BLOCK=head_block,
            BLOCK_T=block_t,
            num_warps=4,
        )
        return
    k, v = kv.chunk(2, dim=-1)
    k_cache.copy_(k.view(1, src_len, num_heads, head_dim).transpose(1, 2))
    v_cache.copy_(v.view(1, src_len, num_heads, head_dim).transpose(1, 2))


def _whisper_qkv_scatter_step(
    qkv: torch.Tensor,
    pos_id: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    num_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """Split 1-token QKV into Q (1, H, 1, D) and scatter K/V into static KV cache in one kernel."""
    if (
        _STT_TRITON_AVAILABLE
        and qkv.is_cuda
        and k_cache.stride(-1) == 1
        and v_cache.stride(-1) == 1
    ):
        qkv_flat = qkv.reshape(-1).contiguous()
        q_out = torch.empty((1, num_heads, 1, head_dim), device=qkv.device, dtype=qkv.dtype)
        head_block = triton.next_power_of_2(head_dim)
        _whisper_qkv_scatter_step_kernel[(3 * num_heads,)](
            qkv_flat,
            pos_id,
            q_out,
            k_cache,
            v_cache,
            k_cache.stride(1),
            k_cache.stride(2),
            v_cache.stride(1),
            v_cache.stride(2),
            NUM_HEADS=num_heads,
            HEAD_DIM=head_dim,
            HEAD_BLOCK=head_block,
            num_warps=2 if head_block <= 128 else 4,
        )
        return q_out
    q, k, v = qkv.chunk(3, dim=-1)
    q = q.view(1, 1, num_heads, head_dim).transpose(1, 2)
    k = k.view(1, 1, num_heads, head_dim).transpose(1, 2)
    v = v.view(1, 1, num_heads, head_dim).transpose(1, 2)
    k_cache.index_copy_(2, pos_id, k)
    v_cache.index_copy_(2, pos_id, v)
    return q


def _whisper_cross_attn_step_fp8_linear(
    cq: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    out_proj: FP8Linear,
    part_acc: torch.Tensor,
    part_m: torch.Tensor,
    part_l: torch.Tensor,
    num_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """2-stage Split-KV Flash-Decoding cross-attention fused with rowwise FP8 quant -> out_proj."""
    embed_dim = num_heads * head_dim
    num_splits = int(part_m.shape[1])
    if (
        _STT_TRITON_AVAILABLE
        and cq.is_cuda
        and k_cache.stride(-1) == 1
        and v_cache.stride(-1) == 1
    ):
        block_cols = triton.next_power_of_2(embed_dim)
        if block_cols <= 8192:
            q_flat = cq.reshape(-1).contiguous()
            src_len = int(k_cache.shape[2])
            head_block = triton.next_power_of_2(head_dim)
            _whisper_cross_attn_splitkv_stage1_kernel[(num_heads, num_splits)](
                q_flat,
                k_cache,
                v_cache,
                part_acc,
                part_m,
                part_l,
                src_len,
                k_cache.stride(1),
                k_cache.stride(2),
                v_cache.stride(1),
                v_cache.stride(2),
                part_acc.stride(0),
                part_acc.stride(1),
                HEAD_DIM=head_dim,
                HEAD_BLOCK=head_block,
                BLOCK_T=64,
                NUM_SPLITS=num_splits,
                num_warps=4,
            )
            y_fp8 = torch.empty((1, embed_dim), device=cq.device, dtype=torch.float8_e4m3fn)
            s_out = torch.empty((1, 1), device=cq.device, dtype=torch.float32)
            _whisper_cross_attn_splitkv_stage2_fp8_kernel[(1,)](
                part_acc,
                part_m,
                part_l,
                y_fp8,
                s_out,
                embed_dim,
                part_acc.stride(0),
                part_acc.stride(1),
                HEAD_DIM=head_dim,
                NUM_SPLITS=num_splits,
                BLOCK_COLS=block_cols,
                num_warps=4 if block_cols <= 2048 else 8,
            )
            return _fp8_mm_prequant(y_fp8, s_out, out_proj, cq.shape)
    cq4 = cq.view(1, 1, num_heads, head_dim).transpose(1, 2)
    cross_out = F.scaled_dot_product_attention(
        cq4, k_cache, v_cache, dropout_p=0.0, is_causal=False, scale=1.0
    )
    return out_proj(cross_out.transpose(1, 2).reshape(1, 1, embed_dim))


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


def _has_speech_activity_tensor(audio: torch.Tensor, sample_rate: int = 16000) -> bool:
    """Exact torch equivalent of _has_speech_activity with a single host sync."""
    n = int(audio.numel())
    if n < int(sample_rate * 0.2):
        return False
    rms = torch.sqrt(torch.mean(audio * audio))
    peak = torch.max(torch.abs(audio))
    frame_len = max(1, int(sample_rate * 0.02))
    n_frames = n // frame_len
    if n_frames < 5:
        return bool(((rms >= 1.5e-3) & (peak >= 3e-3)).item())

    framed = audio[: n_frames * frame_len].view(n_frames, frame_len)
    frame_rms = torch.sqrt(torch.mean(framed * framed, dim=1))
    noise_floor = torch.quantile(frame_rms, 0.20)
    threshold = torch.clamp(noise_floor * 2.2, min=0.0025)
    voiced_frames = (frame_rms >= threshold).sum()
    active = (
        (rms >= 4e-4)
        & (peak >= 3e-3)
        & ((voiced_frames >= 3) | ((noise_floor >= 0.008) & (peak >= 0.02)))
    )
    return bool(active.item())


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
    return _sdpa_out_fp8_linear(attn_output, self.out_proj, bsz, q_len, self.embed_dim), None


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
    return _sdpa_out_fp8_linear(attn_output, self.out_proj, bsz, q_len, self.embed_dim), None


def _fused_encoder_layer_forward(
    self: nn.Module,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    **kwargs,
) -> torch.Tensor:
    """WhisperEncoderLayer forward with fused Triton ResidualAdd+LN+FP8, SDPA-out+FP8, and GELU+FP8."""
    bsz, q_len, embed_dim = hidden_states.shape
    attn = self.self_attn
    qkv = _layernorm_fp8_linear(hidden_states, self.self_attn_layer_norm, attn.qkv_proj)
    q, k, v = qkv.chunk(3, dim=-1)
    q = q.view(bsz, q_len, attn.num_heads, attn.head_dim).transpose(1, 2)
    k = k.view(bsz, q_len, attn.num_heads, attn.head_dim).transpose(1, 2)
    v = v.view(bsz, q_len, attn.num_heads, attn.head_dim).transpose(1, 2)
    attn_out = F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=attention_mask,
        dropout_p=0.0,
        is_causal=False,
        scale=1.0,
    )
    attn_proj = _sdpa_out_fp8_linear(attn_out, attn.out_proj, bsz, q_len, embed_dim)
    hidden_states, ffn_fp8, ffn_scale = _residual_add_layernorm_fp8_quant(
        hidden_states, attn_proj, self.final_layer_norm
    )
    ffn = _fp8_mm_prequant(ffn_fp8, ffn_scale, self.fc1, hidden_states.shape)
    return hidden_states + _gelu_fp8_linear(ffn, self.activation_fn, self.fc2)


def _can_fuse_encoder_pipeline(enc: nn.Module | None) -> bool:
    if (
        not _STT_TRITON_AVAILABLE
        or enc is None
        or not hasattr(enc, "layers")
        or len(enc.layers) == 0
        or not hasattr(enc, "layer_norm")
        or enc.layer_norm.weight is None
        or enc.layer_norm.bias is None
    ):
        return False
    embed_dim = int(enc.config.d_model)
    if triton.next_power_of_2(embed_dim) > 8192:
        return False
    for layer in enc.layers:
        attn = getattr(layer, "self_attn", None)
        if not (
            attn is not None
            and isinstance(getattr(attn, "qkv_proj", None), FP8Linear)
            and getattr(attn, "_q_scaled", False)
            and isinstance(getattr(attn, "out_proj", None), FP8Linear)
            and isinstance(getattr(layer, "fc1", None), FP8Linear)
            and isinstance(getattr(layer, "fc2", None), FP8Linear)
            and layer.self_attn_layer_norm.weight is not None
            and layer.self_attn_layer_norm.bias is not None
            and layer.final_layer_norm.weight is not None
            and layer.final_layer_norm.bias is not None
        ):
            return False
    return True


def _run_fused_encoder_body(
    enc: nn.Module,
    input_features: torch.Tensor,
    *,
    return_fp8: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Cross-layer fused WhisperEncoder (ResidualAdd+LN+FP8 across all 32 layers + final LN)."""
    inputs_embeds = F.gelu(enc.conv1(input_features))
    inputs_embeds = F.gelu(enc.conv2(inputs_embeds))
    pos_w = enc.embed_positions.weight.to(dtype=inputs_embeds.dtype).unsqueeze(0)
    h = inputs_embeds.permute(0, 2, 1) + pos_w

    bsz, q_len, embed_dim = h.shape
    orig_shape = h.shape
    layers = enc.layers
    n_layers = len(layers)
    x_fp8, x_scale = _layernorm_fp8_quant(h, layers[0].self_attn_layer_norm)

    for idx, layer in enumerate(layers):
        attn = layer.self_attn
        qkv = _fp8_mm_prequant(x_fp8, x_scale, attn.qkv_proj, orig_shape)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(bsz, q_len, attn.num_heads, attn.head_dim).transpose(1, 2)
        k = k.view(bsz, q_len, attn.num_heads, attn.head_dim).transpose(1, 2)
        v = v.view(bsz, q_len, attn.num_heads, attn.head_dim).transpose(1, 2)
        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=0.0,
            is_causal=False,
            scale=1.0,
        )
        attn_proj = _sdpa_out_fp8_linear(attn_out, attn.out_proj, bsz, q_len, embed_dim)
        h, ffn_fp8, ffn_scale = _residual_add_layernorm_fp8_quant(
            h, attn_proj, layer.final_layer_norm
        )
        ffn = _fp8_mm_prequant(ffn_fp8, ffn_scale, layer.fc1, orig_shape)
        mlp_out = _gelu_fp8_linear(ffn, layer.activation_fn, layer.fc2)
        if idx + 1 < n_layers:
            h, x_fp8, x_scale = _residual_add_layernorm_fp8_quant(
                h, mlp_out, layers[idx + 1].self_attn_layer_norm
            )
        elif return_fp8:
            return _residual_add_layernorm_fp8_only(h, mlp_out, enc.layer_norm)
        else:
            return _residual_add_layernorm_bf16(h, mlp_out, enc.layer_norm)
    return enc.layer_norm(h)


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

    if _STT_TRITON_AVAILABLE and enc is not None and hasattr(enc, "layers"):
        for layer in enc.layers:
            if (
                hasattr(layer, "self_attn")
                and isinstance(getattr(layer.self_attn, "qkv_proj", None), FP8Linear)
                and isinstance(getattr(layer, "fc1", None), FP8Linear)
                and isinstance(getattr(layer, "fc2", None), FP8Linear)
            ):
                layer.forward = types.MethodType(_fused_encoder_layer_forward, layer)

    return count


class _FastWhisperDecoder:
    """Static-KV-cache + CUDA-graphed greedy Whisper decoder (fused enc+prefill graph + 1-token step graph)."""

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
        self.encoder = getattr(inner, "encoder", None)
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
        self._enc_shape = torch.Size((1, self.src_len, self.embed_dim))

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

        self._all_decoder_fp8 = self._check_decoder_fp8_fusion()
        self.has_fused_encoder = (
            self._all_decoder_fp8
            and _can_fuse_encoder_pipeline(self.encoder)
            and self.encoder is not None
            and hasattr(self.encoder, "config")
        )
        self.expected_feature_shape: tuple[int, int, int] | None = None
        self._static_input_features: torch.Tensor | None = None
        self._static_enc_hidden: torch.Tensor | None = None
        if self.has_fused_encoder and self.encoder is not None:
            n_mels = int(getattr(self.encoder.config, "num_mel_bins", 128))
            mel_seq_len = (
                int(self.encoder.config.max_source_positions)
                * self.encoder.conv1.stride[0]
                * self.encoder.conv2.stride[0]
            )
            self.expected_feature_shape = (1, n_mels, mel_seq_len)
            self._static_input_features = torch.zeros(
                self.expected_feature_shape, device=device, dtype=dtype
            )
        else:
            self._static_enc_hidden = torch.zeros(
                (1, self.src_len, self.embed_dim), device=device, dtype=dtype
            )

        # Pre-allocated static buffers owned for CUDA graph lifetime
        self._self_k = [
            torch.zeros(
                (1, self.num_heads, self.max_cache_len, self.head_dim), device=device, dtype=dtype
            )
            for _ in range(self.num_layers)
        ]
        self._self_v = [
            torch.zeros(
                (1, self.num_heads, self.max_cache_len, self.head_dim), device=device, dtype=dtype
            )
            for _ in range(self.num_layers)
        ]
        self._cross_k = [
            torch.zeros(
                (1, self.num_heads, self.src_len, self.head_dim), device=device, dtype=dtype
            )
            for _ in range(self.num_layers)
        ]
        self._cross_v = [
            torch.zeros(
                (1, self.num_heads, self.src_len, self.head_dim), device=device, dtype=dtype
            )
            for _ in range(self.num_layers)
        ]
        self._arange = torch.arange(self.max_cache_len, device=device, dtype=torch.long)
        self._cur_token = torch.zeros((1, 1), device=device, dtype=torch.long)
        self._pos_id = torch.zeros((1,), device=device, dtype=torch.long)
        self._host_token = torch.empty((1, 1), dtype=torch.long, pin_memory=torch.cuda.is_available())
        # Pre-allocated L2-resident scratch for Split-KV cross-attention decoding (8 splits x 20 heads = 160 CTAs)
        self._cross_num_splits = 8
        self._cross_part_acc = torch.empty(
            (self.num_heads, self._cross_num_splits, self.head_dim),
            device=device,
            dtype=torch.float32,
        )
        self._cross_part_m = torch.empty(
            (self.num_heads, self._cross_num_splits), device=device, dtype=torch.float32
        )
        self._cross_part_l = torch.empty(
            (self.num_heads, self._cross_num_splits), device=device, dtype=torch.float32
        )

        self._enc_prefill_graph: torch.cuda.CUDAGraph | None = None
        self._prefill_graph: torch.cuda.CUDAGraph | None = None
        self._step_graph: torch.cuda.CUDAGraph | None = None
        self._capture_graphs()

    def _check_decoder_fp8_fusion(self) -> bool:
        if (
            not _STT_TRITON_AVAILABLE
            or self.num_layers == 0
            or triton.next_power_of_2(self.embed_dim) > 8192
            or self.decoder.layer_norm.weight is None
            or self.decoder.layer_norm.bias is None
        ):
            return False
        for layer in self.decoder.layers:
            sa = getattr(layer, "self_attn", None)
            ea = getattr(layer, "encoder_attn", None)
            if not (
                sa is not None
                and ea is not None
                and isinstance(getattr(sa, "qkv_proj", None), FP8Linear)
                and getattr(sa, "_q_scaled", False)
                and isinstance(getattr(sa, "out_proj", None), FP8Linear)
                and isinstance(getattr(ea, "q_proj", None), FP8Linear)
                and getattr(ea, "_q_scaled", False)
                and isinstance(getattr(ea, "kv_proj", None), FP8Linear)
                and isinstance(getattr(ea, "out_proj", None), FP8Linear)
                and isinstance(getattr(layer, "fc1", None), FP8Linear)
                and isinstance(getattr(layer, "fc2", None), FP8Linear)
                and layer.self_attn_layer_norm.weight is not None
                and layer.self_attn_layer_norm.bias is not None
                and layer.encoder_attn_layer_norm.weight is not None
                and layer.encoder_attn_layer_norm.bias is not None
                and layer.final_layer_norm.weight is not None
                and layer.final_layer_norm.bias is not None
            ):
                return False
        return True

    def _build_token_mask(self, token_ids: list[int] | tuple[int, ...] | None) -> torch.Tensor | None:
        if not token_ids:
            return None
        mask = torch.zeros(self.vocab_size, device=self.device, dtype=torch.bool)
        valid = [int(t) for t in token_ids if 0 <= int(t) < self.vocab_size]
        if not valid:
            return None
        mask[torch.tensor(valid, device=self.device, dtype=torch.long)] = True
        return mask

    def _project_self_qkv_ln(
        self, attn: nn.Module, ln: nn.LayerNorm, h: torch.Tensor, seq_len: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if hasattr(attn, "qkv_proj"):
            qkv = _layernorm_fp8_linear(h, ln, attn.qkv_proj)
            q, k, v = qkv.chunk(3, dim=-1)
            if not getattr(attn, "_q_scaled", False):
                q = q * attn.scaling
        else:
            normed = ln(h)
            q = attn.q_proj(normed)
            if not getattr(attn, "_q_scaled", False):
                q = q * attn.scaling
            k = attn.k_proj(normed)
            v = attn.v_proj(normed)
        q = q.view(1, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(1, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(1, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        return q, k, v

    def _project_cross_kv(
        self,
        enc_attn: nn.Module,
        enc_hidden: torch.Tensor,
        enc_fp8_pair: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        kv_proj = getattr(enc_attn, "kv_proj", None)
        if kv_proj is not None:
            if enc_fp8_pair is not None and isinstance(kv_proj, FP8Linear):
                kv = _fp8_mm_prequant(enc_fp8_pair[0], enc_fp8_pair[1], kv_proj, enc_hidden.shape)
            else:
                kv = kv_proj(enc_hidden)
            k, v = kv.chunk(2, dim=-1)
        else:
            k = enc_attn.k_proj(enc_hidden)
            v = enc_attn.v_proj(enc_hidden)
        k = k.view(1, self.src_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(1, self.src_len, self.num_heads, self.head_dim).transpose(1, 2)
        return k, v

    def _project_cross_q_ln(
        self, enc_attn: nn.Module, ln: nn.LayerNorm, h: torch.Tensor, seq_len: int
    ) -> torch.Tensor:
        q = _layernorm_fp8_linear(h, ln, enc_attn.q_proj)
        if not getattr(enc_attn, "_q_scaled", False):
            q = q * enc_attn.scaling
        return q.view(1, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def _prefill_fused_fp8(self, enc_fp8_pair: tuple[torch.Tensor, torch.Tensor]) -> None:
        h = self.decoder.embed_tokens(self._static_prefix_ids) + self._static_prefix_pos
        p_len = self.prefix_len
        orig_shape = h.shape
        layers = self.decoder.layers
        n_layers = self.num_layers
        enc_fp8, enc_scale = enc_fp8_pair

        x_fp8, x_scale = _layernorm_fp8_quant(h, layers[0].self_attn_layer_norm)
        for idx, layer in enumerate(layers):
            qkv = _fp8_mm_prequant(x_fp8, x_scale, layer.self_attn.qkv_proj, orig_shape)
            q, k, v = qkv.chunk(3, dim=-1)
            q = q.view(1, p_len, self.num_heads, self.head_dim).transpose(1, 2)
            k = k.view(1, p_len, self.num_heads, self.head_dim).transpose(1, 2)
            v = v.view(1, p_len, self.num_heads, self.head_dim).transpose(1, 2)
            self._self_k[idx][:, :, :p_len, :].copy_(k)
            self._self_v[idx][:, :, :p_len, :].copy_(v)
            attn_out = F.scaled_dot_product_attention(
                q, k, v, dropout_p=0.0, is_causal=True, scale=1.0
            )
            self_proj = _sdpa_out_fp8_linear(
                attn_out, layer.self_attn.out_proj, 1, p_len, self.embed_dim
            )

            h, cross_fp8, cross_scale = _residual_add_layernorm_fp8_quant(
                h, self_proj, layer.encoder_attn_layer_norm
            )
            kv = _fp8_mm_prequant(enc_fp8, enc_scale, layer.encoder_attn.kv_proj, self._enc_shape)
            _split_kv_transpose_into_cache(
                kv, self._cross_k[idx], self._cross_v[idx], self.num_heads, self.head_dim
            )
            cq = _fp8_mm_prequant(cross_fp8, cross_scale, layer.encoder_attn.q_proj, orig_shape)
            cq = cq.view(1, p_len, self.num_heads, self.head_dim).transpose(1, 2)
            cross_out = F.scaled_dot_product_attention(
                cq,
                self._cross_k[idx],
                self._cross_v[idx],
                dropout_p=0.0,
                is_causal=False,
                scale=1.0,
            )
            cross_proj = _sdpa_out_fp8_linear(
                cross_out, layer.encoder_attn.out_proj, 1, p_len, self.embed_dim
            )

            h, ffn_fp8, ffn_scale = _residual_add_layernorm_fp8_quant(
                h, cross_proj, layer.final_layer_norm
            )
            ffn = _fp8_mm_prequant(ffn_fp8, ffn_scale, layer.fc1, orig_shape)
            mlp_out = _gelu_fp8_linear(ffn, layer.activation_fn, layer.fc2)

            if idx + 1 < n_layers:
                h, x_fp8, x_scale = _residual_add_layernorm_fp8_quant(
                    h, mlp_out, layers[idx + 1].self_attn_layer_norm
                )
            else:
                last_h = _residual_add_layernorm_bf16(
                    h[:, -1:, :], mlp_out[:, -1:, :], self.decoder.layer_norm
                )

        logits = self.proj_out(last_h).view(-1)
        if self._suppress_mask is not None:
            logits = logits.masked_fill(self._suppress_mask, float("-inf"))
        if self._begin_suppress_mask is not None:
            logits = logits.masked_fill(self._begin_suppress_mask, float("-inf"))
        first_tok = logits.argmax(dim=-1)
        self._cur_token.copy_(first_tok.view(1, 1))
        self._pos_id.fill_(p_len)

    def _enc_prefill_fn(self) -> None:
        assert self.encoder is not None and self._static_input_features is not None
        enc_fp8_pair = _run_fused_encoder_body(
            self.encoder, self._static_input_features, return_fp8=True
        )
        assert isinstance(enc_fp8_pair, tuple)
        self._prefill_fused_fp8(enc_fp8_pair)

    def _prefill_fn(self) -> None:
        assert self._static_enc_hidden is not None
        if self._all_decoder_fp8:
            enc_fp8_pair = quantize_rowwise_fp8(
                self._static_enc_hidden.reshape(-1, self.embed_dim)
            )
            self._prefill_fused_fp8(enc_fp8_pair)
            return

        h = self.decoder.embed_tokens(self._static_prefix_ids) + self._static_prefix_pos
        p_len = self.prefix_len
        first_enc = self.decoder.layers[0].encoder_attn if self.decoder.layers else None
        enc_fp8_pair = (
            quantize_rowwise_fp8(self._static_enc_hidden.reshape(-1, self.embed_dim))
            if first_enc is not None and isinstance(getattr(first_enc, "kv_proj", None), FP8Linear)
            else None
        )

        for idx, layer in enumerate(self.decoder.layers):
            residual = h
            q, k, v = self._project_self_qkv_ln(layer.self_attn, layer.self_attn_layer_norm, h, p_len)
            self._self_k[idx][:, :, :p_len, :].copy_(k)
            self._self_v[idx][:, :, :p_len, :].copy_(v)
            attn_out = F.scaled_dot_product_attention(
                q, k, v, dropout_p=0.0, is_causal=True, scale=1.0
            )
            attn_out = attn_out.transpose(1, 2).contiguous().view(1, p_len, self.embed_dim)
            h = residual + layer.self_attn.out_proj(attn_out)

            residual = h
            ck, cv = self._project_cross_kv(
                layer.encoder_attn, self._static_enc_hidden, enc_fp8_pair
            )
            self._cross_k[idx].copy_(ck)
            self._cross_v[idx].copy_(cv)
            cq = self._project_cross_q_ln(layer.encoder_attn, layer.encoder_attn_layer_norm, h, p_len)
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
            ffn = _layernorm_fp8_linear(h, layer.final_layer_norm, layer.fc1)
            h = residual + _gelu_fp8_linear(ffn, layer.activation_fn, layer.fc2)

        last_h = self.decoder.layer_norm(h[:, -1:, :])
        logits = self.proj_out(last_h).view(-1)
        if self._suppress_mask is not None:
            logits = logits.masked_fill(self._suppress_mask, float("-inf"))
        if self._begin_suppress_mask is not None:
            logits = logits.masked_fill(self._begin_suppress_mask, float("-inf"))
        first_tok = logits.argmax(dim=-1)
        self._cur_token.copy_(first_tok.view(1, 1))
        self._pos_id.fill_(p_len)

    def _step_fn(self) -> None:
        pos_embed = F.embedding(self._pos_id, self.decoder.embed_positions.weight).unsqueeze(0)
        h = self.decoder.embed_tokens(self._cur_token) + pos_embed
        attn_mask = (self._arange <= self._pos_id).view(1, 1, 1, -1)

        if self._all_decoder_fp8:
            orig_shape = h.shape
            layers = self.decoder.layers
            n_layers = self.num_layers
            x_fp8, x_scale = _layernorm_fp8_quant(h, layers[0].self_attn_layer_norm)

            for idx, layer in enumerate(layers):
                qkv = _fp8_mm_prequant(x_fp8, x_scale, layer.self_attn.qkv_proj, orig_shape)
                q = _whisper_qkv_scatter_step(
                    qkv,
                    self._pos_id,
                    self._self_k[idx],
                    self._self_v[idx],
                    self.num_heads,
                    self.head_dim,
                )
                attn_out = F.scaled_dot_product_attention(
                    q,
                    self._self_k[idx],
                    self._self_v[idx],
                    attn_mask=attn_mask,
                    dropout_p=0.0,
                    is_causal=False,
                    scale=1.0,
                )
                self_proj = layer.self_attn.out_proj(
                    attn_out.transpose(1, 2).reshape(1, 1, self.embed_dim)
                )

                h, cross_fp8, cross_scale = _residual_add_layernorm_fp8_quant(
                    h, self_proj, layer.encoder_attn_layer_norm
                )
                cq = _fp8_mm_prequant(cross_fp8, cross_scale, layer.encoder_attn.q_proj, orig_shape)
                cross_proj = _whisper_cross_attn_step_fp8_linear(
                    cq,
                    self._cross_k[idx],
                    self._cross_v[idx],
                    layer.encoder_attn.out_proj,
                    self._cross_part_acc,
                    self._cross_part_m,
                    self._cross_part_l,
                    self.num_heads,
                    self.head_dim,
                )

                h, ffn_fp8, ffn_scale = _residual_add_layernorm_fp8_quant(
                    h, cross_proj, layer.final_layer_norm
                )
                ffn = _fp8_mm_prequant(ffn_fp8, ffn_scale, layer.fc1, orig_shape)
                mlp_out = _gelu_fp8_linear(ffn, layer.activation_fn, layer.fc2)

                if idx + 1 < n_layers:
                    h, x_fp8, x_scale = _residual_add_layernorm_fp8_quant(
                        h, mlp_out, layers[idx + 1].self_attn_layer_norm
                    )
                else:
                    h = _residual_add_layernorm_bf16(h, mlp_out, self.decoder.layer_norm)

            logits = self.proj_out(h).view(-1)
            if self._suppress_mask is not None:
                logits = logits.masked_fill(self._suppress_mask, float("-inf"))
            next_tok = logits.argmax(dim=-1)
            self._cur_token.copy_(next_tok.view(1, 1))
            self._pos_id.add_(1)
            return

        for idx, layer in enumerate(self.decoder.layers):
            residual = h
            q, k, v = self._project_self_qkv_ln(layer.self_attn, layer.self_attn_layer_norm, h, 1)
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
            cq = self._project_cross_q_ln(layer.encoder_attn, layer.encoder_attn_layer_norm, h, 1)
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
            ffn = _layernorm_fp8_linear(h, layer.final_layer_norm, layer.fc1)
            h = residual + _gelu_fp8_linear(ffn, layer.activation_fn, layer.fc2)

        h = self.decoder.layer_norm(h)
        logits = self.proj_out(h).view(-1)
        if self._suppress_mask is not None:
            logits = logits.masked_fill(self._suppress_mask, float("-inf"))
        next_tok = logits.argmax(dim=-1)
        self._cur_token.copy_(next_tok.view(1, 1))
        self._pos_id.add_(1)

    def _capture_graphs(self) -> None:
        prefill_entry = self._enc_prefill_fn if self.has_fused_encoder else self._prefill_fn
        with CUDA_GRAPH_LOCK:
            torch.cuda.synchronize()
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.inference_mode(), torch.cuda.stream(stream):
                for _ in range(3):
                    prefill_entry()
                for _ in range(3):
                    self._pos_id.fill_(self.prefix_len)
                    self._step_fn()
            torch.cuda.current_stream().wait_stream(stream)

            first_graph = torch.cuda.CUDAGraph()
            with torch.inference_mode(), torch.cuda.graph(first_graph):
                prefill_entry()

            self._pos_id.fill_(self.prefix_len)
            step_graph = torch.cuda.CUDAGraph()
            with torch.inference_mode(), torch.cuda.graph(step_graph):
                self._step_fn()

            torch.cuda.synchronize()

        if self.has_fused_encoder:
            self._enc_prefill_graph = first_graph
        else:
            self._prefill_graph = first_graph
        self._step_graph = step_graph
        for k_buf, v_buf in zip(self._self_k, self._self_v, strict=True):
            k_buf.zero_()
            v_buf.zero_()

    def _read_cur_token(self) -> int:
        self._host_token.copy_(self._cur_token)
        return int(self._host_token.item())

    def _run_step_loop(self, max_steps: int) -> list[int]:
        assert self._step_graph is not None
        tok = self._read_cur_token()
        if tok == self.eos_token_id:
            return []

        tokens = [tok]
        for _ in range(max_steps - 1):
            self._step_graph.replay()
            tok = self._read_cur_token()
            if tok == self.eos_token_id:
                break
            tokens.append(tok)
        return tokens

    @torch.inference_mode()
    def decode_from_features(
        self, input_features: torch.Tensor, max_new_tokens: int = 160
    ) -> list[int]:
        assert self._enc_prefill_graph is not None and self._static_input_features is not None
        max_steps = min(int(max_new_tokens), self.max_cache_len - self.prefix_len)
        if max_steps <= 0:
            return []

        self._static_input_features.copy_(input_features)
        self._enc_prefill_graph.replay()
        return self._run_step_loop(max_steps)

    @torch.inference_mode()
    def decode(self, encoder_hidden_states: torch.Tensor, max_new_tokens: int = 160) -> list[int]:
        assert self._prefill_graph is not None and self._static_enc_hidden is not None
        max_steps = min(int(max_new_tokens), self.max_cache_len - self.prefix_len)
        if max_steps <= 0:
            return []

        self._static_enc_hidden.copy_(encoder_hidden_states)
        self._prefill_graph.replay()
        return self._run_step_loop(max_steps)


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

        self._resamplers: dict[int, nn.Module] = {}
        if self.device.startswith("cuda") and torch.cuda.is_available():
            self._get_resampler(48000)

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
        inner = getattr(self.model, "model", self.model)
        enc = getattr(inner, "encoder", None)

        if self.dtype == torch.bfloat16 and probe_fp8_supported(self.device):
            try:
                n_quant = _quantize_whisper_fp8(inner)
                torch.cuda.empty_cache()
                self.fp8_enabled = n_quant > 0
            except (RuntimeError, ValueError, TypeError, AttributeError, OSError) as exc:
                logger.debug("Whisper FP8 acceleration skipped: %s", exc)

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

        if enc is not None and hasattr(enc, "config"):
            n_mels = int(getattr(enc.config, "num_mel_bins", 128))
            seq_len = (
                int(enc.config.max_source_positions)
                * enc.conv1.stride[0]
                * enc.conv2.stride[0]
            )
            raw_enc_fwd = enc.forward
            can_fuse_enc = _can_fuse_encoder_pipeline(enc)

            if self._fast_decoder is not None and self._fast_decoder.has_fused_encoder:
                self.encoder_graphed = True

                def _fast_fused_enc_eager(
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
                        out = _run_fused_encoder_body(enc, input_features, return_fp8=False)
                        assert isinstance(out, torch.Tensor)
                        return BaseModelOutput(last_hidden_state=out)
                    return raw_enc_fwd(input_features, attention_mask=attention_mask, **kwargs)

                enc.forward = _fast_fused_enc_eager  # type: ignore[method-assign]
            elif self.fp8_enabled:
                try:
                    sample_feats = torch.zeros(
                        (1, n_mels, seq_len), device=self.device, dtype=self.dtype
                    )

                    def _enc_body(feats: torch.Tensor) -> torch.Tensor:
                        if can_fuse_enc:
                            out = _run_fused_encoder_body(enc, feats, return_fp8=False)
                            assert isinstance(out, torch.Tensor)
                            return out
                        return raw_enc_fwd(feats).last_hidden_state

                    with torch.inference_mode():
                        replay_enc = capture_cuda_graph(_enc_body, (sample_feats,))

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
                    logger.debug("Whisper encoder CUDA-graph skipped: %s", exc)

        logger.info(
            "Loaded Whisper STT (%s) on %s (%s%s%s) in %.2fs",
            model_id,
            self.device,
            self.dtype,
            " + FP8" if self.fp8_enabled else "",
            " + CG-Decoder" if self.decoder_graphed else "",
            time.perf_counter() - t0,
        )

    def _get_resampler(self, orig_sr: int) -> nn.Module | None:
        """Return a cached Kaiser-best polyphase FIR resampler (orig_sr -> 16 kHz) on target device."""
        resampler = self._resamplers.get(orig_sr)
        if resampler is not None:
            return resampler
        try:
            import torchaudio
        except (ImportError, OSError):
            return None
        resampler = torchaudio.transforms.Resample(
            orig_freq=orig_sr,
            new_freq=16000,
            resampling_method="sinc_interp_kaiser",
            lowpass_filter_width=64,
            rolloff=0.9475937167399596,
            beta=14.769656459379492,
        ).to(device=self.device, dtype=torch.float32)
        self._resamplers[orig_sr] = resampler
        return resampler

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

    def _extract_input_features_tensor(self, wav_16k: torch.Tensor) -> torch.Tensor:
        """Compute exact Whisper log-mel spectrogram directly from a 1D float32 tensor."""
        if (
            self._mel_window is None
            or self._mel_filters_t is None
            or self._waveform_buf is None
        ):
            inputs = self.processor(
                wav_16k.detach().cpu().numpy(),
                sampling_rate=16000,
                return_tensors="pt",
                return_attention_mask=False,
            )
            return inputs.input_features.to(device=self.device, dtype=self.dtype)

        n = min(int(wav_16k.numel()), self._mel_n_samples)
        wav = self._waveform_buf
        if n < self._mel_n_samples:
            wav[0, n:].zero_()
        if n > 0:
            wav[0, :n].copy_(wav_16k[:n])
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
        src = torch.from_numpy(np.ascontiguousarray(audio_16k[:n], dtype=np.float32))
        return self._extract_input_features_tensor(src)

    def _generate_greedy(self, input_features: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """Run fast CUDA-graphed greedy decoding with clean GenerationMixin fallback."""
        if self._fast_decoder is not None:
            if (
                self._fast_decoder.has_fused_encoder
                and input_features.shape == self._fast_decoder.expected_feature_shape
                and input_features.dtype == self.dtype
            ):
                tokens = self._fast_decoder.decode_from_features(
                    input_features, max_new_tokens=max_new_tokens
                )
            else:
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
        """Warm Whisper FP8/CUDA-graph encoder, decoder, and 48kHz resampler kernels."""
        t0 = time.perf_counter()
        res48 = self._resamplers.get(48000)
        if res48 is not None:
            _ = res48(torch.zeros(4800, device=self.device, dtype=torch.float32))
        t = np.linspace(0.0, 1.0, 16000, endpoint=False, dtype=np.float32)
        probe = (0.04 * np.sin(2.0 * np.pi * 220.0 * t)).astype(np.float32)
        input_features = self._extract_input_features(probe)
        _ = self._generate_greedy(input_features, max_new_tokens=64)
        elapsed = time.perf_counter() - t0
        logger.info("Warmed up Whisper STT (%s) in %.2fs", self.model_id, elapsed)
        return elapsed

    @torch.inference_mode()
    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """Transcribe a mono/multi-channel float waveform (auto-resamples to 16 kHz on GPU)."""
        arr = np.asarray(audio, dtype=np.float32)
        if arr.ndim > 1:
            arr = arr.mean(axis=-1).astype(np.float32, copy=False)
        else:
            arr = arr.reshape(-1)
        if arr.size < int(sample_rate * 0.2) or not np.all(np.isfinite(arr)):
            return ""

        if sample_rate != 16000:
            resampler = self._get_resampler(sample_rate)
            if resampler is not None:
                wav_t = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32)).to(
                    device=self.device, dtype=torch.float32
                )
                wav_t = resampler(wav_t)
                wav_t = wav_t - wav_t.mean()
                if not _has_speech_activity_tensor(wav_t, 16000):
                    return ""
                input_features = self._extract_input_features_tensor(wav_t)
            else:
                import librosa

                arr = librosa.resample(arr, orig_sr=sample_rate, target_sr=16000).astype(
                    np.float32, copy=False
                )
                arr = arr - float(np.mean(arr))
                if not _has_speech_activity(arr, 16000):
                    return ""
                input_features = self._extract_input_features(arr)
        else:
            arr = arr - float(np.mean(arr))
            if not _has_speech_activity(arr, 16000):
                return ""
            input_features = self._extract_input_features(arr)

        predicted_ids = self._generate_greedy(input_features, max_new_tokens=160)
        text = self.processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return text.strip()
