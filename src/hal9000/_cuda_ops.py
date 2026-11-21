"""Shared Blackwell/CUDA inference kernels: native FP8, Split-K INT4 GEMV, and CUDA Graph wrappers."""

from __future__ import annotations

import logging
import threading

import torch
import torch.nn.functional as F
from torch import nn

logger = logging.getLogger(__name__)
CUDA_GRAPH_LOCK = threading.Lock()
_FP8_PROBE_CACHE: dict[str, bool] = {}

_FP8_MAX = 448.0

try:
    import triton
    import triton.language as tl

    @triton.jit
    def _rowwise_fp8_quant_kernel(
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
        amax = tl.max(tl.abs(x), axis=0)
        scale = tl.maximum(amax, 1e-12) / 448.0
        tl.store(s_ptr + row, scale)
        q = tl.clamp(x / scale, -448.0, 448.0).to(tl.float8e4nv)
        tl.store(y_ptr + row * stride_y + cols, q, mask=mask)

    @triton.jit
    def _rmsnorm_rowwise_fp8_quant_kernel(
        x_ptr,
        w_ptr,
        y_ptr,
        s_ptr,
        n_cols,
        stride_x,
        stride_y,
        EPS: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        x = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        mean_sq = tl.sum(x * x, axis=0) / n_cols
        rstd = tl.rsqrt(mean_sq + EPS)
        normed = (x * rstd) * w
        amax = tl.max(tl.abs(normed), axis=0)
        scale = tl.maximum(amax, 1e-12) / 448.0
        tl.store(s_ptr + row, scale)
        q = tl.clamp(normed / scale, -448.0, 448.0).to(tl.float8e4nv)
        tl.store(y_ptr + row * stride_y + cols, q, mask=mask)

    @triton.jit
    def _silu_mul_rowwise_fp8_quant_kernel(
        gu_ptr,
        y_ptr,
        s_ptr,
        n_cols,
        stride_gu,
        stride_y,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        row_ptr = gu_ptr + row * stride_gu
        g = tl.load(row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        u = tl.load(row_ptr + n_cols + cols, mask=mask, other=0.0).to(tl.float32)
        act = (g * tl.sigmoid(g)) * u
        amax = tl.max(tl.abs(act), axis=0)
        scale = tl.maximum(amax, 1e-12) / 448.0
        tl.store(s_ptr + row, scale)
        q = tl.clamp(act / scale, -448.0, 448.0).to(tl.float8e4nv)
        tl.store(y_ptr + row * stride_y + cols, q, mask=mask)

    @triton.jit
    def _int4_gemv_splitk_kernel(
        x_ptr,
        qw_ptr,
        s_ptr,
        part_ptr,
        N,
        K,
        stride_qw_n,
        stride_s_n,
        GROUP_SIZE: tl.constexpr,
        BLOCK_N: tl.constexpr,
        SPLIT_K: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        pid_k = tl.program_id(1)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N

        num_groups = K // GROUP_SIZE
        groups_per_split = tl.cdiv(num_groups, SPLIT_K)
        g_start = pid_k * groups_per_split
        g_end = tl.minimum(g_start + groups_per_split, num_groups)

        half_g: tl.constexpr = GROUP_SIZE // 2
        offs_gh = tl.arange(0, half_g)

        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for g in range(g_start, g_end):
            k_base = g * GROUP_SIZE
            packed = tl.load(
                qw_ptr + offs_n[:, None] * stride_qw_n + (k_base // 2 + offs_gh)[None, :],
                mask=mask_n[:, None],
                other=0,
            )
            lo = (packed & 0x0F).to(tl.float32) - 8.0
            hi = ((packed >> 4) & 0x0F).to(tl.float32) - 8.0

            k_even = k_base + offs_gh * 2
            x_even = tl.load(x_ptr + k_even).to(tl.float32)
            x_odd = tl.load(x_ptr + k_even + 1).to(tl.float32)

            s = tl.load(s_ptr + offs_n * stride_s_n + g, mask=mask_n, other=0.0).to(tl.float32)
            dot = tl.sum(lo * x_even[None, :] + hi * x_odd[None, :], axis=1)
            acc += dot * s

        if SPLIT_K == 1:
            tl.store(part_ptr + offs_n, acc.to(tl.bfloat16), mask=mask_n)
        else:
            tl.store(part_ptr + pid_k * N + offs_n, acc, mask=mask_n)

    @triton.jit
    def _snake1d_fp32_kernel(
        x_ptr,
        alpha_ptr,
        inv_alpha_ptr,
        y_ptr,
        T,
        BLOCK_T: tl.constexpr,
    ):
        pid_t = tl.program_id(0)
        pid_c = tl.program_id(1)
        t_offs = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
        mask = t_offs < T
        row_off = pid_c * T + t_offs
        x = tl.load(x_ptr + row_off, mask=mask, other=0.0).to(tl.float32)
        a = tl.load(alpha_ptr + pid_c).to(tl.float32)
        inv_a = tl.load(inv_alpha_ptr + pid_c).to(tl.float32)
        s = tl.sin(a * x)
        y = x + inv_a * (s * s)
        tl.store(y_ptr + row_off, y, mask=mask)

    @triton.jit
    def _sr_cond_snake1d_fp32_kernel(
        x_ptr,
        scale_ptr,
        bias_ptr,
        alpha_ptr,
        inv_alpha_ptr,
        y_ptr,
        T,
        BLOCK_T: tl.constexpr,
    ):
        pid_t = tl.program_id(0)
        pid_c = tl.program_id(1)
        t_offs = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
        mask = t_offs < T
        row_off = pid_c * T + t_offs
        x = tl.load(x_ptr + row_off, mask=mask, other=0.0).to(tl.float32)
        sc = tl.load(scale_ptr + pid_c).to(tl.float32)
        bi = tl.load(bias_ptr + pid_c).to(tl.float32)
        xs = x * sc + bi
        a = tl.load(alpha_ptr + pid_c).to(tl.float32)
        inv_a = tl.load(inv_alpha_ptr + pid_c).to(tl.float32)
        s = tl.sin(a * xs)
        y = xs + inv_a * (s * s)
        tl.store(y_ptr + row_off, y, mask=mask)

    TRITON_AVAILABLE = True
except (ImportError, RuntimeError, OSError):
    TRITON_AVAILABLE = False


def quantize_rowwise_fp8(x2d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    m_rows, n_cols = x2d.shape
    x2d = x2d.contiguous()
    if TRITON_AVAILABLE:
        block = triton.next_power_of_2(n_cols)
        if block <= 8192:
            y = torch.empty((m_rows, n_cols), device=x2d.device, dtype=torch.float8_e4m3fn)
            s = torch.empty((m_rows, 1), device=x2d.device, dtype=torch.float32)
            _rowwise_fp8_quant_kernel[(m_rows,)](
                x2d, y, s, n_cols, x2d.stride(0), y.stride(0), BLOCK_SIZE=block
            )
            return y, s
    amax = x2d.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
    s = (amax / _FP8_MAX).to(torch.float32)
    y = (x2d / s.to(x2d.dtype)).clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)
    return y, s


def rmsnorm_quantize_rowwise_fp8(
    x2d: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    m_rows, n_cols = x2d.shape
    x2d = x2d.contiguous()
    if TRITON_AVAILABLE and x2d.is_cuda:
        block = triton.next_power_of_2(n_cols)
        if block <= 8192:
            w = weight.contiguous()
            y = torch.empty((m_rows, n_cols), device=x2d.device, dtype=torch.float8_e4m3fn)
            s = torch.empty((m_rows, 1), device=x2d.device, dtype=torch.float32)
            _rmsnorm_rowwise_fp8_quant_kernel[(m_rows,)](
                x2d,
                w,
                y,
                s,
                n_cols,
                x2d.stride(0),
                y.stride(0),
                EPS=float(eps),
                BLOCK_SIZE=block,
                num_warps=8 if block >= 4096 else 4,
            )
            return y, s
    xf = x2d.float()
    normed = (xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)).to(x2d.dtype) * weight
    return quantize_rowwise_fp8(normed)


def silu_mul_quantize_rowwise_fp8(gu2d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    m_rows, two_n = gu2d.shape
    n_cols = two_n // 2
    gu2d = gu2d.contiguous()
    if TRITON_AVAILABLE and gu2d.is_cuda:
        block = triton.next_power_of_2(n_cols)
        if block <= 8192:
            y = torch.empty((m_rows, n_cols), device=gu2d.device, dtype=torch.float8_e4m3fn)
            s = torch.empty((m_rows, 1), device=gu2d.device, dtype=torch.float32)
            _silu_mul_rowwise_fp8_quant_kernel[(m_rows,)](
                gu2d,
                y,
                s,
                n_cols,
                gu2d.stride(0),
                y.stride(0),
                BLOCK_SIZE=block,
                num_warps=8 if block >= 4096 else 4,
            )
            return y, s
    g, u = gu2d.chunk(2, dim=-1)
    return quantize_rowwise_fp8(F.silu(g) * u)


class FP8Linear(nn.Module):
    """Native CUDA FP8 (float8_e4m3fn) linear layer backed by ``torch._scaled_mm``."""

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor | None = None) -> None:
        super().__init__()
        out_f, in_f = weight.shape
        self.out_features = out_f
        self.in_features = in_f
        with torch.no_grad():
            wf = weight.float()
            w_amax = wf.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
            w_scale = (w_amax / _FP8_MAX).to(torch.float32)
            w_fp8 = (wf / w_scale).clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)
            self.register_buffer("w_t", w_fp8.t())
            self.register_buffer("w_scale", w_scale.t().contiguous())
        if bias is not None:
            self.register_buffer("bias", bias.detach().to(torch.bfloat16))
        else:
            self.bias = None

    def forward_prequant(
        self,
        x_fp8: torch.Tensor,
        x_scale: torch.Tensor,
        orig_prefix_shape: tuple[int, ...],
    ) -> torch.Tensor:
        out = torch._scaled_mm(
            x_fp8,
            self.w_t,
            scale_a=x_scale,
            scale_b=self.w_scale,
            bias=self.bias,
            out_dtype=torch.bfloat16,
        )
        return out.view(*orig_prefix_shape, self.out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_fp8, x_scale = quantize_rowwise_fp8(x.reshape(-1, self.in_features))
        return self.forward_prequant(x_fp8, x_scale, orig_shape[:-1])

    def forward_rmsnorm(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        orig_shape = x.shape
        x_fp8, x_scale = rmsnorm_quantize_rowwise_fp8(
            x.reshape(-1, self.in_features), weight, eps
        )
        return self.forward_prequant(x_fp8, x_scale, orig_shape[:-1])

    def forward_silu_mul(self, gate_up: torch.Tensor) -> torch.Tensor:
        orig_shape = gate_up.shape
        x_fp8, x_scale = silu_mul_quantize_rowwise_fp8(
            gate_up.reshape(-1, self.in_features * 2)
        )
        return self.forward_prequant(x_fp8, x_scale, orig_shape[:-1])


class Int4Linear(nn.Module):
    """Group-128 symmetric INT4 linear with Triton W4A16 GEMV for M=1 and BF16 fallback for prefill."""

    def __init__(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        group_size: int = 128,
    ) -> None:
        super().__init__()
        out_f, in_f = weight.shape
        if in_f % group_size != 0:
            raise ValueError(f"in_features ({in_f}) must be divisible by group_size ({group_size})")
        self.out_features = out_f
        self.in_features = in_f
        self.group_size = group_size
        self.split_k = 4 if (in_f >= 4096 and out_f <= 4096) else 1
        self.block_n = 16 if out_f >= 4096 else 8
        with torch.no_grad():
            wf = weight.float().reshape(out_f, in_f // group_size, group_size)
            amax = wf.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
            scale = amax / 7.0
            q = torch.round(wf / scale).clamp(-8, 7).to(torch.int32) + 8
            q = q.reshape(out_f, in_f)
            packed = (q[:, 0::2].to(torch.uint8) | (q[:, 1::2].to(torch.uint8) << 4)).contiguous()
            self.register_buffer("qw", packed)
            self.register_buffer("scales", scale.squeeze(-1).to(torch.bfloat16).contiguous())
            if self.split_k > 1:
                self.register_buffer(
                    "_part",
                    torch.empty((self.split_k, out_f), device=weight.device, dtype=torch.float32),
                    persistent=False,
                )
            else:
                self._part = None
        if bias is not None:
            self.register_buffer("bias", bias.detach().to(torch.bfloat16))
        else:
            self.bias = None

    def _dequant(self) -> torch.Tensor:
        lo = (self.qw & 0x0F).to(torch.bfloat16) - 8.0
        hi = ((self.qw >> 4) & 0x0F).to(torch.bfloat16) - 8.0
        w = torch.stack([lo, hi], dim=-1).reshape(
            self.out_features, self.in_features // self.group_size, self.group_size
        )
        return (w * self.scales.unsqueeze(-1)).reshape(self.out_features, self.in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x2d = x.reshape(-1, self.in_features)
        if TRITON_AVAILABLE and x2d.shape[0] == 1 and x2d.is_cuda:
            grid = (triton.cdiv(self.out_features, self.block_n), self.split_k)
            if self.split_k == 1:
                y = torch.empty((1, self.out_features), device=x.device, dtype=torch.bfloat16)
                _int4_gemv_splitk_kernel[grid](
                    x2d[0].contiguous(),
                    self.qw,
                    self.scales,
                    y[0],
                    self.out_features,
                    self.in_features,
                    self.qw.stride(0),
                    self.scales.stride(0),
                    GROUP_SIZE=self.group_size,
                    BLOCK_N=self.block_n,
                    SPLIT_K=1,
                    num_warps=4,
                )
            else:
                _int4_gemv_splitk_kernel[grid](
                    x2d[0].contiguous(),
                    self.qw,
                    self.scales,
                    self._part,
                    self.out_features,
                    self.in_features,
                    self.qw.stride(0),
                    self.scales.stride(0),
                    GROUP_SIZE=self.group_size,
                    BLOCK_N=self.block_n,
                    SPLIT_K=self.split_k,
                    num_warps=4,
                )
                y = self._part.sum(dim=0, keepdim=True).to(torch.bfloat16)
            if self.bias is not None:
                y = y + self.bias
            return y.view(*orig_shape[:-1], self.out_features)
        return F.linear(x2d, self._dequant(), self.bias).view(*orig_shape[:-1], self.out_features)


def snake1d_fp32(x: torch.Tensor, alpha_1d: torch.Tensor, inv_alpha_1d: torch.Tensor) -> torch.Tensor:
    """Exact lossless FP32 Snake1d activation: x + (alpha + 1e-9)^-1 * sin(alpha * x)^2."""
    if TRITON_AVAILABLE and x.is_cuda and x.ndim == 3 and x.shape[0] == 1 and x.dtype == torch.float32:
        x_c = x.contiguous()
        _, c_dim, t_dim = x_c.shape
        y = torch.empty_like(x_c)
        block_t = min(1024, max(16, triton.next_power_of_2(t_dim)))
        grid = (triton.cdiv(t_dim, block_t), c_dim)
        _snake1d_fp32_kernel[grid](
            x_c,
            alpha_1d,
            inv_alpha_1d,
            y,
            t_dim,
            BLOCK_T=block_t,
            num_warps=4 if block_t >= 256 else 2,
        )
        return y
    a = alpha_1d.view(1, -1, 1)
    inv_a = inv_alpha_1d.view(1, -1, 1)
    return x + inv_a * torch.sin(a * x).pow(2)


def sr_cond_snake1d_fp32(
    x: torch.Tensor,
    scale_1d: torch.Tensor,
    bias_1d: torch.Tensor,
    alpha_1d: torch.Tensor,
    inv_alpha_1d: torch.Tensor,
) -> torch.Tensor:
    """Exact lossless FP32 fused SampleRateConditionLayer (scale/bias) + Snake1d activation."""
    if TRITON_AVAILABLE and x.is_cuda and x.ndim == 3 and x.shape[0] == 1 and x.dtype == torch.float32:
        x_c = x.contiguous()
        _, c_dim, t_dim = x_c.shape
        y = torch.empty_like(x_c)
        block_t = min(1024, max(16, triton.next_power_of_2(t_dim)))
        grid = (triton.cdiv(t_dim, block_t), c_dim)
        _sr_cond_snake1d_fp32_kernel[grid](
            x_c,
            scale_1d,
            bias_1d,
            alpha_1d,
            inv_alpha_1d,
            y,
            t_dim,
            BLOCK_T=block_t,
            num_warps=4 if block_t >= 256 else 2,
        )
        return y
    xs = x * scale_1d.view(1, -1, 1) + bias_1d.view(1, -1, 1)
    a = alpha_1d.view(1, -1, 1)
    inv_a = inv_alpha_1d.view(1, -1, 1)
    return xs + inv_a * torch.sin(a * xs).pow(2)


def probe_fp8_supported(device: str | torch.device) -> bool:
    dev_key = str(device)
    if dev_key in _FP8_PROBE_CACHE:
        return _FP8_PROBE_CACHE[dev_key]
    if not dev_key.startswith("cuda") or not torch.cuda.is_available():
        _FP8_PROBE_CACHE[dev_key] = False
        return False
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) < (8, 9):
        _FP8_PROBE_CACHE[dev_key] = False
        return False
    with CUDA_GRAPH_LOCK:
        if dev_key in _FP8_PROBE_CACHE:
            return _FP8_PROBE_CACHE[dev_key]
        try:
            probe = FP8Linear(torch.zeros(64, 64, device=device, dtype=torch.bfloat16))
            _ = probe(torch.zeros(1, 64, device=device, dtype=torch.bfloat16))
            del probe
            _FP8_PROBE_CACHE[dev_key] = True
            return True
        except (RuntimeError, ValueError, TypeError, OSError) as exc:
            logger.debug("FP8 probe failed: %s", exc)
            _FP8_PROBE_CACHE[dev_key] = False
            return False


def capture_cuda_graph(fn, sample_args: tuple[torch.Tensor, ...]):
    with CUDA_GRAPH_LOCK:
        torch.cuda.synchronize()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                fn(*sample_args)
        torch.cuda.current_stream().wait_stream(stream)

        static_in = [arg.clone() for arg in sample_args]
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_out = fn(*static_in)
        torch.cuda.synchronize()

    def replay(*args: torch.Tensor) -> torch.Tensor:
        for dst, src in zip(static_in, args):
            dst.copy_(src)
        graph.replay()
        return static_out.clone()

    return replay


class CUDAGraphModule(nn.Module):
    def __init__(self, fn, sample_args: tuple[torch.Tensor, ...]) -> None:
        super().__init__()
        self._replay = capture_cuda_graph(fn, sample_args)

    def forward(self, *args: torch.Tensor) -> torch.Tensor:
        return self._replay(*args)
