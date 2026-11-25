"""CUDA fast paths for SliceGPT-specific ops; other platforms keep the eager torch code."""

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - Triton ships with CUDA vLLM builds
    triton = None


if triton is not None:

    @triton.jit
    def _slicegpt_rmsn_kernel(
        x_ptr, y_ptr, x_stride, y_stride, n_cols, mean_dim, eps, BLOCK: tl.constexpr
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        mask = cols < n_cols
        x = tl.load(x_ptr + row * x_stride + cols, mask=mask, other=0.0).to(tl.float32)
        variance = tl.sum(x * x, axis=0) / mean_dim
        y = x * tl.math.rsqrt(variance + eps)
        tl.store(y_ptr + row * y_stride + cols, y.to(y_ptr.dtype.element_ty), mask=mask)


def fused_rmsn_available(x: torch.Tensor) -> bool:
    return triton is not None and x.is_cuda


def slicegpt_rmsn(x: torch.Tensor, mean_dim: int, eps: float) -> torch.Tensor:
    """One-kernel SliceGPTRMSN: fp32 ``x * rsqrt(sum(x^2) / mean_dim + eps)``."""
    n_cols = x.shape[-1]
    x2 = x.reshape(-1, n_cols)
    if x2.stride(-1) != 1:
        x2 = x2.contiguous()
    y = torch.empty_like(x2)
    if x2.shape[0]:
        block = triton.next_power_of_2(n_cols)
        _slicegpt_rmsn_kernel[(x2.shape[0],)](
            x2,
            y,
            x2.stride(0),
            y.stride(0),
            n_cols,
            float(mean_dim),
            eps,
            BLOCK=block,
            num_warps=8 if block >= 4096 else 4,
        )
    return y.view(x.shape)


def residual_project(
    residual: torch.Tensor, shortcut_q: torch.Tensor, hidden_states: torch.Tensor
) -> torch.Tensor:
    """Fuse the add on CUDA; preserve the eager path on other platforms."""
    if residual.dim() == 2 and residual.is_cuda:
        return torch.addmm(hidden_states, residual, shortcut_q)
    return torch.matmul(residual, shortcut_q) + hidden_states


# SliceGPTRMSN divides by the original width D while x has width d. Since
#   x * rsqrt(sum(x^2) / D + eps) = sqrt(D / d) * x * rsqrt(mean(x^2) + eps * D / d),
# vLLM's C++ RMSNorm with unit weight and eps' = eps * D / d gives the same
# normalized direction; the constant sqrt(D / d) is folded into the consuming
# linear weights at load time. This also lets the post-rotation residual add use
# fused_add_rms_norm.


def stock_rmsn_eps(width: int, mean_dim: int, eps: float) -> float:
    return eps * mean_dim / width


def stock_rmsn_scale(width: int, mean_dim: int) -> float:
    return (mean_dim / width) ** 0.5


def stock_norm_available(x: torch.Tensor) -> bool:
    """A fused host RMSNorm kernel exists for this tensor's device."""
    return x.is_cuda or x.device.type == "npu"


_TORCH_NPU = None


def _get_torch_npu():
    global _TORCH_NPU
    if _TORCH_NPU is None:
        import torch_npu

        _TORCH_NPU = torch_npu
    return _TORCH_NPU


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    if x.device.type == "npu":
        return _get_torch_npu().npu_rms_norm(x, weight, eps)[0]
    from vllm import _custom_ops as ops

    out = torch.empty_like(x)
    ops.rms_norm(out, x, weight, eps)
    return out


def fused_add_rms_norm(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(rms_norm(x + residual) * weight, x + residual)``.

    On CUDA this is in place (both returned tensors alias ``x`` and ``residual``);
    on NPU ``torch_npu.npu_add_rms_norm`` returns fresh tensors. Callers must use
    the returned handles rather than assuming ``x``/``residual`` were updated.
    """
    if x.device.type == "npu":
        normed, _, summed = _get_torch_npu().npu_add_rms_norm(x, residual, weight, eps)
        return normed, summed
    from vllm import _custom_ops as ops

    ops.fused_add_rms_norm(x, residual, weight, eps)
    return x, residual


def build_npu_rope_cos_sin(
    cos_sin_cache: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand vLLM's ``[max_pos, head_dim]`` ``[cos, sin]`` half-cache into
    duplicated ``cos_full`` and ``sin_full`` tables for ``npu_rotary_mul``."""
    cos_half, sin_half = cos_sin_cache.chunk(2, dim=-1)
    cos_full = torch.cat((cos_half, cos_half), dim=-1).contiguous()
    sin_full = torch.cat((sin_half, sin_half), dim=-1).contiguous()
    return cos_full, sin_full


def npu_fused_qk_rope(
    qk: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    num_qk_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """Apply NeoX (half) RoPE to contiguous Q+K heads in one ``npu_rotary_mul``."""
    n_tok = qk.shape[0]
    out = _get_torch_npu().npu_rotary_mul(
        qk.contiguous().view(1, n_tok, num_qk_heads, head_dim),
        cos,
        sin,
        rotary_mode="half",
    )
    return out.view(n_tok, qk.shape[1])


def npu_direct_linear(
    x: torch.Tensor,
    weight_t: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Direct ``mm``/``addmm`` against a pre-transposed 2D weight ``[in, out]``."""
    if x.ndim == 2:
        if bias is not None:
            return torch.addmm(bias, x, weight_t)
        return torch.mm(x, weight_t)
    out = torch.matmul(x, weight_t)
    return out + bias if bias is not None else out


def npu_swiglu(x: torch.Tensor) -> torch.Tensor:
    return _get_torch_npu().npu_swiglu(x)
