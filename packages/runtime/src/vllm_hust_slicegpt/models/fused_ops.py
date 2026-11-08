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
