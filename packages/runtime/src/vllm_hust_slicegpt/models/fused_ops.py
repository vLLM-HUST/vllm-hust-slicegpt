"""CUDA and Ascend NPU fast paths for SliceGPT-specific ops; other platforms keep eager torch."""

import os

import torch
from torch import nn

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


def npu_apply_qk_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply fused NeoX RoPE to already-contiguous Q and K in one CANN op."""
    n_tok = q.shape[0]
    q_out, k_out = _get_torch_npu().npu_apply_rotary_pos_emb(
        q.view(1, n_tok, num_heads, head_dim),
        k.view(1, n_tok, num_kv_heads, head_dim),
        cos,
        sin,
    )
    return q_out.view(n_tok, q.shape[1]), k_out.view(n_tok, k.shape[1])


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


_ACL_FORMAT_ND = 2
_CUDA_SIDE_STREAM: torch.cuda.Stream | None = None


def cuda_side_stream() -> torch.cuda.Stream:
    """Return a lazy process-wide CUDA side stream for shortcut GEMM overlap."""
    global _CUDA_SIDE_STREAM
    if _CUDA_SIDE_STREAM is None:
        _CUDA_SIDE_STREAM = torch.cuda.Stream()
    return _CUDA_SIDE_STREAM


def pretranspose_nd_weight(param: nn.Parameter) -> torch.Tensor | None:
    """In-place transpose a 2D ND NPU weight to ``[in, out]`` for ``torch.mm``."""
    w = param.data
    if w.device.type != "npu" or w.ndim != 2:
        return None
    if _get_torch_npu().get_npu_format(w) != _ACL_FORMAT_ND:
        return None
    param.data = w.t().contiguous()
    return param.data


def pretranspose_split_qkv_nd_weight(
    param: nn.Parameter,
    bias: nn.Parameter | None,
    q_size: int,
    kv_size: int,
) -> (
    tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]
    | None
):
    """Split a 2D ND NPU QKV weight into contiguous pre-transposed Q, K, V
    weights (and optional biases), freeing the original storage for zero net HBM
    increase so Q, K, V projections produce contiguous outputs directly."""
    w = param.data
    if w.device.type != "npu" or w.ndim != 2 or w.shape[0] != q_size + 2 * kv_size:
        return None
    if _get_torch_npu().get_npu_format(w) != _ACL_FORMAT_ND:
        return None
    q_end = q_size
    k_end = q_size + kv_size
    q_w_t = w[:q_end].t().contiguous()
    k_w_t = w[q_end:k_end].t().contiguous()
    v_w_t = w[k_end:].t().contiguous()
    param.data = torch.empty(0, dtype=w.dtype, device=w.device)

    q_b = k_b = v_b = None
    if bias is not None:
        b = bias.data
        q_b = b[:q_end].contiguous()
        k_b = b[q_end:k_end].contiguous()
        v_b = b[k_end:].contiguous()
        bias.data = torch.empty(0, dtype=b.dtype, device=b.device)
    return q_w_t, k_w_t, v_w_t, q_b, k_b, v_b


def enable_npu_direct_attention(layers: nn.ModuleList | list[nn.Module]) -> None:
    """Bypass ``torch.ops.vllm.unified_attention_with_output`` custom-op boxing
    on model-owned eager ``Attention`` instances on NPU."""
    if os.environ.get("SLICEGPT_NPU_DIRECT_ATTN") == "0":
        return
    for layer in layers:
        attn = getattr(layer.self_attn, "attn", None)
        if attn is not None and hasattr(attn, "use_direct_call"):
            attn.use_direct_call = True


def prepare_npu_shared_rope_cache(
    layers: nn.ModuleList | list[nn.Module],
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Build shared duplicated ``(cos_full, sin_full)`` tables when all layers
    share an identical NeoX full-head-dim ``cos_sin_cache`` on NPU."""
    if (
        not layers
        or os.environ.get("SLICEGPT_NPU_NATIVE_ROPE") == "1"
        or os.environ.get("SLICEGPT_NPU_HOST_ROPE") == "1"
    ):
        return None, None
    ref_rope = getattr(layers[0].self_attn.rotary_emb, "embedding", None)
    ref_cache = getattr(ref_rope, "cos_sin_cache", None)
    if (
        ref_cache is None
        or ref_cache.device.type != "npu"
        or not getattr(ref_rope, "is_neox_style", False)
        or getattr(ref_rope, "rotary_dim", None) != getattr(ref_rope, "head_size", None)
        or ref_cache.shape[-1] != ref_rope.head_size
    ):
        return None, None
    for layer in layers[1:]:
        rope = getattr(layer.self_attn.rotary_emb, "embedding", None)
        cache = getattr(rope, "cos_sin_cache", None)
        if cache is ref_cache:
            continue
        if (
            cache is None
            or cache.shape != ref_cache.shape
            or not torch.equal(cache, ref_cache)
        ):
            return None, None
    return build_npu_rope_cos_sin(ref_cache)


def select_npu_rope_cos_sin(
    cos_full: torch.Tensor | None,
    sin_full: torch.Tensor | None,
    positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Index shared NPU RoPE tables once per forward step."""
    if cos_full is None or sin_full is None or positions.ndim != 1:
        return None
    n_tok = positions.shape[0]
    head_dim = cos_full.shape[-1]
    cos = cos_full.index_select(0, positions).view(1, n_tok, 1, head_dim)
    sin = sin_full.index_select(0, positions).view(1, n_tok, 1, head_dim)
    return cos, sin
