"""Fused SliceGPT ops must keep the eager semantics; non-CUDA tensors keep the torch path."""

import pytest


def reference_rmsn(torch, x, mean_dim, eps):
    orig_dtype = x.dtype
    x = x.to(torch.float32)
    variance = x.pow(2).sum(-1, keepdim=True) / mean_dim
    return (x * torch.rsqrt(variance + eps)).to(orig_dtype)


def test_cpu_tensors_keep_eager_rmsn():
    torch = pytest.importorskip("torch")
    from vllm_hust_slicegpt.models.fused_ops import fused_rmsn_available

    assert not fused_rmsn_available(torch.zeros(2, 4))


@pytest.mark.parametrize("shape", [(5, 6), (2, 3, 6)])
def test_residual_project_matches_matmul_add(shape):
    torch = pytest.importorskip("torch")
    from vllm_hust_slicegpt.models.fused_ops import residual_project

    generator = torch.Generator().manual_seed(0)
    residual = torch.randn(*shape, generator=generator, dtype=torch.float64)
    hidden = torch.randn(*shape[:-1], 4, generator=generator, dtype=torch.float64)
    shortcut = torch.randn(6, 4, generator=generator, dtype=torch.float64)
    torch.testing.assert_close(
        residual_project(residual, shortcut, hidden),
        torch.matmul(residual, shortcut) + hidden,
    )


def test_cpu_residual_project_does_not_use_cuda_addmm(monkeypatch):
    torch = pytest.importorskip("torch")
    from vllm_hust_slicegpt.models.fused_ops import residual_project

    residual = torch.ones(2, 3)
    shortcut = torch.ones(3, 4)
    hidden = torch.ones(2, 4)

    def reject_addmm(*args, **kwargs):
        raise AssertionError("CPU/Ascend fallback must keep matmul + add")

    monkeypatch.setattr(torch, "addmm", reject_addmm)
    torch.testing.assert_close(
        residual_project(residual, shortcut, hidden),
        torch.matmul(residual, shortcut) + hidden,
    )


@pytest.mark.parametrize("n_cols", [3840, 5120])
def test_cuda_rmsn_matches_eager(n_cols):
    torch = pytest.importorskip("torch")
    pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for the fused kernel")
    from vllm_hust_slicegpt.models.fused_ops import fused_rmsn_available, slicegpt_rmsn

    x = torch.randn(33, n_cols, device="cuda", dtype=torch.float16) * 4
    assert fused_rmsn_available(x)
    torch.testing.assert_close(
        slicegpt_rmsn(x, 5120, 1e-6), reference_rmsn(torch, x, 5120, 1e-6)
    )


def _npu_available(torch):
    return hasattr(torch, "npu") and torch.npu.is_available()


def test_npu_rms_norm_matches_eager():
    torch = pytest.importorskip("torch")
    if not _npu_available(torch):
        pytest.skip("Ascend NPU required")
    from vllm_hust_slicegpt.models.fused_ops import (
        rms_norm,
        stock_rmsn_eps,
        stock_rmsn_scale,
    )

    x = torch.randn(48, 3840, device="npu", dtype=torch.float16) * 4
    w = torch.ones(3840, device="npu", dtype=torch.float16)
    got = rms_norm(x, w, stock_rmsn_eps(3840, 5120, 1e-6)) * stock_rmsn_scale(
        3840, 5120
    )
    torch.testing.assert_close(
        got, reference_rmsn(torch, x, 5120, 1e-6), atol=8e-3, rtol=0
    )


def test_npu_fused_add_rms_norm_matches_eager():
    torch = pytest.importorskip("torch")
    if not _npu_available(torch):
        pytest.skip("Ascend NPU required")
    from vllm_hust_slicegpt.models.fused_ops import (
        fused_add_rms_norm,
        stock_rmsn_eps,
        stock_rmsn_scale,
    )

    attn = torch.randn(48, 3840, device="npu", dtype=torch.float16)
    rotated = torch.randn(48, 3840, device="npu", dtype=torch.float16)
    w = torch.ones(3840, device="npu", dtype=torch.float16)
    normed, summed = fused_add_rms_norm(
        attn, rotated, w, stock_rmsn_eps(3840, 5120, 1e-6)
    )
    torch.testing.assert_close(
        normed * stock_rmsn_scale(3840, 5120),
        reference_rmsn(torch, attn + rotated, 5120, 1e-6),
        atol=8e-3,
        rtol=0,
    )
    torch.testing.assert_close(summed, attn + rotated)


def test_build_npu_rope_cos_sin_expands_half_cache():
    torch = pytest.importorskip("torch")
    from vllm_hust_slicegpt.models.fused_ops import build_npu_rope_cos_sin

    cos_h = torch.arange(12, dtype=torch.float32).view(3, 4)
    sin_h = cos_h + 100.0
    cos_full, sin_full = build_npu_rope_cos_sin(torch.cat((cos_h, sin_h), dim=-1))
    torch.testing.assert_close(cos_full, torch.cat((cos_h, cos_h), dim=-1))
    torch.testing.assert_close(sin_full, torch.cat((sin_h, sin_h), dim=-1))
    assert cos_full.is_contiguous() and sin_full.is_contiguous()


@pytest.mark.parametrize("shape", [(5, 6), (2, 3, 6)])
@pytest.mark.parametrize("with_bias", [False, True])
def test_npu_direct_linear_matches_f_linear(shape, with_bias):
    torch = pytest.importorskip("torch")
    from vllm_hust_slicegpt.models.fused_ops import npu_direct_linear

    g = torch.Generator().manual_seed(1)
    x = torch.randn(*shape, generator=g, dtype=torch.float64)
    w = torch.randn(4, 6, generator=g, dtype=torch.float64)
    bias = torch.randn(4, generator=g, dtype=torch.float64) if with_bias else None
    got = npu_direct_linear(x, w.t().contiguous(), bias)
    expected = torch.nn.functional.linear(x, w, bias)
    torch.testing.assert_close(got, expected)


def test_npu_fused_qk_rope_matches_neox_reference():
    torch = pytest.importorskip("torch")
    if not _npu_available(torch):
        pytest.skip("Ascend NPU required")
    from vllm_hust_slicegpt.models.fused_ops import (
        build_npu_rope_cos_sin,
        npu_fused_qk_rope,
    )

    bs, n_q, n_kv, head_dim = 17, 8, 2, 128
    inv_freq = 1.0 / (
        1000000.0 ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    )
    t = torch.arange(128, dtype=torch.float32)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    cos_h = freqs.cos().to(torch.float16).to("npu")
    sin_h = freqs.sin().to(torch.float16).to("npu")
    cos_full, sin_full = build_npu_rope_cos_sin(torch.cat((cos_h, sin_h), dim=-1))

    pos = torch.randint(0, 128, (bs,), device="npu", dtype=torch.int64)
    qkv = torch.randn(
        bs, (n_q + 2 * n_kv) * head_dim, device="npu", dtype=torch.float16
    )
    qk, _ = qkv.split([(n_q + n_kv) * head_dim, n_kv * head_dim], dim=-1)
    cos = cos_full.index_select(0, pos).view(1, bs, 1, head_dim)
    sin = sin_full.index_select(0, pos).view(1, bs, 1, head_dim)
    got = npu_fused_qk_rope(qk, cos, sin, n_q + n_kv, head_dim)

    ch = cos_h.index_select(0, pos).unsqueeze(1).float()
    sh = sin_h.index_select(0, pos).unsqueeze(1).float()
    x = qk.view(bs, n_q + n_kv, head_dim).float()
    x1, x2 = x.chunk(2, dim=-1)
    ref = (
        torch.cat((x1 * ch - x2 * sh, x2 * ch + x1 * sh), dim=-1)
        .to(torch.float16)
        .view(bs, -1)
    )
    torch.testing.assert_close(got, ref, atol=0, rtol=0)


def test_stock_norm_scale_folding_matches_slicegpt_rmsn():
    """Verify the math identity shared by both Qwen2 and Llama stock-norm paths."""
    torch = pytest.importorskip("torch")
    from vllm_hust_slicegpt.models.fused_ops import stock_rmsn_eps, stock_rmsn_scale

    g = torch.Generator().manual_seed(7)
    orig_hidden, sliced_dim, out_dim, eps = 64, 48, 32, 1e-6
    x = torch.randn(11, sliced_dim, generator=g, dtype=torch.float64) * 3.0
    w = torch.randn(out_dim, sliced_dim, generator=g, dtype=torch.float64)

    # Eager SliceGPTRMSN followed by linear
    eager_norm = x * torch.rsqrt(x.pow(2).sum(-1, keepdim=True) / orig_hidden + eps)
    expected = torch.nn.functional.linear(eager_norm, w)

    # Unit-weight stock RMSNorm (mean over sliced_dim with eps * D/d) + folded weight
    eps_prime = stock_rmsn_eps(sliced_dim, orig_hidden, eps)
    scale = stock_rmsn_scale(sliced_dim, orig_hidden)
    stock_norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps_prime)
    got = torch.nn.functional.linear(stock_norm, w * scale)
    torch.testing.assert_close(got, expected, atol=1e-12, rtol=1e-12)


def test_pretranspose_nd_weight_ignores_non_npu():
    torch = pytest.importorskip("torch")
    from vllm_hust_slicegpt.models.fused_ops import pretranspose_nd_weight

    p = torch.nn.Parameter(torch.randn(4, 6))
    assert pretranspose_nd_weight(p) is None
    assert tuple(p.shape) == (4, 6)


def test_prepare_and_select_npu_shared_rope_cache_guards(monkeypatch):
    from types import SimpleNamespace

    torch = pytest.importorskip("torch")
    from vllm_hust_slicegpt.models.fused_ops import (
        prepare_npu_shared_rope_cache,
        select_npu_rope_cos_sin,
    )

    # Non-NPU tensors must cleanly decline shared NPU table creation.
    cache = torch.randn(16, 8)
    rope = SimpleNamespace(
        cos_sin_cache=cache, is_neox_style=True, rotary_dim=8, head_size=8
    )
    layers = [
        SimpleNamespace(
            self_attn=SimpleNamespace(rotary_emb=SimpleNamespace(embedding=rope))
        )
    ]
    assert prepare_npu_shared_rope_cache(layers) == (None, None)

    # Env opt-out toggles must disable even before checking device.
    monkeypatch.setenv("SLICEGPT_NPU_HOST_ROPE", "1")
    assert prepare_npu_shared_rope_cache(layers) == (None, None)
    monkeypatch.delenv("SLICEGPT_NPU_HOST_ROPE")

    # select_npu_rope_cos_sin reshapes 1D positions cleanly and skips 2D positions.
    cos_full = torch.arange(32, dtype=torch.float32).view(4, 8)
    sin_full = cos_full + 10.0
    pos = torch.tensor([2, 0, 3], dtype=torch.int64)
    cos, sin = select_npu_rope_cos_sin(cos_full, sin_full, pos)
    assert tuple(cos.shape) == (1, 3, 1, 8)
    assert tuple(sin.shape) == (1, 3, 1, 8)
    torch.testing.assert_close(cos[0, :, 0, :], cos_full[pos])
    assert select_npu_rope_cos_sin(cos_full, sin_full, pos.unsqueeze(0)) is None
