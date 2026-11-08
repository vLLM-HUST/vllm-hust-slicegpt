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
