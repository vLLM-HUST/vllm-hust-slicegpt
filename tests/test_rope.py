"""Prove that choosing a native path affects only the model-owned object."""

import pytest


def test_native_rope_is_model_local():
    torch = pytest.importorskip("torch")
    from vllm_hust_slicegpt.models.rope import SliceGPTRotaryEmbedding

    class Rope(torch.nn.Module):
        def forward(self, positions, query, key):
            return query + 10, key + 10

        def forward_native(self, positions, query, key):
            return query + 1, key + 1

    underlying = Rope()
    native = SliceGPTRotaryEmbedding(underlying, native=True)
    normal = SliceGPTRotaryEmbedding(underlying, native=False)
    inputs = torch.zeros(1)
    torch.testing.assert_close(native(inputs, inputs, inputs)[0], inputs + 1)
    torch.testing.assert_close(normal(inputs, inputs, inputs)[0], inputs + 10)
    torch.testing.assert_close(underlying(inputs, inputs, inputs)[0], inputs + 10)


def test_missing_native_rope_fails_closed():
    torch = pytest.importorskip("torch")
    from vllm_hust_slicegpt.models.rope import SliceGPTRotaryEmbedding

    with pytest.raises(RuntimeError, match="no native"):
        SliceGPTRotaryEmbedding(torch.nn.Identity(), native=True)
