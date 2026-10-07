"""Model-local native RoPE on Ascend, without modifying platform/global classes."""

from torch import nn


class SliceGPTRotaryEmbedding(nn.Module):
    def __init__(self, rotary_embedding, *, native: bool):
        super().__init__()
        if native and not callable(getattr(rotary_embedding, "forward_native", None)):
            raise RuntimeError("Host RoPE has no native implementation")
        self.embedding = rotary_embedding
        self.native = native

    def forward(self, positions, query, key):
        if self.native:
            return self.embedding.forward_native(positions, query, key)
        return self.embedding(positions, query, key)
