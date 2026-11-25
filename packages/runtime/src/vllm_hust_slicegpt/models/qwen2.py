# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only SliceGPT-compressed Qwen2/Qwen2.5 model."""

import os
from collections.abc import Iterable

import torch
from torch import nn
from transformers import Qwen2Config
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.attention import Attention, EncoderOnlyAttention
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.utils import (
    extract_layer_index,
    make_layers,
    maybe_prefix,
)
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.config import set_default_rope_theta
from vllm.v1.attention.backend import AttentionType

from vllm_hust_slicegpt.artifact import checked_weights, validate_config
from vllm_hust_slicegpt.compatibility import check_backend, check_execution_config
from vllm_hust_slicegpt.models.fused_ops import (
    build_npu_rope_cos_sin,
    fused_add_rms_norm,
    fused_rmsn_available,
    npu_direct_linear,
    npu_fused_qk_rope,
    npu_swiglu,
    residual_project,
    rms_norm,
    slicegpt_rmsn,
    stock_rmsn_eps,
    stock_rmsn_scale,
)
from vllm_hust_slicegpt.models.rope import SliceGPTRotaryEmbedding

_ACL_FORMAT_ND = 2


def _pretranspose_nd_weight(param: nn.Parameter) -> torch.Tensor | None:
    """In-place transpose a 2D ND NPU weight to ``[in, out]`` for ``torch.mm``."""
    import torch_npu

    w = param.data
    if w.device.type != "npu" or w.ndim != 2:
        return None
    if torch_npu.get_npu_format(w) != _ACL_FORMAT_ND:
        return None
    param.data = w.t().contiguous()
    return param.data


class SliceGPTRMSN(nn.Module):
    def __init__(self, mean_dim: int, eps: float) -> None:
        super().__init__()
        self.mean_dim = mean_dim
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if fused_rmsn_available(x):
            return slicegpt_rmsn(x, self.mean_dim, self.eps)
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).sum(-1, keepdim=True) / self.mean_dim
        x = x * torch.rsqrt(variance + self.eps)
        return x.to(orig_dtype)


class SliceGPTQwen2MLP(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        intermediate_size: int,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=input_size,
            output_sizes=[intermediate_size] * 2,
            bias=False,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=output_size,
            bias=False,
            prefix=f"{prefix}.down_proj",
        )
        self.act_fn = SiluAndMul()
        self._fast_linear = False
        self._gu_w_t: torch.Tensor | None = None
        self._dn_w_t: torch.Tensor | None = None

    def enable_npu_fast_linear(self) -> None:
        if self._fast_linear:
            return
        gu_t = _pretranspose_nd_weight(self.gate_up_proj.weight)
        if gu_t is None:
            return
        dn_t = _pretranspose_nd_weight(self.down_proj.weight)
        if dn_t is None:
            # Undo gate_up transpose if down_proj was not ND.
            self.gate_up_proj.weight.data = gu_t.t().contiguous()
            return
        self._gu_w_t = gu_t
        self._dn_w_t = dn_t
        self._fast_linear = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._fast_linear:
            x = npu_direct_linear(x, self._gu_w_t)
            x = npu_swiglu(x)
            return npu_direct_linear(x, self._dn_w_t)
        x, _ = self.gate_up_proj(x)
        x = self.act_fn(x)
        x, _ = self.down_proj(x)
        return x


class SliceGPTQwen2Attention(nn.Module):
    def __init__(
        self,
        config: Qwen2Config,
        input_size: int,
        output_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position_embeddings: int,
        cache_config,
        prefix: str = "",
        attn_type: str = AttentionType.DECODER,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)

        head_dim = getattr(config, "head_dim", None)
        self.head_dim = head_dim or config.hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.qk_size = self.q_size + self.kv_size
        self.total_qk_heads = self.num_heads + self.num_kv_heads
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            hidden_size=input_size,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=True,
            quant_config=None,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            input_size=self.total_num_heads * self.head_dim,
            output_size=output_size,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.o_proj",
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=max_position_embeddings,
            rope_parameters=getattr(config, "rope_parameters", None),
            is_neox_style=True,
        )
        # This model-local RoPE dispatches to the host's fused forward_oot (on NPU:
        # AscendRotaryEmbedding, which only reads its own cos_sin_cache).
        # SLICEGPT_NPU_NATIVE_ROPE=1 forces forward_native on NPU as a fallback.
        self.rotary_emb = SliceGPTRotaryEmbedding(
            self.rotary_emb,
            native=current_platform.device_type == "npu"
            and os.environ.get("SLICEGPT_NPU_NATIVE_ROPE") == "1",
        )
        attn_cls = (
            EncoderOnlyAttention
            if attn_type == AttentionType.ENCODER_ONLY
            else Attention
        )
        self.attn = attn_cls(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=None,
            attn_type=attn_type,
            prefix=f"{prefix}.attn",
        )
        self._fast_linear = False
        self._qkv_w_t: torch.Tensor | None = None
        self._qkv_bias: torch.Tensor | None = None
        self._o_w_t: torch.Tensor | None = None

    def enable_npu_fast_linear(self) -> None:
        if self._fast_linear:
            return
        qkv_t = _pretranspose_nd_weight(self.qkv_proj.weight)
        if qkv_t is None:
            return
        o_t = _pretranspose_nd_weight(self.o_proj.weight)
        if o_t is None:
            self.qkv_proj.weight.data = qkv_t.t().contiguous()
            return
        self._qkv_w_t = qkv_t
        self._qkv_bias = (
            self.qkv_proj.bias.data if self.qkv_proj.bias is not None else None
        )
        self._o_w_t = o_t
        self._fast_linear = True

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        rope_cos_sin: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if self._fast_linear:
            qkv = npu_direct_linear(hidden_states, self._qkv_w_t, self._qkv_bias)
        else:
            qkv, _ = self.qkv_proj(hidden_states)
        if rope_cos_sin is not None:
            cos, sin = rope_cos_sin
            qk, v = qkv.split([self.qk_size, self.kv_size], dim=-1)
            qk = npu_fused_qk_rope(qk, cos, sin, self.total_qk_heads, self.head_dim)
            q, k = qk.split([self.q_size, self.kv_size], dim=-1)
        else:
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        if self._fast_linear:
            return npu_direct_linear(attn_output, self._o_w_t)
        output, _ = self.o_proj(attn_output)
        return output


_SIDE_STREAM: torch.cuda.Stream | None = None


def _side_stream() -> torch.cuda.Stream:
    global _SIDE_STREAM
    if _SIDE_STREAM is None:
        _SIDE_STREAM = torch.cuda.Stream()
    return _SIDE_STREAM


class SliceGPTQwen2DecoderLayer(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config: Qwen2Config = vllm_config.model_config.hf_config.get_text_config()
        cache_config = vllm_config.cache_config
        sc = config.slicing_config
        idx = extract_layer_index(prefix)

        set_default_rope_theta(config, default_theta=1000000)

        attn_in = sc["attention_input_dims"][idx]
        attn_out = sc["attention_output_dims"][idx]
        mlp_in = sc["mlp_input_dims"][idx]
        mlp_out = sc["mlp_output_dims"][idx]

        self.self_attn = SliceGPTQwen2Attention(
            config=config,
            input_size=attn_in,
            output_size=attn_out,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position_embeddings=config.max_position_embeddings,
            cache_config=cache_config,
            prefix=f"{prefix}.self_attn",
        )
        self.mlp = SliceGPTQwen2MLP(
            input_size=mlp_in,
            output_size=mlp_out,
            intermediate_size=config.intermediate_size,
            prefix=f"{prefix}.mlp",
        )
        orig_hidden = config.slicegpt_artifact["original_hidden_size"]
        self.input_layernorm = SliceGPTRMSN(orig_hidden, eps=config.rms_norm_eps)
        self.post_attention_layernorm = SliceGPTRMSN(
            orig_hidden, eps=config.rms_norm_eps
        )
        self.attn_shortcut_Q = nn.Parameter(torch.empty(attn_in, attn_out))
        self.mlp_shortcut_Q = nn.Parameter(torch.empty(mlp_in, mlp_out))

        # Stock-kernel path (CUDA): unit-weight norms; sqrt(D / d) is folded into
        # qkv_proj / gate_up_proj by SliceGPTQwen2ForCausalLM.load_weights.
        eps = config.rms_norm_eps
        self.attn_input_scale = stock_rmsn_scale(attn_in, orig_hidden)
        self.mlp_input_scale = stock_rmsn_scale(mlp_in, orig_hidden)
        self.mlp_norm_eps = stock_rmsn_eps(mlp_in, orig_hidden, eps)
        self.out_norm_eps = stock_rmsn_eps(mlp_out, orig_hidden, eps)
        self.register_buffer("mlp_norm_weight", torch.ones(mlp_in), persistent=False)
        self.register_buffer("out_norm_weight", torch.ones(mlp_out), persistent=False)
        # Shortcut GEMM / block overlap uses CUDA streams; NPU runs sequentially.
        self.overlap = current_platform.device_type == "cuda"

    def enable_npu_fast_linear(self) -> None:
        self.self_attn.enable_npu_fast_linear()
        self.mlp.enable_npu_fast_linear()

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        rope_cos_sin: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            rope_cos_sin=rope_cos_sin,
        )
        hidden_states = residual_project(residual, self.attn_shortcut_Q, hidden_states)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual_project(residual, self.mlp_shortcut_Q, hidden_states)
        return hidden_states

    def forward_stock_norm(
        self,
        positions: torch.Tensor,
        normed: torch.Tensor,
        residual: torch.Tensor,
        rope_cos_sin: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Take the normalized input and raw residual; return the same for the next
        layer (the last layer's output norm is the model's final norm)."""
        rotated = self._rotate(
            residual, self.attn_shortcut_Q, positions, normed, True, rope_cos_sin
        )
        hidden_states, residual = fused_add_rms_norm(
            self._block_out, rotated, self.mlp_norm_weight, self.mlp_norm_eps
        )
        rotated = self._rotate(
            residual, self.mlp_shortcut_Q, positions, hidden_states, False, None
        )
        hidden_states, residual = fused_add_rms_norm(
            self._block_out, rotated, self.out_norm_weight, self.out_norm_eps
        )
        return hidden_states, residual

    def _rotate(
        self,
        residual: torch.Tensor,
        shortcut_q: torch.Tensor,
        positions: torch.Tensor,
        block_in: torch.Tensor,
        is_attn: bool,
        rope_cos_sin: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Compute ``residual @ shortcut_q`` while the attention/MLP block runs.

        On CUDA the rotation runs on a side stream, overlapping the block on the
        main stream; on NPU both run sequentially (cross-stream ACL event sync
        regressed throughput on 910B2). Output is stashed on ``self._block_out``
        so the caller can feed it to the fused add+norm.
        """
        if self.overlap:
            main, side = torch.cuda.current_stream(), _side_stream()
            side.wait_stream(main)
            with torch.cuda.stream(side):
                rotated = torch.matmul(residual, shortcut_q)
            residual.record_stream(side)
            self._block_out = (
                self.self_attn(positions=positions, hidden_states=block_in)
                if is_attn
                else self.mlp(block_in)
            )
            main.wait_stream(side)
            rotated.record_stream(main)
            return rotated
        rotated = (
            torch.mm(residual, shortcut_q)
            if residual.ndim == 2
            else torch.matmul(residual, shortcut_q)
        )
        self._block_out = (
            self.self_attn(
                positions=positions,
                hidden_states=block_in,
                rope_cos_sin=rope_cos_sin,
            )
            if is_attn
            else self.mlp(block_in)
        )
        return rotated


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": {0: "b"},
        "positions": {0: "b"},
        "inputs_embeds": {0: "b"},
    },
)
class SliceGPTQwen2Model(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config: Qwen2Config = vllm_config.model_config.hf_config.get_text_config()
        sc = config.slicing_config
        self.config = config
        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(self.vocab_size, sc["embedding_dim"])
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: SliceGPTQwen2DecoderLayer(
                vllm_config=vllm_config, prefix=prefix
            ),
            prefix=f"{prefix}.layers",
        )
        orig_hidden = config.slicegpt_artifact["original_hidden_size"]
        self.norm = SliceGPTRMSN(orig_hidden, eps=config.rms_norm_eps)

        # Fused host-kernel norm path (CUDA/NPU). SLICEGPT_EAGER_NORM=1 forces the
        # portable eager SliceGPTRMSN path, for A/B comparison and as a fallback.
        self.stock_norm = (
            current_platform.device_type in ("cuda", "npu")
            and os.environ.get("SLICEGPT_EAGER_NORM") != "1"
        )
        emb = sc["embedding_dim"]
        self.input_norm_eps = stock_rmsn_eps(emb, orig_hidden, config.rms_norm_eps)
        self.final_scale = stock_rmsn_scale(sc["final_norm_dim"], orig_hidden)
        self.register_buffer("input_norm_weight", torch.ones(emb), persistent=False)
        self._npu_need_prepare = current_platform.device_type == "npu"
        self._npu_cos_full: torch.Tensor | None = None
        self._npu_sin_full: torch.Tensor | None = None

    def _prepare_npu_fast_path(self) -> None:
        """One-time NPU setup after weights finish loading and post-processing."""
        self._npu_need_prepare = False
        if os.environ.get("SLICEGPT_NPU_FAST_LINEAR") != "0":
            for layer in self.layers:
                layer.enable_npu_fast_linear()
        if (
            self.layers
            and os.environ.get("SLICEGPT_NPU_NATIVE_ROPE") != "1"
            and os.environ.get("SLICEGPT_NPU_HOST_ROPE") != "1"
        ):
            ref_rope = self.layers[0].self_attn.rotary_emb.embedding
            ref_cache = getattr(ref_rope, "cos_sin_cache", None)
            if (
                ref_cache is not None
                and ref_cache.device.type == "npu"
                and getattr(ref_rope, "is_neox_style", False)
                and getattr(ref_rope, "rotary_dim", None)
                == getattr(ref_rope, "head_size", None)
                and ref_cache.shape[-1] == ref_rope.head_size
                and all(
                    (
                        c := getattr(
                            layer.self_attn.rotary_emb.embedding, "cos_sin_cache", None
                        )
                    )
                    is ref_cache
                    or (
                        c is not None
                        and c.shape == ref_cache.shape
                        and torch.equal(c, ref_cache)
                    )
                    for layer in self.layers
                )
            ):
                self._npu_cos_full, self._npu_sin_full = build_npu_rope_cos_sin(
                    ref_cache
                )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self._npu_need_prepare:
            self._prepare_npu_fast_path()
        rope_cos_sin = None
        if self._npu_cos_full is not None and positions.ndim == 1:
            n_tok = positions.shape[0]
            head_dim = self._npu_cos_full.shape[-1]
            cos = self._npu_cos_full.index_select(0, positions).view(
                1, n_tok, 1, head_dim
            )
            sin = self._npu_sin_full.index_select(0, positions).view(
                1, n_tok, 1, head_dim
            )
            rope_cos_sin = (cos, sin)
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embed_input_ids(input_ids)
        if self.stock_norm:
            residual = hidden_states
            hidden_states = rms_norm(
                residual, self.input_norm_weight, self.input_norm_eps
            )
            for layer in self.layers:
                hidden_states, residual = layer.forward_stock_norm(
                    positions, hidden_states, residual, rope_cos_sin
                )
            return hidden_states
        for layer in self.layers:
            hidden_states = layer(positions, hidden_states, rope_cos_sin)
        hidden_states = self.norm(hidden_states)
        return hidden_states


class SliceGPTQwen2ForCausalLM(nn.Module):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        check_execution_config(vllm_config)
        check_backend(current_platform.device_type)
        if vllm_config.parallel_config.pipeline_parallel_size > 1:
            raise ValueError(
                "SliceGPTQwen2ForCausalLM does not support pipeline parallelism. "
                "Use pipeline_parallel_size=1."
            )
        config: Qwen2Config = vllm_config.model_config.hf_config.get_text_config()
        self._artifact_config = config.to_dict()
        validate_config(self._artifact_config)
        sc = config.slicing_config
        self.config = config

        self.model = SliceGPTQwen2Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            sc["final_norm_dim"],
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if intermediate_tensors is not None:
            raise ValueError(
                "SliceGPTQwen2ForCausalLM does not support pipeline "
                "intermediate_tensors. Use pipeline_parallel_size=1."
            )
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            (".qkv_proj.weight", ".q_proj.weight", "q"),
            (".qkv_proj.weight", ".k_proj.weight", "k"),
            (".qkv_proj.weight", ".v_proj.weight", "v"),
            (".qkv_proj.bias", ".q_proj.bias", "q"),
            (".qkv_proj.bias", ".k_proj.bias", "k"),
            (".qkv_proj.bias", ".v_proj.bias", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded: set[str] = set()
        for name, w in checked_weights(self._artifact_config, weights):
            if "rotary_emb.inv_freq" in name:
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                mapped = name.replace(weight_name, param_name)
                if mapped not in params_dict:
                    continue
                param = params_dict[mapped]
                param.weight_loader(param, w, shard_id)
                loaded.add(mapped)
                break
            else:
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, w)
                loaded.add(name)
        if self.model.stock_norm:
            self._fold_norm_scales()
        return loaded

    @torch.no_grad()
    def _fold_norm_scales(self) -> None:
        """Fold sqrt(D / d) of each unit-weight norm into the linear it feeds."""
        if getattr(self, "_norm_scales_folded", False):
            raise RuntimeError("SliceGPT norm scales are already folded")
        for layer in self.model.layers:
            layer.self_attn.qkv_proj.weight.mul_(layer.attn_input_scale)
            layer.mlp.gate_up_proj.weight.mul_(layer.mlp_input_scale)
        if self.model.final_scale != 1.0:
            if self.config.tie_word_embeddings:
                raise ValueError("Cannot fold the final norm scale into tied lm_head")
            self.lm_head.weight.mul_(self.model.final_scale)
        self._norm_scales_folded = True
