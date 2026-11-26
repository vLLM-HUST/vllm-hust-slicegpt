# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only SliceGPT-compressed LLaMA model.

SliceGPT (https://github.com/microsoft/TransformerCompression) compresses a Llama
model by reducing the *residual-stream* (embedding) dimension. Compared to a stock
Llama this model has three differences:

  1. The residual width is reduced (e.g. 4096 -> 3072) and may differ per layer.
     All per-layer dims are read from `config.slicing_config` (embedded in config.json
     by tools/convert_slicegpt_to_vllm.py). Attention inner dims (num_heads*head_dim)
     and MLP intermediate_size are UNCHANGED.
  2. Before each residual add, the residual is rotated by a dense matrix:
        hidden = residual @ attn_shortcut_Q + attn_out
        hidden = residual @ mlp_shortcut_Q  + mlp_out
  3. The RMSNorm layers have NO affine weight (folded into the linears). CRUCIALLY,
     SliceGPT's RMSN normalizes by the *original* (pre-slice) hidden size, not the
     reduced per-layer width: `var = x.pow(2).sum(-1) / orig_hidden` even when the
     input is only `attn_in` (e.g. 3072) wide. vLLM's stock RMSNorm divides by the
     actual width, which would scale every activation by sqrt(reduced/orig) ~= 0.866
     and corrupt perplexity. We therefore use a faithful SliceGPTRMSN below with a
     fixed divisor = num_attention_heads * head_dim (the original residual width).
     These norms are called in the *un-fused* single-arg form (the residual add cannot
     be fused because the residual is rotated first).

Only tensor_parallel_size=1 / pipeline_parallel_size=1 is supported.
"""

import os
from collections.abc import Iterable

import torch
from torch import nn
from transformers import LlamaConfig
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.llama import LlamaAttention
from vllm.model_executor.models.utils import (
    extract_layer_index,
    make_layers,
    maybe_prefix,
)
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors

from vllm_hust_slicegpt.artifact import checked_weights, validate_config
from vllm_hust_slicegpt.compatibility import check_backend, check_execution_config
from vllm_hust_slicegpt.models.fused_ops import (
    cuda_side_stream,
    enable_npu_direct_attention,
    fused_add_rms_norm,
    fused_rmsn_available,
    npu_apply_qk_rope,
    npu_direct_linear,
    npu_fused_qk_rope,
    npu_swiglu,
    prepare_npu_shared_rope_cache,
    pretranspose_nd_weight,
    pretranspose_split_qkv_nd_weight,
    residual_project,
    rms_norm,
    select_npu_rope_cos_sin,
    slicegpt_rmsn,
    stock_rmsn_eps,
    stock_rmsn_scale,
)
from vllm_hust_slicegpt.models.rope import SliceGPTRotaryEmbedding


class SliceGPTRMSN(nn.Module):
    """RMSNorm without affine weight, normalizing by a FIXED divisor.

    Faithful reimplementation of SliceGPT's `slicegpt.modules.RMSN`: the variance is
    `x.pow(2).sum(-1) / mean_dim` where `mean_dim` is the ORIGINAL (pre-slice) hidden
    size, regardless of the actual (possibly reduced) width of `x`. Computed in float32
    like the reference, then cast back. There is no learnable weight.
    """

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


class SliceGPTMLP(nn.Module):
    """Llama gated MLP allowing different input/output residual dims."""

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
        gu_t = pretranspose_nd_weight(self.gate_up_proj.weight)
        if gu_t is None:
            return
        dn_t = pretranspose_nd_weight(self.down_proj.weight)
        if dn_t is None:
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


class SliceGPTDecoderLayer(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config: LlamaConfig = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        sc = config.slicing_config
        idx = extract_layer_index(prefix)

        attn_in = sc["attention_input_dims"][idx]
        attn_out = sc["attention_output_dims"][idx]
        mlp_in = sc["mlp_input_dims"][idx]
        mlp_out = sc["mlp_output_dims"][idx]
        # We reuse vLLM's LlamaAttention, whose o_proj output is forced to equal its
        # hidden_size input. SliceGPT (sequential) keeps attn_in == attn_out,
        # so this holds.
        assert attn_in == attn_out, (
            f"layer {idx}: attn_in({attn_in}) != attn_out({attn_out}); "
            "LlamaAttention reuse requires equal attention in/out dims"
        )

        self.self_attn = LlamaAttention(
            config=config,
            hidden_size=attn_in,
            num_heads=config.num_attention_heads,
            num_kv_heads=getattr(
                config, "num_key_value_heads", config.num_attention_heads
            ),
            max_position_embeddings=getattr(config, "max_position_embeddings", 4096),
            quant_config=None,
            bias=False,
            cache_config=cache_config,
            prefix=f"{prefix}.self_attn",
        )
        self.self_attn.rotary_emb = SliceGPTRotaryEmbedding(
            self.self_attn.rotary_emb,
            native=current_platform.device_type == "npu"
            and os.environ.get("SLICEGPT_NPU_NATIVE_ROPE") == "1",
        )
        self.mlp = SliceGPTMLP(
            input_size=mlp_in,
            output_size=mlp_out,
            intermediate_size=config.intermediate_size,
            prefix=f"{prefix}.mlp",
        )
        # SliceGPT normalizes by the ORIGINAL hidden size, not the sliced width.
        orig_hidden = config.slicegpt_artifact["original_hidden_size"]
        self.input_layernorm = SliceGPTRMSN(orig_hidden, eps=config.rms_norm_eps)
        self.post_attention_layernorm = SliceGPTRMSN(
            orig_hidden, eps=config.rms_norm_eps
        )

        # Dense residual rotations, used as `residual @ Q`.
        self.attn_shortcut_Q = nn.Parameter(torch.empty(attn_in, attn_out))
        self.mlp_shortcut_Q = nn.Parameter(torch.empty(mlp_in, mlp_out))

        # Stock-kernel path (CUDA/NPU): unit-weight norms; sqrt(D / d) is folded
        # into qkv_proj / gate_up_proj by SliceGPTLlamaForCausalLM.load_weights.
        eps = config.rms_norm_eps
        self.attn_input_scale = stock_rmsn_scale(attn_in, orig_hidden)
        self.mlp_input_scale = stock_rmsn_scale(mlp_in, orig_hidden)
        self.mlp_norm_eps = stock_rmsn_eps(mlp_in, orig_hidden, eps)
        self.out_norm_eps = stock_rmsn_eps(mlp_out, orig_hidden, eps)
        self.register_buffer("mlp_norm_weight", torch.ones(mlp_in), persistent=False)
        self.register_buffer("out_norm_weight", torch.ones(mlp_out), persistent=False)
        self.overlap = current_platform.device_type == "cuda"

        self._attn_fast_linear = False
        self._split_qkv = False
        self._qkv_w_t: torch.Tensor | None = None
        self._q_w_t: torch.Tensor | None = None
        self._k_w_t: torch.Tensor | None = None
        self._v_w_t: torch.Tensor | None = None
        self._o_w_t: torch.Tensor | None = None
        self._qk_size = self.self_attn.q_size + self.self_attn.kv_size
        self._total_qk_heads = self.self_attn.num_heads + self.self_attn.num_kv_heads

    def enable_npu_fast_linear(self) -> None:
        self.mlp.enable_npu_fast_linear()
        if self._attn_fast_linear:
            return
        o_t = pretranspose_nd_weight(self.self_attn.o_proj.weight)
        if o_t is None:
            return
        if os.environ.get("SLICEGPT_NPU_SPLIT_QKV") != "0":
            split = pretranspose_split_qkv_nd_weight(
                self.self_attn.qkv_proj.weight,
                self.self_attn.qkv_proj.bias,
                self.self_attn.q_size,
                self.self_attn.kv_size,
            )
            if split is not None:
                self._q_w_t, self._k_w_t, self._v_w_t = split[:3]
                self._o_w_t = o_t
                self._split_qkv = True
                self._attn_fast_linear = True
                return
        qkv_t = pretranspose_nd_weight(self.self_attn.qkv_proj.weight)
        if qkv_t is None:
            self.self_attn.o_proj.weight.data = o_t.t().contiguous()
            return
        self._qkv_w_t = qkv_t
        self._o_w_t = o_t
        self._attn_fast_linear = True

    def _run_attn(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        rope_cos_sin: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if not self._attn_fast_linear and rope_cos_sin is None:
            return self.self_attn(positions=positions, hidden_states=hidden_states)
        attn = self.self_attn
        if self._split_qkv:
            q = npu_direct_linear(hidden_states, self._q_w_t)
            k = npu_direct_linear(hidden_states, self._k_w_t)
            v = npu_direct_linear(hidden_states, self._v_w_t)
            if rope_cos_sin is not None:
                cos, sin = rope_cos_sin
                q, k = npu_apply_qk_rope(
                    q, k, cos, sin, attn.num_heads, attn.num_kv_heads, attn.head_dim
                )
            else:
                q, k = attn.rotary_emb(positions, q, k)
        else:
            if self._attn_fast_linear:
                qkv = npu_direct_linear(hidden_states, self._qkv_w_t)
            else:
                qkv, _ = attn.qkv_proj(hidden_states)
            if rope_cos_sin is not None:
                cos, sin = rope_cos_sin
                qk, v = qkv.split([self._qk_size, attn.kv_size], dim=-1)
                qk = npu_fused_qk_rope(
                    qk, cos, sin, self._total_qk_heads, attn.head_dim
                )
                q, k = qk.split([attn.q_size, attn.kv_size], dim=-1)
            else:
                q, k, v = qkv.split([attn.q_size, attn.kv_size, attn.kv_size], dim=-1)
                q, k = attn.rotary_emb(positions, q, k)
        attn_output = attn.attn(q, k, v)
        if self._attn_fast_linear:
            return npu_direct_linear(attn_output, self._o_w_t)
        output, _ = attn.o_proj(attn_output)
        return output

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        rope_cos_sin: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        # Attention block (un-fused residual: rotate residual, then add).
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self._run_attn(positions, hidden_states, rope_cos_sin)
        hidden_states = residual_project(residual, self.attn_shortcut_Q, hidden_states)

        # MLP block.
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
        if self.overlap:
            main, side = torch.cuda.current_stream(), cuda_side_stream()
            side.wait_stream(main)
            with torch.cuda.stream(side):
                rotated = torch.matmul(residual, shortcut_q)
            residual.record_stream(side)
            self._block_out = (
                self._run_attn(positions, block_in) if is_attn else self.mlp(block_in)
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
            self._run_attn(positions, block_in, rope_cos_sin)
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
class SliceGPTLlamaModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config: LlamaConfig = vllm_config.model_config.hf_config
        sc = config.slicing_config
        self.config = config
        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(self.vocab_size, sc["embedding_dim"])
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: SliceGPTDecoderLayer(vllm_config=vllm_config, prefix=prefix),
            prefix=f"{prefix}.layers",
        )
        # Final norm also divides by the original hidden size (final_norm_dim happens
        # to equal it when do_slice_head=false, but be explicit for correctness).
        orig_hidden = config.slicegpt_artifact["original_hidden_size"]
        self.norm = SliceGPTRMSN(orig_hidden, eps=config.rms_norm_eps)

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
        enable_npu_direct_attention(self.layers)
        self._npu_cos_full, self._npu_sin_full = prepare_npu_shared_rope_cache(
            self.layers
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
        rope_cos_sin = select_npu_rope_cos_sin(
            self._npu_cos_full, self._npu_sin_full, positions
        )
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


class SliceGPTLlamaForCausalLM(nn.Module):
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
                "SliceGPTLlamaForCausalLM does not support pipeline parallelism. "
                "Use pipeline_parallel_size=1."
            )
        config: LlamaConfig = vllm_config.model_config.hf_config
        self._artifact_config = config.to_dict()
        validate_config(self._artifact_config)
        self.config = config
        sc = config.slicing_config

        self.model = SliceGPTLlamaModel(
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
                "SliceGPTLlamaForCausalLM does not support pipeline "
                "intermediate_tensors. Use pipeline_parallel_size=1."
            )
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
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
            if getattr(self.config, "tie_word_embeddings", False):
                raise ValueError("Cannot fold the final norm scale into tied lm_head")
            self.lm_head.weight.mul_(self.model.final_scale)
        self._norm_scales_folded = True
