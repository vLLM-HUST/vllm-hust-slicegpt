"""Versioned artifact validation without importing torch or vLLM."""

import hashlib
import json
import math
from pathlib import Path

from . import __version__

SCHEMA = "slicegpt-artifact/1"
LEGACY_SOURCE_COMMIT = "19b61cb8b95c9e9c2feb3dfab49cb5bc8f2623ee"
ARCHITECTURES = {
    "SliceGPTLlamaForCausalLM": "llama",
    "SliceGPTQwen2ForCausalLM": "qwen2",
}
_DIMS = (
    "attention_input_dims",
    "attention_output_dims",
    "mlp_input_dims",
    "mlp_output_dims",
)
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "generation_config.json",
    "merges.txt",
    "vocab.json",
)


class ArtifactError(ValueError):
    pass


def _positive(value, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ArtifactError(f"{name} must be a positive integer")
    return value


def _nonempty(value, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or value.lower() in {"unknown", "main", "latest"}
    ):
        raise ArtifactError(
            f"{name} must be explicit (use a pinned revision, not main/latest)"
        )
    return value


def validate_config(config: dict) -> dict[str, tuple[int, ...]]:
    """Return exact tensor shapes after validating the version and residual graph."""
    meta = config.get("slicegpt_artifact")
    if not isinstance(meta, dict) or meta.get("schema_version") != SCHEMA:
        raise ArtifactError(
            "Missing/unknown SliceGPT artifact version; migrate legacy artifacts explicitly"
        )
    if set(meta) != {
        "schema_version",
        "producer",
        "base_model",
        "original_hidden_size",
        "calibration",
        "weights",
    }:
        raise ArtifactError("Unexpected/missing SliceGPT artifact metadata fields")
    if not isinstance(meta["producer"], dict) or set(meta["producer"]) != {
        "name",
        "version",
        "legacy_source_commit",
    }:
        raise ArtifactError("producer requires name, version and legacy_source_commit")
    for key, value in meta["producer"].items():
        _nonempty(value, f"producer.{key}")
    base = meta["base_model"]
    if not isinstance(base, dict) or set(base) != {"id", "revision", "license"}:
        raise ArtifactError("base_model requires id, revision and license")
    for key, value in base.items():
        _nonempty(value, f"base_model.{key}")
    if not isinstance(meta["calibration"], dict):
        raise ArtifactError(
            "calibration must be a record (legacy details may be explicitly unavailable)"
        )
    weights = meta["weights"]
    if not isinstance(weights, dict) or set(weights) != {"model.safetensors"}:
        raise ArtifactError("Artifact v1 requires exactly model.safetensors")
    digest = weights["model.safetensors"]
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(c not in "0123456789abcdef" for c in digest)
    ):
        raise ArtifactError("weights requires a SHA-256 digest")
    architectures = config.get("architectures")
    if (
        not isinstance(architectures, list)
        or len(architectures) != 1
        or architectures[0] not in ARCHITECTURES
    ):
        raise ArtifactError("Unsupported SliceGPT architecture")
    family = ARCHITECTURES[architectures[0]]
    if config.get("model_type") != family or config.get("auto_map"):
        raise ArtifactError("Use the canonical llama/qwen2 config without auto_map")
    if (
        config.get("quantization_config") is not None
        or config.get("tie_word_embeddings") is not False
    ):
        raise ArtifactError("Artifact v1 requires unquantized, untied weights")
    dtype = config.get("torch_dtype", config.get("dtype"))
    if dtype not in {"float16", "bfloat16", "float32"}:
        raise ArtifactError("Unsupported artifact dtype")
    if config.get("hidden_act") != "silu" or config.get("use_sliding_window", False):
        raise ArtifactError("Artifact v1 requires silu and full attention")
    if family == "llama" and (
        config.get("attention_bias", False) or config.get("mlp_bias", False)
    ):
        raise ArtifactError("Llama attention/MLP bias is unsupported")
    eps = config.get("rms_norm_eps")
    if (
        isinstance(eps, bool)
        or not isinstance(eps, (int, float))
        or not math.isfinite(eps)
        or eps <= 0
    ):
        raise ArtifactError("rms_norm_eps must be finite and positive")
    layers = _positive(config.get("num_hidden_layers"), "num_hidden_layers")
    heads = _positive(config.get("num_attention_heads"), "num_attention_heads")
    kv_heads = _positive(
        config.get("num_key_value_heads", heads), "num_key_value_heads"
    )
    if heads % kv_heads:
        raise ArtifactError("Query heads must be divisible by KV heads")
    vocab = _positive(config.get("vocab_size"), "vocab_size")
    intermediate = _positive(config.get("intermediate_size"), "intermediate_size")
    original = _positive(meta["original_hidden_size"], "original_hidden_size")
    sc = config.get("slicing_config")
    if not isinstance(sc, dict) or set(sc) != set(_DIMS) | {
        "embedding_dim",
        "final_norm_dim",
        "attn_head_dim",
    }:
        raise ArtifactError("Unexpected/missing slicing_config fields")
    for key in _DIMS:
        if not isinstance(sc[key], list) or len(sc[key]) != layers:
            raise ArtifactError(f"{key} must describe all {layers} layers")
        for dim in sc[key]:
            if _positive(dim, key) > original:
                raise ArtifactError(f"{key} exceeds original_hidden_size")
    embedding = _positive(sc["embedding_dim"], "embedding_dim")
    final = _positive(sc["final_norm_dim"], "final_norm_dim")
    head_dim = _positive(sc["attn_head_dim"], "attn_head_dim")
    if head_dim % 2 or original != heads * head_dim:
        raise ArtifactError(
            "Original width and unchanged attention head dimensions disagree"
        )
    if config.get("hidden_size") != embedding or config.get("head_dim") != head_dim:
        raise ArtifactError("HF hidden_size/head_dim disagree with slicing_config")
    if embedding != sc["attention_input_dims"][0] or final != sc["mlp_output_dims"][-1]:
        raise ArtifactError("Embedding/final residual dimensions are disconnected")
    shapes = {
        "model.embed_tokens.weight": (vocab, embedding),
        "lm_head.weight": (vocab, final),
    }
    for i in range(layers):
        ai, ao, mi, mo = (sc[key][i] for key in _DIMS)
        if ao != mi or (i + 1 < layers and mo != sc["attention_input_dims"][i + 1]):
            raise ArtifactError(f"Residual dimensions are disconnected at layer {i}")
        if family == "llama" and ai != ao:
            raise ArtifactError(
                "Llama runtime requires equal attention input/output widths"
            )
        prefix = f"model.layers.{i}"
        for proj, width in (
            ("q", heads * head_dim),
            ("k", kv_heads * head_dim),
            ("v", kv_heads * head_dim),
        ):
            shapes[f"{prefix}.self_attn.{proj}_proj.weight"] = (width, ai)
            if family == "qwen2":
                shapes[f"{prefix}.self_attn.{proj}_proj.bias"] = (width,)
        shapes.update(
            {
                f"{prefix}.self_attn.o_proj.weight": (ao, heads * head_dim),
                f"{prefix}.mlp.gate_proj.weight": (intermediate, mi),
                f"{prefix}.mlp.up_proj.weight": (intermediate, mi),
                f"{prefix}.mlp.down_proj.weight": (mo, intermediate),
                f"{prefix}.attn_shortcut_Q": (ai, ao),
                f"{prefix}.mlp_shortcut_Q": (mi, mo),
            }
        )
    return shapes


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_directory(path: Path, *, verify_hash: bool = True) -> dict:
    from safetensors import safe_open

    config = json.loads((path / "config.json").read_text())
    shapes = validate_config(config)
    tensor_path = path / "model.safetensors"
    if (
        verify_hash
        and sha256(tensor_path)
        != config["slicegpt_artifact"]["weights"][tensor_path.name]
    ):
        raise ArtifactError("model.safetensors SHA-256 mismatch")
    expected_dtype = {"float16": "F16", "bfloat16": "BF16", "float32": "F32"}[
        config.get("torch_dtype", config.get("dtype"))
    ]
    with safe_open(tensor_path, framework="np") as weights:
        if set(weights.keys()) != set(shapes):
            raise ArtifactError("Checkpoint has missing or unexpected tensors")
        for name, shape in shapes.items():
            tensor = weights.get_slice(name)
            if (
                tuple(tensor.get_shape()) != shape
                or tensor.get_dtype() != expected_dtype
            ):
                raise ArtifactError(f"Tensor shape/dtype mismatch: {name}")
    return config


def seal_directory(
    path: Path,
    *,
    original_hidden_size: int,
    base_model_id: str,
    base_revision: str,
    model_license: str,
    calibration: dict,
) -> None:
    config = json.loads((path / "config.json").read_text())
    config["slicegpt_artifact"] = {
        "schema_version": SCHEMA,
        "producer": {
            "name": "vllm-hust-slicegpt",
            "version": __version__,
            "legacy_source_commit": LEGACY_SOURCE_COMMIT,
        },
        "base_model": {
            "id": base_model_id,
            "revision": base_revision,
            "license": model_license,
        },
        "original_hidden_size": original_hidden_size,
        "calibration": calibration,
        "weights": {"model.safetensors": sha256(path / "model.safetensors")},
    }
    validate_config(config)
    temporary = path / ".config.json.tmp"
    temporary.write_text(json.dumps(config, indent=2) + "\n")
    temporary.replace(path / "config.json")
    validate_directory(path, verify_hash=False)


def checked_weights(config: dict, weights):
    """Validate streamed source tensors before vLLM packs their QKV/MLP names."""
    shapes = validate_config(config)
    seen = set()
    for name, tensor in weights:
        if name not in shapes or name in seen:
            raise ArtifactError(f"Unexpected or duplicate runtime tensor: {name}")
        if tuple(tensor.shape) != shapes[name]:
            raise ArtifactError(f"Runtime tensor shape mismatch: {name}")
        seen.add(name)
        yield name, tensor
    if seen != set(shapes):
        raise ArtifactError(f"Missing runtime tensors: {sorted(set(shapes) - seen)}")
