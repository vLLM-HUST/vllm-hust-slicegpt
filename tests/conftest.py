import json

import numpy as np
import pytest
from safetensors.numpy import save_file
from vllm_hust_slicegpt.artifact import SCHEMA, seal_directory, validate_config


def model_config(family="qwen2"):
    sc = {
        "embedding_dim": 6,
        "final_norm_dim": 4,
        "attn_head_dim": 4,
        "attention_input_dims": [6, 4],
        "attention_output_dims": [6, 4],
        "mlp_input_dims": [6, 4],
        "mlp_output_dims": [4, 4],
    }
    return {
        "architectures": [
            "SliceGPTQwen2ForCausalLM"
            if family == "qwen2"
            else "SliceGPTLlamaForCausalLM"
        ],
        "model_type": family,
        "hidden_size": 6,
        "head_dim": 4,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "num_hidden_layers": 2,
        "intermediate_size": 12,
        "vocab_size": 16,
        "rms_norm_eps": 1e-6,
        "hidden_act": "silu",
        "torch_dtype": "float16",
        "max_position_embeddings": 128,
        "rope_theta": 10000.0,
        "tie_word_embeddings": False,
        "slicing_config": sc,
        "slicegpt_artifact": {
            "schema_version": SCHEMA,
            "producer": {
                "name": "test",
                "version": "0.1",
                "legacy_source_commit": "test-revision",
            },
            "base_model": {
                "id": "synthetic",
                "revision": "seed-42",
                "license": "test-fixture",
            },
            "original_hidden_size": 8,
            "calibration": {"status": "synthetic"},
            "weights": {"model.safetensors": "0" * 64},
        },
    }


def create_artifact(path, family="qwen2"):
    path.mkdir()
    config = model_config(family)
    arrays = {
        name: np.ones(shape, dtype=np.float16)
        for name, shape in validate_config(config).items()
    }
    save_file(arrays, path / "model.safetensors")
    (path / "config.json").write_text(json.dumps(config))
    seal_directory(
        path,
        original_hidden_size=8,
        base_model_id="synthetic",
        base_revision="seed-42",
        model_license="test-fixture",
        calibration={"status": "synthetic"},
    )
    return path


@pytest.fixture
def artifact(tmp_path):
    return create_artifact(tmp_path / "model")
