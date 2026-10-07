import copy
import json
from types import SimpleNamespace

import pytest
from conftest import model_config
from vllm_hust_slicegpt.artifact import (
    ArtifactError,
    checked_weights,
    validate_config,
    validate_directory,
)
from vllm_hust_slicegpt.artifact_cli import migrate


def test_valid_directory_and_content_corruption(artifact):
    validate_directory(artifact)
    path = artifact / "model.safetensors"
    with path.open("r+b") as stream:
        stream.seek(-1, 2)
        byte = stream.read(1)
        stream.seek(-1, 2)
        stream.write(bytes([byte[0] ^ 1]))
    with pytest.raises(ArtifactError, match="SHA-256"):
        validate_directory(artifact)


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", "slicegpt-artifact/99"),
        ("original_hidden_size", 6),
        ("original_hidden_size", True),
        ("unexpected", "field"),
    ],
)
def test_unknown_or_invalid_metadata(field, value):
    config = model_config()
    config["slicegpt_artifact"][field] = value
    with pytest.raises(ArtifactError):
        validate_config(config)


@pytest.mark.parametrize(
    "key,value",
    [
        ("attention_input_dims", [6]),
        ("mlp_input_dims", [4, 4]),
        ("mlp_output_dims", [6, 4]),
        ("attention_input_dims", [6.0, 4]),
        ("attn_head_dim", 3),
    ],
)
def test_residual_connections_and_dimensions(key, value):
    config = model_config()
    config["slicing_config"][key] = value
    with pytest.raises(ArtifactError):
        validate_config(config)


def test_runtime_stream_rejects_missing_duplicate_and_bad_shapes():
    config = model_config()
    tensors = [
        (name, SimpleNamespace(shape=shape))
        for name, shape in validate_config(config).items()
    ]
    assert len(list(checked_weights(config, iter(tensors)))) == len(tensors)
    for invalid in [
        tensors[:-1],
        tensors + [tensors[0]],
        [("unknown", tensors[0][1])],
        [(tensors[0][0], SimpleNamespace(shape=(1,)))] + tensors[1:],
    ]:
        with pytest.raises(ArtifactError):
            list(checked_weights(config, iter(invalid)))


def test_explicit_legacy_migration_preserves_weights_and_source(artifact, tmp_path):
    source = json.loads((artifact / "config.json").read_text())
    source.pop("slicegpt_artifact")
    source["model_type"] = "slicegpt_qwen2"
    source["auto_map"] = {
        "AutoConfig": "configuration_slicegpt_qwen2.SliceGPTQwen2Config"
    }
    original = json.dumps(source)
    (artifact / "config.json").write_text(original)
    destination = tmp_path / "migrated"
    arguments = dict(
        original_hidden_size=8,
        base_model_id="synthetic",
        base_revision="seed-42",
        model_license="test-fixture",
        calibration={"status": "not-recorded"},
    )
    migrate(artifact, destination, **arguments)
    result = validate_directory(destination)
    assert result["model_type"] == "qwen2" and "auto_map" not in result
    assert (artifact / "config.json").read_text() == original
    assert (artifact / "model.safetensors").read_bytes() == (
        destination / "model.safetensors"
    ).read_bytes()
    with pytest.raises(ArtifactError, match="already exists"):
        migrate(artifact, destination, **arguments)
    with pytest.raises(ArtifactError, match="already has artifact"):
        migrate(destination, tmp_path / "another", **arguments)


def test_failed_migration_does_not_publish_output(artifact, tmp_path):
    config = json.loads((artifact / "config.json").read_text())
    del config["slicegpt_artifact"]
    config["slicing_config"]["mlp_input_dims"] = [4, 4]
    (artifact / "config.json").write_text(json.dumps(config))
    with pytest.raises(ArtifactError):
        migrate(
            artifact,
            tmp_path / "bad",
            original_hidden_size=8,
            base_model_id="synthetic",
            base_revision="seed-42",
            model_license="test-fixture",
            calibration={},
        )
    assert not (tmp_path / "bad").exists()


def test_no_implicit_legacy_acceptance():
    config = copy.deepcopy(model_config())
    config.pop("slicegpt_artifact")
    with pytest.raises(ArtifactError, match="migrate"):
        validate_config(config)
