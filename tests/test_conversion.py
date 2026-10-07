import json
import subprocess
import sys

import pytest
from conftest import model_config
from vllm_hust_slicegpt.artifact import validate_config, validate_directory


@pytest.mark.parametrize("family", ["llama", "qwen2"])
def test_converter_exports_real_safetensors_atomically(tmp_path, family):
    torch = pytest.importorskip("torch")
    config = model_config(family)
    state = {
        key: torch.arange(torch.tensor(shape).prod()).reshape(shape).float()
        for key, shape in validate_config(config).items()
    }
    state["model.norm.weight"] = torch.ones(8)
    sliced = tmp_path / "sliced"
    sliced.mkdir()
    torch.save(state, sliced / "model.pt")
    sc = config["slicing_config"]
    raw = {
        key.replace("_dims", "_dimensions"): dict(enumerate(sc[key]))
        for key in (
            "attention_input_dims",
            "attention_output_dims",
            "mlp_input_dims",
            "mlp_output_dims",
        )
    }
    raw.update(embedding_dimensions={0: 6}, head_dimension=4)
    (sliced / "model.json").write_text(json.dumps(raw))
    base = tmp_path / "base"
    base.mkdir()
    config.pop("slicegpt_artifact")
    config.pop("slicing_config")
    config["hidden_size"] = 8
    (base / "config.json").write_text(json.dumps(config))
    destination = tmp_path / "out"
    module = "slicegpt_toolkit.convert_slicegpt_to_vllm" + (
        "_qwen2" if family == "qwen2" else ""
    )
    command = [
        sys.executable,
        "-m",
        module,
        "--sliced-dir",
        str(sliced),
        "--pt-name",
        "model.pt",
        "--base-model-path",
        str(base),
        "--out-dir",
        str(destination),
        "--base-model-id",
        "synthetic",
        "--base-revision",
        "seed-42",
        "--model-license",
        "test-fixture",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    validate_directory(destination)
    from safetensors.torch import load_file

    exported = load_file(destination / "model.safetensors")
    assert "model.norm.weight" not in exported
    for key, value in exported.items():
        torch.testing.assert_close(value, state[key].half())
    assert subprocess.run(command, capture_output=True).returncode != 0
    state.pop("lm_head.weight")
    torch.save(state, sliced / "model.pt")
    command[command.index(str(destination))] = str(tmp_path / "invalid")
    assert subprocess.run(command, capture_output=True).returncode != 0
    assert not (tmp_path / "invalid").exists()
