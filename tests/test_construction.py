"""Real CPU calibration/rotation on tiny random models, with no downloads."""

import json
import subprocess
import sys

import pytest


@pytest.mark.toolkit
@pytest.mark.parametrize("family", ["llama", "qwen2"])
@pytest.mark.parametrize("width", [16, 12])
def test_rotation_preserves_full_width_logits_and_slices(family, width, tmp_path):
    import torch
    import transformers

    if transformers.__version__ != "4.41.0":
        pytest.skip("Run in the toolkit environment pinned to Transformers 4.41.0")
    from slicegpt import layernorm_fusion, rotate
    from slicegpt.adapters.llama_adapter import LlamaModelAdapter
    from slicegpt.adapters.qwen2_adapter import Qwen2ModelAdapter
    from slicegpt.config import config
    from slicegpt.slicing_scheduler import ConstSlicingScheduler

    torch.set_num_threads(1)
    torch.manual_seed(42)
    config.device = torch.device("cpu")
    config.dtype = torch.float32
    config_cls, model_cls, adapter_cls = (
        (transformers.LlamaConfig, transformers.LlamaForCausalLM, LlamaModelAdapter)
        if family == "llama"
        else (
            transformers.Qwen2Config,
            transformers.Qwen2ForCausalLM,
            Qwen2ModelAdapter,
        )
    )
    hf_config = config_cls(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=32,
        max_position_embeddings=32,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
        use_cache=False,
    )
    hf_config._attn_implementation = "eager"
    model = model_cls(hf_config).eval()
    tokens = torch.randint(0, 32, (2, 8))
    with torch.no_grad():
        reference = model(tokens).logits.clone()
    adapter = adapter_cls(model)
    layernorm_fusion.replace_layers(adapter)
    layernorm_fusion.fuse_modules(adapter)
    batches = [{"input_ids": tokens, "attention_mask": torch.ones_like(tokens)}]
    rotate.rotate_and_slice(
        adapter,
        batches,
        ConstSlicingScheduler(width, do_slice_head=True),
        final_orientation="pca",
    )
    with torch.no_grad():
        actual = model(tokens).logits
    assert actual.shape == reference.shape and torch.isfinite(actual).all()
    assert model.model.embed_tokens.weight.shape[1] == width
    if width == 16:
        torch.testing.assert_close(actual, reference, atol=2e-5, rtol=2e-4)
    # Exercise the public conversion command on the actual rotated state, not
    # just a synthetic shape fixture.
    sliced = tmp_path / "sliced"
    sliced.mkdir()
    base = tmp_path / "base"
    base.mkdir()
    torch.save(model.state_dict(), sliced / "model.pt")
    (sliced / "model.json").write_text(adapter.slicing_conf.to_json_string())
    (base / "config.json").write_text(json.dumps(hf_config.to_dict()))
    module = "slicegpt_toolkit.convert_slicegpt_to_vllm" + (
        "_qwen2" if family == "qwen2" else ""
    )
    destination = tmp_path / "exported"
    result = subprocess.run(
        [
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
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    from vllm_hust_slicegpt.artifact import validate_directory

    exported = validate_directory(destination)
    assert exported["slicing_config"]["embedding_dim"] == width
