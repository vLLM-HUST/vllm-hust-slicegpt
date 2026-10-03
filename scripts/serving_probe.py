"""Run via Manager on a qualified candidate host; this is not an ECPA observer."""

import argparse
import importlib.metadata
import json
import os
import platform
from pathlib import Path

from vllm_hust_slicegpt.artifact import validate_directory
from vllm_hust_slicegpt.plugin import BUNDLE_ID, ENABLED_BUNDLES_ENV


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    args = parser.parse_args()
    if BUNDLE_ID not in os.getenv(ENABLED_BUNDLES_ENV, "").split(","):
        parser.error("Run with the SliceGPT Bundle enabled through vllm-hust-ext run")
    if args.output.exists():
        parser.error("Output already exists; keep acceptance records immutable")
    config = validate_directory(args.model)
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=str(args.model.resolve()),
        skip_tokenizer_init=True,
        enforce_eager=True,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        max_model_len=128,
        max_num_seqs=2,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    outputs = llm.generate(
        [{"prompt_token_ids": [1, 2, 3]}, {"prompt_token_ids": [3, 2, 1]}],
        SamplingParams(temperature=0, max_tokens=8, ignore_eos=True),
    )
    tokens = [list(result.outputs[0].token_ids) for result in outputs]
    if len(tokens) != 2 or any(len(row) != 8 for row in tokens):
        raise RuntimeError("Generation did not complete both eight-token requests")
    evidence = {
        "schema": "slicegpt-serving-smoke/1",
        "result": "passed",
        "observation_kind": "front_end_generation_smoke",
        "runtime_effective": None,
        "host": platform.node(),
        "front_end_pid": os.getpid(),
        "vllm": importlib.metadata.version("vllm"),
        "runtime": importlib.metadata.version("vllm-hust-slicegpt"),
        "artifact": config["slicegpt_artifact"],
        "requested_architecture": config["architectures"],
        "generated_token_ids": tokens,
        "note": "No owning-worker observer proof or quality/performance claim is made",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(evidence, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
