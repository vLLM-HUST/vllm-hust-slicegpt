"""Narrow source-candidate compatibility, separate from hardware qualification."""

from importlib.metadata import PackageNotFoundError, version

from packaging.version import InvalidVersion, Version

HOST_VERSION = "0.29.1.post1.dev298"
HOST_COMMIT = "3e67bc6b5734f98600bb171039b375f2c4f2d2ea"
ASCEND_COMMIT = "83a519c8e915938724369c6d6232d73a425a1034"


def check_host_version(value: str) -> None:
    try:
        parsed = Version(value)
    except InvalidVersion as exc:
        raise RuntimeError(f"Unknown vLLM version: {value!r}") from exc
    local = (parsed.local or "").split(".")
    commit = local[0].removeprefix("g")
    if (
        parsed.public != HOST_VERSION
        or not local[0].startswith("g")
        or len(commit) < 9
        or not HOST_COMMIT.startswith(commit)
        or "dirty" in local
        or any(
            part not in {"precompiled", "cu130", "cu129", "cu128"} for part in local[1:]
        )
    ):
        raise RuntimeError(
            f"Unsupported vLLM {value!r}; SliceGPT targets the clean source candidate "
            f"{HOST_VERSION} at {HOST_COMMIT}. Hardware qualification is separate."
        )


def check_installed_host() -> None:
    try:
        installed = version("vllm")
    except PackageNotFoundError as exc:
        raise RuntimeError("Install the documented vLLM-HUST host first") from exc
    check_host_version(installed)


def check_execution_config(vllm_config) -> None:
    parallel = vllm_config.parallel_config
    if parallel.tensor_parallel_size != 1 or parallel.pipeline_parallel_size != 1:
        raise ValueError("SliceGPT currently requires TP=1 and PP=1")
    if not vllm_config.model_config.enforce_eager:
        raise ValueError("SliceGPT currently requires enforce_eager=True")
    if vllm_config.quant_config is not None:
        raise ValueError("SliceGPT runtime does not support quantized weights")


def check_backend(device_type: str) -> None:
    if device_type == "cuda":
        return
    if device_type != "npu":
        raise RuntimeError(f"Unqualified SliceGPT backend: {device_type}")
    try:
        parsed = Version(version("vllm-ascend"))
    except (PackageNotFoundError, InvalidVersion) as exc:
        raise RuntimeError(
            "A traceable vLLM-Ascend candidate build is required"
        ) from exc
    local = (parsed.local or "").split(".")
    commits = [part[1:] for part in local if part.startswith("g")]
    if "dirty" in local or not any(
        len(commit) >= 8 and ASCEND_COMMIT.startswith(commit) for commit in commits
    ):
        raise RuntimeError(
            f"Unqualified vLLM-Ascend build; expected source {ASCEND_COMMIT}"
        )
