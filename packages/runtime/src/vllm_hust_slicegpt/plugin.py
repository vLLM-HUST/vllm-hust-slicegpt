"""A gated general plugin. Installation alone never changes the registry."""

import os
from threading import RLock

from .compatibility import check_host_version, check_installed_host

BUNDLE_ID = "org.vllm-hust.slicegpt"
ENABLED_BUNDLES_ENV = "VLLMHUST_EXT_ENABLED_BUNDLES"
_MODELS = {
    "SliceGPTLlamaForCausalLM": "vllm_hust_slicegpt.models.llama:SliceGPTLlamaForCausalLM",
    "SliceGPTQwen2ForCausalLM": "vllm_hust_slicegpt.models.qwen2:SliceGPTQwen2ForCausalLM",
}
_lock = RLock()
_registered_with = None


def register() -> None:
    enabled = {item.strip() for item in os.getenv(ENABLED_BUNDLES_ENV, "").split(",")}
    if BUNDLE_ID not in enabled:
        return
    check_installed_host()
    from vllm import ModelRegistry
    from vllm.version import __version__ as loaded_host_version

    check_host_version(loaded_host_version)

    global _registered_with
    with _lock:
        if _registered_with is ModelRegistry:
            return
        collisions = set(_MODELS).intersection(ModelRegistry.get_supported_archs())
        if collisions:
            raise RuntimeError(
                f"SliceGPT architectures already owned by another implementation: "
                f"{sorted(collisions)}. Use a host without legacy built-in SliceGPT."
            )
        for architecture, implementation in _MODELS.items():
            ModelRegistry.register_model(architecture, implementation)
        _registered_with = ModelRegistry
