import importlib
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
from vllm_hust_slicegpt import compatibility, plugin


@pytest.mark.parametrize("enabled", ["", "org.example.other"])
def test_disabled_import_has_no_host_or_model_imports(enabled):
    environment = dict(os.environ, VLLMHUST_EXT_ENABLED_BUNDLES=enabled)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from vllm_hust_slicegpt.plugin import register
register()
assert not any(n == 'vllm' or n.startswith('vllm.') or n == 'torch' or
               n.startswith('vllm_hust_slicegpt.models') for n in sys.modules)
""",
        ],
        env=environment,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.fixture
def registry(monkeypatch):
    class Registry:
        models = {}

        @classmethod
        def get_supported_archs(cls):
            return set(cls.models)

        @classmethod
        def register_model(cls, name, implementation):
            assert name not in cls.models
            cls.models[name] = implementation

    monkeypatch.setenv(plugin.ENABLED_BUNDLES_ENV, plugin.BUNDLE_ID)
    monkeypatch.setattr(plugin, "_registered_with", None)
    monkeypatch.setattr(
        compatibility, "version", lambda name: "0.29.1.post1.dev298+g3e67bc6b5"
    )
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(ModelRegistry=Registry))
    monkeypatch.setitem(
        sys.modules,
        "vllm.version",
        SimpleNamespace(__version__="0.29.1.post1.dev298+g3e67bc6b5"),
    )
    return Registry


def test_enabled_registration_is_lazy_and_idempotent(registry):
    plugin.register()
    first = dict(registry.models)
    plugin.register()
    assert registry.models == first
    assert set(first) == {"SliceGPTLlamaForCausalLM", "SliceGPTQwen2ForCausalLM"}
    assert all(isinstance(value, str) for value in first.values())


def test_foreign_registry_entries_are_not_overwritten(registry):
    registry.models["SliceGPTQwen2ForCausalLM"] = "other:Model"
    with pytest.raises(RuntimeError, match="already owned"):
        plugin.register()
    assert registry.models == {"SliceGPTQwen2ForCausalLM": "other:Model"}


def test_shadowed_source_host_is_rejected(registry, monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "vllm.version",
        SimpleNamespace(__version__="0.20.1+g39fef6206.dirty"),
    )
    with pytest.raises(RuntimeError, match="Unsupported"):
        plugin.register()
    assert registry.models == {}


@pytest.mark.parametrize(
    "backend,version",
    [
        ("cpu", "0.29.1"),
        ("xpu", "0.29.1"),
        ("npu", "dev"),
        ("npu", "0.29.1"),
        ("npu", "0.29.1+g83a519c8.dirty"),
    ],
)
def test_unknown_backends_fail_closed(monkeypatch, backend, version):
    monkeypatch.setattr(compatibility, "version", lambda name: version)
    with pytest.raises(RuntimeError):
        compatibility.check_backend(backend)


@pytest.mark.parametrize(
    "version",
    [
        "dev",
        "0.29.1",
        "0.29.1.post1.dev298",
        "0.29.1.post1.dev298+g3e67bc6b5.dirty",
        "0.29.1.post1.dev298+gfffffffff",
        "0.30.0",
        "0.29.1.post1.dev298+g3e67bc6b5.rocm",
    ],
)
def test_unknown_hosts_fail_before_registration(registry, monkeypatch, version):
    monkeypatch.setattr(compatibility, "version", lambda name: version)
    with pytest.raises(RuntimeError):
        plugin.register()
    assert registry.models == {}


@pytest.mark.parametrize(
    "tp,pp,eager,quant",
    [
        (2, 1, True, None),
        (1, 2, True, None),
        (1, 1, False, None),
        (1, 1, True, object()),
    ],
)
def test_unsupported_execution_modes(tp, pp, eager, quant):
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tp, pipeline_parallel_size=pp
        ),
        model_config=SimpleNamespace(enforce_eager=eager),
        quant_config=quant,
    )
    with pytest.raises(ValueError):
        compatibility.check_execution_config(config)


def test_manifest_is_importable_without_activation():
    importlib.import_module("vllm_hust_slicegpt.manifests")
