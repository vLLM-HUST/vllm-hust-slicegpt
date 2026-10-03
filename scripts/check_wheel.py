"""Run inside a clean runtime+Manager wheel environment (no vLLM or torch).

This checks packaging and lifecycle intent. It does not qualify a serving host.
"""

import importlib.metadata
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def main():
    assert importlib.util.find_spec("vllm") is None, "Use a clean environment"
    assert importlib.util.find_spec("torch") is None, "Use a clean environment"
    bundle_id = "org.vllm-hust.slicegpt"
    entries = importlib.metadata.entry_points(group="vllm_hust.extension_bundles")
    assert len([entry for entry in entries if entry.name == bundle_id]) == 1
    with tempfile.TemporaryDirectory(prefix="slicegpt-wheel-") as directory:
        config_path = Path(directory) / "manager.json"
        env = {
            key: value
            for key, value in os.environ.items()
            if key not in {"PYTHONPATH", "VLLM_PLUGINS", "VLLMHUST_EXT_ENABLED_BUNDLES"}
        }
        env["VLLM_HUST_EXT_CONFIG"] = str(config_path)
        env.pop("VLLM_HUST_EXT_EVIDENCE_PATH", None)

        def manager(*args, expect_success=True):
            result = subprocess.run(
                [str(Path(sys.executable).parent / "vllm-hust-ext"), *args],
                env=env,
                capture_output=True,
                text=True,
                cwd=directory,
            )
            assert (result.returncode == 0) == expect_success, (
                result.stderr + result.stdout
            )
            if not expect_success:
                assert "refusing to launch unverified" in result.stderr, result.stderr
            return result.stdout

        def no_injection(extra=None):
            child_env = dict(env, **(extra or {}))
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    """
import sys
from vllm_hust_slicegpt.plugin import register
register()
assert 'vllm' not in sys.modules and 'torch' not in sys.modules
assert not any(n.startswith('vllm_hust_slicegpt.models') for n in sys.modules)
""",
                ],
                env=child_env,
                cwd=directory,
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0, result.stderr

        manager("extension", "list", "--json")
        for action in ("inspect", "validate", "check", "plan", "render"):
            json.loads(manager("extension", action, bundle_id))
        no_injection()
        status = json.loads(manager("extension", "check", bundle_id))
        assert "runtime_effective" not in status["states"]
        assert "compatible" not in status["states"]  # no installed host
        manager("extension", "enable", bundle_id)
        activated = json.loads(manager("extension", "env"))
        assert activated["VLLMHUST_EXT_ENABLED_BUNDLES"] == bundle_id
        assert {"ascend", "slicegpt"}.issubset(activated["VLLM_PLUGINS"].split(","))
        manager(
            "run",
            "--",
            sys.executable,
            "-c",
            "raise AssertionError('must not launch')",
            expect_success=False,
        )
        manager("extension", "disable", bundle_id)
        disabled_snapshot = config_path.read_bytes()
        no_injection(json.loads(manager("extension", "env")))
        manager("extension", "enable", bundle_id)
        config_path.write_bytes(disabled_snapshot)  # explicit configuration rollback
        no_injection(json.loads(manager("extension", "env")))
        status = json.loads(manager("extension", "status", bundle_id))
        assert (
            "enabled" not in status["states"]
            and "runtime_effective" not in status["states"]
        )
        manager("extension", "forget", bundle_id)
        assert bundle_id not in json.loads(config_path.read_text())["extensions"]
        no_injection()
    print(
        "PASS: clean-wheel discovery/inspect/check/plan/render, no injection, unknown-host refusal, disable/configuration rollback/forget"
    )


if __name__ == "__main__":
    main()
