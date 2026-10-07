# Pluginization acceptance and recovery

Reference Manager: `98903e416bdb593186b8245fd95180dafde995b9`.
ECPA 0.3 remains experimental; this project does not claim a stable ECPA v1 API.

The Bundle entry point is a static module directory. The manifest and callable
entry point must belong to the same installed distribution. Discovery must not
import torch, vLLM, model implementations or accelerator runtimes. Resource names
`vllm.model-registry.slicegpt-llama` and `vllm.model-registry.slicegpt-qwen2` are this
bundle's exclusive architecture claims, not a new host protocol. `protocols=[]`
is deliberate: the inspected host has no independently versioned model-registry
protocol. We do not claim ownership of the full registry, devices, KV cache,
global Ascend RoPE, or a host observer that we have not installed.

## CPU packaging gate

Build both wheels, create a new environment without torch or vLLM, and install
the Runtime wheel plus the pinned Manager. Execute:

```bash
.venv-clean/bin/python scripts/check_wheel.py
uv pip uninstall --python .venv-clean/bin/python vllm-hust-slicegpt
.venv-clean/bin/python -c 'import importlib.util; assert importlib.util.find_spec("vllm_hust_slicegpt") is None'
```

The script checks discover/inspect/validate/check/plan/render, disabled imports,
explicit enable intent, unknown-host launch refusal, disabled next-process
behavior, configuration rollback and forget. `check` returns diagnostic JSON;
its exit code alone is not compatibility evidence. Plan/render may describe an
unverified candidate and do not activate it. No fake host metadata is installed
to make these checks appear compatible.

Unit tests use an explicitly synthetic registry for active-registration and
collision behavior. Those tests do not count as real vLLM integration evidence.

## Required serving gate (still pending)

Use a clean host without legacy built-in SliceGPT. Pin and record host/package
versions and hashes, hardware, driver/runtime versions, model revision/license,
dataset revision/license and the complete command line.

1. With the wheel installed and disabled, start a dense model and verify no
   SliceGPT registration/injection. Preserve other enabled plugins.
2. Validate an artifact, enable and start the candidate. Confirm class selection,
   complete weight loading, real prefill/decode and nonempty generated output.
3. Compare logits/PPL with the same compressed reference artifact; report dense
   quality separately. Repeat CUDA and Ascend independently. Do not transfer
   historical performance numbers onto this new implementation.
4. Collect owning-process evidence of actual SliceGPT work. The host observer
   must bind plan id, launch id, process start identity and selected implementation
   to a real model execution. Import/registration/constructor success is insufficient.
   Do not call private host event emitters from `plugin.register()` to forge this.
5. Disable, stop the process tree, restart the dense baseline and check the native
   path. Repeat after configuration rollback, forget and wheel uninstall.
6. Reject future artifact versions, unsupported source versions, TP/PP/graph/
   quantization modes, duplicate architecture ownership and corrupt checkpoints.

Until gate 4 exists and passes, `runtime_effective` is intentionally unavailable.
There is no custom fallback that interprets an environment variable or a local
success boolean as runtime evidence.

The pending two-request serving probe is executable on a prepared target:

```bash
vllm-hust-ext run -- python scripts/serving_probe.py "$MODEL_DIR" --output artifacts/serving-smoke.json
```

It validates the artifact, runs real generation and records front-end evidence.
It explicitly records `runtime_effective: null`; the front-end PID is not worker
observer evidence. Adjust the memory fraction to the dedicated test device.

## Rollback and uninstall

Keep the previously accepted wheels, their checksums, the corresponding model
artifact and a backup of Manager configuration. Disable the Bundle and stop its
serving process tree. Reinstall the previous wheel and restore the matching
configuration/artifact; then start a new process and rerun the acceptance probe.
The first migration release has no previously qualified plugin wheel, so its
rollback target is the disabled native host and original dense model.

`forget` removes Manager-owned intent/configuration only. Uninstall affects the
selected environment only; no source patches, `.pth` files or `sitecustomize`
hooks are written. Artifact directories are not deleted. A process that already
imported the plugin must be stopped before verifying removal.
