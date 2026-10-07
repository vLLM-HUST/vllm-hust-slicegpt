# Validation record — 2026-10-03

This record separates CPU functionality and packaging from pending hardware
qualification. No new model quality, throughput or memory claim is made.

## Executed checks

Environment: Linux x86_64, Python 3.11.15, Torch 2.11.0,
Transformers 4.41.0, NumPy 1.26.4, safetensors 0.8.0, pytest 9.0.3.
Tests run with `CUDA_VISIBLE_DEVICES=''` and do not reserve a GPU.

| Check | Result |
| --- | --- |
| `CUDA_VISIBLE_DEVICES='' .venv-toolkit/bin/python -m pytest -q` | 44 passed |
| Tiny Llama/Qwen2 CPU calibration, full-width rotation and actual slicing | Passed; full-width logits checked against the original random models |
| Real rotated tiny state to canonical safetensors via both public converter CLIs | Passed |
| Corrupt hash, malformed residual graph, missing/duplicate tensors, unknown schema | Rejected as expected |
| Disabled entrypoint in a fresh subprocess without vLLM/torch imports | Passed |
| Active registration/idempotence, foreign ownership, unknown/shadowed host versions | Passed with an explicitly synthetic registry, not a serving host |
| Model-local native RoPE dispatch | Unit test passed; not an NPU numerical result |
| Four command-line `--help` entrypoints | Passed |
| Runtime and Toolkit sdist + wheel builds | Passed using hatchling 1.32.4 |
| Wheel archive inspection | Includes entrypoints, static manifest and license files; no weights, caches or environments |
| Ruff lint and format checks | Passed for new runtime, CLI, tests and scripts |

The original legacy source is retained under its notices. Numerical tests found
two reproducible inherited failures: RMSNorm model embeddings were centered even
though those models do not subtract the mean, and the Qwen2 compressed decoder
returned a Tensor instead of the tuple consumed by Transformers 4.41. Both were
fixed; the zero-slice tests now pass for Llama and Qwen2. This is a correctness
result on synthetic models, not evidence of improved 14B perplexity.

## Clean wheel environment

The Runtime wheel was installed in a fresh environment containing only it,
safetensors, packaging, platformdirs and Manager `0.2.0.dev0`, built from
`52e96021c8017938b133ddba895795a13f707568`. Neither torch nor vLLM was installed.

The same clean-wheel lifecycle is now a Python 3.10/3.12 CI matrix pinned to
Manager `98903e416bdb593186b8245fd95180dafde995b9`; this newer packaging check
does not change the pending serving and observer gates below.

`scripts/check_wheel.py` passed static discovery, inspect/validate/check/plan/render,
disabled import, enable-intent/environment composition, unknown-host launch
refusal, disable, explicit saved-configuration rollback and forget. Uninstall
then removed both the importable runtime package and the Bundle registration.
Compatibility correctly remained unverified in the absence of a host.

The check uses an isolated temporary Manager config. It does not alter a user's
existing enabled extensions. The wheel was tested away from its source directory.

## Pending gates and interpretation

- Current CUDA host candidate: `3e67bc6b5734f98600bb171039b375f2c4f2d2ea`.
- Current Ascend source candidate: `83a519c8e915938724369c6d6232d73a425a1034`.
- Clean-host registration/model import, GPU/NPU model loading, real request
  generation, PPL, throughput/memory and native-path restart recovery are pending.
- `scripts/serving_probe.py` is provided for the target hosts but was not run.
- A host-owned SliceGPT model-execution observer is not implemented here.
  `runtime_effective` must remain absent until that evidence exists.

The inspected local vLLM checkout had compiled-version metadata
`0.20.1.post1.dev315+g39fef6206.dirty` despite a newer source branch, and had no
installed vLLM distribution metadata in its development environment. It cannot
qualify the pinned 0.29.1 source candidate. No old `.so` files were copied into a
new host and no existing research environment or workload was changed.

Historical Qwen2.5-14B v12cuda logs contain 2994.97 to 3105.11 total tok/s
(+3.68%) for the matched single-GPU serial comparison. Its historical full
WikiText-2 PPL was recorded as 5.693846 to 8.479382 (+48.92%). Those are prior
experiments, not results of this plugin wheel or the corrected toolkit.
