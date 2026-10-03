# Provenance

Source archives:

- [intellistream/vllm-hust-legacy-20260831](https://github.com/intellistream/vllm-hust-legacy-20260831)
- [intellistream/vllm-ascend-hust-legacy-20260831](https://github.com/intellistream/vllm-ascend-hust-legacy-20260831)

Primary history:

- [Core/runtime PR #158](https://github.com/intellistream/vllm-hust-legacy-20260831/pull/158)
- [Ascend PR #150](https://github.com/intellistream/vllm-ascend-hust-legacy-20260831/pull/150)

The migration must preserve the original model conversion and runtime authors,
separate offline dependencies from serving dependencies, and record the
artifact format used by each extracted commit.

## Extraction implemented in 0.1.0a1

Original runtime/toolkit contribution owner: **@qingfengyuhuoda**. Upstream
SliceGPT is Microsoft [TransformerCompression](https://github.com/microsoft/TransformerCompression),
reviewed revision `6b12cdee6ad51791d7c776b3a046bc408b9e77e9`, under MIT.
Transformer-derived portions retain their original notices. The vLLM runtime
files retain Apache-2.0 and vLLM contributor attribution. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), [LICENSE](LICENSE), and
[LICENSE-MIT](LICENSE-MIT); model and dataset licenses are separate.

The source of truth for extraction is the final head of merged legacy PR #158,
`19b61cb8b95c9e9c2feb3dfab49cb5bc8f2623ee`. Preserved history includes:

- `792caf8934`: Llama runtime integration;
- `14c0ee25f5`: Qwen2 runtime;
- `e4782d7084`: offline toolkit;
- `c2d8a1e44a`: strict conversion contracts;
- `b6fdd3837f`: subsequent toolkit/CI corrections;
- `19b61cb8b9`: registry-resolution regression coverage.

Ascend's historical compatibility strategy and tests are traced to merged PR
#150, final head `793a46d6ebb019c26586749086e3e4c3ffb7fb77`, including
`a07ff805ca`, `b4c0f9e051`, `df58594f9f`, and `5fa5231741`.
The old platform/RoPE patches are not copied into the new host. The extracted
runtime uses model-local native RoPE on the pinned Ascend source candidate.

Legacy code wrote unversioned `.pt` plus slicing JSON, then unversioned HF
safetensors directories. This package exports `slicegpt-artifact/1` and provides
an explicit legacy migration command; it does not silently reinterpret unknown
formats. It adds atomic conversion, runtime validation, opt-in registration,
ECPA metadata, and separate environments. CPU numerical regression tests also
identified and fixed unconditional RMSNorm embedding centering and the Qwen2
decoder return contract for the pinned Transformers 4.41 toolkit environment.
Existing large-model quality/performance results therefore are historical
evidence only, not measurements of this revised toolkit.

The original research workspace and model/log archives were inspected but not
modified or copied wholesale. Model weights, private paths and experiment data
are not included. This extraction and its tests were prepared with Codex AI
assistance; maintainer review and hardware acceptance are required for release.
