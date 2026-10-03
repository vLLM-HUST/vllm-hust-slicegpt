# vLLM-HUST SliceGPT

SliceGPT 离线工具链与按需启用的 vLLM 模型插件，包含 Llama 和 Qwen2/Qwen2.5
实现。作者、上游代码及 legacy PR/commit 见 [PROVENANCE.md](PROVENANCE.md)。

**状态：实验性迁移版本，尚未完成新宿主 CUDA/NPU serving 验收。**
两个 wheel 可独立安装；Runtime 带 ECPA 0.3 静态 Bundle，安装后默认不注册模型。
本版本不宣称 `runtime_effective`、质量或性能已通过新宿主验证。
已运行检查及限制见 [验证记录](docs/validation.md)。

## 包与兼容边界

| 包 | 内容 | 环境 |
| --- | --- | --- |
| `vllm-hust-slicegpt-toolkit` | 校准、旋转、裁剪、转换 CLI | Transformers 4.41.0；Torch 2.6–2.11 |
| `vllm-hust-slicegpt` | 产物校验、模型实现、插件入口、ECPA Manifest | 宿主单独安装；不引入校准依赖 |

Toolkit 依赖轻量 Runtime wheel 以共用产物格式，但不安装 vLLM。
Toolkit 和 serving 使用独立环境，避免 Transformers 版本冲突。
静态发现和包导入均不加载模型或初始化设备。

代码审查候选宿主固定为 vLLM-HUST `3e67bc6b5734f98600bb171039b375f2c4f2d2ea`，
对应 `0.29.1.post1.dev298+g3e67bc6b5` 干净构建。
Manifest 固定 public version；入口另检查 commit 后缀，拒绝未知、dirty、
官方同名版本以及旧宿主内置 SliceGPT 的重复注册。版本匹配不是硬件实测承诺。
当前运行模式限制为 **TP=1、PP=1、eager、未量化权重**。

## 构建和安装

在仓库根目录执行：

```bash
uv venv .venv-build --python 3.11
uv pip install --python .venv-build/bin/python build
.venv-build/bin/python -m build packages/runtime --outdir dist
.venv-build/bin/python -m build packages/toolkit --outdir dist

uv venv .venv-toolkit --python 3.11
uv pip install --python .venv-toolkit/bin/python dist/vllm_hust_slicegpt-0.1.0a1-py3-none-any.whl dist/vllm_hust_slicegpt_toolkit-0.1.0a1-py3-none-any.whl
```

Serving 环境先按照宿主构建说明安装上述固定 commit。需要匹配的 Torch、CUDA
或 CANN 编译产物；不要复制旧环境的 `.so` 到新源码。完成宿主安装后执行：

```bash
uv pip install --python .venv-serving/bin/python dist/vllm_hust_slicegpt-0.1.0a1-py3-none-any.whl
uv pip install --python .venv-serving/bin/python 'vllm-hust-ext @ git+https://github.com/vLLM-HUST/extension-manager.git@52e96021c8017938b133ddba895795a13f707568'
```

这里 `.venv-serving` 表示已完成宿主安装的环境。
无 vLLM 的环境也能安装和检查 Bundle；兼容状态应为 unverified/degraded，不能启动推理。

## 从原始模型构造产物

先将原始模型及 tokenizer 的固定 revision 下载到本地。以下变量由实验者填写，
revision 不要使用浮动 `main`：

```bash
export BASE_MODEL_DIR=/path/to/pinned/Qwen2.5-model
export BASE_MODEL_ID=Qwen/Qwen2.5-0.5B
export BASE_REVISION=REPLACE_WITH_IMMUTABLE_MODEL_REVISION
export MODEL_LICENSE=REPLACE_WITH_MODEL_LICENSE_ID_OR_URL
export DATASET_REVISION=REPLACE_WITH_IMMUTABLE_WIKITEXT_REVISION
export DATASET_LICENSE=REPLACE_WITH_DATASET_LICENSE_ID_OR_URL

.venv-toolkit/bin/slicegpt-compress \
  --model "$BASE_MODEL_ID" --model-path "$BASE_MODEL_DIR" \
  --dtype fp32 --device cuda:0 --sparsity 0.25 --round-interval 8 \
  --cal-dataset wikitext2 --cal-dataset-revision "$DATASET_REVISION" \
  --cal-dataset-license "$DATASET_LICENSE" \
  --cal-nsamples 128 --cal-batch-size 1 --cal-max-seqlen 2048 \
  --seed 42 --final-orientation pca --save-dir artifacts/sliced \
  --skip-final-eval --no-wandb

.venv-toolkit/bin/slicegpt-convert-qwen2 \
  --sliced-dir artifacts/sliced --pt-name Qwen2.5-0.5B_0.25.pt \
  --base-model-path "$BASE_MODEL_DIR" --out-dir artifacts/qwen2-v1 \
  --base-model-id "$BASE_MODEL_ID" --base-revision "$BASE_REVISION" \
  --model-license "$MODEL_LICENSE" \
  --calibration-record artifacts/sliced/calibration.json

.venv-toolkit/bin/slicegpt-artifact validate artifacts/qwen2-v1
```

Llama 使用相同构造流程和 `slicegpt-convert-llama`。中间文件名为
`<model 名称>_<sparsity>.pt`；转换使用 `weights_only=True`。
正式产物为 safetensors，不执行模型目录的 Python 代码。输出目录必须不存在，
通过验证后才原子发布。校准记录保存 dataset revision/license/fingerprint 和采样参数。
未提供 `--calibration-record` 时明确记录 `not-recorded`，不补造实验记录。

## 迁移已有 v12cuda / legacy checkpoint

历史 `model_type=slicegpt_qwen2` 和 `model_type=qwen2` 的 SliceGPT 产物通过显式迁移统一：

```bash
.venv-toolkit/bin/slicegpt-artifact migrate /path/to/v12cuda artifacts/v12-v1 \
  --original-hidden-size 5120 \
  --base-model-id Qwen/Qwen2.5-14B-Instruct \
  --base-revision "$BASE_REVISION" --model-license "$MODEL_LICENSE" \
  --calibration-record /path/to/verified-calibration.json
```

迁移保留原目录，复制权重到新目录，需预留磁盘空间；不会重新执行 PCA。
仅接受已知 SliceGPT 架构，不能将 dense 模型改名冒充。见 [产物格式](docs/artifact.md)。

## 启用、运行、停用

在 serving 环境内执行；`MODEL_DIR` 是已验证的本地 v1 产物：

```bash
vllm-hust-ext extension list
vllm-hust-ext extension inspect org.vllm-hust.slicegpt
vllm-hust-ext extension check org.vllm-hust.slicegpt
vllm-hust-ext extension plan org.vllm-hust.slicegpt
vllm-hust-ext extension render org.vllm-hust.slicegpt
slicegpt-artifact validate "$MODEL_DIR"
vllm-hust-ext extension enable org.vllm-hust.slicegpt
vllm-hust-ext run -- vllm serve "$MODEL_DIR" --enforce-eager --tensor-parallel-size 1 --pipeline-parallel-size 1
```

`enable` 保存下次启动意图，不等于当前兼容或模型已经执行。
入口仅在 Manager 的 `VLLMHUST_EXT_ENABLED_BUNDLES` 包含本 Bundle 时注册。
Manager 合并白名单并保留 Ascend；本包不覆盖 `VLLM_PLUGINS`。无需 `--trust-remote-code`。

```bash
vllm-hust-ext extension disable org.vllm-hust.slicegpt
# 停止旧 serving 进程，再启动普通 dense 模型验证原生路径。
vllm-hust-ext extension forget org.vllm-hust.slicegpt
uv pip uninstall --python .venv-serving/bin/python vllm-hust-slicegpt
```

停用和回滚需要重启，不能热卸载已导入的模型。禁用后压缩模型应拒绝加载；
原生路径恢复指普通 dense 模型恢复原有行为。Toolkit 环境卸载两个 SliceGPT 包。
卸载不删除用户模型目录。回滚步骤见 [验收与恢复](docs/acceptance.md)。

## Ascend 与运行证据

Ascend 候选 `83a519c8e915938724369c6d6232d73a425a1034` 已更换旧 RoPE 路径。
本包在 SliceGPT 自有 RoPE 对象上选择 `forward_native`，不修改宿主 platform、
全局 RoPE 类或架构判断。NPU 入口要求 Ascend distribution version 包含该候选
commit 的干净 `g<sha>` 后缀；不可追溯的版本拒绝运行。仍须真实 NPU 验证。

插件不自行发送 `runtime_effective`。当前 general-plugin loader 的
`discovered/resolved/invoked` 只证明加载生命周期。完整验收需要宿主在实际执行
进程中观察 SliceGPT 处理请求，并绑定 plan/launch/process 身份；此观测点尚未交付。

## 测试

```bash
uv pip install --python .venv-toolkit/bin/python pytest
CUDA_VISIBLE_DEVICES='' .venv-toolkit/bin/python -m pytest -q
```

测试包含真实 tiny Llama/Qwen2 的 CPU 校准与旋转，检查零裁剪 logits 一致性、
切片后前向计算，以及真实 safetensors 转换。合成模型结果仅验证功能，不能作为
真实模型精度、吞吐或显存结论。CI 另验干净 wheel 生命周期。
