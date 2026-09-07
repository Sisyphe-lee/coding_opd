# Coding OPD

公开版为不含原始开发历史的代码快照。机器部署路径仅为示例，私有模型、数据、镜像和
凭据不随仓库提供；请阅读 [`docs/remote_runtime.md`](docs/remote_runtime.md) 并配置自己的环境。
文中的历史性能记录不是对新机器的性能保证。第三方依赖遵循各自许可证，当前未另行
指定本项目的开源许可证。

面向 repository-level agentic coding 的 On-Policy Distillation 实验代码库。主训练环境
是 **R2E-Gym**，使用 Qwen3.8-27B Teacher 蒸馏 Qwen3.5-9B Student，并在单机
8×B300 上运行 veRL `separate_async` 训练流水线。外部评测使用 SWE-bench Verified 和
DeepSWE。

SWE-Smith 只保留为已经验证过的基础设施/吞吐 smoke fixture，不再是研究训练集或评测集。

## 当前已验证基线

| 项目 | 默认设置 |
|---|---|
| Teacher / Student | Qwen3.8-27B / Qwen3.5-9B，BF16 |
| GPU 拓扑 | 4×Actor/FSDP2 + 2×standalone rollout + 2×Teacher replica |
| 训练时序 | veRL V1 `separate_async`，batch 32，minibatch 16，`parameter_sync_step=2` |
| Policy lag | 实测平均 0.91、最大 1；不是严格同步 on-policy |
| OPD loss | `k3` sampled-token reverse-KL，直接反向传播 |
| Teacher 监督 | 每个位置只返回 Student 已采样 token 的 logprob，不传 Top-K/全词表 |
| 研究训练集 | R2E-Gym 128-task pilot / 512-task main pool |
| 外部评测 | SWE-bench Verified 500（默认）/ Verified-64 + DeepSWE 113/Tura20 |
| 已测性能入口 | `scripts/run_swesmith_opd_async_smoke.sh`（只用于基础设施性能复现） |

当前基线不混入 policy-gradient loss 或 executable task reward。Top-K、full-vocabulary、
PG 和 reward mixing 只作为后续消融。

稳态性能来自相同八卡机器上的第二个 optimizer step：

- 503,590 tokens / 88.42 秒；
- 711.96 token/s/GPU；
- Actor update 69.17 秒，0.137 ms/token；
- Actor MFU 9.25%，权重同步 18.75 秒。

完整优化记录见 [`docs/performance_optimization.md`](docs/performance_optimization.md)，远端路径、
模型、GPU、镜像、训练和评测入口的交接表见
[`docs/remote_runtime.md`](docs/remote_runtime.md)。

## 默认优化配置

八卡 launcher 已默认启用目前有明确正收益的设置：

- 全部 8 张 GPU，32 个 agent worker/concurrent session；
- FlashAttention、Qwen3.5 GDN Triton、fused Triton kernels、`torch.compile`；
- remove-padding、dynamic token batching、1,024-token padding bucket；
- Actor token budget 24K，关闭 gradient checkpointing；
- `reshard_after_forward=false`；
- rollout/standalone rollout/Teacher 显存比例为 0.8/0.7/0.5；
- decode CUDA Graph 和持久编译缓存；
- NCCL multi-sender 权重同步，8 GiB bucket；
- 不计算未使用的 rollout old-logprob、entropy、PG loss 和 task-reward loss。

Liger、fused AdamW、FSDP2 forward prefetch 和 deferred gradient sync 默认关闭。四项一起
开启时吞吐下降 33.1%，所以只保留为显式实验开关。

## Checkout 与环境

```bash
git clone git@github.com:Sisyphe-lee/coding_opd.git
cd coding_opd
git switch main
git submodule update --init
```

不要递归初始化 `third_party/uni-agent/verl`；本项目统一使用顶层固定版本
`third_party/verl`。

本地 CPU 开发环境：

```bash
bash scripts/bootstrap_python_env.sh dev
.venv/bin/pytest -q
```

B300 完整训练环境与嵌套 Docker：

```bash
bash scripts/bootstrap_python_env.sh train
bash scripts/bootstrap_docker_daemon.sh
```

本地不保存模型 checkpoint，也不执行正式训练。模型、数据集、Docker archive、日志和
checkpoint 都不进入 Git。机器路径、镜像下载和安全约束见 [`AGENTS.md`](AGENTS.md)。

## 当前数据

训练和评测 task IDs 已冻结在 `configs/dataset_manifests/`：

- R2E-Gym training：R2E-128 pilot 和严格包含它的 R2E-512 main pool；
- SWE-bench Verified evaluation：官方 500-task full set（“SWE-bench”的默认含义）和固定
  Verified-64 quick panel（包含历史 Verified-50）；
- DeepSWE evaluation：公开的 Tura20 quick panel 和官方 113-task full set。

完整的来源 revision、抽样方法、repository/language 配额、manifest 哈希、获取方式和使用边界
见 [`docs/datasets.md`](docs/datasets.md)。Verified 和 DeepSWE 永不进入梯度。R2E task
pool 大小与 rollout budget 是两个概念，不能为了缩短一次运行而改变冻结 IDs。

当前四个 manifest 已完成，远端 R2E-128 parquet 与镜像已物化；R2E-512 和新版外部评测
runtime bundle 仍待物化。旧的 SWE-Smith debug/eval bundle 与 R2E-2,048 split 均不属于
当前研究数据协议。

## R2E-128 checkpoint / resume 短训

`run_r2e128_checkpoint_resume.sh` 默认先训练到 global step 1 并保存 FSDP model、optimizer、
LR/RNG 和 dataloader 状态，再由新进程显式恢复到 step 2。恢复阶段默认额外导出完整 HF
权重，后续评测直接读取 `actor/huggingface/`：

```bash
RAY_ADDRESS=127.0.0.1:<ray-port> \
RUN_NAME=<unique-name> \
bash scripts/run_r2e128_checkpoint_resume.sh
```

当前固定的 TransferQueue 0.1.9 没有队列 checkpoint API，因此这个受控短训将
`ASYNC_WARMUP_BATCHES=0`，保证 dataloader 不会越过尚未更新的预取样本。它验证可恢复训练
状态，不作为吞吐基准。`validate_training_checkpoint.py` 会逐 rank 检查 actor shard、
optimizer、extra state、dataloader marker，以及可选的 HF 权重。

该流程已在 B300 上用冻结 R2E-128 pool 完成首次两进程实测：step 1 checkpoint 经独立校验后，
新进程恢复全部四个 actor rank 的 model、optimizer、RNG、LR scheduler 和 dataloader 状态，
完成 step 2 并导出可供评测使用的 HF 权重。该结果只验证 save/resume 正确性，不代表模型质量。

## 外部评测入口

新版评测 bundle 严格从冻结 manifest 生成，quick/full 分别对应 Verified-64/500 和
DeepSWE-Tura20/113：

```bash
.venv/bin/python scripts/materialize_swebench_verified.py \
  data/manifest_sources/swebench_verified \
  /personal/coding_opd_runtime/datasets/coding_opd_eval_v3/swebench_verified

.venv/bin/python scripts/materialize_deepswe_eval.py \
  <pinned-deep-swe>/tasks <pinned-tasks.json> \
  /personal/coding_opd_runtime/datasets/coding_opd_eval_v2/deepswe
```

DeepSWE 还需从每个 task 的 `tests/` 离线构建独立 verifier image；构建时使用
`--network=none`，agent 容器看不到隐藏 verifier：

```bash
.venv/bin/python scripts/prepare_deepswe_verifier_images.py \
  <pinned-deep-swe>/tasks \
  /personal/coding_opd_runtime/datasets/coding_opd_eval_v2/deepswe/quick.parquet
```

Verified-64 使用独立的新 bundle，不覆盖仍被运行中评测引用的 v2/Verified-50 数据。
当前 Codex 入口不需要 Ray；快速评测必须显式设置 `EVAL_TIER=quick`：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TASKS_PER_REPLICA=8 \
MODEL_PATH=<checkpoint>/actor/huggingface SAMPLING_PROFILE=student \
EVAL_ROOT=/personal/coding_opd_runtime/datasets/coding_opd_eval_v3/swebench_verified \
EVAL_TIER=quick bash scripts/run_swebench_codex_eval.sh
```

历史 ReAct 入口需要独立 Ray head，以下为其 quick 用法（不等同于 Codex 协议）：

```bash
RAY_ADDRESS=127.0.0.1:<ray-port> \
MODEL_PATH=<checkpoint>/actor/huggingface \
EVAL_ROOT=/personal/coding_opd_runtime/datasets/coding_opd_eval_v3/swebench_verified \
EVAL_TIER=quick \
bash scripts/run_external_eval.sh swebench_verified

RAY_ADDRESS=127.0.0.1:<ray-port> \
MODEL_PATH=<checkpoint>/actor/huggingface \
EVAL_TIER=quick \
bash scripts/run_external_eval.sh deepswe
```

评测结果按 task ID 写入 `coding_opd_runtime/eval_results/`；失败或缺失的 rollout 按 0 计入
分母，避免只对成功返回的 session 求均值。

## 运行已验证的八卡性能 smoke

先确认没有接入其他项目的 Ray cluster，然后在远端启动独立八卡 Ray head：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 .venv/bin/ray start --head \
  --node-ip-address=127.0.0.1 \
  --port=<ray-port> \
  --num-cpus=128 \
  --num-gpus=8 \
  --disable-usage-stats
```

正式运行前确保本地和远端 branch、完整 SHA 与 tracked-tree cleanliness 一致。当前已测的
SWE-Smith fixture 运行方式为：

```bash
RAY_ADDRESS=127.0.0.1:<ray-port> \
RUN_NAME=<unique-name> \
bash scripts/run_swesmith_opd_async_smoke.sh
```

脚本默认使用全部八卡、batch 32 和 64 条 trajectory，共两个 optimizer step。第一步包含
JIT/warm-up，第二步用于稳态吞吐判断。调整预算时必须是 32 的正整数倍，例如：

```bash
ROLLOUT_BUDGET=128 bash scripts/run_swesmith_opd_async_smoke.sh
```

这个 smoke 只复现训练系统吞吐，不能用于当前 R2E 数据方案的质量结论。
`scripts/run_swesmith_opd_debug.sh` 是旧的低资源同步诊断入口。

## 代码结构

- `third_party/verl`：OPD、FSDP2、vLLM 和训练后端；
- `third_party/uni-agent`：agent orchestration 与 task/sandbox 接口；
- `third_party/R2E-Gym`：R2E 兼容参考；
- `src/coding_opd`：R2E/SWE-Smith adapter、数据抽样、Ray-safe sandbox、训练入口和性能补丁；
- `scripts`：环境、数据、镜像、训练和 profiling 入口；
- `docs`：环境、评测与性能证据。

当前运行配置以本 README、`AGENTS.md`、`docs/datasets.md` 和实际 launcher 为准。

## 开发规则

- 本地编辑、测试、提交和 push；远端只 fast-forward pull 和执行。
- 正式运行前必须通过 Git parity 检查。
- 不停止、修改或复用其他项目的 Ray/process tree。
- GPU 6/7 可与低显存、间歇运行的 PhysicalAgent 共存，但 Coding OPD 正式训练仍使用全部
  八卡，只为共享负载保留必要显存余量。
