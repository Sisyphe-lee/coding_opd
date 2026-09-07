# Coding OPD 性能优化与运行配置

本文记录截至 2026-09-03 的 Coding OPD 基础设施性能结果。主要吞吐数字来自同一台 8×B300
机器上的 veRL `separate_async` smoke，不包含评测时间。SWE-Smith 是当时用于优化训练系统的
固定性能 fixture，并非当前研究训练集；这一定位不影响下列系统吞吐和优化前后对比指标。

## 结论

- 相比最早可完整运行的八卡异步 baseline，稳态吞吐从 **66.68** 提升到
  **711.96 token/s/GPU**，约为原来的 **10.7 倍**。
- 单个 optimizer step 从 **484.86 秒** 降到 **88.42 秒**，快 **5.5 倍**；同时
  每步处理的 token 从 258,630 增加到 503,590，接近原来的两倍。
- Actor 的单位 token 更新时间从 **1.109 ms** 降到 **0.137 ms**，快 **8.1 倍**。
- 最终 Actor MFU 为 **9.25%**。早期日志因机器被错误识别为 `L20D` 而虚高；按
  B300 dense BF16 2,250 TFLOPS/GPU 统一重算，初始 Actor MFU 约为 **1.15%**。
- 最近一次完整 GPU 采样的整机平均 utilization 约为 **41.8%**。它与 Actor MFU
  不是同一指标：前者包含 rollout、teacher 和同步，后者只估算 Actor 有效模型 FLOPs。

## 当时的基准运行入口

`scripts/run_swesmith_opd_async_smoke.sh` 是复现这组测量的基准入口。它默认使用全部八卡、
batch 32、minibatch 16、`parameter_sync_step=2`、最大 off-policy threshold 2、24K Actor
token budget 和 64 条 trajectory。64 条会产生两个 optimizer step，第二步才用于稳态比较。
`run_swesmith_opd_debug.sh` 的四卡同步配置只用于低资源故障诊断。

## 对比口径

初始 baseline 是
`swesmith_opd_async_fla_gc32k_20260902_121714`：4 张 Actor GPU、2 张 rollout
GPU、2 张 Teacher GPU，batch 16，32K token packing，开启 gradient checkpointing，
权重同步 bucket 为 2GB。

最终配置是
`swesmith_opd_fast8_norshard24k_sync2_bucket8g_20260902_192300`：相同的
4+2+2 八卡拓扑，batch 32，24K token packing，关闭 gradient checkpointing，
每两个 16 条 minibatch 同步一次权重，允许最多两步 off-policy threshold，权重同步
bucket 为 8GB。

| 指标 | 初始 baseline | 最终配置 | 改善 |
|---|---:|---:|---:|
| 每步 trajectories | 16 | 32 | 2.0× |
| 每步总 token | 258,630 | 503,590 | 1.95× |
| Step 时间 | 484.86 s | 88.42 s | 5.5× faster |
| 吞吐 | 66.68 token/s/GPU | 711.96 token/s/GPU | 10.7× |
| Actor update | 286.75 s | 69.17 s | 4.1× faster |
| Actor update / token | 1.109 ms | 0.137 ms | 8.1× faster |
| Actor MFU（统一 B300 口径） | ≈1.15% | 9.25% | ≈8.1× |

不同 run 的 trajectory 长度并不完全相同，因此 step 时间只能作为端到端参考；
`token/s/GPU` 和 Actor update/token 是更公平的吞吐指标。

### 未采用的 Actor fast-path 组合

`swesmith_opd_actorfast_nosync_20260902_205000` 同时开启了 Liger、fused AdamW、
FSDP2 forward prefetch 和安全 deferred gradient sync。组合实验没有做单项 ablation，
结果明显退化，因此这些开关保留为 opt-in、默认关闭：

| 指标 | 最终配置 | fast-path 组合 | 变化 |
|---|---:|---:|---:|
| 稳态 Step | 88.42 s | 126.22 s | 慢 42.8% |
| Actor update/token | 0.137 ms | 0.196 ms | 慢 42.8% |
| 吞吐 | 711.96 token/s/GPU | 476.39 token/s/GPU | 下降 33.1% |
| Actor MFU | 9.25% | 8.99% | 下降 0.26 个百分点 |
| Actor 峰值显存 | 216.02 GiB | 189.09 GiB | 降低 12.5% |

该结果只能判定整个组合不应进入默认配置，不能据此归因到其中某一个开关。

## 已采用的优化

### 训练和流水线

- 使用全部 8 张 GPU：4×Actor/FSDP2、2×standalone vLLM rollout、2×Teacher replica。
- 使用 veRL `separate_async`，rollout 与 Actor update 重叠；`parameter_sync_step=2`，
  实测第二步 trajectory staleness 平均 0.91、最大 1，trajectory span 最大 2。
- batch 从 16 提升到 32，并使用 32 个 agent worker/concurrent session。
- 不计算未使用的 rollout old logprob、entropy、policy-gradient loss 和 task-reward loss。

### Actor

- FlashAttention + Qwen3.5 GDN Triton kernel、fused Triton kernels 和 `torch.compile`。
- FSDP2 remove-padding、dynamic token packing 和 1,024-token padding bucket。
- 关闭 gradient checkpointing。B300 显存足够保留激活，避免重算。
- `reshard_after_forward=false`，避免每个 packed microbatch 重复 FSDP all-gather；
  在相同缓存 trajectory 上，Actor update 快 **7.7%**，额外占用约 15.2GB 显存。
- token budget 固定为 24K。相同 trajectory 的 32K 实验在 MLP forward 达到约
  264.4GiB 后 OOM，因此没有继续尝试更大的 budget。

### vLLM、Teacher 和权重同步

- vLLM 使用 decode CUDA Graph、prefix caching、chunked prefill 和编译缓存。
- hybrid rollout memory utilization 为 0.8，standalone rollout 为 0.7，Teacher 为 0.5。
- Teacher 使用两个 TP=1 replica，FlashAttention，49,152 max batched tokens。
- NCCL multi-sender 权重同步 bucket 从 4GB 提升到 8GB。在其他配置不变时：
  - 两步平均同步时间从 **28.37 秒** 降到 **19.64 秒**，降低 **30.8%**；
  - 稳态同步从 **30.07 秒** 降到 **18.75 秒**，降低 **37.6%**；
  - 稳态 step 从 99.18 秒降到 88.42 秒。由于两次 rollout token 数不同，不能把
    全部 step/吞吐差异都归因于 bucket，但同步时间的收益是明确的。

### R2E parameter-sync allocator 修复

R2E-128 四步诊断发现，veRL NCCL checkpoint engine 在每次同步后都会对 Actor worker
执行全局 `torch.cuda.empty_cache()`。该操作逐卡回收约 160–200GiB FSDP/compile allocator
缓存，导致 `timing_s/update_weights` 在 19.83–47.72 秒间抖动；实际 NCCL payload 发送只需
数秒。Coding OPD 现在通过 checkpoint engine 的 `custom_backend_module` 扩展点，在持久
NCCL group 下复用 Actor sender 的两块 transport buffer，并保留 rollout consumer 原有的
释放/清缓存行为，以便恢复 vLLM KV cache。

相同 8GiB bucket、4+2+2 拓扑和 4-step R2E smoke 的对比如下。轨迹 token 分布仍有差异，
因此只把 parameter-sync 指标作为直接 A/B；端到端 step 时间仅供参考。

| 指标 | 修复前 | 修复后 | 改善 |
|---|---:|---:|---:|
| 四步 parameter sync | 19.83 / 35.71 / 47.72 / 20.92 s | 4.79 / 3.15 / 12.11 / 3.52 s | — |
| Parameter sync 平均 | 31.04 s | 5.89 s | 降低 81.0% |
| Parameter sync 中位数 | 28.31 s | 4.15 s | 降低 85.3% |
| Parameter sync 最大值 | 47.72 s | 12.11 s | 降低 74.6% |

修复前 run 为 `r2e128_syncdiag_8g_4step_20260903_174000`，修复后 run 为
`r2e128_syncfix_8g_4step_20260903_184500`。修复后 Actor 最大 allocated/reserved 分别为
206.94/211.52GiB；`nvidia-smi` 观察到的峰值约 246GiB，仍低于每卡约 275GiB 的容量。
四步训练和最终退出均正常，没有 OOM。Actor 显存不再在每次同步时降到 32–48GiB；
剩余 3–12 秒波动来自 rollout consumer buffer 回收、KV cache 恢复及同步 barrier。

## SWE-Smith fixture 迭代效率估算

当前 debug 数据的稳态是约 32 条 trajectory/88.4 秒，单次进程初始化约 9–10 分钟。
下面是基于当前 token 分布的外推，不包含 evaluation、checkpoint 写盘和首次下载镜像：

| Rollout budget | 纯计算下界（含 10 分钟初始化） | 建议规划值 |
|---:|---:|---:|
| 128 | ≈16 分钟 | 20–30 分钟 |
| 512 | ≈34 分钟 | 45–75 分钟 |
| 2,048 | ≈1.7 小时 | 2.5–4 小时 |

相同假设下，初始 baseline 跑完这些 budget 约需 1.25、4.5 和 17.4 小时。因此：

- 很短的 pilot 受初始化支配，端到端约快 **4–5 倍**；
- 512 条以上的开发/主训练，初始化被摊薄，预计快 **8–10 倍**；
- 为项目排期建议保守按 **6–8 倍总体迭代效率提升**，不要直接把 10.7× 稳态
  token 吞吐等同于所有任务上的 wall-clock 提升。

这些数字只外推同类 SWE-Smith fixture 的系统吞吐。R2E-Gym 的 repository、镜像大小和
轨迹长度更加异质，首次 image unpack、长尾 agent episode、容器故障和评测都可能改变
实际速度；R2E-128/R2E-512 物化后需要重新做 pilot 校准，但无需否定本文的优化对比指标。

## 当前机器和软件配置

| 项目 | 配置 |
|---|---|
| 本地 checkout | 自行选择的项目目录；公开快照使用 `main` |
| 远端 checkout | `/personal/coding_opd` |
| 远端 runtime | `/personal/coding_opd_runtime` |
| GPU | 8×NVIDIA B300，约 275GiB/GPU，GPU 间 NV18 |
| CPU / RAM | 256 CPU，约 1TiB RAM |
| CUDA / 架构 | CUDA 12.8，compute capability `sm_103a` |
| Python / PyTorch | Python 3.12，PyTorch 2.11.0+cu130 |
| 训练栈 | veRL 0.10.0.dev0，vLLM 0.24.0，Ray 2.55.1 |
| Student / Teacher | Qwen3.5-9B / Qwen3.8-27B，均只存放在远端 |
| Sandbox | nested rootful Docker，task container 使用 `--network none` |

`nvidia-smi`、CUDA 和 Ray 可能把 B300 错误显示为 `NVIDIA L20D`。内核和显存配置必须
按实际 B300 选择；MFU 使用 `CODING_OPD_DEVICE_PEAK_TFLOPS=2250` 校准。GPU 6/7
可以与低显存、间歇负载的 PhysicalAgent 共存，但正式训练仍使用全部 8 张 GPU。

本地 `.venv` 是轻量 CPU 开发环境，用于编辑、单元测试、数据准备和配置检查，不保存
Student/Teacher checkpoint，也不执行正式训练。完整 CUDA/FSDP/vLLM 环境只在远端。
