# Training infrastructure optimization log

只记录正式采用或明确回退的基础设施改动及实测结果。早期实验细节见
`docs/performance_optimization.md`；完整原始日志保留在远端 runtime。

## 当前正式配置

- 8×B300：2 Actor/FSDP2 + 4 rollout/TP1 + 2 Teacher/TP1。
- R2E batch=minibatch=32，每批一次更新；32K context，gradient checkpointing。
- Vanilla/TCOD one-batch prefetch，`parameter_sync_step=1`，实际 policy lag 最大 1；
  Adaptive 使用完整 batch barrier。
- 32 agent workers；shell timeout 120 秒。
- Step 16 首次保存，之后每 64 step 保存。

## 已采用且有测量

| 日期 | 优化 | 实测结果 | Evidence |
|---|---|---|---|
| 2026-09-02 | 八卡异步流水线、batch 32、24K packing、dynamic batching、remove-padding、compile/cache 等组合优化 | SWE-Smith fixture：step 484.86→88.42s；吞吐 66.68→711.96 token/s/GPU（10.7×） | `swesmith_opd_fast8_norshard24k_sync2_bucket8g_20260902_192300` |
| 2026-09-02 | `reshard_after_forward=false` | 相同缓存轨迹 Actor update 快 7.7% | `docs/performance_optimization.md` |
| 2026-09-02 | NCCL multi-sender，bucket 4→8 GiB | 稳态 weight sync 30.07→18.75s（-37.6%） | `docs/performance_optimization.md` |
| 2026-09-03 | 复用 Actor sender transport buffer，避免同步后全局 `empty_cache()` | R2E weight sync 平均 31.04→5.89s（-81.0%） | `r2e128_syncfix_8g_4step_20260903_184500` |
| 2026-09-10 | 32K + gradient checkpointing | 对比 20K/no-GC：单位 token 慢 23.2%；峰值 allocated 216.00→69.96 GiB | `actor_context_ab_20260910` |
| 2026-09-10 | Podman SQLite WAL | 32 containers / 6,400 exec：81.47→20.73s（3.93×） | commit `0ab8e06` |
| 2026-09-10 | conmon exec cleanup 改走 libpod API；`run/exec/cp/rm` 全部复用 Podman service | 32 containers / 6,400 exec：23.99s，零错误；训练 create/rm 恢复到约 0.1–0.9s | commit `74a7216` |
| 2026-09-10 | One-batch prefetch + shell timeout 120s | R2E step 2：338.16→195.98s（-42.0%）；658.79 token/s/GPU；staleness mean/max=1，span max=1 | `r2e512_vanilla_react32k_242_lag1_120s_probe_20260910_1822` |

最后一项同时改变了 prefetch 和 shell timeout，42.0% 是组合收益；单项收益尚未拆分。

## 已采用，未做独立 A/B

- Actor：FlashAttention、Qwen3.5 GDN Triton、fused Triton kernels、`torch.compile`。
- vLLM：decode CUDA graph、chunked prefill、持久 compile cache。
- Teacher、standalone rollout、Actor 初始化并行。
- 关闭 baseline 不使用的 rollout old-logprob、entropy、PG、task reward 和 rollout 后测试。
- Sandbox `OPENBLAS_NUM_THREADS=1`；独立本地 NVMe store；启动前 image preflight。
- Checkpoint-safe prefetch：保存边界停止预取并检查无未消费 prompt。
- 独立八卡 Ray head、五秒 GPU 采样、轻量 JSONL pipeline spans、Git SHA parity preflight。

## 已回退或否决

| 实验 | 结果 |
|---|---|
| Liger + fused AdamW + FSDP forward prefetch + deferred gradient sync | 吞吐下降 33.1% |
| 3 Actor + 3 rollout + 2 Teacher | batch 32 不能被 Actor DP=3 整除 |
| 4 Actor + 2 rollout + 2 Teacher（32K R2E） | 首步 473.78s，未显示收益；停止继续测试 |
| 只开 WAL | cleanup 洪峰后仍出现约 100s 阻塞 |
| conmon cleanup delay 300→5s | 32 个 remove 全部超过 115s |
| 超过 32 个 agent workers | batch 只有 32；slot wait 已低于 1ms |
| Oversampling 后只取最快 32 条 | 会偏向短任务，未采用 |

## 当前剩余瓶颈

- 工具调用长尾：慢 batch 的最后几条曾额外等待 109–243s。
- Rollout GPU 经常欠载：平均 outstanding model requests 约 6–8，峰值才到 30–32。
- Lag 1 能隐藏 Actor update，不能跨过同一 batch 的最后一条长尾。
