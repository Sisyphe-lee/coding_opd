# Evaluation Result

Teacher：**Qwen3.8-27B**；Student：**Qwen3.5-9B**。

## Base Model

以下为固定的 OPD 前基线，供后续训练结果反复对比。成功率后括号为成功题数；未测填 `—`。

### Teacher

| 评测设置 | Verified-500 | Verified-64 | DeepSWE-113 | Tura20 |
|---|---:|---:|---:|---:|
| Codex 256K | 72.80%（364/500） | 73.44%（47/64） | 42.48%（48/113） | 45.00%（9/20） |
| ReAct 64K | 48.00%（240/500） | 57.81%（37/64） | — | — |
| ReAct 256K | 69.20%（346/500） | 68.75%（44/64） | — | — |
| Shared training ReAct 64K | — | 50.00%（32/64） | — | — |

### Student

| 评测设置 | Verified-500 | Verified-64 | DeepSWE-113 | Tura20 |
|---|---:|---:|---:|---:|
| Codex 256K | 52.00%（260/500） | 53.13%（34/64） | 0.00%（0/113） | 0.00%（0/20） |
| ReAct 64K | 50.40%（252/500） | 56.25%（36/64） | — | — |
| ReAct 256K | 53.20%（266/500） | 51.56%（33/64） | — | — |
| Shared training ReAct 64K | — | 46.88%（30/64） | — | — |

## OPD

后续训练结果单独记录在此，用“算法 / 训练 run / step / 有效轨迹数”标识 checkpoint，并注明评测设置。

| 算法 / Checkpoint | 评测设置 | Verified-500 | Verified-64 | DeepSWE-113 | Tura20 |
|---|---|---:|---:|---:|---:|
| Vanilla OPD / R2E-512 / step 256 / 8,192 trajectories | Shared training ReAct 64K | — | 31.25%（20/64） | — | — |
| TCOD / R2E-512 / step 256 / 8,192 trajectories | Shared training ReAct 64K | — | 37.50%（24/64） | — | — |

## 评测口径与备注

Base Model 记录日期：Codex 为 2026-09-07（Baseline v1），ReAct 为 2026-09-10。
协议变更或异常补测另记版本，不覆盖固定基线。

口径：每题一次测量，固定分母，失败与执行异常不剔除。Subset 使用冻结的
[Verified-64](../configs/dataset_manifests/swebench_verified_64.json) /
[Tura20](../configs/dataset_manifests/deepswe_tura20.json)；上述 baseline subset 从对应 full 结果抽取。
数据与抽样定义见 [datasets.md](datasets.md)。

Baseline v1 使用本项目 Codex + vLLM harness，并非厂商官方成绩；上下文 262,144，
Student / Teacher 温度分别为 0.6 / 1.0。保留原始异常计分：Verified Teacher / Student
分别有 2 / 1 个顶层执行错误，另有 Codex 非零退出（含 Student 连接失败）；未完成异常修正。
这些成绩不等于无基础设施异常的能力测量。

ReAct 四条均为未经过 OPD 训练的 base model，使用 Uni-Agent ReAct scaffold 和本项目独立
SWE-bench grader；不是 Codex harness，也不是厂商官方成绩。64K / 256K 分别为
65,536 / 262,144 tokens（此前口头所称“264K”实际为 256K），最多 100 个 agent steps，
Student / Teacher 均为 temperature=0.8、top_p=0.9。64K Student 的 agent 超时为
45 分钟；其余最新 resume 配置为 50 分钟，保留此前已完成结果，因此不是完全统一预算的重测。
Teacher 256K 包含异常补测及保存补丁的重新判分，不能把四条差异解释成严格的 context-only ablation。

ReAct 异常计分状态（2026-09-10 核对）：

- 64K Teacher：497/500 有效判分，另 3 题按固定分母暂计失败，尚未完成异常修正。
- 64K Student：494/500 有效判分，另 6 题按固定分母暂计失败，尚未完成异常修正。
- 256K Teacher：500/500 有效判分，成功 346 题。
- 256K Student：原始结果为 498/500 有效判分。剩余 `django__django-14155`、
  `django__django-9296` 已由日志确认是模型补丁破坏测试启动，应算模型失败；
  判分分类代码已修正，但尚未重跑更新原始结果。按失败计入后成功数仍为 266/500。
- 四条 ReAct 的 Verified-64 均已有效判分 64/64，成绩从各自 full 结果按冻结 manifest 抽取。
- `Shared training ReAct 64K` 使用当前训练共用的 `configs/coding_react.yaml`，直接运行
  Verified-64；上下文 65,536 tokens、最多 100 steps、temperature=0.8、top_p=0.9、
  top_k=-1，并启用原生重复停止。四条严格对照均使用 8 GPU、两个 TP=4 serving replica，
  64/64 有效判分，无基础设施失败。

## 结果来源

原始文件：`<runtime_root>/eval_results/<run>/result.json`（不随 Git 发布）。

| 结果版本 | Verified run | DeepSWE run |
|---|---|---|
| Teacher · Baseline v1 | `verified_teacher_full_6gpu_c4_resume_20260906` | `deepswe_teacher_full_4gpu_c2_20260906` |
| Student · Baseline v1 | `verified_student_full_8gpu_c4_threads2_20260907` | `deepswe_student_full_4gpu_c8_20260906` |
| Teacher · ReAct base 64K | `teacher_react64k_c64_50min_resume_20260909_181221` | — |
| Student · ReAct base 64K | `base_student_teacher_upstream_react64k100_20260909_125617/student` | — |
| Teacher · ReAct base 256K | `teacher_react256k_retry4_20260910` | — |
| Student · ReAct base 256K | `student_react256k_c32_resume_20260910_032649` | — |
| Student · Shared training ReAct base 64K | `base_student_coding_react64k_verified64_8gpu_20260911` | — |
| Teacher · Shared training ReAct base 64K | `teacher_coding_react64k_verified64_8gpu_20260911` | — |
| Vanilla OPD · Shared training ReAct 64K | `r2e512_vanilla_step256_swebench_verified64_react64k_20260911` | — |
| TCOD · Shared training ReAct 64K | `r2e512_tcod_step256_swebench_verified64_react64k_20260911` | — |
