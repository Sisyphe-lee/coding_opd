# Evaluation Result

长期维护的 Coding OPD 测评成绩表。成功率后括号为成功题数；未测填 `—`，不推算 full 成绩。

| 模型 / Checkpoint | Verified-500 | Verified-64 | DeepSWE-113 | Tura20 | 记录日期 / 版本 |
|---|---:|---:|---:|---:|---|
| Teacher · Qwen3.8-27B | 72.80%（364/500） | 73.44%（47/64） | 42.48%（48/113） | 45.00%（9/20） | 2026-09-07 · Baseline v1 |
| Student · Qwen3.5-9B · OPD 前 | 52.00%（260/500） | 53.13%（34/64） | 0.00%（0/113） | 0.00%（0/20） | 2026-09-07 · Baseline v1 |

口径：每题一次测量，固定分母，失败与执行异常不剔除。Subset 使用冻结的
[Verified-64](../configs/dataset_manifests/swebench_verified_64.json) /
[Tura20](../configs/dataset_manifests/deepswe_tura20.json)；上述 baseline subset 从对应 full 结果抽取。
数据与抽样定义见 [datasets.md](datasets.md)。

Baseline v1 使用本项目 Codex + vLLM harness，并非厂商官方成绩；上下文 262,144，
Student / Teacher 温度分别为 0.6 / 1.0。保留原始异常计分：Verified Teacher / Student
分别有 2 / 1 个顶层执行错误，另有 Codex 非零退出（含 Student 连接失败）；未完成异常修正。
这些成绩不等于无基础设施异常的能力测量。

新增 checkpoint 用“训练 run / step / 有效轨迹数”标识，固定保留四列成绩。
协议变更或异常补测另记版本，不覆盖旧结果。

## 结果来源

原始文件：`<runtime_root>/eval_results/<run>/result.json`（不随 Git 发布）。

| 结果版本 | Verified run | DeepSWE run |
|---|---|---|
| Teacher · Baseline v1 | `verified_teacher_full_6gpu_c4_resume_20260906` | `deepswe_teacher_full_4gpu_c2_20260906` |
| Student · Baseline v1 | `verified_student_full_8gpu_c4_threads2_20260907` | `deepswe_student_full_4gpu_c8_20260906` |
