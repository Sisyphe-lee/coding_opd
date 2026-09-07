# Evaluation Result

Coding OPD 实验成绩登记表。Baseline 于 **2026-09-07** 从已完成的逐任务结果核算；
只记录已完成的测评，不把正在训练的 checkpoint 或预计成绩列为结果。

## 结果总表

指标为成功率，单元格同时记录成功题数 / 固定任务数。`—` 表示未测，不代表零分。
每个模型或 checkpoint 固定保留四个成绩栏；只测两个 subset 时，full 栏保留 `—`。

| 模型 / Checkpoint | SWE-bench Verified-500 full | SWE-bench Verified-64 subset | DeepSWE-113 full | DeepSWE-Tura20 subset | 结果版本 |
|---|---:|---:|---:|---:|---|
| Teacher：Qwen3.8-27B（原始权重） | 72.80%（364/500） | 73.44%（47/64） | 42.48%（48/113） | 45.00%（9/20） | Baseline v1，含下述异常 |
| Student：Qwen3.5-9B（未进行 Coding OPD） | 52.00%（260/500） | 53.13%（34/64） | 0.00%（0/113） | 0.00%（0/20） | Baseline v1，含下述异常 |

这里的 baseline 是**本项目 harness 下的测量值**，不是模型厂商发布的官方成绩。
后续 OPD checkpoint 应记录 run 名、global step、有效训练轨迹数、权重路径及评测协议版本。
相同 checkpoint 若更换 agent、上下文压缩策略或其他实质协议，应新增结果版本，不覆盖旧行。

## Subset 与计分口径

- Verified subset 使用冻结的 [swebench_verified_64.json](../configs/dataset_manifests/swebench_verified_64.json)，
  不是历史 Verified-50。Task-ID SHA-256：`7a684848015a2f24522bcd8747ba6e671ec18abdda8977649b46a5243aa750ee`。
- DeepSWE subset 使用冻结的 [deepswe_tura20.json](../configs/dataset_manifests/deepswe_tura20.json)。
  Task-ID SHA-256：`ff015d8b828d5f9cf225c95cdad6f038ea9e2581d97cf592e3e4c6c3cd4baa10`。
- 抽样方法、数据来源及 quick/full 使用政策见 [datasets.md](datasets.md)。不按模型成绩重新选题。
- 上表全部 subset 成绩均为**从对应同一次 full 结果按 task ID 抽取**，不是独立重跑。
- 每题一次测量（`n=1`）；成功数为逐任务二元 `score` 之和。分母始终为 500、64、113、20，
  不以成功执行或成功评分的任务数代替，不删除超时、空补丁或执行失败记录。
- 四组 full 均核验 `complete=true`、任务 ID 唯一、数量正确、score 为 0/1；
  两个模型的 subset 匹配覆盖分别为 64/64 和 20/20。百分比保留两位小数。
- Subset 有采样噪声：Verified-64 一题为 1.5625 个百分点，Tura20 一题为 5 个百分点。
  subset 改善不能直接解释为 full 同幅改善，也不能把两个 benchmark 混成一个平均成功率。

## Baseline v1 协议与限制

两组模型均通过 Codex CLI + 本地 vLLM 运行。Verified 使用官方 SWE-bench grading；
DeepSWE 使用 v1.1 协议。上下文窗口为 262,144，reasoning effort 为 `xhigh`、summary 为 `auto`。
Student 温度 0.6，Teacher 温度 1.0；均为 top-p 0.95、top-k 20、min-p 0，
presence penalty 0、repetition penalty 1，vLLM seed 0，无投机采样。
`reasoning_summary=auto` 不等于已验证的上下文自动压缩策略；本版不包含后续待验证的显式压缩阈值改动。

Verified 结果经过断点续跑，保留已完成任务，不按多次尝试中的最高分选取。
GPU 分配、并发和 CPU 线程限制在续跑期间调整过；Student 最后一次续跑设置为八个 TP1 副本、
总并发 32、数值库线程限制 2。不能将最后一次 run identity 当成所有历史任务完全相同的资源配置，
也不能用最终 result 的 `wall_seconds` 当作整个 500 题的总耗时。
原始结果中的 `resume_provenance` 与逐任务 artifact 保留历史来源。

本版是**保留异常、可追溯的原始 baseline**，不是“全部基础设施异常已修正”的结果：

- Verified Teacher 有 2 个 `status=error`：`scikit-learn__scikit-learn-14710`、
  `sphinx-doc__sphinx-9711`；Student 有 1 个：`sphinx-doc__sphinx-9711`。
  均保留零分，且这两个任务不在 Verified-64 中。Student 的已评分 session 数为 499，
  但成功率分母仍是 500。
- Student Verified 已确认 `sympy__sympy-13878` 出现模型连接失败后空补丁零分，尚未补测。
  此外，Codex 非零退出并不总被上层标成 `status=error`；不可只靠顶层 status 宣称无异常。
- 逐任务 Codex exit code=1 的记录数：Verified Teacher 8、Student 4；DeepSWE Teacher 0、
  Student 6。部分记录没有 exit code；缺失不等于正常退出，非零退出也不自动等同基础设施故障。
  本次登记未重新诊断这些任务，保持原始 score，不推测其修正后成绩。
- 因此 DeepSWE Student 的 0/113 是此协议与本次运行下的结果，不应表述为模型普遍没有 coding 能力。
  后续若复核或补测异常，另列 v2，记录替换任务与原因，保留 v1。

## 原始结果索引

以下 run 目录位于实验环境的 `<runtime_root>/eval_results/`（部署时自行配置路径），
主文件为 `<run>/result.json`。模型、逐任务日志和原始结果不提交进 Git。
下表代码 SHA 是原始实验开发仓库的溯源标识，不保证存在于这个 fresh-history 公共仓库中。

| 标识 | Run | 最终 result 的代码 SHA |
|---|---|---|
| DS-S | `deepswe_student_full_4gpu_c8_20260906` | `58b845cd961905135ad6f1aec3c93cd4af356adc` |
| DS-T | `deepswe_teacher_full_4gpu_c2_20260906` | `58b845cd961905135ad6f1aec3c93cd4af356adc` |
| SV-S | `verified_student_full_8gpu_c4_threads2_20260907` | `5d688e88cf312d2e337f14cd6c82d3fa3f5339b1` |
| SV-T | `verified_teacher_full_6gpu_c4_resume_20260906` | `a643b064017da46703ec775baeacf8dcd2562dab` |

登记时 `result.json` 文件 SHA-256（用于识别后续覆盖或修订）：

```text
DS-S  7e65981d388813200c4101d82b934bc2e84dbdb0f58ca5a88c26768d714c90cc
DS-T  38874c935fb0cf7d6b1300ea562d246b26f06d7b0ed4487713293e84aab8134d
SV-S  b94d14ce8a091ba365f5d50363c00c7d57da6e70cd23970fcdcb82b12a175320
SV-T  5d25d8c43c7acce55a333b3b234f25b8c44d61f7f2e027e11f6a1f0abd17011e
```

## 复算方法

在可访问原始结果的环境中，把 `result_path` 和 `manifest_path` 分别设为上面的结果与冻结
subset manifest 路径。直接按冻结 manifest 抽取，避免旧 runtime manifest 中的 Verified-50
被误当成 Verified-64：

```python
import json
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

def summarize(result_path, manifest_path, full_count):
    result = json.loads(Path(result_path).read_text())
    manifest = json.loads(Path(manifest_path).read_text())
    rows = result["tasks"]
    by_id = {row["task_id"]: row for row in rows}
    ids = [row["task_id"] for row in manifest["tasks"]]
    assert result["complete"]
    assert len(rows) == len(by_id) == full_count
    assert len(ids) == len(set(ids)) == manifest["count"]
    assert set(ids) <= by_id.keys()
    assert all(row["score"] in (0, 1) for row in rows)
    for label, selected in (("full", list(by_id)), ("subset", ids)):
        passed = sum(int(by_id[task_id]["score"]) for task_id in selected)
        percent = (Decimal(100) * passed / len(selected)).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )
        print(label, f"{percent}% ({passed}/{len(selected)})")
```

新增结果时同时检查原始文件 hash、任务覆盖、异常清单及协议是否可比。
仅 subset 测评需核验恰好覆盖冻结 subset；不填充或推算未测 full 的成绩。
