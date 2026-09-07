# Coding OPD 数据协议

本文档是 Coding OPD 训练集与评测集的唯一说明。任务身份以
`configs/dataset_manifests/*.json` 为准；README 只提供摘要，运行时 parquet、Docker
镜像或旧的远端 bundle 都不能覆盖 manifest 中冻结的 task ID。

## 已确定的数据组合

| 用途 | Quick / Pilot | Full | 是否进入梯度 |
|---|---:|---:|---|
| R2E-Gym training | 128 | 512，严格包含 128 | 是 |
| SWE-bench Verified evaluation | 固定 64 | 官方 500 | 否 |
| DeepSWE v1.1 evaluation | Tura20 | 官方 113 | 否 |

R2E-128 用于开发迭代，R2E-512 用于更稳定的主实验。这里的 128/512 是 unique task
pool 大小，不是 rollout 数量：同一个 pool 可以跨 checkpoint、seed 或 epoch 产生多条
trajectory，不能为了缩短运行时间而悄悄改变 task IDs。

Verified-64 和 DeepSWE-Tura20 是快速评测面板，不是新的训练集。重要 checkpoint 才运行
Verified-500 和 DeepSWE-113；最终结论必须报告完整评测，不能用 quick panel 代替。
在本项目中，不加限定的 “SWE-bench” 一律指 SWE-bench Verified 全量 500；64 条面板必须
明确写作 `Verified-64` 或 quick panel。历史 Verified-50 清单和结果保留原名。

## R2E-Gym training

母集是 `PrimeIntellect/R2E-Gym-Subset-Validated`。HF 当前将它解析为
`PrimeIntellect/R2E-Gym-Subset-Verified`，冻结 revision 为：

```text
151f9950e62cac613e07be1bb92e5dd19687315e
```

该 train split 包含 4,522 条通过 gold-patch 端到端校验的任务；原始 4,578 条中连续重试仍
无法正确评分的 56 条已被移除。R2E-Gym subset 本身已经去掉与 SWE-bench 测试仓库重叠
的 repository，因此它适合作为训练集，而 Verified 保持外部评测。

官方没有发布标准的 R2E-128 或 R2E-512。项目采用 seed 42 的固定嵌套抽样：

1. 按 repository 数量的平方根分配 quota，即 `quota ∝ sqrt(N_repo)`，在原始比例和完全
   均匀之间折中。
2. 在每个 repository 内，根据 `num_non_test_files`、
   `num_non_test_func_methods` 和 `num_non_test_lines` 构造四个等频复杂度档。
3. 在每个 `(repository, complexity_bin)` 内按 task ID 的 SHA-256 固定排序。
4. 先冻结 512 档的 quota，再在相同排序前缀中取 128 档，保证
   `R2E-128 ⊂ R2E-512`。

静态 complexity bin 只用于保持 patch 规模覆盖，不代表 Student 的真实求解难度。真实
难度应在 rollout 后用成功率、episode 长度、工具调用数和超时率分析；后续可以改变任务
采样权重，但不能改变这两个冻结 pool 的成员。

| Repository | R2E-128 | R2E-512 |
|---|---:|---:|
| pandas | 25 | 99 |
| numpy | 18 | 73 |
| pillow | 16 | 65 |
| orange3 | 14 | 57 |
| aiohttp | 11 | 44 |
| tornado | 10 | 40 |
| scrapy | 9 | 38 |
| pyramid | 9 | 35 |
| datalad | 9 | 34 |
| coveragepy | 7 | 27 |

每个 R2E task 对应一个唯一 Docker image，因此两个档位分别需要 128 和 512 个镜像。

## SWE-bench Verified evaluation

完整评测使用 `SWE-bench/SWE-bench_Verified` 官方 test split 的 500 条，冻结 revision：

```text
78f471bf655a3137b2e8a75af1501690ec009ec3
```

Verified-64 先保证全部 12 个 repository 至少一条，再按完整 500 条的 repository 数量
比例分配剩余 quota；repository 内使用 seed 42 对 instance ID 做固定哈希排序。它是
Verified-500 的严格子集，并完整包含历史 Verified-50 的 50 题。2026-09-06 将 quick
从 50 扩为 64，沿用原抽样规则及 seed，而非根据正在运行的模型成绩挑题。

冻结源的难度分布（工程师预计修复时间，非模型难度）如下：

| 难度 | Verified-500 | 历史 Verified-50 | Verified-64 |
|---|---:|---:|---:|
| <15 min | 194 | 21 | 26 |
| 15 min–1 hour | 261 | 25 | 34 |
| 1–4 hours | 42 | 4 | 4 |
| >4 hours | 3 | 0 | 0 |

协议接受该近似分布，不强制 >4 hours 覆盖，不增加难度配额或 repository×难度全覆盖
约束。保证小仓库可见会产生不等比例抽样，quick 原始成功率不应当作无偏全量估计。
64 个任务能填满八个 TP1 副本各 8 个任务槽的初始调度，不保证后期没有拖尾。

64 条面板中一个任务等于 1.5625 个百分点，统计噪声仍较大。它适合比较明显回归或大幅改善，不能
支撑小幅性能结论。Verified 的 task、patch、test 和 trajectory 均不得进入训练或 OPD
loss。

## DeepSWE evaluation

完整评测是 DataCurve DeepSWE v1.1 的 113 条任务，覆盖 91 个 repository 和 5 种语言。
这里的 DeepSWE 是 2026 年的 benchmark，不是 Together AI 的 DeepSWE-Preview 模型。

DeepSWE 官方没有命名为 quick 的 subset。项目采用公开且可复现的 Tura20：

- benchmark：DeepSWE v1.1；
- 20 个固定 task；
- Go、Python、TypeScript、Rust、JavaScript 各 4 条；
- easy、medium-easy、medium-hard、hard 各 5 条；
- selection source：`Tura-AI/benchmark@da58a15d35b5284db2abfd233378093a2689dc5c`。

文档和报告必须写 `DeepSWE-Tura20`，不能称为 DataCurve 官方 quick subset。一个 Tura20
任务等于 5 个百分点，只适合低成本回归检查。DeepSWE-113 应保留给重要 checkpoint 和最终
评测，避免反复查看结果后对它做自适应调参；DeepSWE 数据绝不能进入训练。

## 冻结 manifest

| Manifest | Count | Task-ID SHA-256 |
|---|---:|---|
| [`r2e_train_128.json`](../configs/dataset_manifests/r2e_train_128.json) | 128 | `09e0c5632227b87883078bc4d53824920bdec69b53016dcd221798f7f6f91355` |
| [`r2e_train_512.json`](../configs/dataset_manifests/r2e_train_512.json) | 512 | `683c16872947e243b80d8c48010ce5c8e361ae2d75765fd3d10f7708fbab1b36` |
| [`swebench_verified_50.json`](../configs/dataset_manifests/swebench_verified_50.json) | 50 | `f3ce5666276619992387d195d2e954c65fca667f0b9b9b2dd24c625dacd46281` |
| [`swebench_verified_64.json`](../configs/dataset_manifests/swebench_verified_64.json) | 64 | `7a684848015a2f24522bcd8747ba6e671ec18abdda8977649b46a5243aa750ee` |
| [`deepswe_tura20.json`](../configs/dataset_manifests/deepswe_tura20.json) | 20 | `ff015d8b828d5f9cf225c95cdad6f038ea9e2581d97cf592e3e4c6c3cd4baa10` |

Manifest 记录 ordered task IDs、source revision、适用的 repository/language/difficulty
分布和抽样规则。R2E 与 Verified 清单还直接记录镜像名；DeepSWE 镜像从固定上游任务配置
解析。下载环境时应直接消费 `tasks[]`，不要重新抽样，例如：

```bash
jq -r '.tasks[].docker_image' configs/dataset_manifests/r2e_train_128.json
```

## 获取源 metadata 与复现 manifest

源 parquet 只用于复现抽样，不进入 Git。工作站下载 Hugging Face 数据时关闭代理并使用
HF mirror；对于约 930 MB 的 R2E parquet，使用 HFD/aria2 的整文件并发下载，不要使用大量
小 range request。固定 revision 后运行：

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    -u http_proxy -u https_proxy -u all_proxy \
    HF_ENDPOINT=https://hf-mirror.com \
    bash /path/to/hfd.sh PrimeIntellect/R2E-Gym-Subset-Verified \
      --dataset --include 'data/train-*.parquet' \
      --revision 151f9950e62cac613e07be1bb92e5dd19687315e \
      --local-dir data/manifest_sources/r2e_validated_4522 -x 10 -j 8
```

Verified metadata 使用相同方式下载：

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    -u http_proxy -u https_proxy -u all_proxy \
    HF_ENDPOINT=https://hf-mirror.com \
    bash /path/to/hfd.sh SWE-bench/SWE-bench_Verified \
      --dataset --include 'data/test-*.parquet' \
      --revision 78f471bf655a3137b2e8a75af1501690ec009ec3 \
      --local-dir data/manifest_sources/swebench_verified -x 10 -j 8
```

随后运行：

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    -u http_proxy -u https_proxy -u all_proxy \
    .venv/bin/python scripts/freeze_dataset_manifests.py
```

生成脚本会核验 R2E=4,522、Verified=500、DeepSWE=113、Tura20=20，并校验 DeepSWE/Tura
源 JSON 的内容哈希。项目测试还会检查 task 唯一性、128/512 嵌套和 quick eval 覆盖。

## Task ID、镜像与 runtime 数据的对应关系

| 题集 | 冻结身份 | 镜像映射 / 物化入口 |
|---|---|---|
| R2E-128 / R2E-512 | manifest 的 `task_id`、`commit_hash` | `tasks[].docker_image`；`scripts/materialize_r2e_manifest.py` |
| Verified-64（历史 50） | manifest 的 `task_id` | `tasks[].docker_image`；`scripts/materialize_swebench_verified.py` |
| Verified-500 | 固定 HF revision 的全部 500 个 instance ID | `coding_opd.eval_data.verified_image()`；同一 Verified materializer |
| DeepSWE-Tura20 / 113 | subset manifest / 固定官方 `tasks.json` | 上游 `task.toml` 的 `[environment].docker_image`；`scripts/materialize_deepswe_eval.py` |

Verified 镜像规则为 `swebench/sweb.eval.x86_64.<instance_id>`，instance ID 转小写并将
`__` 替换为 `_1776_`。DeepSWE 还需运行 `scripts/prepare_deepswe_verifier_images.py`
构建独立 verifier；agent 环境不能看到隐藏测试。所有 materializer 的参数以 `--help` 为准。

DeepSWE task corpus 固定为 `datacurve-ai/deep-swe@0b9fabbb63b9104d678fe965e1632f2dd9eaa2ea`，
`tasks/` tree SHA-256 为 `207e1309bfbaebdf9186123ea4e74732c9348c395687af9bd48e01193a8bd5d9`。
Materializer 同时核验 tree 和官方 `tasks.json`，避免同名任务对应不同 verifier。

这里区分三类资产：

- **Git 中的冻结 manifest**：定义成员、顺序和数据源，不包含 parquet 或镜像层。
- **部署生成的 runtime bundle**：包含 parquet、运行时 manifest 和完整镜像列表；不得通过
  修改 runtime 数据改变冻结 subset。Verified-64 与历史 Verified-50 应使用不同目录。
- **镜像与 verifier 构建产物**：部署时获取/构建；名称/tag 不是不可变 digest。严格复现时
  另行记录 agent/verifier 镜像 digest、构建输入和 OCI archive 哈希。

源 metadata、runtime parquet、模型、镜像层和结果不进入 Git。下载后检查固定 revision，
物化后运行镜像完整性检查。旧 SWE-Smith fixture 和 R2E-2,048 不属于研究数据协议。

## 评测执行协议

- 两个 benchmark 均使用本项目 Codex harness：模型通过本地 vLLM 执行编码任务，
  不是 mini-SWE-agent，也不宣称与 Claude Code 的 agent 行为完全相同。
- Student 使用 `SAMPLING_PROFILE=student`（temperature 0.6），Teacher 使用
  `SAMPLING_PROFILE=teacher`（temperature 1.0）；两者 top_p 0.95、top_k 20、min_p 0、
  presence_penalty 0、repetition_penalty 1。角色必须显式指定，不能从 checkpoint 目录猜测。
- 上下文上限 262144，reasoning effort `xhigh`、summary `auto`；固定配置以
  `configs/deepswe_codex.yaml`、`configs/swebench_codex.yaml` 和 launcher 为准。
- Agent 预算 10800 秒，verifier 预算 1800 秒；sandbox runtime timeout 11100 秒。
  Agent 正常退出或达到预算后，测评器独立评分，不以模型自述“完成”作为成功。
- Verified 收集相对 base commit 的补丁，不要求 git commit；DeepSWE 使用其任务指令和
  已提交补丁。不要为提高分数追加自动提交或重复循环纠正等未纳入协议的行为。
- 使用 benchmark 的测试规则判定成功；完整报告保留所有选中任务作为分母，执行失败计零。
  未完成运行不能当成最终成功率。`canary` 仅用于接通检查，不是固定 quick panel。
- Task 容器无网络，通过本地 socket 访问模型。示例配置禁用 cgroup；不能据此宣称
  与官方云端 CPU、RAM、磁盘资源限制完全等价，也不作资源等价的耗时比较。
- GPU/TP/任务并发允许按机器调整，数据成员、采样和评分语义保持不变。资源-only resume
  保留全部已完成结果（包括零分），未完成任务重新开始，不筛掉失败结果。

训练仅消费冻结 R2E pool。Verified 和 DeepSWE 不进入梯度、Teacher 监督或奖励塑形；
quick panel 用于预先约定的回归检查，全量用于重要 checkpoint 和最终报告，不根据正在
查看的分数改题、改 verifier 或自适应调参。
