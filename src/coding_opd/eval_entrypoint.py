"""External coding-benchmark evaluation through the same veRL/Uni-Agent rollout path."""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from copy import copy
from pathlib import Path
from uuid import uuid4

import numpy as np
import ray
from datasets import load_dataset
from omegaconf import OmegaConf

import verl

try:
    import transfer_queue as tq
except ImportError:
    from verl.utils.transferqueue_utils import tq

from coding_opd.opd_framework import OPDAgentFrameworkRolloutAdapter
from coding_opd.rollout_config import configure_coding_rollout
from uni_agent.tasks import TaskConfigResolver
from verl.utils import tensordict_utils as tu
from verl.workers.rollout.llm_server import LLMServerManager

from coding_opd.dataset_selection import id_sha256
from coding_opd.eval_data import sha256_file

# Registration side effects for the only authoritative external benchmarks.
from coding_opd.deepswe_task import DeepSWETask as DeepSWETask  # noqa: F401
from coding_opd.swe_bench_task import SWEBenchTask as SWEBenchTask  # noqa: F401

logger = logging.getLogger(__name__)
PARTITION_ID = "val"
TASK_RUNNERS = {
    "deep_swe": "coding_opd.deepswe_task.run_deepswe_task",
    "swe_bench": "coding_opd.swe_bench_task.run_swe_bench_task",
}


def _task_id(sample: dict) -> str:
    try:
        return str(sample["extra_info"]["task_id"])
    except (KeyError, TypeError) as error:
        raise ValueError("every evaluation row must contain extra_info.task_id") from error


def validate_runtime_dataset(
    data_path: Path, samples: list[dict], *, manifest_path: Path, split: str
) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    try:
        expected = manifest["splits"][split]
    except (KeyError, TypeError) as error:
        raise ValueError(f"runtime manifest has no {split!r} split") from error
    task_ids = [_task_id(sample) for sample in samples]
    if len(task_ids) != int(expected["count"]):
        raise ValueError(f"expected {expected['count']} {split} rows, found {len(task_ids)}")
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("evaluation parquet has duplicate task IDs")
    if id_sha256(task_ids) != expected["task_id_sha256"]:
        raise ValueError("evaluation parquet task order/hash does not match runtime manifest")
    if sha256_file(data_path) != expected["sha256"]:
        raise ValueError("evaluation parquet bytes do not match runtime manifest")
    return manifest


def task_runner_fqn(task_configs: list[dict]) -> str:
    """Select a runner that imports the project's task and sandbox registrations."""
    task_names = {str(entry.get("name") or "") for entry in task_configs}
    if len(task_names) != 1:
        raise ValueError(f"evaluation config must define exactly one task name, got {sorted(task_names)}")
    task_name = task_names.pop()
    try:
        return TASK_RUNNERS[task_name]
    except KeyError as error:
        raise ValueError(f"unsupported external-evaluation task name: {task_name!r}") from error


def init_config(args: argparse.Namespace, *, task_configs: list[dict], served_model_name: str):
    from hydra import compose, initialize_config_dir

    config_dir = str(Path(verl.__file__).resolve().parent / "trainer" / "config")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        config = compose(config_name="ppo_trainer")

    model_cfgs = [entry.get("agent", {}).get("model", {}) for entry in task_configs]
    temperature = float(model_cfgs[0].get("temperature", 0.0))
    top_p = float(model_cfgs[0].get("top_p", 1.0))
    response_length = max(
        (int(entry.get("max_total_tokens", 65536)) for entry in model_cfgs),
        default=65536,
    )
    rollout = config.actor_rollout_ref.rollout
    rollout.temperature = temperature
    rollout.top_p = top_p
    rollout.val_kwargs.temperature = temperature
    rollout.val_kwargs.top_p = top_p
    rollout.top_k = int(model_cfgs[0].get("top_k", -1))
    rollout.val_kwargs.top_k = rollout.top_k
    rollout.n = args.n
    rollout.val_kwargs.n = args.n
    rollout.nnodes = 1
    rollout.n_gpus_per_node = args.n_gpus
    config.trainer.nnodes = 1
    config.trainer.n_gpus_per_node = args.n_gpus
    config.actor_rollout_ref.model.path = str(args.model_path)
    rollout.name = "vllm"
    rollout.mode = "async"
    rollout.load_format = "auto"
    rollout.prompt_length = 4096
    rollout.response_length = response_length
    # The task budget already includes prompt, observations and generated text.
    rollout.max_model_len = response_length
    rollout.tensor_model_parallel_size = args.tensor_parallel_size
    rollout.gpu_memory_utilization = args.gpu_memory_utilization
    rollout.max_num_seqs = args.concurrency
    rollout.max_num_batched_tokens = 8192
    rollout.calculate_log_probs = False
    rollout.disable_log_stats = False
    rollout.free_cache_engine = False
    OmegaConf.update(config, "actor_rollout_ref.rollout.enable_sleep_mode", False, force_add=True)
    OmegaConf.update(config, "actor_rollout_ref.rollout.multi_turn.format", args.tool_parser, force_add=True)
    OmegaConf.update(
        config,
        "actor_rollout_ref.rollout.engine_kwargs.vllm.attention_backend",
        "FLASH_ATTN",
        force_add=True,
    )
    OmegaConf.update(
        config,
        "actor_rollout_ref.rollout.engine_kwargs.vllm.gdn_prefill_backend",
        "triton",
        force_add=True,
    )
    OmegaConf.update(
        config,
        "actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_mode",
        "FULL_DECODE_ONLY",
        force_add=True,
    )

    framework = {
        "gateway_count": args.gateway_count,
        "log_dir": str(args.log_dir),
        "agent_runners": {
            "task": {
                "runner_fqn": task_runner_fqn(task_configs),
                "dispatch_mode": "ray_task",
                "max_concurrent_sessions": args.concurrency,
                "session_timeout_seconds": max(
                    int(entry.get("agent_timeout", 900)) + int(entry.get("eval_timeout", 1800)) + 600
                    for entry in task_configs
                ),
                "runner_kwargs": {
                    "task_config_path": str(args.task_config),
                    "model_name": served_model_name,
                    "report_reward": True,
                    "verification_root": str(args.log_dir.parent / "verification"),
                },
            }
        },
    }
    OmegaConf.update(config, "actor_rollout_ref.rollout.custom.agent_framework", framework, force_add=True)
    OmegaConf.update(config, "transfer_queue.enable", True, force_add=True)
    config.data.return_raw_chat = True
    config.data.max_prompt_length = 4096
    config.data.max_response_length = response_length
    return config


def _build_prompts(samples: list[dict], uids: list[str]):
    return tu.get_tensordict(
        tensor_dict={
            "raw_prompt": [sample["prompt"] for sample in samples],
            "uid": uids,
            "tools_kwargs": [sample["extra_info"]["tools_kwargs"] for sample in samples],
        },
        non_tensor_dict={"global_steps": None, "validate": True},
    )


def _read_scores(uids: list[str]) -> tuple[dict[str, list[float]], dict[str, str]]:
    uid_set = set(uids)
    partition = (tq.kv_list(PARTITION_ID) or {}).get(PARTITION_ID, {}) or {}
    final: dict[tuple[str, str], tuple[int, str]] = {}
    statuses: dict[str, str] = {}
    for key, tag in partition.items():
        tag = tag or {}
        if key in uid_set:
            statuses[key] = str(tag.get("status") or "unknown")
            continue
        parts = key.rsplit("_", 2)
        if len(parts) != 3 or parts[0] not in uid_set or tag.get("status") != "success":
            continue
        try:
            index = int(parts[2])
        except ValueError:
            continue
        session_key = (parts[0], parts[1])
        if session_key not in final or final[session_key][0] < index:
            final[session_key] = (index, key)

    items = sorted(final.items())
    keys = [value[1] for _, value in items]
    per_uid: dict[str, list[float]] = defaultdict(list)
    if keys:
        data = tq.kv_batch_get(keys=keys, partition_id=PARTITION_ID, select_fields=["rm_scores"])
        scores = [float(value) for value in data["rm_scores"].sum(dim=-1).tolist()]
        for ((uid, _), _), score in zip(items, scores, strict=True):
            per_uid[uid].append(score)
    return dict(per_uid), statuses


def _write_result(
    args: argparse.Namespace,
    *,
    samples: list[dict],
    uids: list[str],
    scores_by_uid: dict[str, list[float]],
    statuses: dict[str, str],
    wall_seconds: float,
    served_model_name: str,
    runtime_manifest: dict,
) -> None:
    records = []
    all_scores = []
    for sample, uid in zip(samples, uids, strict=True):
        scores = scores_by_uid.get(uid, [])[: args.n]
        padded = scores + [0.0] * (args.n - len(scores))
        all_scores.extend(padded)
        records.append(
            {
                "task_id": _task_id(sample),
                "score": float(np.mean(padded)),
                "scored_sessions": len(scores),
                "session_scores": scores,
                "status": statuses.get(uid, "missing"),
            }
        )
    payload = {
        "benchmark": runtime_manifest["benchmark"],
        "data_path": str(args.data_path),
        "model_path": str(args.model_path),
        "runtime_manifest": str(args.runtime_manifest),
        "runtime_manifest_sha256": sha256_file(args.runtime_manifest),
        "served_model_name": served_model_name,
        "task_config": str(args.task_config),
        "task_config_sha256": sha256_file(args.task_config),
        "protocol": "shared-react-v1",
        "split": args.split,
        "num_tasks": len(samples),
        "n": args.n,
        "expected_sessions": len(samples) * args.n,
        "scored_sessions": sum(record["scored_sessions"] for record in records),
        "completed_tasks": sum(record["scored_sessions"] >= args.n for record in records),
        "mean_score": float(np.mean(all_scores)) if all_scores else 0.0,
        "wall_seconds": wall_seconds,
        "tasks": records,
    }
    args.result_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.result_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.result_path)
    print(json.dumps({key: payload[key] for key in ("mean_score", "num_tasks", "scored_sessions", "wall_seconds")}, sort_keys=True))


def evaluation_stages(samples, quick_ids):
    """One pass over each task: the frozen quick panel first, then its complement."""
    if not quick_ids:
        return [samples]
    by_id = {_task_id(sample): sample for sample in samples}
    quick_set = set(quick_ids)
    if not quick_set <= by_id.keys():
        raise ValueError("quick panel must be contained in the full runtime")
    return [[by_id[task_id] for task_id in quick_ids],
            [sample for sample in samples if _task_id(sample) not in quick_set]]


def _evaluate(args, config, samples, runtime_manifest, start_rank=0):
    uids = [str(uuid4()) for _ in samples]
    uid_by_id = {_task_id(sample): uid for sample, uid in zip(samples, uids, strict=True)}
    scores, statuses = {}, {}
    if args.result_path.exists():
        previous = json.loads(args.result_path.read_text())
        expected = {"protocol": "shared-react-v1", "model_path": str(args.model_path), "n": args.n,
                    "task_config_sha256": sha256_file(args.task_config),
                    "runtime_manifest_sha256": sha256_file(args.runtime_manifest)}
        if any(previous.get(key) != value for key, value in expected.items()):
            raise ValueError("evaluation resume settings differ from the saved result")
        for record in previous["tasks"]:
            if record["scored_sessions"] >= args.n:
                uid = uid_by_id[record["task_id"]]
                scores[uid], statuses[uid] = record["session_scores"], record["status"]
    manager = LLMServerManager.create(config=config, start_rank=start_rank)
    adapter = OPDAgentFrameworkRolloutAdapter.create(config=config, llm_client=manager.get_client())
    begin = time.perf_counter()
    quick_ids = runtime_manifest["splits"]["quick"]["task_ids"] if args.quick_first else []
    stages = evaluation_stages(samples, quick_ids)

    def save(selected=samples, output_args=args):
        _write_result(output_args, samples=selected, uids=[uid_by_id[_task_id(s)] for s in selected],
                      scores_by_uid=scores, statuses=statuses, wall_seconds=time.perf_counter() - begin,
                      served_model_name=args.served_model_name, runtime_manifest=runtime_manifest)

    save()
    with ThreadPoolExecutor(max_workers=1) as executor:
        for index, stage in enumerate(stages):
            pending = [s for s in stage if len(scores.get(uid_by_id[_task_id(s)], [])) < args.n]
            if pending:
                stage_uids = [uid_by_id[_task_id(s)] for s in pending]
                future = executor.submit(adapter.generate_sequences_and_wait, _build_prompts(pending, stage_uids))
                count = -1
                while True:
                    done = future.done()
                    new_scores, new_statuses = _read_scores(stage_uids)
                    scores.update(new_scores)
                    statuses.update(new_statuses)
                    current = sum(len(v) for v in scores.values())
                    if current != count or done:
                        save()
                        count = current
                    if done:
                        future.result()
                        break
                    time.sleep(5)
            if args.quick_first and index == 0:
                quick_args = copy(args)
                quick_args.split = "quick"
                quick_args.result_path = args.result_path.with_name("quick_result.json")
                save(stage, quick_args)
                print(f"QUICK_COMPLETE model={args.served_model_name}; continuing remaining tasks", flush=True)
    save()
    print(f"EVALUATION_COMPLETE model={args.served_model_name}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--split", choices=("quick", "full"), required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--compare-model-path", type=Path)
    parser.add_argument("--quick-first", action="store_true")
    parser.add_argument("--vllm-cache-root", type=Path, required=True)
    parser.add_argument("--task-config", type=Path, required=True)
    parser.add_argument("--result-path", type=Path, required=True)
    parser.add_argument("--served-model-name")
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--n-gpus", type=int, default=4)
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--gateway-count", type=int, default=2)
    parser.add_argument("--tool-parser", default="qwen3_coder")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--log-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.quick_first and args.split != "full":
        parser.error("--quick-first requires --split full")
    if args.n < 1 or args.n_gpus < 1 or args.tensor_parallel_size < 1:
        parser.error("--n, --n-gpus and --tensor-parallel-size must be positive")
    if args.n_gpus % args.tensor_parallel_size:
        parser.error("--n-gpus must be divisible by --tensor-parallel-size")

    dataset = load_dataset("parquet", data_files=str(args.data_path), split="train")
    samples = [dict(row) for row in dataset]
    runtime_manifest = validate_runtime_dataset(
        args.data_path, samples, manifest_path=args.runtime_manifest, split=args.split
    )
    served_model_name = args.served_model_name or args.model_path.name
    resolver = TaskConfigResolver.from_file(str(args.task_config))
    task_names = {sample["extra_info"]["tools_kwargs"]["task"]["name"] for sample in samples}
    task_configs = [resolver.defaults_by_name[name] for name in sorted(task_names)]

    args.vllm_cache_root.mkdir(parents=True, exist_ok=True)
    ray.init(runtime_env={"env_vars": {"VLLM_CACHE_ROOT": str(args.vllm_cache_root)}})
    args.served_model_name = served_model_name
    jobs = [args]
    if args.compare_model_path:
        other = copy(args)
        args.result_path = args.result_path.parent / "student" / "result.json"
        args.log_dir = args.log_dir / "student" / "agents"
        other.model_path = args.compare_model_path
        other.served_model_name = other.model_path.name
        other.result_path = other.result_path.parent / "teacher" / "result.json"
        other.log_dir = other.log_dir / "teacher" / "agents"
        jobs.append(other)
    configs = []
    for job in jobs:
        config = init_config(job, task_configs=task_configs, served_model_name=job.served_model_name)
        configure_coding_rollout(config, job.task_config, task_configs[0]["name"])
        configs.append(config)
    tq.init(configs[0].transfer_queue)
    # One driver owns TransferQueue for both models until both evaluations finish.
    with ThreadPoolExecutor(max_workers=len(jobs)) as executor:
        futures = [executor.submit(_evaluate, job, config, samples, runtime_manifest,
                                   index * args.n_gpus // args.tensor_parallel_size)
                   for index, (job, config) in enumerate(zip(jobs, configs, strict=True))]
        for future in futures:
            future.result()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    main()
