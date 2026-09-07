"""External coding-benchmark evaluation through the same veRL/Uni-Agent rollout path."""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from collections import defaultdict
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

from uni_agent.framework.entry import AgentFrameworkRolloutAdapter
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
    rollout.max_model_len = response_length + 4097
    rollout.tensor_model_parallel_size = args.tensor_parallel_size
    rollout.gpu_memory_utilization = args.gpu_memory_utilization
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
                "runner_kwargs": {
                    "task_config_path": str(args.task_config),
                    "model_name": served_model_name,
                    "report_reward": True,
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
        "split": args.split,
        "num_tasks": len(samples),
        "n": args.n,
        "expected_sessions": len(samples) * args.n,
        "scored_sessions": sum(record["scored_sessions"] for record in records),
        "mean_score": float(np.mean(all_scores)) if all_scores else 0.0,
        "wall_seconds": wall_seconds,
        "tasks": records,
    }
    args.result_path.parent.mkdir(parents=True, exist_ok=True)
    args.result_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("mean_score", "num_tasks", "scored_sessions", "wall_seconds")}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--split", choices=("quick", "full"), required=True)
    parser.add_argument("--model-path", type=Path, required=True)
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
    task_configs = list(resolver.defaults_by_name.values())

    args.vllm_cache_root.mkdir(parents=True, exist_ok=True)
    ray.init(runtime_env={"env_vars": {"VLLM_CACHE_ROOT": str(args.vllm_cache_root)}})
    config = init_config(args, task_configs=task_configs, served_model_name=served_model_name)
    tq.init(config.transfer_queue)
    manager = LLMServerManager.create(config=config)
    adapter = AgentFrameworkRolloutAdapter.create(config=config, llm_client=manager.get_client())
    uids = [str(uuid4()) for _ in samples]
    begin = time.perf_counter()
    adapter.generate_sequences_and_wait(_build_prompts(samples, uids))
    wall = time.perf_counter() - begin
    scores_by_uid, statuses = _read_scores(uids)
    _write_result(
        args,
        samples=samples,
        uids=uids,
        scores_by_uid=scores_by_uid,
        statuses=statuses,
        wall_seconds=wall,
        served_model_name=served_model_name,
        runtime_manifest=runtime_manifest,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    main()
