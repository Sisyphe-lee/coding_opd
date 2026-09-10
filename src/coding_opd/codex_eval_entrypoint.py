"""Run restartable DeepSWE evaluation with Codex and sticky vLLM replicas."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import subprocess
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
from datasets import load_dataset
from uni_agent.tasks import TaskConfigResolver, get_task

from coding_opd.codex_agent import (
    CodexCliAgent as CodexCliAgent,  # noqa: F401
    resolve_auto_compact_limit,
    summarize_codex_artifacts,
)
from coding_opd.deepswe_task import DeepSWETask as DeepSWETask  # noqa: F401
from coding_opd.swebench_codex_task import VerifiedCodexTask as VerifiedCodexTask  # noqa: F401
from coding_opd.eval_data import sha256_file
from coding_opd.sandbox_resources import thread_limit_args
from coding_opd.eval_entrypoint import _task_id, validate_runtime_dataset

logger = logging.getLogger(__name__)
PROTOCOL = "DeepSWE v1.1 + Codex CLI + local vLLM Responses API"


def _protocol(args) -> str:
    if getattr(args, "benchmark", "deepswe") == "swebench_verified":
        return "SWE-bench Verified + Codex CLI + official SWE-bench grading"
    return PROTOCOL
CONTAINER_AGENT_LOG_DIR = "/opt/coding-opd/agent-logs"


@dataclass(frozen=True)
class Backend:
    backend_id: str
    socket_path: str


def _safe_name(value: str) -> str:
    readable = "".join(char if char.isalnum() or char in "._-" else "_" for char in value)
    return f"{readable[:80]}-{hashlib.sha256(value.encode()).hexdigest()[:10]}"


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _git_sha() -> str | None:
    repo_root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        capture_output=True,
        check=False,
        text=True,
        timeout=10,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _run_identity(args: argparse.Namespace, runtime_manifest_sha256: str) -> dict[str, Any]:
    model_files = {}
    for name in ("config.json", "generation_config.json", "model.safetensors.index.json"):
        path = args.model_path / name
        if path.is_file():
            model_files[name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    return {
        "protocol": _protocol(args),
        "runtime_manifest_sha256": runtime_manifest_sha256,
        "task_config_sha256": sha256_file(args.task_config),
        "served_model_name": args.served_model_name,
        "model_path": str(args.model_path),
        "model_files": model_files,
        "sampling_config": json.loads(args.sampling_config_json),
        "repo_git_sha": _git_sha(),
        "codex_reasoning_effort": args.reasoning_effort,
        "codex_reasoning_summary": args.reasoning_summary,
        "codex_context_window": args.context_window,
        "codex_auto_compact_token_limit": args.auto_compact_token_limit,
        "backend_count": len(args.model_socket),
        "concurrency": args.concurrency,
        "tasks_per_replica": args.tasks_per_replica,
        "task_cpu_threads": getattr(args, "task_cpu_threads", 2),
        "vllm_seed": args.vllm_seed,
        "serving_config": json.loads(args.serving_config_json),
        "selected_task_ids": args.task_id,
    }


def _fingerprint(identity: dict[str, Any]) -> str:
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _select_tasks(samples: list[dict], task_ids: list[str]) -> list[dict]:
    if not task_ids:
        return samples
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("Duplicate selected task IDs")
    available = {_task_id(sample) for sample in samples}
    if missing := set(task_ids) - available:
        raise ValueError(f"Unknown selected task IDs: {sorted(missing)}")
    return [sample for sample in samples if _task_id(sample) in set(task_ids)]


def _record_path(records_dir: Path, task_id: str) -> Path:
    return records_dir / f"{_safe_name(task_id)}.json"


def _load_records(
    records_dir: Path, *, expected_task_ids: set[str], run_fingerprint: str
) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    if not records_dir.exists():
        return records
    for path in sorted(records_dir.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        task_id = str(record.get("task_id") or "")
        if task_id not in expected_task_ids:
            raise ValueError(f"unexpected task record in {records_dir}: {task_id!r}")
        if record.get("run_fingerprint") != run_fingerprint:
            raise ValueError(f"task record was produced by a different run configuration: {path}")
        if task_id in records:
            raise ValueError(f"duplicate completed record for {task_id}")
        records[task_id] = record
    return records


def _resume_records(args, identity, expected_task_ids):
    """Import completed records without rewriting their original provenance."""
    source = getattr(args, "resume_from_result", None)
    if source is None:
        return {}, None
    if source.resolve() == args.result_path.resolve():
        raise ValueError("resume source must be a separate, stopped run")
    payload = json.loads(source.read_text())
    previous = payload["run_identity"]
    fingerprint = payload["run_fingerprint"]
    if _fingerprint(previous) != fingerprint:
        raise ValueError("resume source identity fingerprint is invalid")
    old, new = deepcopy(previous), deepcopy(identity)
    approved_sha = getattr(args, "resume_source_git_sha", None)
    if old.get("repo_git_sha") != new.get("repo_git_sha"):
        if not approved_sha or approved_sha != old.get("repo_git_sha"):
            raise ValueError("resume source code change requires its explicitly reviewed full SHA")
        old.pop("repo_git_sha")
        new.pop("repo_git_sha")
    reviewed_config = getattr(args, "resume_source_task_config_sha256", None)
    if reviewed_config is not None:
        if reviewed_config != old.get("task_config_sha256"):
            raise ValueError("resume source task config does not match the reviewed hash")
        old.pop("task_config_sha256", None)
        new.pop("task_config_sha256", None)
    for key in ("backend_count", "concurrency", "tasks_per_replica", "task_cpu_threads"):
        old.pop(key, None)
        new.pop(key, None)
    for config in (old, new):
        serving = config.get("serving_config", {})
        for key in ("gpu_groups", "replica_count", "tasks_per_replica",
                    "total_task_concurrency", "max_num_seqs"):
            serving.pop(key, None)
    if old != new:
        raise ValueError("resume may change resource allocation only, not evaluation semantics")
    # Records are authoritative: a task can finish just before result.json is updated.
    records = _load_records(source.parent / "records", expected_task_ids=expected_task_ids,
                            run_fingerprint=fingerprint)
    # Also retain records inherited by an earlier resource-resume phase.
    provenance = payload.get("resume_provenance") or {}
    allowed = {fingerprint, *provenance.get("inherited_fingerprints", [])}
    for record in payload["tasks"]:
        task_id = record["task_id"]
        if task_id not in expected_task_ids or record["run_fingerprint"] not in allowed:
            raise ValueError("invalid inherited task record")
        if task_id in records and records[task_id] != record:
            raise ValueError("resume source record disagrees with result")
        records[task_id] = record
    return records, {
        "source_result": str(source.resolve()),
        "source_run_identity": previous,
        "source_run_fingerprint": fingerprint,
        "inherited_fingerprints": sorted({r["run_fingerprint"] for r in records.values()}),
        "inherited_tasks": len(records),
        "source_wall_seconds": payload.get("wall_seconds", 0),
        "previous_resume_provenance": payload.get("resume_provenance"),
        "reviewed_source_git_sha": approved_sha,
        "reviewed_source_task_config_sha256": reviewed_config,
        "unfinished_tasks_restart_from_scratch": True,
    }


def _result_payload(
    args: argparse.Namespace,
    *,
    samples: list[dict[str, Any]],
    records: dict[str, dict[str, Any]],
    runtime_manifest: dict[str, Any],
    identity: dict[str, Any],
    run_fingerprint: str,
    started_at: float,
) -> dict[str, Any]:
    ordered = [records[_task_id(sample)] for sample in samples if _task_id(sample) in records]
    scores = [float(record["score"]) for record in ordered]
    total = len(samples)
    complete = len(ordered) == total
    return {
        "benchmark": runtime_manifest["benchmark"],
        "protocol": _protocol(args),
        "split": args.split,
        "n": 1,
        "is_canary": bool(runtime_manifest.get("is_canary")),
        "complete": complete,
        "data_path": str(args.data_path),
        "runtime_manifest": str(args.runtime_manifest),
        "runtime_manifest_sha256": identity["runtime_manifest_sha256"],
        "task_config": str(args.task_config),
        "model_path": str(args.model_path),
        "served_model_name": args.served_model_name,
        "run_identity": identity,
        "run_fingerprint": run_fingerprint,
        "backend_count": len(args.model_socket),
        "num_tasks": total,
        "completed_tasks": len(ordered),
        "pending_tasks": total - len(ordered),
        "scored_sessions": sum(record["status"] == "success" for record in ordered),
        "mean_score": float(np.mean(scores)) if complete and scores else (0.0 if complete else None),
        "mean_score_completed": float(np.mean(scores)) if scores else None,
        "mean_score_lower_bound": float(sum(scores) / total) if total else 0.0,
        "wall_seconds": time.time() - started_at,
        "resume_provenance": getattr(args, "resume_provenance", None),
        "tasks": ordered,
    }


async def _run_one(
    sample: dict[str, Any],
    *,
    resolver: TaskConfigResolver,
    model_name: str,
    log_dir: Path,
    records_dir: Path,
    backend_queue: asyncio.Queue[Backend],
    semaphore: asyncio.Semaphore,
    run_fingerprint: str,
) -> dict[str, Any]:
    task_id = _task_id(sample)
    async with semaphore:
        backend = await backend_queue.get()
        try:
            attempt_log_dir = log_dir / _safe_name(task_id) / f"attempt-{uuid4().hex[:10]}"
            attempt_log_dir.mkdir(parents=True, exist_ok=False)
            attempt_log_dir.chmod(0o700)
            begin = time.perf_counter()
            try:
                sample_config = dict(sample["extra_info"]["tools_kwargs"]["task"])
                if sample_config["name"] == "swe_bench":
                    sample_config["name"] = "swe_bench_codex"
                sample_config["prompt"] = sample["prompt"]
                resolved = deepcopy(resolver.resolve(
                    sample_config,
                    runtime_model={
                        "base_url": "http://127.0.0.1:8000/v1",
                        "api_key": "EMPTY",
                        "model_name": model_name,
                    },
                ))
                sandbox_kwargs = resolved.setdefault("sandbox", {}).setdefault(
                    "sandbox_kwargs", {}
                )
                run_args = list(sandbox_kwargs.get("run_args") or [])
                run_args.extend(
                    [
                        "--volume",
                        f"{attempt_log_dir.resolve()}:{CONTAINER_AGENT_LOG_DIR}:rw",
                    ]
                )
                sandbox_kwargs["run_args"] = run_args
                agent_config = resolved.setdefault("agent", {})
                agent_config.update(
                    {
                        "log_dir": str(attempt_log_dir.resolve()),
                        "container_log_dir": CONTAINER_AGENT_LOG_DIR,
                        "model_socket": backend.socket_path,
                    }
                )
                logger.info("Codex DeepSWE start: %s backend=%s", task_id, backend.backend_id)
                result = await get_task(resolved).run()
                status = "success"
                score = float(result.reward)
                finished = result.finished
                extra_info = result.extra_info or {}
            except Exception as error:  # Infrastructure failures are benchmark zeros.
                logger.exception("Codex DeepSWE failed: %s", task_id)
                status = "error"
                score = 0.0
                finished = False
                extra_info = {"error": f"{type(error).__name__}: {error}"}
            wall_seconds = time.perf_counter() - begin
            try:
                artifacts = summarize_codex_artifacts(attempt_log_dir)
            except Exception as error:
                # Auxiliary audit metadata must not discard a completed verifier score
                # or abort every other task. Keep the raw files and explicit failure.
                logger.exception("Artifact summary failed for %s", task_id)
                artifacts = {"summary_error": f"{type(error).__name__}: {error}",
                             "raw_jsonl_path": str(attempt_log_dir / "codex-cli.jsonl")}
            record = {
                "task_id": task_id,
                "score": score,
                "status": status,
                "scored_sessions": int(status == "success"),
                "finished": finished,
                "wall_seconds": wall_seconds,
                "backend_id": backend.backend_id,
                "backend_socket": backend.socket_path,
                "run_fingerprint": run_fingerprint,
                "attempt_log_dir": str(attempt_log_dir),
                "agent_artifacts": artifacts,
                "extra_info": extra_info,
            }
            _atomic_write_json(_record_path(records_dir, task_id), record)
            logger.info(
                "Codex DeepSWE done: %s backend=%s status=%s score=%s wall=%.1fs",
                task_id,
                backend.backend_id,
                status,
                score,
                wall_seconds,
            )
            return record
        finally:
            backend_queue.put_nowait(backend)


async def _run(
    args: argparse.Namespace,
    samples: list[dict[str, Any]],
    *,
    records: dict[str, dict[str, Any]],
    runtime_manifest: dict[str, Any],
    identity: dict[str, Any],
    run_fingerprint: str,
    started_at: float,
) -> dict[str, dict[str, Any]]:
    resolver = deepcopy(TaskConfigResolver.from_file(str(args.task_config)))
    for defaults in resolver.defaults_by_name.values():
        sandbox_kwargs = defaults.setdefault("sandbox", {}).setdefault("sandbox_kwargs", {})
        sandbox_kwargs["run_args"] = list(sandbox_kwargs.get("run_args") or []) + thread_limit_args(
            getattr(args, "task_cpu_threads", 2)
        )
        defaults["agent"].update(
            context_window=args.context_window,
            auto_compact_token_limit=args.auto_compact_token_limit,
            reasoning_effort=args.reasoning_effort,
            reasoning_summary=args.reasoning_summary,
        )
    semaphore = asyncio.Semaphore(args.concurrency)
    backend_queue: asyncio.Queue[Backend] = asyncio.Queue()
    for _ in range(args.tasks_per_replica):
        for index, socket_path in enumerate(args.model_socket):
            backend_queue.put_nowait(Backend(backend_id=f"replica-{index}", socket_path=socket_path))

    pending = [sample for sample in samples if _task_id(sample) not in records]
    tasks = [
        asyncio.create_task(
            _run_one(
                sample,
                resolver=resolver,
                model_name=args.served_model_name,
                log_dir=args.log_dir,
                records_dir=args.records_dir,
                backend_queue=backend_queue,
                semaphore=semaphore,
                run_fingerprint=run_fingerprint,
            )
        )
        for sample in pending
    ]
    for completed in asyncio.as_completed(tasks):
        record = await completed
        records[record["task_id"]] = record
        _atomic_write_json(
            args.result_path,
            _result_payload(
                args,
                samples=samples,
                records=records,
                runtime_manifest=runtime_manifest,
                identity=identity,
                run_fingerprint=run_fingerprint,
                started_at=started_at,
            ),
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--split", choices=("quick", "full"), required=True)
    parser.add_argument("--task-config", type=Path, required=True)
    parser.add_argument("--result-path", type=Path, required=True)
    parser.add_argument("--records-dir", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--served-model-name", required=True)
    parser.add_argument("--model-socket", action="append", required=True)
    parser.add_argument("--sampling-config-json", required=True)
    parser.add_argument("--reasoning-effort", default="xhigh")
    parser.add_argument("--reasoning-summary", default="auto")
    parser.add_argument("--context-window", type=int, default=262_144)
    parser.add_argument("--task-cpu-threads", type=int, default=2)
    parser.add_argument("--auto-compact-token-limit", type=int,
                        help="Defaults to 90%% of context window; lower overrides are allowed")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--tasks-per-replica", type=int, default=2)
    parser.add_argument("--vllm-seed", type=int, default=0)
    parser.add_argument("--serving-config-json", default="{}")
    parser.add_argument("--resume-from-result", type=Path,
                        help="Stopped source run; retain all completed scores, including zeros")
    parser.add_argument("--resume-source-task-config-sha256",
                        help="Reviewed source config hash; retain completed tasks under their original budget")
    parser.add_argument("--resume-source-git-sha",
                        help="Explicitly reviewed source full SHA when only orchestration code changed")
    parser.add_argument("--benchmark", choices=("deepswe", "swebench_verified"), default="deepswe")
    parser.add_argument("--task-id", action="append", default=[],
                        help="Retry selected tasks from a non-reportable canary only")
    args = parser.parse_args()
    if args.task_cpu_threads < 1:
        parser.error("--task-cpu-threads must be positive")
    try:
        args.auto_compact_token_limit = resolve_auto_compact_limit(
            args.context_window, args.auto_compact_token_limit)
    except ValueError as error:
        parser.error(str(error))
    if args.concurrency < 1:
        parser.error("--concurrency must be positive")
    if args.tasks_per_replica < 1:
        parser.error("--tasks-per-replica must be positive")
    if args.concurrency > len(args.model_socket) * args.tasks_per_replica:
        parser.error("--concurrency exceeds available replica task slots")
    try:
        sampling = json.loads(args.sampling_config_json)
    except json.JSONDecodeError as error:
        parser.error(f"--sampling-config-json is invalid JSON: {error}")
    if not isinstance(sampling, dict):
        parser.error("--sampling-config-json must be a JSON object")

    dataset = load_dataset("parquet", data_files=str(args.data_path), split="train")
    samples = [dict(row) for row in dataset]
    runtime_manifest = validate_runtime_dataset(
        args.data_path,
        samples,
        manifest_path=args.runtime_manifest,
        split=args.split,
    )
    if args.task_id and not runtime_manifest.get("is_canary"):
        parser.error("--task-id requires a non-reportable canary manifest")
    samples = _select_tasks(samples, args.task_id)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    args.records_dir.mkdir(parents=True, exist_ok=True)
    identity = _run_identity(args, sha256_file(args.runtime_manifest))
    run_fingerprint = _fingerprint(identity)
    records = _load_records(
        args.records_dir,
        expected_task_ids={_task_id(sample) for sample in samples},
        run_fingerprint=run_fingerprint,
    )
    inherited, args.resume_provenance = _resume_records(
        args, identity, {_task_id(sample) for sample in samples})
    if records.keys() & inherited.keys():
        raise ValueError("completed inherited tasks were rerun in this phase")
    records = {**inherited, **records}
    started_at = time.time()
    _atomic_write_json(
        args.result_path,
        _result_payload(
            args,
            samples=samples,
            records=records,
            runtime_manifest=runtime_manifest,
            identity=identity,
            run_fingerprint=run_fingerprint,
            started_at=started_at,
        ),
    )
    records = asyncio.run(
        _run(
            args,
            samples,
            records=records,
            runtime_manifest=runtime_manifest,
            identity=identity,
            run_fingerprint=run_fingerprint,
            started_at=started_at,
        )
    )
    payload = _result_payload(
        args,
        samples=samples,
        records=records,
        runtime_manifest=runtime_manifest,
        identity=identity,
        run_fingerprint=run_fingerprint,
        started_at=started_at,
    )
    _atomic_write_json(args.result_path, payload)
    print(
        json.dumps(
            {
                key: payload[key]
                for key in (
                    "complete",
                    "mean_score",
                    "num_tasks",
                    "completed_tasks",
                    "scored_sessions",
                    "wall_seconds",
                )
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    main()
