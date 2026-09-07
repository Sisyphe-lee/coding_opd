"""Validate full external-evaluation results and derive fixed quick panels."""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence

from coding_opd.dataset_selection import id_sha256


def _task_ids(records: Sequence[Mapping[str, Any]], *, label: str) -> list[str]:
    task_ids = [str(record["task_id"]) for record in records]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError(f"{label} has duplicate task IDs")
    return task_ids


def _summarize_records(records: Sequence[Mapping[str, Any]], *, n: int) -> dict[str, Any]:
    if n < 1:
        raise ValueError("evaluation n must be positive")
    task_ids = _task_ids(records, label="evaluation records")
    scores = [float(record["score"]) for record in records]
    if any(score < 0.0 or score > 1.0 for score in scores):
        raise ValueError("evaluation task scores must be in [0, 1]")
    scored_sessions = sum(int(record.get("scored_sessions", 0)) for record in records)
    expected_sessions = len(records) * n
    if scored_sessions < 0 or scored_sessions > expected_sessions:
        raise ValueError("scored session count is outside the expected range")
    score_sum = sum(scores)
    return {
        "num_tasks": len(records),
        "task_id_sha256": id_sha256(task_ids),
        "mean_score": score_sum / len(records) if records else 0.0,
        "score_percent": 100.0 * score_sum / len(records) if records else 0.0,
        "score_sum": score_sum,
        "expected_sessions": expected_sessions,
        "scored_sessions": scored_sessions,
        "missing_sessions": expected_sessions - scored_sessions,
        "status_counts": dict(
            sorted(Counter(str(record.get("status", "missing")) for record in records).items())
        ),
    }


def build_external_eval_summary(
    result: Mapping[str, Any], runtime_manifest: Mapping[str, Any]
) -> dict[str, Any]:
    """Build full and quick metrics from one complete, task-level result."""
    if result.get("split") != "full":
        raise ValueError("subset derivation requires a full evaluation result")
    try:
        full_manifest = runtime_manifest["splits"]["full"]
        quick_manifest = runtime_manifest["splits"]["quick"]
    except (KeyError, TypeError) as error:
        raise ValueError("runtime manifest must contain full and quick splits") from error

    records = list(result.get("tasks", []))
    full_ids = _task_ids(records, label="full result")
    if len(full_ids) != int(full_manifest["count"]):
        raise ValueError(
            f"full result has {len(full_ids)} tasks; expected {full_manifest['count']}"
        )
    if id_sha256(full_ids) != full_manifest["task_id_sha256"]:
        raise ValueError("full result task order/hash does not match runtime manifest")
    if int(result.get("num_tasks", -1)) != len(records):
        raise ValueError("result num_tasks does not match task records")

    quick_ids = [str(task_id) for task_id in quick_manifest.get("task_ids", [])]
    if len(quick_ids) != int(quick_manifest["count"]):
        raise ValueError("quick manifest count does not match its task IDs")
    if id_sha256(quick_ids) != quick_manifest["task_id_sha256"]:
        raise ValueError("quick manifest task order/hash is invalid")
    by_id = {str(record["task_id"]): record for record in records}
    missing_quick = [task_id for task_id in quick_ids if task_id not in by_id]
    if missing_quick:
        raise ValueError(f"full result is missing quick tasks: {missing_quick[:5]}")
    quick_records = [by_id[task_id] for task_id in quick_ids]

    n = int(result.get("n", 0))
    full_summary = _summarize_records(records, n=n)
    quick_summary = _summarize_records(quick_records, n=n)
    if abs(float(result.get("mean_score", -1.0)) - full_summary["mean_score"]) > 1e-12:
        raise ValueError("reported full mean_score does not match task records")

    return {
        "benchmark": runtime_manifest.get("benchmark", result.get("benchmark")),
        "model_path": result.get("model_path"),
        "served_model_name": result.get("served_model_name"),
        "source_result": result.get("result_path"),
        "n": n,
        "wall_seconds": float(result.get("wall_seconds", 0.0)),
        "full": full_summary,
        "quick": quick_summary,
        "quick_tasks": quick_records,
    }
