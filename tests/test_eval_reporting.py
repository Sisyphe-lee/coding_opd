from __future__ import annotations

import pytest

from coding_opd.dataset_selection import id_sha256
from coding_opd.eval_reporting import build_external_eval_summary


def _manifest() -> dict:
    full_ids = ["task-a", "task-b", "task-c"]
    quick_ids = ["task-c", "task-a"]
    return {
        "benchmark": "example",
        "splits": {
            "full": {
                "count": len(full_ids),
                "task_ids": full_ids,
                "task_id_sha256": id_sha256(full_ids),
            },
            "quick": {
                "count": len(quick_ids),
                "task_ids": quick_ids,
                "task_id_sha256": id_sha256(quick_ids),
            },
        },
    }


def _result() -> dict:
    return {
        "split": "full",
        "num_tasks": 3,
        "n": 1,
        "mean_score": 1 / 3,
        "wall_seconds": 12.5,
        "tasks": [
            {"task_id": "task-a", "score": 1.0, "scored_sessions": 1, "status": "success"},
            {"task_id": "task-b", "score": 0.0, "scored_sessions": 0, "status": "missing"},
            {"task_id": "task-c", "score": 0.0, "scored_sessions": 1, "status": "success"},
        ],
    }


def test_full_result_derives_quick_subset_without_rerun() -> None:
    summary = build_external_eval_summary(_result(), _manifest())
    assert summary["full"]["num_tasks"] == 3
    assert summary["full"]["score_percent"] == pytest.approx(100 / 3)
    assert summary["full"]["missing_sessions"] == 1
    assert summary["quick"]["num_tasks"] == 2
    assert summary["quick"]["score_percent"] == 50.0
    assert [row["task_id"] for row in summary["quick_tasks"]] == ["task-c", "task-a"]


def test_quick_only_result_is_rejected() -> None:
    result = _result()
    result["split"] = "quick"
    with pytest.raises(ValueError, match="requires a full"):
        build_external_eval_summary(result, _manifest())


def test_full_result_order_mismatch_is_rejected() -> None:
    result = _result()
    result["tasks"] = list(reversed(result["tasks"]))
    with pytest.raises(ValueError, match="order/hash"):
        build_external_eval_summary(result, _manifest())
