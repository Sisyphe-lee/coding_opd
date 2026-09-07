from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


MANIFEST_ROOT = Path(__file__).resolve().parents[1] / "configs" / "dataset_manifests"


def _load(name: str) -> dict:
    return json.loads((MANIFEST_ROOT / f"{name}.json").read_text(encoding="utf-8"))


def test_frozen_training_manifests_are_unique_and_nested() -> None:
    small = _load("r2e_train_128")
    large = _load("r2e_train_512")
    small_ids = {task["task_id"] for task in small["tasks"]}
    large_ids = {task["task_id"] for task in large["tasks"]}

    assert small["count"] == len(small_ids) == 128
    assert large["count"] == len(large_ids) == 512
    assert small_ids <= large_ids
    assert set(small["repo_counts"]) == set(large["repo_counts"])
    assert small["image_count"] == 128
    assert large["image_count"] == 512


def test_frozen_quick_eval_manifests_have_declared_coverage() -> None:
    verified = _load("swebench_verified_50")
    deepswe = _load("deepswe_tura20")

    assert verified["count"] == len({task["task_id"] for task in verified["tasks"]}) == 50
    assert len(verified["repo_counts"]) == 12
    assert min(verified["repo_counts"].values()) >= 1
    assert verified["source"]["full_count"] == 500

    assert deepswe["count"] == len({task["task_id"] for task in deepswe["tasks"]}) == 20
    assert deepswe["language_counts"] == {
        "go": 4,
        "javascript": 4,
        "python": 4,
        "rust": 4,
        "typescript": 4,
    }
    assert deepswe["difficulty_counts"] == {
        "easy": 5,
        "hard": 5,
        "medium-easy": 5,
        "medium-hard": 5,
    }
    assert deepswe["source"]["official_full_count"] == 113


def test_verified64_preserves_legacy50_and_frozen_identity() -> None:
    from coding_opd.dataset_selection import id_sha256

    old, new = _load("swebench_verified_50"), _load("swebench_verified_64")
    ids = [task["task_id"] for task in new["tasks"]]
    assert len(ids) == len(set(ids)) == new["count"] == 64
    assert {t["task_id"] for t in old["tasks"]} < set(ids)
    assert new["source"] == old["source"]
    assert new["seed"] == old["seed"] == 42
    assert new["task_id_sha256"] == id_sha256(ids)
    assert dict(Counter(t["repo"] for t in new["tasks"])) == new["repo_counts"]
    assert len(new["repo_counts"]) == 12
    assert len({t["docker_image"] for t in new["tasks"]}) == new["image_count"] == 64
