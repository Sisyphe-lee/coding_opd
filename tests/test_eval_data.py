from __future__ import annotations

import json
import importlib.util
from pathlib import Path

from coding_opd.swesmith_data import build_eval_splits

_SCRIPT = Path(__file__).parents[1] / "scripts" / "prepare_coding_opd_eval.py"
_SPEC = importlib.util.spec_from_file_location("prepare_coding_opd_eval", _SCRIPT)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
_convert_verified_row = _MODULE._convert_verified_row
_select_verified = _MODULE._select_verified
_task_id_from_row = _MODULE._task_id_from_row


def _smith_row(repo: str, index: int) -> dict:
    return {
        "instance_id": f"{repo}.func_basic__{index}",
        "repo": repo,
        "problem_statement": f"Fix {repo} {index}",
        "image_name": f"swebench/{repo}",
        "FAIL_TO_PASS": [f"tests/test_x.py::test_{index}"],
        "PASS_TO_PASS": ["tests/test_x.py::test_existing"],
    }


def _verified_row(repo: str, index: int) -> dict:
    return {
        "instance_id": f"{repo}__repo-{index}",
        "repo": repo,
        "version": "1.0",
        "base_commit": "a" * 40,
        "patch": "diff --git a/x b/x\n",
        "test_patch": "diff --git a/tests/test_x.py b/tests/test_x.py\n",
        "problem_statement": f"Fix {repo} {index}",
        "FAIL_TO_PASS": ["tests/test_x.py::test_fix"],
        "PASS_TO_PASS": ["tests/test_x.py::test_old"],
    }


def test_smith_eval_splits_are_nested_and_exclude_training_ids() -> None:
    rows = [_smith_row(repo, index) for repo in "abcdefgh" for index in range(20)]
    splits = build_eval_splits(rows, smoke_size=8, fast_size=32, full_size=128, repositories=8, seed=42,
                               exclude_ids={"a.func_basic__0"})
    assert len(splits["smoke"]) == 8
    assert len(splits["fast"]) == 32
    assert len(splits["full"]) == 128
    assert [row["instance_id"] for row in splits["smoke"]] == [row["instance_id"] for row in splits["full"][:8]]
    assert [row["instance_id"] for row in splits["fast"]] == [row["instance_id"] for row in splits["full"][:32]]
    assert "a.func_basic__0" not in {row["instance_id"] for row in splits["full"]}


def test_verified_selection_is_nested_and_prompt_hides_patches() -> None:
    rows = [_verified_row(repo, index) for repo in "abcdefghij" for index in range(10)]
    splits = _select_verified(rows, fast_size=8, full_size=50, seed=42)
    assert len(splits["fast"]) == 8
    assert len(splits["full"]) == 50
    assert [row["instance_id"] for row in splits["fast"]] == [row["instance_id"] for row in splits["full"][:8]]
    converted = _convert_verified_row(splits["fast"][0], "fast")
    metadata = converted["extra_info"]["tools_kwargs"]["task"]["metadata"]
    assert converted["prompt"][0]["content"] == splits["fast"][0]["problem_statement"]
    assert metadata["patch"] not in json.dumps(converted["prompt"])
    assert json.loads(metadata["FAIL_TO_PASS"]) == ["tests/test_x.py::test_fix"]


def test_training_exclusion_accepts_converted_opd_rows() -> None:
    assert _task_id_from_row({"instance_id": "raw-1"}) == "raw-1"
    assert _task_id_from_row({"extra_info": {"task_id": "converted-1"}}) == "converted-1"
    assert _task_id_from_row(
        {"extra_info": {"tools_kwargs": {"task": {"metadata": {"instance_id": "nested-1"}}}}}
    ) == "nested-1"
