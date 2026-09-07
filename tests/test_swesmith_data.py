from __future__ import annotations

from datasets import Dataset

from coding_opd.swesmith_data import add_repeated_debug_fixture, convert_row, select_debug_tasks, write_debug_bundle


def _row(repo: str, index: int) -> dict:
    return {
        "instance_id": f"{repo}.func_basic__{index}",
        "repo": repo,
        "problem_statement": f"Fix bug {index}",
        "image_name": f"swebench/{repo}",
        "patch": "unused by the agent runtime",
        "FAIL_TO_PASS": [f"tests/test_x.py::test_{index}"],
        "PASS_TO_PASS": ["tests/test_x.py::test_existing"],
    }


def test_select_debug_tasks_balances_repositories() -> None:
    rows = [_row(repo, index) for repo in ("a", "b", "c") for index in range(8)]
    selected = select_debug_tasks(rows, size=8, repositories=2, seed=7, include_repos=["a", "c"])
    assert sum(row["repo"] == "a" for row in selected) == 4
    assert sum(row["repo"] == "c" for row in selected) == 4
    assert selected == select_debug_tasks(rows, size=8, repositories=2, seed=7, include_repos=["a", "c"])


def test_convert_row_does_not_expose_bug_patch() -> None:
    row = _row("example", 1)
    converted = convert_row(row)
    task = converted["extra_info"]["tools_kwargs"]["task"]
    assert converted["data_source"] == "swe_smith"
    assert task["sandbox"]["image"] == "swebench/example"
    assert task["metadata"]["problem_statement"] == row["problem_statement"]
    assert "patch" not in task["metadata"]


def test_debug_bundle_writes_batch_32_repeat_fixture(tmp_path) -> None:
    rows = [_row(repo, index) for repo in ("a", "b", "c", "d") for index in range(4)]
    manifest = write_debug_bundle(rows, tmp_path, source="test", seed=42)

    assert manifest["splits"]["debug"]["count"] == 16
    assert manifest["splits"]["debug_repeat2"] == {
        "count": 32,
        "unique_tasks": 16,
        "repeat_factor": 2,
        "source_split": "debug",
        "path": "debug_repeat2.parquet",
    }
    repeated = Dataset.from_parquet(str(tmp_path / "debug_repeat2.parquet"))
    assert len(repeated) == 32
    task_ids = [extra_info["task_id"] for extra_info in repeated["extra_info"]]
    assert len(set(task_ids)) == 16


def test_repeat_fixture_can_reuse_a_frozen_debug_split(tmp_path) -> None:
    rows = [_row(repo, index) for repo in ("a", "b", "c", "d") for index in range(4)]
    write_debug_bundle(rows, tmp_path, source="test", seed=42, repeat_factor=1)

    manifest = add_repeated_debug_fixture(tmp_path, repeat_factor=3)

    assert manifest["splits"]["debug"]["count"] == 16
    assert manifest["splits"]["debug_repeat3"]["count"] == 48
    assert len(Dataset.from_parquet(str(tmp_path / "debug_repeat3.parquet"))) == 48
