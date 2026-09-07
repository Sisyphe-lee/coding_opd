from __future__ import annotations

from datasets import Dataset

from coding_opd.r2e_data import (
    build_r2e_splits,
    convert_row,
    load_r2e_source,
    select_frozen_manifest_rows,
    task_id,
)


def _rows() -> list[dict]:
    rows = []
    for repo, count in (("alpha", 12), ("beta", 8), ("gamma", 4)):
        for index in range(count):
            rows.append(
                {
                    "repo_name": repo,
                    "commit_hash": f"{repo}-{index:02d}",
                    "problem_statement": f"Fix issue {index} in {repo}",
                    "expected_output_json": "{}",
                    "docker_image": f"example/{repo}:{index}",
                }
            )
    return rows


def test_split_is_deterministic_stratified_and_nested() -> None:
    first = build_r2e_splits(
        _rows(), train_size=12, dev_size=6, pilot_size=6, debug_size=4, smoke_size=2, seed=7
    )
    second = build_r2e_splits(
        _rows(), train_size=12, dev_size=6, pilot_size=6, debug_size=4, smoke_size=2, seed=7
    )

    first_ids = {name: [task_id(row) for row in rows] for name, rows in first.items()}
    second_ids = {name: [task_id(row) for row in rows] for name, rows in second.items()}
    assert first_ids == second_ids
    assert set(first_ids["train"]).isdisjoint(first_ids["dev"])
    assert set(first_ids["pilot"]) <= set(first_ids["train"])
    assert set(first_ids["debug"]) <= set(first_ids["pilot"])
    assert set(first_ids["smoke"]) <= set(first_ids["debug"])
    assert {row["repo_name"] for row in first["dev"]} == {"alpha", "beta", "gamma"}


def test_convert_row_records_stable_task_identity() -> None:
    row = _rows()[0]
    converted = convert_row(row, data_split="pilot")
    uid = task_id(row)
    assert converted["extra_info"]["task_id"] == uid
    assert converted["extra_info"]["split"] == "pilot"
    assert converted["extra_info"]["tools_kwargs"]["task"]["metadata"]["task_id"] == uid


def test_duplicate_task_ids_are_rejected() -> None:
    rows = _rows()
    rows.append(dict(rows[0]))
    try:
        build_r2e_splits(rows, train_size=12, dev_size=6, pilot_size=6, smoke_size=2)
    except ValueError as error:
        assert "duplicate R2E task id" in str(error)
    else:
        raise AssertionError("expected duplicate task id validation to fail")


def test_load_local_shard_directory(tmp_path) -> None:
    rows = _rows()[:4]
    Dataset.from_list(rows[:2]).to_parquet(tmp_path / "train-00000.parquet")
    Dataset.from_list(rows[2:]).to_parquet(tmp_path / "train-00001.parquet")
    Dataset.from_list(_rows()[4:5]).to_parquet(tmp_path / "dropped-00000.parquet")

    loaded = load_r2e_source(str(tmp_path), split="train")

    assert len(loaded) == 4
    assert set(loaded["commit_hash"]) == {row["commit_hash"] for row in rows}


def test_frozen_manifest_selection_preserves_order_and_checks_images() -> None:
    rows = _rows()
    wanted = [rows[3], rows[0]]
    manifest = {
        "schema": "coding_opd.dataset_manifest.v1",
        "count": 2,
        "tasks": [
            {
                "task_id": task_id(row),
                "repo_name": row["repo_name"],
                "commit_hash": row["commit_hash"],
                "docker_image": row["docker_image"],
            }
            for row in wanted
        ],
    }

    selected = select_frozen_manifest_rows(rows, manifest)

    assert [task_id(row) for row in selected] == [task_id(row) for row in wanted]

    manifest["tasks"][0]["docker_image"] = "example/wrong:tag"
    try:
        select_frozen_manifest_rows(rows, manifest)
    except ValueError as error:
        assert "docker_image" in str(error)
    else:
        raise AssertionError("expected a manifest/source image mismatch")
