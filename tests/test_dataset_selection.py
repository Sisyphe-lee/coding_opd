from __future__ import annotations

from coding_opd.dataset_selection import (
    build_nested_r2e_subsets,
    largest_remainder_quotas,
    select_verified_quick,
)
from coding_opd.r2e_data import task_id


def _r2e_rows() -> list[dict]:
    rows = []
    for repo, count in (("large", 80), ("medium", 48), ("small", 32)):
        for index in range(count):
            rows.append(
                {
                    "repo_name": repo,
                    "commit_hash": f"{repo}-{index:03d}",
                    "problem_statement": f"Fix {repo} issue {index}",
                    "expected_output_json": "{}",
                    "docker_image": f"example/{repo}:{index}",
                    "num_non_test_files": index % 5 + 1,
                    "num_non_test_func_methods": index % 11 + 1,
                    "num_non_test_lines": index + 1,
                }
            )
    return rows


def test_largest_remainder_is_deterministic_and_respects_capacity() -> None:
    quotas = largest_remainder_quotas(
        {"a": 5, "b": 3, "c": 2},
        7,
        capacities={"a": 3, "b": 3, "c": 2},
    )
    assert quotas == {"a": 3, "b": 2, "c": 2}


def test_r2e_tiers_are_deterministic_stratified_and_nested() -> None:
    first, metadata = build_nested_r2e_subsets(
        _r2e_rows(), small_size=32, large_size=96, seed=7
    )
    second, _ = build_nested_r2e_subsets(
        _r2e_rows(), small_size=32, large_size=96, seed=7
    )
    first_ids = {name: [task_id(row) for row in rows] for name, rows in first.items()}
    second_ids = {name: [task_id(row) for row in rows] for name, rows in second.items()}
    assert first_ids == second_ids
    assert set(first_ids["r2e_train_128"]) <= set(first_ids["r2e_train_512"])
    assert sum(metadata["repo_quotas"]["r2e_train_128"].values()) == 32
    assert sum(metadata["repo_quotas"]["r2e_train_512"].values()) == 96
    assert set(metadata["repo_quotas"]["r2e_train_128"]) == {"large", "medium", "small"}


def test_verified_quick_is_repository_proportional() -> None:
    rows = [
        {"instance_id": f"owner/repo-a__{index}", "repo": "owner/repo-a"}
        for index in range(80)
    ] + [
        {"instance_id": f"owner/repo-b__{index}", "repo": "owner/repo-b"}
        for index in range(20)
    ]
    selected, quotas = select_verified_quick(rows, size=20, seed=9)
    assert len(selected) == 20
    assert quotas == {"owner/repo-a": 15, "owner/repo-b": 5}
    assert len({row["instance_id"] for row in selected}) == 20
