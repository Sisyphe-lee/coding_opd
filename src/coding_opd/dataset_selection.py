from __future__ import annotations

import bisect
import hashlib
import math
from collections import Counter, defaultdict
from typing import Any, Iterable, Mapping, Sequence

from coding_opd.r2e_data import task_id, validate_source_rows


R2E_COMPLEXITY_FIELDS = (
    "num_non_test_files",
    "num_non_test_func_methods",
    "num_non_test_lines",
)


def stable_rank(seed: int, label: str, value: str) -> bytes:
    return hashlib.sha256(f"{seed}\0{label}\0{value}".encode()).digest()


def id_sha256(task_ids: Sequence[str]) -> str:
    payload = "".join(f"{task_id_}\n" for task_id_ in task_ids).encode()
    return hashlib.sha256(payload).hexdigest()


def largest_remainder_quotas(
    weights: Mapping[str, float],
    total: int,
    *,
    capacities: Mapping[str, int] | None = None,
) -> dict[str, int]:
    """Allocate an integer total deterministically using largest remainder."""
    if total < 0:
        raise ValueError("total must be non-negative")
    keys = sorted(weights)
    if not keys or any(float(weights[key]) < 0 for key in keys):
        raise ValueError("weights must be a non-empty mapping of non-negative values")
    if capacities is None:
        capacities = {key: total for key in keys}
    if set(capacities) != set(keys):
        raise ValueError("capacities and weights must have identical keys")
    if total > sum(int(capacities[key]) for key in keys):
        raise ValueError("requested total exceeds capacity")

    positive_weight = sum(float(weights[key]) for key in keys if capacities[key] > 0)
    if total and positive_weight <= 0:
        raise ValueError("positive capacity requires positive weight")
    exact = {
        key: (total * float(weights[key]) / positive_weight if capacities[key] > 0 else 0.0)
        for key in keys
    }
    quotas = {
        key: min(int(capacities[key]), math.floor(exact[key]))
        for key in keys
    }
    remaining = total - sum(quotas.values())
    while remaining:
        candidates = [key for key in keys if quotas[key] < int(capacities[key])]
        if not candidates:
            raise RuntimeError("unable to satisfy quota allocation")
        # The first pass is the conventional largest-remainder assignment. The
        # deficit form also behaves sensibly if a capacity binds.
        selected = min(candidates, key=lambda key: (-(exact[key] - quotas[key]), key))
        quotas[selected] += 1
        remaining -= 1
    return quotas


def temperature_repo_quotas(
    repo_sizes: Mapping[str, int],
    total: int,
    *,
    alpha: float,
    capacities: Mapping[str, int] | None = None,
) -> dict[str, int]:
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be in [0, 1]")
    if capacities is None:
        capacities = repo_sizes
    weights = {repo: float(size) ** alpha for repo, size in repo_sizes.items()}
    return largest_remainder_quotas(weights, total, capacities=capacities)


def _numeric(row: Mapping[str, Any], field: str) -> int:
    value = row.get(field, 0)
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid {field} for {task_id(dict(row))}: {value!r}") from error


def _complexity_bins(repo_rows: Sequence[dict[str, Any]], *, seed: int) -> dict[str, int]:
    """Assign equal-frequency bins from within-repository static patch proxies."""
    sorted_values = {
        field: sorted(math.log1p(_numeric(row, field)) for row in repo_rows)
        for field in R2E_COMPLEXITY_FIELDS
    }

    def score(row: dict[str, Any]) -> float:
        percentiles = []
        for field, values in sorted_values.items():
            value = math.log1p(_numeric(row, field))
            percentiles.append(bisect.bisect_right(values, value) / len(values))
        return sum(percentiles) / len(percentiles)

    ordered = sorted(
        repo_rows,
        key=lambda row: (score(row), stable_rank(seed, "complexity-tie", task_id(row))),
    )
    return {
        task_id(row): min(3, index * 4 // len(ordered))
        for index, row in enumerate(ordered)
    }


def build_nested_r2e_subsets(
    rows: Iterable[dict[str, Any]],
    *,
    small_size: int = 128,
    large_size: int = 512,
    seed: int = 42,
    repo_temperature: float = 0.5,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Build deterministic repository/complexity-stratified nested task pools."""
    pool = validate_source_rows(rows)
    if not 0 < small_size <= large_size <= len(pool):
        raise ValueError(f"expected 0 < {small_size=} <= {large_size=} <= {len(pool)}")

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pool:
        missing = set(R2E_COMPLEXITY_FIELDS) - row.keys()
        if missing:
            raise ValueError(f"R2E row is missing complexity fields: {sorted(missing)}")
        groups[str(row["repo_name"])].append(row)

    repo_sizes = {repo: len(repo_rows) for repo, repo_rows in groups.items()}
    large_repo_quotas = temperature_repo_quotas(
        repo_sizes, large_size, alpha=repo_temperature
    )
    small_repo_quotas = temperature_repo_quotas(
        repo_sizes,
        small_size,
        alpha=repo_temperature,
        capacities=large_repo_quotas,
    )

    row_by_id = {task_id(row): row for row in pool}
    bin_by_id: dict[str, int] = {}
    small_ids: set[str] = set()
    large_ids: set[str] = set()
    stratum_quotas: dict[str, dict[str, int]] = {}

    for repo in sorted(groups):
        repo_bins = _complexity_bins(groups[repo], seed=seed)
        bin_by_id.update(repo_bins)
        strata: dict[int, list[str]] = defaultdict(list)
        for uid, bin_index in repo_bins.items():
            strata[bin_index].append(uid)
        bin_sizes = {str(index): len(strata[index]) for index in range(4)}
        large_bin_quotas = largest_remainder_quotas(
            bin_sizes,
            large_repo_quotas[repo],
            capacities=bin_sizes,
        )
        small_bin_quotas = largest_remainder_quotas(
            bin_sizes,
            small_repo_quotas[repo],
            capacities=large_bin_quotas,
        )
        stratum_quotas[repo] = {
            f"q128_bin_{index}": small_bin_quotas[str(index)] for index in range(4)
        } | {
            f"q512_bin_{index}": large_bin_quotas[str(index)] for index in range(4)
        }

        for index in range(4):
            ranked = sorted(
                strata[index],
                key=lambda uid: stable_rank(seed, "r2e-task", uid),
            )
            large_selected = ranked[: large_bin_quotas[str(index)]]
            small_selected = ranked[: small_bin_quotas[str(index)]]
            large_ids.update(large_selected)
            small_ids.update(small_selected)

    if len(small_ids) != small_size or len(large_ids) != large_size:
        raise AssertionError("R2E tier sizes do not match requested sizes")
    if not small_ids <= large_ids:
        raise AssertionError("small R2E tier is not nested in large tier")

    def materialize(ids: set[str], label: str) -> list[dict[str, Any]]:
        return [
            row_by_id[uid]
            for uid in sorted(ids, key=lambda uid: stable_rank(seed, f"{label}:order", uid))
        ]

    subsets = {
        "r2e_train_128": materialize(small_ids, "r2e128"),
        "r2e_train_512": materialize(large_ids, "r2e512"),
    }
    metadata = {
        "source_pool_count": len(pool),
        "source_repo_counts": dict(sorted(repo_sizes.items())),
        "repo_temperature_alpha": repo_temperature,
        "repo_quotas": {
            "r2e_train_128": dict(sorted(small_repo_quotas.items())),
            "r2e_train_512": dict(sorted(large_repo_quotas.items())),
        },
        "stratum_quotas": stratum_quotas,
        "complexity_bin_by_task_id": bin_by_id,
    }
    return subsets, metadata


def select_verified_quick(
    rows: Iterable[dict[str, Any]],
    *,
    size: int = 50,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Select a deterministic repository-proportional subset of Verified."""
    pool = [dict(row) for row in rows]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: set[str] = set()
    for row in pool:
        uid = str(row["instance_id"])
        if uid in seen:
            raise ValueError(f"duplicate SWE-bench Verified instance_id: {uid}")
        seen.add(uid)
        groups[str(row["repo"])].append(row)
    if not 0 < size <= len(pool):
        raise ValueError(f"expected 0 < size <= {len(pool)}")
    if size < len(groups):
        raise ValueError(f"size {size} cannot cover all {len(groups)} repositories")

    repo_sizes = {repo: len(repo_rows) for repo, repo_rows in groups.items()}
    # Give every official repository one slot, then distribute the remaining
    # slots proportionally. This keeps tiny repositories visible in a 50-task
    # panel without pretending the panel is a simple random sample.
    extra_capacities = {repo: repo_size - 1 for repo, repo_size in repo_sizes.items()}
    extra = temperature_repo_quotas(
        repo_sizes,
        size - len(groups),
        alpha=1.0,
        capacities=extra_capacities,
    )
    quotas = {repo: extra[repo] + 1 for repo in sorted(repo_sizes)}
    selected: list[dict[str, Any]] = []
    for repo in sorted(groups):
        ranked = sorted(
            groups[repo],
            key=lambda row: stable_rank(seed, "verified-task", str(row["instance_id"])),
        )
        selected.extend(ranked[: quotas[repo]])
    selected.sort(
        key=lambda row: stable_rank(seed, "verified50:order", str(row["instance_id"]))
    )
    return selected, dict(sorted(quotas.items()))


def r2e_task_record(row: dict[str, Any], *, complexity_bin: int) -> dict[str, Any]:
    return {
        "task_id": task_id(row),
        "repo_name": str(row["repo_name"]),
        "commit_hash": str(row["commit_hash"]),
        "docker_image": str(row["docker_image"]),
        "complexity_bin": int(complexity_bin),
        "complexity_proxy": {
            field: _numeric(row, field) for field in R2E_COMPLEXITY_FIELDS
        },
    }


def split_counts(rows: Iterable[Mapping[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row[field]) for row in rows).items()))
