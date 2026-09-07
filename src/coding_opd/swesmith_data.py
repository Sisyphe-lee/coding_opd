from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from glob import glob
from pathlib import Path
from typing import Iterable, Sequence

from datasets import Dataset, load_dataset


DEFAULT_SOURCE = "SWE-bench/SWE-smith-py"
REQUIRED_COLUMNS = {
    "instance_id",
    "repo",
    "problem_statement",
    "image_name",
    "FAIL_TO_PASS",
    "PASS_TO_PASS",
}


def load_swesmith_source(source: str = DEFAULT_SOURCE, *, split: str = "train") -> Dataset:
    """Load SWE-smith from Hugging Face or local parquet files."""
    source_path = Path(source)
    if source_path.is_file():
        return Dataset.from_parquet(str(source_path))
    if source_path.is_dir():
        paths = sorted(source_path.glob(f"{split}-*.parquet")) or sorted(source_path.glob("*.parquet"))
        if not paths:
            raise ValueError(f"no parquet files found in {source_path}")
        return Dataset.from_parquet([str(path) for path in paths])
    if any(character in source for character in "*?["):
        paths = sorted(glob(source))
        if not paths:
            raise ValueError(f"parquet glob matched no files: {source}")
        return Dataset.from_parquet(paths)
    return load_dataset(source, split=split)


def validate_rows(rows: Iterable[dict]) -> list[dict]:
    validated: list[dict] = []
    seen: set[str] = set()
    for index, raw_row in enumerate(rows):
        row = dict(raw_row)
        missing = REQUIRED_COLUMNS - row.keys()
        if missing:
            raise ValueError(f"row {index} is missing required columns: {sorted(missing)}")
        instance_id = str(row["instance_id"])
        if instance_id in seen:
            raise ValueError(f"duplicate SWE-smith instance_id: {instance_id}")
        if not str(row["problem_statement"]).strip():
            raise ValueError(f"empty problem statement for {instance_id}")
        if not str(row["image_name"]).strip():
            raise ValueError(f"empty image name for {instance_id}")
        if not list(row["FAIL_TO_PASS"]):
            raise ValueError(f"no FAIL_TO_PASS tests for {instance_id}")
        seen.add(instance_id)
        validated.append(row)
    return validated


def _rank(seed: int, label: str, value: str) -> bytes:
    return hashlib.sha256(f"{seed}\0{label}\0{value}".encode()).digest()


def select_debug_tasks(
    rows: Iterable[dict],
    *,
    size: int = 16,
    repositories: int = 4,
    seed: int = 42,
    include_repos: Sequence[str] | None = None,
) -> list[dict]:
    """Select a deterministic repository-balanced debug set.

    A small number of repositories intentionally keeps image downloads small while
    still exercising image sharing and task-specific branch checkout.
    """
    pool = validate_rows(rows)
    if not 0 < repositories <= size:
        raise ValueError("expected 0 < repositories <= size")
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in pool:
        groups[str(row["repo"])].append(row)

    if include_repos:
        missing = set(include_repos) - groups.keys()
        if missing:
            raise ValueError(f"requested repositories not found: {sorted(missing)}")
        chosen_repos = list(include_repos)
        if len(chosen_repos) != repositories:
            raise ValueError("include_repos count must equal repositories")
    else:
        eligible = [repo for repo, repo_rows in groups.items() if len(repo_rows) >= size // repositories]
        chosen_repos = sorted(eligible, key=lambda repo: _rank(seed, "repo", repo))[:repositories]
    if len(chosen_repos) < repositories:
        raise ValueError(f"only {len(chosen_repos)} repositories can satisfy the requested allocation")

    base, remainder = divmod(size, repositories)
    selected: list[dict] = []
    for index, repo in enumerate(chosen_repos):
        count = base + (index < remainder)
        ranked = sorted(groups[repo], key=lambda row: _rank(seed, repo, str(row["instance_id"])))
        if len(ranked) < count:
            raise ValueError(f"repository {repo} has only {len(ranked)} rows; need {count}")
        selected.extend(ranked[:count])
    return sorted(selected, key=lambda row: _rank(seed, "order", str(row["instance_id"])))


def build_eval_splits(
    rows: Iterable[dict],
    *,
    smoke_size: int = 8,
    fast_size: int = 32,
    full_size: int = 128,
    repositories: int = 8,
    seed: int = 42,
    exclude_ids: Iterable[str] = (),
) -> dict[str, list[dict]]:
    """Build nested, repository-balanced SWE-smith evaluation splits.

    The selected repositories are intentionally bounded so a full evaluation
    does not require one cold image per task.  ``fast`` and ``smoke`` are
    prefixes of the same deterministic ``full`` ordering, which keeps scores
    comparable across checkpoints and tiers.
    """
    if not 0 < smoke_size <= fast_size <= full_size:
        raise ValueError("expected 0 < smoke_size <= fast_size <= full_size")
    if not 0 < repositories <= full_size:
        raise ValueError("repositories must be positive and no larger than full_size")

    excluded = {str(value) for value in exclude_ids}
    pool = [row for row in validate_rows(rows) if str(row["instance_id"]) not in excluded]
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in pool:
        groups[str(row["repo"])].append(row)
    eligible = [repo for repo, repo_rows in groups.items() if len(repo_rows) >= full_size // repositories]
    if len(eligible) < repositories:
        raise ValueError(
            f"only {len(eligible)} repositories have enough rows for "
            f"{full_size // repositories} tasks each; need {repositories}"
        )
    chosen_repos = sorted(eligible, key=lambda repo: _rank(seed, "eval-repo", repo))[:repositories]
    per_repo, remainder = divmod(full_size, repositories)
    selected_by_repo: dict[str, list[dict]] = {}
    for index, repo in enumerate(chosen_repos):
        count = per_repo + (index < remainder)
        ranked = sorted(
            groups[repo],
            key=lambda row: _rank(seed, "eval-task", str(row["instance_id"])),
        )
        selected_by_repo[repo] = ranked[:count]

    # Round-robin by repository so the nested prefixes retain broad coverage.
    ordered: list[dict] = []
    repo_order = sorted(chosen_repos, key=lambda repo: _rank(seed, "eval-order", repo))
    for offset in range(max(map(len, selected_by_repo.values()))):
        for repo in repo_order:
            repo_rows = selected_by_repo[repo]
            if offset < len(repo_rows):
                ordered.append(repo_rows[offset])
    if len(ordered) != full_size:
        raise RuntimeError("internal evaluation split allocation error")
    return {"smoke": ordered[:smoke_size], "fast": ordered[:fast_size], "full": ordered}


def convert_row(row: dict, *, data_split: str = "debug") -> dict:
    metadata = {
        "instance_id": str(row["instance_id"]),
        "repo": str(row["repo"]),
        # Task prompt templates are rendered from metadata inside Uni-Agent.
        # Keep this alongside the top-level prompt, which remains the
        # framework's authoritative input.
        "problem_statement": str(row["problem_statement"]),
        "FAIL_TO_PASS": list(row["FAIL_TO_PASS"]),
        "PASS_TO_PASS": list(row["PASS_TO_PASS"]),
    }
    task = {
        "name": "swe_smith",
        "sandbox": {"image": str(row["image_name"])},
        "metadata": metadata,
    }
    return {
        "data_source": "swe_smith",
        "prompt": [{"role": "user", "content": str(row["problem_statement"])}],
        "extra_info": {
            "task_id": metadata["instance_id"],
            "split": data_split,
            "tools_kwargs": {"task": task},
        },
    }


def write_debug_bundle(
    rows: Sequence[dict],
    output_dir: Path,
    *,
    source: str,
    seed: int,
    repeat_factor: int = 2,
) -> dict:
    if repeat_factor < 1:
        raise ValueError("repeat_factor must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    converted_rows = [convert_row(row) for row in rows]
    converted = Dataset.from_list(converted_rows)
    parquet_path = output_dir / "debug.parquet"
    converted.to_parquet(str(parquet_path))
    manifest = {
        "format_version": 1,
        "source": source,
        "seed": seed,
        "splits": {
            "debug": {
                "count": len(rows),
                "unique_images": len({str(row["image_name"]) for row in rows}),
                "repo_counts": dict(sorted(Counter(str(row["repo"]) for row in rows).items())),
                "task_ids": [str(row["instance_id"]) for row in rows],
                "images": sorted({str(row["image_name"]) for row in rows}),
            }
        },
    }
    _add_repeated_debug_fixture(converted_rows, output_dir, manifest, repeat_factor=repeat_factor)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def _add_repeated_debug_fixture(
    converted_rows: Sequence[dict],
    output_dir: Path,
    manifest: dict,
    *,
    repeat_factor: int,
) -> None:
    if repeat_factor < 1:
        raise ValueError("repeat_factor must be positive")
    if repeat_factor == 1:
        return
    repeated_name = f"debug_repeat{repeat_factor}"
    repeated_path = output_dir / f"{repeated_name}.parquet"
    Dataset.from_list(list(converted_rows) * repeat_factor).to_parquet(str(repeated_path))
    manifest["splits"][repeated_name] = {
        "count": len(converted_rows) * repeat_factor,
        "unique_tasks": len(converted_rows),
        "repeat_factor": repeat_factor,
        "source_split": "debug",
        "path": repeated_path.name,
    }


def add_repeated_debug_fixture(output_dir: Path, *, repeat_factor: int = 2) -> dict:
    """Repeat an already frozen converted debug split without reselecting tasks."""
    parquet_path = output_dir / "debug.parquet"
    manifest_path = output_dir / "manifest.json"
    if not parquet_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("reuse mode requires debug.parquet and manifest.json")
    converted_rows = [dict(row) for row in Dataset.from_parquet(str(parquet_path))]
    manifest = json.loads(manifest_path.read_text())
    expected_count = int(manifest["splits"]["debug"]["count"])
    if len(converted_rows) != expected_count:
        raise ValueError(f"debug parquet/manifest count mismatch: {len(converted_rows)} != {expected_count}")
    _add_repeated_debug_fixture(converted_rows, output_dir, manifest, repeat_factor=repeat_factor)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest
