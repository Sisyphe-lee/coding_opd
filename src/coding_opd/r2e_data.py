from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from glob import glob
from pathlib import Path
from typing import Iterable

from datasets import Dataset, load_dataset


REQUIRED_COLUMNS = {
    "repo_name",
    "commit_hash",
    "problem_statement",
    "expected_output_json",
    "docker_image",
}


def task_id(row: dict) -> str:
    return f"{row['repo_name']}:{row['commit_hash']}"


def load_r2e_source(source: str, *, split: str = "train") -> Dataset:
    """Load a local parquet file/shard set or a Hugging Face dataset split."""
    source_path = Path(source)
    if source_path.is_file():
        return Dataset.from_parquet(str(source_path))
    if source_path.is_dir():
        shard_paths = sorted(source_path.glob(f"{split}-*.parquet"))
        if not shard_paths:
            shard_paths = sorted(source_path.glob("*.parquet"))
        if not shard_paths:
            raise ValueError(f"no parquet files found in {source_path}")
        return Dataset.from_parquet([str(path) for path in shard_paths])
    if any(character in source for character in "*?["):
        shard_paths = sorted(glob(source))
        if not shard_paths:
            raise ValueError(f"parquet glob matched no files: {source}")
        return Dataset.from_parquet(shard_paths)
    return load_dataset(source, split=split)


def validate_source_rows(rows: Iterable[dict]) -> list[dict]:
    validated: list[dict] = []
    seen: set[str] = set()
    for index, raw_row in enumerate(rows):
        row = dict(raw_row)
        missing = REQUIRED_COLUMNS - row.keys()
        if missing:
            raise ValueError(f"row {index} is missing required columns: {sorted(missing)}")
        uid = task_id(row)
        if uid in seen:
            raise ValueError(f"duplicate R2E task id: {uid}")
        if not row["problem_statement"].strip():
            raise ValueError(f"empty problem statement for {uid}")
        if not row["docker_image"].strip():
            raise ValueError(f"empty docker image for {uid}")
        seen.add(uid)
        validated.append(row)
    return validated


def select_frozen_manifest_rows(rows: Iterable[dict], manifest: dict) -> list[dict]:
    """Select and validate source rows in the exact order of a frozen manifest."""
    if manifest.get("schema") != "coding_opd.dataset_manifest.v1":
        raise ValueError(f"unsupported dataset manifest schema: {manifest.get('schema')!r}")

    tasks = manifest.get("tasks")
    if not isinstance(tasks, list) or manifest.get("count") != len(tasks):
        raise ValueError("manifest count does not match its task list")

    source_by_id = {task_id(row): row for row in validate_source_rows(rows)}
    selected: list[dict] = []
    seen: set[str] = set()
    for item in tasks:
        uid = item.get("task_id")
        if not isinstance(uid, str) or not uid:
            raise ValueError("manifest task is missing task_id")
        if uid in seen:
            raise ValueError(f"duplicate task id in manifest: {uid}")
        seen.add(uid)
        try:
            row = source_by_id[uid]
        except KeyError as error:
            raise ValueError(f"manifest task is missing from source: {uid}") from error
        for field in ("repo_name", "commit_hash", "docker_image"):
            expected = item.get(field)
            if row[field] != expected:
                raise ValueError(
                    f"manifest/source mismatch for {uid}: {field}={row[field]!r}, "
                    f"expected {expected!r}"
                )
        selected.append(row)
    return selected


def _rank(seed: int, label: str, uid: str) -> bytes:
    return hashlib.sha256(f"{seed}\0{label}\0{uid}".encode()).digest()


def _repo_quotas(groups: dict[str, list[dict]], size: int) -> dict[str, int]:
    total = sum(len(rows) for rows in groups.values())
    if size < 0 or size > total:
        raise ValueError(f"requested {size} rows from a pool of {total}")
    exact = {repo: size * len(rows) / total for repo, rows in groups.items()}
    quotas = {repo: min(len(groups[repo]), int(value)) for repo, value in exact.items()}
    remaining = size - sum(quotas.values())
    order = sorted(groups, key=lambda repo: (-(exact[repo] - quotas[repo]), repo))
    while remaining:
        progressed = False
        for repo in order:
            if quotas[repo] < len(groups[repo]):
                quotas[repo] += 1
                remaining -= 1
                progressed = True
                if not remaining:
                    break
        if not progressed:
            raise RuntimeError("unable to allocate repository quotas")
    return quotas


def stratified_select(rows: Iterable[dict], size: int, *, seed: int, label: str) -> list[dict]:
    """Select a deterministic, repository-proportional subset."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["repo_name"]].append(row)
    quotas = _repo_quotas(groups, size)
    selected: list[dict] = []
    for repo in sorted(groups):
        ranked = sorted(groups[repo], key=lambda row: _rank(seed, label, task_id(row)))
        selected.extend(ranked[: quotas[repo]])
    return sorted(selected, key=lambda row: _rank(seed, f"{label}:order", task_id(row)))


def build_r2e_splits(
    rows: Iterable[dict],
    *,
    train_size: int = 2048,
    dev_size: int = 128,
    pilot_size: int = 128,
    debug_size: int | None = None,
    smoke_size: int = 8,
    seed: int = 42,
) -> dict[str, list[dict]]:
    """Build disjoint train/dev splits with pilot and smoke nested in train."""
    pool = validate_source_rows(rows)
    if train_size + dev_size > len(pool):
        raise ValueError(
            f"train_size + dev_size ({train_size + dev_size}) exceeds source pool ({len(pool)})"
        )
    if not 0 < smoke_size <= pilot_size <= train_size:
        raise ValueError("expected 0 < smoke_size <= pilot_size <= train_size")
    if debug_size is not None and not smoke_size <= debug_size <= pilot_size:
        raise ValueError("expected smoke_size <= debug_size <= pilot_size")

    dev = stratified_select(pool, dev_size, seed=seed, label="dev")
    dev_ids = {task_id(row) for row in dev}
    train_candidates = [row for row in pool if task_id(row) not in dev_ids]
    train = stratified_select(train_candidates, train_size, seed=seed, label="train")
    pilot = stratified_select(train, pilot_size, seed=seed, label="pilot")
    smoke = stratified_select(pilot, smoke_size, seed=seed, label="smoke")
    splits = {"train": train, "dev": dev, "pilot": pilot}
    if debug_size is not None:
        smoke_ids = {task_id(row) for row in smoke}
        remaining = [row for row in pilot if task_id(row) not in smoke_ids]
        debug_extra = stratified_select(
            remaining,
            debug_size - smoke_size,
            seed=seed,
            label="debug-extra",
        )
        splits["debug"] = sorted(
            [*smoke, *debug_extra],
            key=lambda row: _rank(seed, "debug:order", task_id(row)),
        )
    splits["smoke"] = smoke
    return splits


def convert_row(row: dict, *, data_split: str | None = None) -> dict:
    uid = task_id(row)
    metadata = {
        "task_id": uid,
        "repo_name": row["repo_name"],
        "commit_hash": row["commit_hash"],
        "problem_statement": row["problem_statement"],
        "expected_output_json": row["expected_output_json"],
    }
    task = {
        "name": "r2e_gym",
        "sandbox": {"image": row["docker_image"]},
        "metadata": metadata,
    }
    extra_info = {"task_id": uid, "tools_kwargs": {"task": task}}
    if data_split is not None:
        extra_info["split"] = data_split
    return {
        "data_source": "r2e_gym",
        "prompt": [{"role": "user", "content": row["problem_statement"]}],
        "extra_info": extra_info,
    }


def write_split_bundle(
    splits: dict[str, list[dict]],
    output_dir: Path,
    *,
    source: str,
    source_split: str,
    seed: int,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    nesting = {"pilot": "train", "smoke": "debug" if "debug" in splits else "pilot"}
    if "debug" in splits:
        nesting["debug"] = "pilot"
    manifest = {
        "format_version": 1,
        "source": source,
        "source_split": source_split,
        "seed": seed,
        "nesting": nesting,
        "splits": {},
    }
    for name, rows in splits.items():
        converted = Dataset.from_list([convert_row(row, data_split=name) for row in rows])
        converted.to_parquet(str(output_dir / f"{name}.parquet"))
        repo_counts = dict(sorted(Counter(row["repo_name"] for row in rows).items()))
        manifest["splits"][name] = {
            "count": len(rows),
            "repo_counts": repo_counts,
            "task_ids": [task_id(row) for row in rows],
        }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest
