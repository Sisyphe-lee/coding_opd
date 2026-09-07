#!/usr/bin/env python3
"""Freeze the agreed Coding OPD training and quick-evaluation task manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from coding_opd.dataset_selection import (
    R2E_COMPLEXITY_FIELDS,
    build_nested_r2e_subsets,
    id_sha256,
    r2e_task_record,
    select_verified_quick,
    split_counts,
)


R2E_REQUESTED_SOURCE = "PrimeIntellect/R2E-Gym-Subset-Validated"
R2E_RESOLVED_SOURCE = "PrimeIntellect/R2E-Gym-Subset-Verified"
R2E_REVISION = "151f9950e62cac613e07be1bb92e5dd19687315e"
VERIFIED_SOURCE = "SWE-bench/SWE-bench_Verified"
VERIFIED_REVISION = "78f471bf655a3137b2e8a75af1501690ec009ec3"
TURA_REVISION = "da58a15d35b5284db2abfd233378093a2689dc5c"
TURA20_URL = (
    "https://raw.githubusercontent.com/Tura-AI/benchmark/"
    f"{TURA_REVISION}/deep_swe/canonical_tasks.json"
)
TURA20_SHA256 = "de633d8d686ecdbfd46a37fb768c4fb207c80a9de86ae0c3599d825a1919dc21"
DEEPSWE_TASKS_URL = "https://deepswe.datacurve.ai/artifacts/v1.1/tasks.json"
DEEPSWE_TASKS_SHA256 = "bae967f6472943564c3fc5232fba3c8e0ac465c1be5ccf9dd4895d4ee9df6242"
DEEPSWE_TASK_REPOSITORY_REVISION = "0b9fabbb63b9104d678fe965e1632f2dd9eaa2ea"
DEEPSWE_TASK_TREE_SHA256 = "207e1309bfbaebdf9186123ea4e74732c9348c395687af9bd48e01193a8bd5d9"


def _fetch_json(url: str) -> tuple[Any, str]:
    request = urllib.request.Request(url, headers={"User-Agent": "coding-opd-manifest/1"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = response.read()
            return json.loads(payload), hashlib.sha256(payload).hexdigest()
        except (OSError, TimeoutError, urllib.error.URLError):
            if attempt == 4:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable JSON retry loop")


def _load_local_parquet_columns(
    directory: Path,
    *,
    split: str,
    columns: list[str],
) -> list[dict[str, Any]]:
    paths = sorted((directory / "data").glob(f"{split}-*.parquet"))
    if not paths:
        paths = sorted(directory.glob(f"{split}-*.parquet"))
    if not paths:
        raise ValueError(f"no local {split} parquet shards found under {directory}")
    rows: list[dict[str, Any]] = []
    for index, path in enumerate(paths, start=1):
        print(f"[{directory}] shard {index}/{len(paths)}: local", flush=True)
        table = pq.read_table(path, columns=columns)
        rows.extend(dict(row) for row in table.to_pylist())
    return rows


def _write(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _base_manifest(name: str, *, count: int, task_ids: list[str]) -> dict[str, Any]:
    return {
        "schema": "coding_opd.dataset_manifest.v1",
        "name": name,
        "frozen_date": "2026-09-02",
        "count": count,
        "task_id_sha256": id_sha256(task_ids),
    }


def _verified_image(instance_id: str) -> str:
    normalized = instance_id.lower().replace("__", "_1776_")
    return f"swebench/sweb.eval.x86_64.{normalized}"


def verified64_manifest(rows, legacy50):
    selected, quotas = select_verified_quick(rows, size=64, seed=legacy50["seed"])
    tasks = [{"task_id": str(row["instance_id"]), "repo": str(row["repo"]),
              "docker_image": _verified_image(str(row["instance_id"]))} for row in selected]
    if not {t["task_id"] for t in legacy50["tasks"]} <= {t["task_id"] for t in tasks}:
        raise ValueError("Verified-64 must preserve every frozen Verified-50 task")
    return {
        **legacy50,
        **_base_manifest("swebench_verified_64", count=64,
                         task_ids=[t["task_id"] for t in tasks]),
        "frozen_date": "2026-09-06", "image_count": 64,
        "contains": "swebench_verified_50",
        "selection": {**legacy50["selection"], "repo_quotas": quotas},
        "repo_counts": quotas, "tasks": tasks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("configs/dataset_manifests"),
    )
    parser.add_argument(
        "--r2e-local-dir",
        type=Path,
        default=Path("data/manifest_sources/r2e_validated_4522"),
        help="HFD local directory for the pinned R2E dataset",
    )
    parser.add_argument(
        "--verified-local-dir",
        type=Path,
        default=Path("data/manifest_sources/swebench_verified"),
        help="HFD local directory for the pinned SWE-bench Verified dataset",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    r2e_columns = [
        "repo_name",
        "docker_image",
        "commit_hash",
        "problem_statement",
        "expected_output_json",
        *R2E_COMPLEXITY_FIELDS,
    ]
    r2e_rows = _load_local_parquet_columns(
        args.r2e_local_dir, split="train", columns=r2e_columns
    )
    if len(r2e_rows) != 4522:
        raise ValueError(f"expected 4522 validated R2E rows, found {len(r2e_rows)}")
    r2e_subsets, r2e_metadata = build_nested_r2e_subsets(r2e_rows, seed=args.seed)
    complexity_bins = r2e_metadata.pop("complexity_bin_by_task_id")

    r2e_manifests: dict[str, dict[str, Any]] = {}
    for name in ("r2e_train_128", "r2e_train_512"):
        rows = r2e_subsets[name]
        tasks = [
            r2e_task_record(row, complexity_bin=complexity_bins[f"{row['repo_name']}:{row['commit_hash']}"])
            for row in rows
        ]
        task_ids = [task["task_id"] for task in tasks]
        manifest = _base_manifest(name, count=len(tasks), task_ids=task_ids)
        manifest.update(
            {
                "purpose": "training",
                "seed": args.seed,
                "source": {
                    "requested_dataset": R2E_REQUESTED_SOURCE,
                    "resolved_dataset": R2E_RESOLVED_SOURCE,
                    "revision": R2E_REVISION,
                    "split": "train",
                    "validated_source_count": 4522,
                },
                "selection": {
                    "repository_allocation": "largest_remainder(N_repo ** 0.5)",
                    "within_repository": "four equal-frequency static complexity bins",
                    "complexity_fields": list(R2E_COMPLEXITY_FIELDS),
                    "within_stratum_order": "sha256(seed, r2e-task, task_id)",
                    "repo_quotas": r2e_metadata["repo_quotas"][name],
                    "stratum_quotas": {
                        repo: {
                            key: value
                            for key, value in quotas.items()
                            if key.startswith("q128" if name.endswith("128") else "q512")
                        }
                        for repo, quotas in r2e_metadata["stratum_quotas"].items()
                    },
                },
                "repo_counts": split_counts(tasks, "repo_name"),
                "image_count": len({task["docker_image"] for task in tasks}),
                "tasks": tasks,
            }
        )
        if name == "r2e_train_128":
            manifest["nested_in"] = "r2e_train_512"
        r2e_manifests[name] = manifest
        _write(args.output_dir / f"{name}.json", manifest)

    ids128 = {task["task_id"] for task in r2e_manifests["r2e_train_128"]["tasks"]}
    ids512 = {task["task_id"] for task in r2e_manifests["r2e_train_512"]["tasks"]}
    if not ids128 <= ids512:
        raise AssertionError("r2e_train_128 is not nested in r2e_train_512")

    verified_rows = _load_local_parquet_columns(
        args.verified_local_dir,
        split="test",
        columns=["instance_id", "repo"],
    )
    if len(verified_rows) != 500:
        raise ValueError(f"expected 500 SWE-bench Verified rows, found {len(verified_rows)}")
    verified50, verified_quotas = select_verified_quick(verified_rows, seed=args.seed)
    verified_tasks = [
        {
            "task_id": str(row["instance_id"]),
            "repo": str(row["repo"]),
            "docker_image": _verified_image(str(row["instance_id"])),
        }
        for row in verified50
    ]
    verified_ids = [task["task_id"] for task in verified_tasks]
    verified_manifest = _base_manifest(
        "swebench_verified_50", count=len(verified_tasks), task_ids=verified_ids
    )
    verified_manifest.update(
        {
            "purpose": "quick_evaluation",
            "nested_in": "official SWE-bench Verified test split (500 tasks)",
            "seed": args.seed,
            "source": {
                "dataset": VERIFIED_SOURCE,
                "revision": VERIFIED_REVISION,
                "split": "test",
                "full_count": len(verified_rows),
            },
            "selection": {
                "repository_allocation": "one per repository, then largest_remainder(N_repo)",
                "within_repository_order": "sha256(seed, verified-task, instance_id)",
                "repo_quotas": verified_quotas,
            },
            "repo_counts": split_counts(verified_tasks, "repo"),
            "image_count": len({task["docker_image"] for task in verified_tasks}),
            "tasks": verified_tasks,
        }
    )
    _write(args.output_dir / "swebench_verified_50.json", verified_manifest)
    _write(args.output_dir / "swebench_verified_64.json",
           verified64_manifest(verified_rows, verified_manifest))

    tura_payload, tura_sha256 = _fetch_json(TURA20_URL)
    deepswe_payload, deepswe_sha256 = _fetch_json(DEEPSWE_TASKS_URL)
    if tura_sha256 != TURA20_SHA256:
        raise ValueError(f"Tura20 source hash changed: {tura_sha256}")
    if deepswe_sha256 != DEEPSWE_TASKS_SHA256:
        raise ValueError(f"DeepSWE v1.1 task source hash changed: {deepswe_sha256}")
    if int(deepswe_payload["n_tasks"]) != 113 or len(deepswe_payload["rows"]) != 113:
        raise ValueError("expected exactly 113 DeepSWE v1.1 tasks")
    if len(tura_payload["tasks"]) != 20:
        raise ValueError("expected exactly 20 Tura canonical DeepSWE tasks")
    full_tasks = {str(row["id"]): dict(row) for row in deepswe_payload["rows"]}
    quick_tasks: list[dict[str, Any]] = []
    for selected in tura_payload["tasks"]:
        uid = str(selected["task_id"])
        if uid not in full_tasks:
            raise ValueError(f"Tura20 task is absent from DeepSWE v1.1: {uid}")
        official = full_tasks[uid]
        if str(official["language"]).lower() != str(selected["language"]).lower():
            raise ValueError(f"language mismatch for DeepSWE task {uid}")
        quick_tasks.append(
            {
                "task_id": uid,
                "language": str(selected["language"]).lower(),
                "difficulty_band": str(selected["difficulty_band"]),
                "repository": str(official["repository"]),
                "base_commit_hash": str(official["base_commit_hash"]),
            }
        )
    deepswe_ids = [task["task_id"] for task in quick_tasks]
    deepswe_manifest = _base_manifest(
        "deepswe_tura20", count=len(quick_tasks), task_ids=deepswe_ids
    )
    deepswe_manifest.update(
        {
            "purpose": "quick_evaluation",
            "nested_in": "DeepSWE v1.1 full evaluation (113 tasks)",
            "source": {
                "benchmark": "DataCurve DeepSWE",
                "benchmark_version": "v1.1",
                "official_tasks_url": DEEPSWE_TASKS_URL,
                "official_tasks_sha256": deepswe_sha256,
                "official_full_count": int(deepswe_payload["n_tasks"]),
                "task_repository": "datacurve-ai/deep-swe",
                "task_repository_revision": DEEPSWE_TASK_REPOSITORY_REVISION,
                "task_tree_sha256": DEEPSWE_TASK_TREE_SHA256,
                "selection_repository": "Tura-AI/benchmark",
                "selection_revision": TURA_REVISION,
                "selection_url": TURA20_URL,
                "selection_sha256": tura_sha256,
            },
            "selection": {
                "method": "public Tura20 canonical subset",
                "language_quota": 4,
                "difficulty_bands": ["easy", "medium-easy", "medium-hard", "hard"],
            },
            "language_counts": split_counts(quick_tasks, "language"),
            "difficulty_counts": split_counts(quick_tasks, "difficulty_band"),
            "tasks": quick_tasks,
        }
    )
    _write(args.output_dir / "deepswe_tura20.json", deepswe_manifest)

    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "r2e_train_128": len(r2e_subsets["r2e_train_128"]),
                "r2e_train_512": len(r2e_subsets["r2e_train_512"]),
                "swebench_verified_50": len(verified_tasks),
                "swebench_verified_64": 64,
                "deepswe_tura20": len(quick_tasks),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
