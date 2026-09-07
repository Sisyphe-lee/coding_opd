#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import Dataset

from coding_opd.dataset_selection import id_sha256
from coding_opd.eval_data import (
    agent_images_from_rows,
    describe_eval_rows,
    load_deepswe_task_ids,
    materialize_deepswe_rows,
    sha256_file,
    sha256_tree,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize DeepSWE-Tura20/113 runtime parquets.")
    parser.add_argument("task_root", type=Path, help="DeepSWE v1.1 tasks/ directory")
    parser.add_argument("official_tasks_json", type=Path, help="Pinned official v1.1 tasks.json")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("configs/dataset_manifests/deepswe_tura20.json"),
    )
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual_tree_hash = sha256_tree(args.task_root)
    expected_tree_hash = manifest["source"]["task_tree_sha256"]
    if actual_tree_hash != expected_tree_hash:
        raise ValueError(
            f"DeepSWE task tree hash mismatch: expected {expected_tree_hash}, found {actual_tree_hash}"
        )
    full_ids = load_deepswe_task_ids(args.official_tasks_json, manifest)
    quick_ids = [str(task["task_id"]) for task in manifest["tasks"]]
    if id_sha256(quick_ids) != manifest["task_id_sha256"]:
        raise ValueError("DeepSWE manifest task ID hash does not match its ordered tasks")
    if not set(quick_ids) <= set(full_ids):
        raise ValueError("DeepSWE-Tura20 is not contained in the pinned full task list")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    runtime_manifest = {
        "format_version": 2,
        "protocol": "coding-opd-external-eval-v2",
        "benchmark": "DataCurve DeepSWE v1.1",
        "frozen_manifest": str(manifest_path),
        "frozen_manifest_name": manifest["name"],
        "official_tasks_json": str(args.official_tasks_json.resolve()),
        "official_tasks_sha256": manifest["source"]["official_tasks_sha256"],
        "task_root": str(args.task_root.resolve()),
        "task_repository_revision": manifest["source"]["task_repository_revision"],
        "task_tree_sha256": actual_tree_hash,
        "splits": {},
    }
    rows_by_split = {}
    for split, task_ids in (("quick", quick_ids), ("full", full_ids)):
        rows = materialize_deepswe_rows(args.task_root, task_ids, split=split)
        rows_by_split[split] = rows
        output_path = args.output_dir / f"{split}.parquet"
        Dataset.from_list(rows).to_parquet(str(output_path))
        runtime_manifest["splits"][split] = {
            **describe_eval_rows(rows),
            "path": output_path.name,
            "sha256": sha256_file(output_path),
        }
    runtime_manifest["agent_images"] = agent_images_from_rows(rows_by_split["full"])
    runtime_manifest["agent_image_count"] = len(runtime_manifest["agent_images"])

    (args.output_dir / "manifest.json").write_text(
        json.dumps(runtime_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value["count"] for key, value in runtime_manifest["splits"].items()}, sort_keys=True))


if __name__ == "__main__":
    main()
