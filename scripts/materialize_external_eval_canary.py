#!/usr/bin/env python3
"""Create a non-reportable canary from a validated external-evaluation split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import Dataset, load_dataset

from coding_opd.dataset_selection import id_sha256
from coding_opd.eval_data import (
    agent_images_from_rows,
    describe_eval_rows,
    select_canary_rows,
    sha256_file,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_parquet", type=Path)
    parser.add_argument("source_manifest", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--source-split", choices=("quick", "full"), default="full")
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument(
        "--sample-seed",
        type=int,
        help="Shuffle with Python random.Random(seed) before truncating, matching Pier.",
    )
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be positive")

    source_manifest = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    expected = source_manifest["splits"][args.source_split]
    dataset = load_dataset("parquet", data_files=str(args.source_parquet), split="train")
    rows = [dict(row) for row in dataset]
    task_ids = [str(row["extra_info"]["task_id"]) for row in rows]
    if len(rows) != int(expected["count"]):
        raise ValueError("source parquet count does not match its runtime manifest")
    if id_sha256(task_ids) != expected["task_id_sha256"]:
        raise ValueError("source parquet task order/hash does not match its runtime manifest")
    if sha256_file(args.source_parquet) != expected["sha256"]:
        raise ValueError("source parquet bytes do not match its runtime manifest")
    if args.count > len(rows):
        parser.error(f"--count exceeds the {len(rows)} source rows")

    canary_rows = select_canary_rows(rows, count=args.count, sample_seed=args.sample_seed)
    for row in canary_rows:
        row["extra_info"]["split"] = "quick"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "quick.parquet"
    Dataset.from_list(canary_rows).to_parquet(str(output_path))
    agent_images = agent_images_from_rows(canary_rows)
    manifest = {
        "format_version": 2,
        "protocol": "coding-opd-external-eval-canary-v1",
        "benchmark": f"{source_manifest['benchmark']} canary (not reportable)",
        "is_canary": True,
        "source_manifest": str(args.source_manifest.resolve()),
        "source_manifest_sha256": sha256_file(args.source_manifest),
        "source_split": args.source_split,
        "sample_seed": args.sample_seed,
        "agent_image_count": len(agent_images),
        "agent_images": agent_images,
        "splits": {
            "quick": {
                **describe_eval_rows(canary_rows),
                "path": output_path.name,
                "sha256": sha256_file(output_path),
            }
        },
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "benchmark": manifest["benchmark"],
                "count": len(canary_rows),
                "manifest": str(manifest_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
