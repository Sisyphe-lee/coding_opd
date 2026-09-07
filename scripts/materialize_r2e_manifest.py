#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from datasets import Dataset

from coding_opd.r2e_data import (
    convert_row,
    load_r2e_source,
    select_frozen_manifest_rows,
    task_id,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialize an exact frozen R2E manifest as Coding OPD runtime parquet."
    )
    parser.add_argument("source", help="Local R2E parquet file/directory/glob or dataset id")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--source-split", default="train")
    parser.add_argument("--split-name", default="train")
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source = load_r2e_source(args.source, split=args.source_split)
    selected = select_frozen_manifest_rows(source, manifest)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"{args.split_name}.parquet"
    Dataset.from_list(
        [convert_row(row, data_split=args.split_name) for row in selected]
    ).to_parquet(str(output_path))

    runtime_manifest = {
        "format_version": 1,
        "frozen_manifest": str(manifest_path),
        "frozen_manifest_name": manifest["name"],
        "frozen_task_id_sha256": manifest["task_id_sha256"],
        "source": args.source,
        "source_revision": manifest["source"]["revision"],
        "source_split": args.source_split,
        "splits": {
            args.split_name: {
                "count": len(selected),
                "repo_counts": dict(
                    sorted(Counter(row["repo_name"] for row in selected).items())
                ),
                "task_ids": [task_id(row) for row in selected],
            }
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(runtime_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"materialized {len(selected)} tasks from {manifest['name']} at {output_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
