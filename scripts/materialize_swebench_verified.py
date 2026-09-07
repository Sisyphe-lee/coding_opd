#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import Dataset, load_dataset

from coding_opd.eval_data import (
    agent_images_from_rows,
    convert_verified_row,
    describe_eval_rows,
    select_verified_manifest_rows,
    sha256_file,
)


def _load_source(source: Path) -> list[dict]:
    if source.is_file():
        paths = [source]
    elif source.is_dir():
        paths = sorted(source.glob("data/test-*.parquet")) or sorted(source.glob("test-*.parquet"))
    else:
        paths = sorted(source.parent.glob(source.name))
    if not paths:
        raise ValueError(f"no Verified parquet files found for {source}")
    dataset = load_dataset("parquet", data_files=[str(path) for path in paths], split="train")
    return [dict(row) for row in dataset]


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize canonical Verified-64/500 runtime parquets.")
    parser.add_argument("source", type=Path, help="Pinned Verified parquet file, directory, or glob")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("configs/dataset_manifests/swebench_verified_64.json"),
    )
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    quick_source, full_source = select_verified_manifest_rows(_load_source(args.source), manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    runtime_manifest = {
        "format_version": 2,
        "protocol": "coding-opd-external-eval-v2",
        "benchmark": "SWE-bench Verified",
        "frozen_manifest": str(manifest_path),
        "frozen_manifest_name": manifest["name"],
        "source": str(args.source),
        "source_revision": manifest["source"]["revision"],
        "splits": {},
    }
    rows_by_split = {}
    for split, source_rows in (("quick", quick_source), ("full", full_source)):
        rows = [convert_verified_row(row, split=split) for row in source_rows]
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
