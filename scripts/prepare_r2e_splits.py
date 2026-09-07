#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from coding_opd.r2e_data import build_r2e_splits, load_r2e_source, write_split_bundle


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create deterministic, repository-stratified R2E OPD splits."
    )
    parser.add_argument(
        "source",
        help="Local parquet file/directory/glob or Hugging Face dataset id.",
    )
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--source-split", default="train")
    parser.add_argument("--train-size", type=int, default=2048)
    parser.add_argument("--dev-size", type=int, default=128)
    parser.add_argument("--pilot-size", type=int, default=128)
    parser.add_argument("--debug-size", type=int, default=16)
    parser.add_argument("--smoke-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset = load_r2e_source(args.source, split=args.source_split)
    splits = build_r2e_splits(
        dataset,
        train_size=args.train_size,
        dev_size=args.dev_size,
        pilot_size=args.pilot_size,
        debug_size=args.debug_size,
        smoke_size=args.smoke_size,
        seed=args.seed,
    )
    manifest = write_split_bundle(
        splits,
        args.output_dir,
        source=args.source,
        source_split=args.source_split,
        seed=args.seed,
    )
    counts = {name: info["count"] for name, info in manifest["splits"].items()}
    print(f"wrote R2E OPD split bundle to {args.output_dir}: {counts}")


if __name__ == "__main__":
    main()
