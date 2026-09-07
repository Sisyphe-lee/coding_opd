#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from datasets import Dataset

from coding_opd.r2e_data import convert_row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--max-samples", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()

    dataset = Dataset.from_parquet(str(args.source))
    if args.max_samples >= 0:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    if args.repeat < 1:
        raise ValueError("--repeat must be at least 1")
    converted_rows = [convert_row(row) for row in dataset]
    converted = Dataset.from_list(converted_rows * args.repeat)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    converted.to_parquet(str(args.output))
    print(f"wrote {len(converted)} R2E samples to {args.output}")


if __name__ == "__main__":
    main()
