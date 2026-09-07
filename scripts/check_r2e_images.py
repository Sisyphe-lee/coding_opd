#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from coding_opd.r2e_images import images_from_parquet, local_images, missing_images


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Fail fast when an R2E split needs uncached images.")
    parser.add_argument("parquet", help="Converted Coding OPD parquet to inspect")
    parser.add_argument("--runtime", default=str(repo_root / "scripts" / "podman_sandbox"))
    parser.add_argument("--show-missing", type=int, default=20)
    parser.add_argument(
        "--min-rows",
        type=int,
        default=1,
        help="Fail when the parquet cannot supply one configured training batch",
    )
    args = parser.parse_args()

    import pyarrow.parquet as pq

    row_count = pq.ParquetFile(args.parquet).metadata.num_rows
    required = images_from_parquet(args.parquet)
    available = local_images(args.runtime)
    missing = missing_images(required, available)
    summary = {
        "parquet": str(Path(args.parquet).resolve()),
        "rows": row_count,
        "min_rows": args.min_rows,
        "required": len(required),
        "available_required": len(required) - len(missing),
        "missing": len(missing),
        "missing_examples": missing[: args.show_missing],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if row_count < args.min_rows:
        raise SystemExit(
            f"training parquet has {row_count} rows, fewer than train_batch_size={args.min_rows}"
        )
    if missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
