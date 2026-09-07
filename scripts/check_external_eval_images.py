#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq

from coding_opd.r2e_images import (
    images_from_extra_info,
    local_images,
    missing_images,
    normalize_image_name,
)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Verify all agent and verifier images for an eval parquet.")
    parser.add_argument("parquet", type=Path)
    parser.add_argument("--runtime", default=str(repo_root / "scripts" / "podman_sandbox"))
    args = parser.parse_args()

    rows = pq.read_table(args.parquet, columns=["extra_info"])["extra_info"].to_pylist()
    required = set(images_from_extra_info(rows))
    for row in rows:
        metadata = row["tools_kwargs"]["task"].get("metadata") or {}
        verifier_image = metadata.get("verifier_image")
        if verifier_image:
            required.add(normalize_image_name(str(verifier_image)))
    absent = missing_images(required, local_images(args.runtime))
    print(
        json.dumps(
            {
                "missing": len(absent),
                "missing_examples": absent[:20],
                "required": len(required),
                "rows": len(rows),
            },
            sort_keys=True,
        )
    )
    if absent:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
