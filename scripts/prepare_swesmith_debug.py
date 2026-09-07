#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from coding_opd.swesmith_data import (
    DEFAULT_SOURCE,
    add_repeated_debug_fixture,
    load_swesmith_source,
    select_debug_tasks,
    write_debug_bundle,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a small, image-efficient SWE-smith debug set")
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--source-label", default=None, help="Stable source recorded in manifest")
    parser.add_argument("--source-split", default="train")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--size", type=int, default=16)
    parser.add_argument("--repositories", type=int, default=4)
    parser.add_argument(
        "--repeat-factor",
        type=int,
        default=2,
        help="Also write debug_repeat<N>.parquet for the optimized batch-32 smoke",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repo", action="append", dest="repos")
    parser.add_argument(
        "--reuse-existing-debug",
        action="store_true",
        help="Keep the frozen debug task IDs and only regenerate the repeated fixture",
    )
    args = parser.parse_args()

    if args.reuse_existing_debug:
        manifest = add_repeated_debug_fixture(args.output_dir, repeat_factor=args.repeat_factor)
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return

    dataset = load_swesmith_source(args.source, split=args.source_split)
    # The canonical source contains a small number of rows with no usable
    # prompt. Match the frozen evaluation-data filter before strict validation
    # and deterministic selection.
    eligible_rows = (row for row in dataset if str(row.get("problem_statement") or "").strip())
    rows = select_debug_tasks(
        eligible_rows,
        size=args.size,
        repositories=args.repositories,
        seed=args.seed,
        include_repos=args.repos,
    )
    manifest = write_debug_bundle(
        rows,
        args.output_dir,
        source=args.source_label or args.source,
        seed=args.seed,
        repeat_factor=args.repeat_factor,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
