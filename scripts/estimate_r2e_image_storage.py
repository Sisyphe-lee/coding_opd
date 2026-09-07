#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

from coding_opd.r2e_images import images_from_parquet


def read_manifest(crane: str, image: str, timeout: float) -> dict:
    result = subprocess.run(
        [crane, "manifest", image],
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        return {"image": image, "error": result.stderr.strip()[-500:]}
    manifest = json.loads(result.stdout)
    if "layers" not in manifest:
        return {"image": image, "error": "manifest index is not a platform image"}
    return {
        "image": image,
        "layers": [(layer["digest"], int(layer["size"])) for layer in manifest["layers"]],
    }


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Estimate registry bytes after layer-digest deduplication.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--parquet")
    source.add_argument("--stdin", action="store_true", help="Read a JSON array of image names from stdin")
    parser.add_argument("--crane", default=str(repo_root / ".tools" / "crane" / "bin" / "crane"))
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=90)
    args = parser.parse_args()

    images = images_from_parquet(args.parquet) if args.parquet else sorted(set(json.load(sys.stdin)))
    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        results = list(executor.map(lambda image: read_manifest(args.crane, image, args.timeout), images))

    errors = [result for result in results if "error" in result]
    unique_layers: dict[str, int] = {}
    image_sizes: list[tuple[str, int]] = []
    for result in results:
        if "layers" not in result:
            continue
        image_sizes.append((result["image"], sum(size for _, size in result["layers"])))
        for digest, size in result["layers"]:
            previous = unique_layers.setdefault(digest, size)
            if previous != size:
                raise RuntimeError(f"inconsistent size for {digest}: {previous} != {size}")

    per_image_bytes = [size for _, size in image_sizes]
    naive_bytes = sum(per_image_bytes)
    unique_bytes = sum(unique_layers.values())

    def percentile(values: list[int], fraction: float) -> int:
        ordered = sorted(values)
        if not ordered:
            return 0
        return ordered[round((len(ordered) - 1) * fraction)]

    repositories: dict[str, list[int]] = defaultdict(list)
    for image, size in image_sizes:
        repository = image.split("/", 1)[-1].split(":", 1)[0].removesuffix("_final")
        repositories[repository].append(size)

    by_repository = {
        repository: {
            "images": len(sizes),
            "mean_compressed_bytes": sum(sizes) / len(sizes),
            "median_compressed_bytes": percentile(sizes, 0.5),
            "min_compressed_bytes": min(sizes),
            "max_compressed_bytes": max(sizes),
        }
        for repository, sizes in sorted(repositories.items())
    }
    summary = {
        "images": len(images),
        "manifests_read": len(per_image_bytes),
        "manifest_errors": len(errors),
        "error_examples": errors[:5],
        "naive_compressed_bytes": naive_bytes,
        "unique_compressed_layer_bytes": unique_bytes,
        "unique_layers": len(unique_layers),
        "mean_image_compressed_bytes": naive_bytes / len(per_image_bytes) if per_image_bytes else 0,
        "median_image_compressed_bytes": percentile(per_image_bytes, 0.5),
        "p10_image_compressed_bytes": percentile(per_image_bytes, 0.1),
        "p90_image_compressed_bytes": percentile(per_image_bytes, 0.9),
        "min_image_compressed_bytes": min(per_image_bytes, default=0),
        "max_image_compressed_bytes": max(per_image_bytes, default=0),
        "by_repository": by_repository,
        "dedup_ratio": naive_bytes / unique_bytes if unique_bytes else 0,
        "elapsed_seconds": time.monotonic() - started,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
