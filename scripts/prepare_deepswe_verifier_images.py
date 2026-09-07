#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

from datasets import load_dataset

from coding_opd.eval_data import (
    deepswe_verifier_image,
    portable_deepswe_verifier_dockerfile,
)


def _task_ids(parquet: Path) -> list[str]:
    dataset = load_dataset("parquet", data_files=str(parquet), split="train")
    return [str(row["extra_info"]["task_id"]) for row in dataset]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build or verify network-isolated DeepSWE verifier images.")
    parser.add_argument("task_root", type=Path)
    parser.add_argument("parquet", type=Path)
    parser.add_argument("--docker-binary", default="scripts/podman_sandbox")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    results = []
    for task_id in _task_ids(args.parquet):
        image = deepswe_verifier_image(task_id)
        context = args.task_root / task_id / "tests"
        if not context.is_dir():
            raise ValueError(f"missing verifier build context: {context}")
        inspect = subprocess.run(
            [args.docker_binary, "image", "inspect", image],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if inspect.returncode != 0:
            if args.check_only:
                raise RuntimeError(f"missing DeepSWE verifier image: {image}")
            with tempfile.TemporaryDirectory(prefix="coding-opd-deepswe-verifier-") as temporary:
                portable = Path(temporary) / "Dockerfile"
                portable.write_text(
                    portable_deepswe_verifier_dockerfile(context), encoding="utf-8"
                )
                subprocess.run(
                    [
                        args.docker_binary,
                        "build",
                        "--pull=never",
                        "--network=none",
                        "--file",
                        str(portable),
                        "-t",
                        image,
                        str(context),
                    ],
                    check=True,
                )
        results.append({"image": image, "task_id": task_id})
    print(json.dumps({"count": len(results), "images": results}, sort_keys=True))


if __name__ == "__main__":
    main()
