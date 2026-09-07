#!/usr/bin/env python3
"""Materialize the frozen three-tier Coding OPD evaluation bundle.

The output is intentionally remote-data friendly: parquet files contain only
the task fields needed by the agent/evaluator, while ``manifest.json`` records
the ordered IDs, repository/image counts, and hashes that define the bundle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from datasets import Dataset, load_dataset

from coding_opd.swesmith_data import (
    build_eval_splits,
    convert_row as convert_swesmith_row,
    load_swesmith_source,
)


def _rank(seed: int, label: str, value: str) -> bytes:
    return hashlib.sha256(f"{seed}\0{label}\0{value}".encode()).digest()


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        parsed = json.loads(value)
        return [str(item) for item in parsed]
    return [str(item) for item in value]


def _load_rows(source: str, *, split: str) -> list[dict[str, Any]]:
    path = Path(source)
    if path.is_file():
        if path.suffix == ".parquet":
            return [dict(row) for row in Dataset.from_parquet(str(path))]
        if path.suffix in {".json", ".jsonl"}:
            if path.suffix == ".json":
                payload = json.loads(path.read_text(encoding="utf-8"))
                return [dict(row) for row in payload]
            return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    dataset = load_dataset(source, split=split)
    return [dict(row) for row in dataset]


def _task_id_from_row(row: dict[str, Any]) -> str:
    """Read IDs from either raw SWE-smith rows or converted OPD parquets."""
    if row.get("instance_id") is not None:
        return str(row["instance_id"])
    extra_info = row.get("extra_info")
    if isinstance(extra_info, dict) and extra_info.get("task_id") is not None:
        return str(extra_info["task_id"])
    tools_kwargs = extra_info.get("tools_kwargs") if isinstance(extra_info, dict) else None
    task = tools_kwargs.get("task") if isinstance(tools_kwargs, dict) else None
    metadata = task.get("metadata") if isinstance(task, dict) else None
    if isinstance(metadata, dict) and metadata.get("instance_id") is not None:
        return str(metadata["instance_id"])
    raise ValueError("training exclusion row has no instance_id or converted task_id")


def _select_verified(rows: Iterable[dict[str, Any]], *, fast_size: int, full_size: int, seed: int) -> dict[str, list[dict[str, Any]]]:
    rows = list(rows)
    required = {"instance_id", "repo", "problem_statement", "base_commit", "patch", "test_patch", "version"}
    missing = required - rows[0].keys() if rows else required
    if missing:
        raise ValueError(f"Verified rows are missing required columns: {sorted(missing)}")
    if not 0 < fast_size <= full_size <= len(rows):
        raise ValueError(f"expected 0 < fast_size <= full_size <= {len(rows)}")

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: set[str] = set()
    for row in rows:
        instance_id = str(row["instance_id"])
        if instance_id in seen:
            raise ValueError(f"duplicate Verified instance_id: {instance_id}")
        seen.add(instance_id)
        groups[str(row["repo"])].append(row)

    # Keep the pilot proportional to the official test distribution.  The
    # round-robin ordering makes the nested fast prefix cover many repositories.
    total = len(rows)
    exact = {repo: full_size * len(repo_rows) / total for repo, repo_rows in groups.items()}
    quotas = {repo: min(len(groups[repo]), int(value)) for repo, value in exact.items()}
    remaining = full_size - sum(quotas.values())
    for repo in sorted(groups, key=lambda item: (-(exact[item] - quotas[item]), item)):
        if remaining == 0:
            break
        if quotas[repo] < len(groups[repo]):
            quotas[repo] += 1
            remaining -= 1

    chosen: dict[str, list[dict[str, Any]]] = {}
    for repo in sorted(quotas):
        ranked = sorted(groups[repo], key=lambda row: _rank(seed, "verified-task", str(row["instance_id"])))
        chosen[repo] = ranked[: quotas[repo]]
    repo_order = sorted(chosen, key=lambda repo: _rank(seed, "verified-repo", repo))
    ordered: list[dict[str, Any]] = []
    for offset in range(max(map(len, chosen.values()))):
        for repo in repo_order:
            if offset < len(chosen[repo]):
                ordered.append(chosen[repo][offset])
    return {"fast": ordered[:fast_size], "full": ordered}


def _verified_image(instance_id: str) -> str:
    return f"swebench/sweb.eval.x86_64.{instance_id.lower().replace('__', '_1776_')}"


def _convert_verified_row(row: dict[str, Any], split: str) -> dict[str, Any]:
    instance_id = str(row["instance_id"])
    metadata = {
        "instance_id": instance_id,
        "repo": str(row["repo"]),
        "version": str(row["version"]),
        "base_commit": str(row["base_commit"]),
        # These are evaluation-only fields. They are never included in the
        # prompt and the Verified runner uses them only after the agent exits.
        "patch": str(row["patch"]),
        "test_patch": str(row["test_patch"]),
        "problem_statement": str(row["problem_statement"]),
        # The native Uni-Agent reward parser consumes the canonical JSON form.
        "FAIL_TO_PASS": json.dumps(_as_list(row.get("FAIL_TO_PASS"))),
        "PASS_TO_PASS": json.dumps(_as_list(row.get("PASS_TO_PASS"))),
    }
    return {
        "data_source": "SWE-bench/SWE-bench_Verified",
        "prompt": [{"role": "user", "content": metadata["problem_statement"]}],
        "extra_info": {
            "task_id": instance_id,
            "split": split,
            "tools_kwargs": {
                "task": {
                    "name": "swe_bench",
                    "sandbox": {"image": _verified_image(instance_id)},
                    "metadata": metadata,
                }
            },
        },
    }


def _write_parquet(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(rows).to_parquet(str(path))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _describe(rows: list[dict[str, Any]], *, id_key: str, repo_key: str, image_getter) -> dict[str, Any]:
    return {
        "count": len(rows),
        "repo_counts": dict(sorted(Counter(str(row[repo_key]) for row in rows).items())),
        "task_ids": [str(row[id_key]) for row in rows],
        "images": sorted({str(image_getter(row)) for row in rows}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--swesmith-source", required=True, help="SWE-smith dataset ID or local parquet")
    parser.add_argument("--swesmith-split", default="train")
    parser.add_argument("--exclude-swesmith-source", help="Training parquet whose instance IDs must be excluded")
    parser.add_argument("--verified-source", default="SWE-bench/SWE-bench_Verified")
    parser.add_argument("--verified-split", default="test")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repositories", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    smith = load_swesmith_source(args.swesmith_source, split=args.swesmith_split)
    smith_rows = [dict(row) for row in smith if str(row.get("problem_statement") or "").strip()]
    exclude_ids: set[str] = set()
    if args.exclude_swesmith_source:
        training = load_swesmith_source(args.exclude_swesmith_source, split="train")
        exclude_ids = {_task_id_from_row(dict(row)) for row in training}
    smith_splits = build_eval_splits(
        smith_rows,
        repositories=args.repositories,
        seed=args.seed,
        exclude_ids=exclude_ids,
    )

    verified_rows = _load_rows(args.verified_source, split=args.verified_split)
    verified_splits = _select_verified(verified_rows, fast_size=8, full_size=50, seed=args.seed)

    manifest: dict[str, Any] = {
        "format_version": 1,
        "protocol": "legacy-swesmith-eval-v1",
        "status": "historical bundle; not part of docs/datasets.md",
        "seed": args.seed,
        "sources": {
            "swesmith": {"source": args.swesmith_source, "split": args.swesmith_split},
            "verified": {"source": args.verified_source, "split": args.verified_split},
        },
        "excludes": {"swesmith_instance_ids": sorted(exclude_ids)},
        "filters": {"swesmith_nonempty_problem_statement": True, "swesmith_rows_after_filter": len(smith_rows)},
        "nesting": {
            "smoke_swesmith": "fast_swesmith",
            "fast_swesmith": "full_swesmith",
            "fast_verified": "full_verified",
        },
        "splits": {},
    }

    for tier, rows in smith_splits.items():
        converted = [convert_swesmith_row(row, data_split=tier) for row in rows]
        path = args.output_dir / tier / "swesmith.parquet"
        _write_parquet(converted, path)
        info = _describe(rows, id_key="instance_id", repo_key="repo", image_getter=lambda row: row["image_name"])
        info["path"] = str(path.relative_to(args.output_dir))
        info["sha256"] = _sha256(path)
        manifest["splits"][f"{tier}_swesmith"] = info

    for tier, rows in verified_splits.items():
        converted = [_convert_verified_row(row, tier) for row in rows]
        path = args.output_dir / tier / "verified.parquet"
        _write_parquet(converted, path)
        info = _describe(rows, id_key="instance_id", repo_key="repo", image_getter=lambda row: _verified_image(str(row["instance_id"])))
        info["path"] = str(path.relative_to(args.output_dir))
        info["sha256"] = _sha256(path)
        manifest["splits"][f"{tier}_verified"] = info

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({name: info["count"] for name, info in manifest["splits"].items()}, sort_keys=True))


if __name__ == "__main__":
    main()
