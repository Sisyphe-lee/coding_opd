from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import re
import tomllib
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from coding_opd.dataset_selection import id_sha256


VERIFIED_REQUIRED_FIELDS = {
    "instance_id",
    "repo",
    "problem_statement",
    "base_commit",
    "patch",
    "test_patch",
    "version",
}


def select_canary_rows(
    rows: Sequence[Mapping[str, Any]], *, count: int, sample_seed: int | None
) -> list[dict[str, Any]]:
    """Match Pier's deterministic shuffle-then-truncate subset semantics."""
    selected_rows = [dict(row) for row in rows]
    if sample_seed is not None:
        random.Random(sample_seed).shuffle(selected_rows)
    return copy.deepcopy(selected_rows[:count])


def _as_json_list(value: Any) -> str:
    if value is None:
        parsed: list[Any] = []
    elif isinstance(value, str):
        loaded = json.loads(value)
        if not isinstance(loaded, list):
            raise ValueError(f"expected a JSON list, got {type(loaded).__name__}")
        parsed = loaded
    else:
        parsed = list(value)
    return json.dumps([str(item) for item in parsed])


def verified_image(instance_id: str) -> str:
    return f"swebench/sweb.eval.x86_64.{instance_id.lower().replace('__', '_1776_')}"


def select_verified_manifest_rows(
    rows: Iterable[Mapping[str, Any]], manifest: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pool = [dict(row) for row in rows]
    if len(pool) != int(manifest["source"]["full_count"]):
        raise ValueError(
            f"expected {manifest['source']['full_count']} Verified rows, found {len(pool)}"
        )
    missing = VERIFIED_REQUIRED_FIELDS if not pool else VERIFIED_REQUIRED_FIELDS - pool[0].keys()
    if missing:
        raise ValueError(f"Verified rows are missing required columns: {sorted(missing)}")

    by_id: dict[str, dict[str, Any]] = {}
    for row in pool:
        task_id = str(row["instance_id"])
        if task_id in by_id:
            raise ValueError(f"duplicate Verified instance_id: {task_id}")
        by_id[task_id] = row

    quick_ids = [str(task["task_id"]) for task in manifest["tasks"]]
    if id_sha256(quick_ids) != manifest["task_id_sha256"]:
        raise ValueError("Verified manifest task ID hash does not match its ordered tasks")
    missing_ids = [task_id for task_id in quick_ids if task_id not in by_id]
    if missing_ids:
        raise ValueError(f"Verified source is missing manifest tasks: {missing_ids[:5]}")
    return [by_id[task_id] for task_id in quick_ids], pool


def convert_verified_row(row: Mapping[str, Any], *, split: str) -> dict[str, Any]:
    instance_id = str(row["instance_id"])
    problem = str(row["problem_statement"])
    metadata = {
        "instance_id": instance_id,
        "repo": str(row["repo"]),
        "version": str(row["version"]),
        "base_commit": str(row["base_commit"]),
        "patch": str(row["patch"]),
        "test_patch": str(row["test_patch"]),
        "problem_statement": problem,
        "FAIL_TO_PASS": _as_json_list(row.get("FAIL_TO_PASS")),
        "PASS_TO_PASS": _as_json_list(row.get("PASS_TO_PASS")),
    }
    return {
        "data_source": "SWE-bench/SWE-bench_Verified",
        "prompt": [{"role": "user", "content": problem}],
        "extra_info": {
            "task_id": instance_id,
            "split": split,
            "tools_kwargs": {
                "task": {
                    "name": "swe_bench",
                    "sandbox": {"image": verified_image(instance_id)},
                    "metadata": metadata,
                }
            },
        },
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(root: Path) -> str:
    """Hash a directory independent of mtimes, ownership, and archive format."""
    digest = hashlib.sha256()
    files = sorted(path for path in root.rglob("*") if path.is_file())
    for path in files:
        relative = path.relative_to(root).as_posix().encode()
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def deepswe_verifier_image(task_id: str) -> str:
    safe = re.sub(r"[^a-z0-9_.-]+", "-", task_id.lower()).strip("-.")
    if not safe:
        raise ValueError(f"cannot derive verifier image tag from task ID {task_id!r}")
    return f"coding-opd/deepswe-verifier:v1.1-{safe}"


def portable_deepswe_verifier_dockerfile(context: Path) -> str:
    """Remove only the redundant RUN that requires unavailable nested cgroups."""
    dockerfile = context / "Dockerfile"
    lines = dockerfile.read_text(encoding="utf-8").splitlines()
    while lines and not lines[-1].strip():
        lines.pop()
    expected_run = "RUN chmod +x /tests/test.sh"
    if not lines or lines[-1].strip() != expected_run:
        raise ValueError(f"unexpected final verifier Dockerfile instruction: {dockerfile}")
    test_script = context / "test.sh"
    if not test_script.is_file() or not os.access(test_script, os.X_OK):
        raise ValueError(f"verifier test.sh must already be executable: {test_script}")
    return "\n".join(lines[:-1]) + "\n"


def load_deepswe_task_ids(official_tasks_json: Path, manifest: Mapping[str, Any]) -> list[str]:
    actual_hash = sha256_file(official_tasks_json)
    expected_hash = str(manifest["source"]["official_tasks_sha256"])
    if actual_hash != expected_hash:
        raise ValueError(f"DeepSWE tasks.json hash mismatch: expected {expected_hash}, found {actual_hash}")
    payload = json.loads(official_tasks_json.read_text(encoding="utf-8"))
    rows = payload.get("rows", [])
    expected_count = int(manifest["source"]["official_full_count"])
    if int(payload.get("n_tasks", -1)) != expected_count or len(rows) != expected_count:
        raise ValueError(f"expected exactly {expected_count} DeepSWE tasks")
    task_ids = [str(row["id"]) for row in rows]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("DeepSWE tasks.json has duplicate task IDs")
    return task_ids


def load_deepswe_task(task_dir: Path) -> dict[str, Any]:
    config_path = task_dir / "task.toml"
    instruction_path = task_dir / "instruction.md"
    tests_dir = task_dir / "tests"
    if not config_path.is_file() or not instruction_path.is_file() or not tests_dir.is_dir():
        raise ValueError(f"incomplete DeepSWE task directory: {task_dir}")
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    metadata = config.get("metadata", {})
    environment = config.get("environment", {})
    verifier = config.get("verifier", {})
    task_id = str(metadata.get("task_id") or task_dir.name)
    base_commit = str(metadata.get("base_commit_hash") or "")
    if task_id != task_dir.name:
        raise ValueError(f"DeepSWE task directory/name mismatch: {task_dir.name} != {task_id}")
    if re.fullmatch(r"[0-9a-f]{7,40}", base_commit) is None:
        raise ValueError(f"invalid DeepSWE base commit for {task_id}: {base_commit!r}")
    if verifier.get("environment_mode") != "separate":
        raise ValueError(f"DeepSWE task {task_id} must use a separate verifier environment")
    image = str(environment.get("docker_image") or "")
    if not image:
        raise ValueError(f"DeepSWE task {task_id} has no agent image")
    return {
        "task_id": task_id,
        "base_commit_hash": base_commit,
        "language": str(metadata.get("language") or ""),
        "repository": str(metadata.get("repository_url") or ""),
        "instruction": instruction_path.read_text(encoding="utf-8"),
        "agent_image": image,
        "verifier_image": deepswe_verifier_image(task_id),
    }


def materialize_deepswe_rows(
    task_root: Path,
    task_ids: Sequence[str],
    *,
    split: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task_id in task_ids:
        task = load_deepswe_task(task_root / task_id)
        rows.append(
            {
                "data_source": "DataCurve/DeepSWE-v1.1",
                "prompt": [{"role": "user", "content": task["instruction"]}],
                "extra_info": {
                    "task_id": task_id,
                    "split": split,
                    "tools_kwargs": {
                        "task": {
                            "name": "deep_swe",
                            "sandbox": {"image": task["agent_image"]},
                            "metadata": {
                                key: value
                                for key, value in task.items()
                                if key != "agent_image"
                            },
                        }
                    },
                },
            }
        )
    return rows


def describe_eval_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    task_ids = [str(row["extra_info"]["task_id"]) for row in rows]
    sources = Counter(str(row["data_source"]) for row in rows)
    return {
        "count": len(rows),
        "data_source_counts": dict(sorted(sources.items())),
        "task_ids": task_ids,
        "task_id_sha256": id_sha256(task_ids),
    }


def agent_images_from_rows(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    images: list[str] = []
    for row in rows:
        try:
            image = str(row["extra_info"]["tools_kwargs"]["task"]["sandbox"]["image"])
        except (KeyError, TypeError) as error:
            raise ValueError("every evaluation row must contain a sandbox image") from error
        if not image:
            raise ValueError("evaluation sandbox image cannot be empty")
        images.append(image)
    if len(images) != len(set(images)):
        raise ValueError("external evaluation requires one unique agent image per task")
    return images
