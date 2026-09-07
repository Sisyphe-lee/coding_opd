from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_checkpoint_resume_launcher_runs_fresh_then_explicit_resume(tmp_path: Path) -> None:
    split = tmp_path / "datasets" / "r2e_train_128"
    split.mkdir(parents=True)
    (split / "train.parquet").write_bytes(b"fixture")
    calls = tmp_path / "calls.jsonl"
    fake_train = tmp_path / "fake_train.sh"
    fake_train.write_text(
        """#!/usr/bin/env bash
set -eu
step=$((ROLLOUT_BUDGET / TRAIN_BATCH_SIZE))
mkdir -p "$CHECKPOINT_DIR/global_step_$step"
printf '%s\\n' "{\\"mode\\":\\"$RESUME_MODE\\",\\"budget\\":$ROLLOUT_BUDGET,\\"resume\\":\\"${RESUME_FROM_PATH:-}\\"}" >> "$CALLS_PATH"
""",
        encoding="utf-8",
    )
    fake_train.chmod(0o755)
    fake_validator = tmp_path / "fake_validator.py"
    fake_validator.write_text("raise SystemExit(0)\n", encoding="utf-8")

    environment = os.environ | {
        "REPO_ROOT": str(ROOT),
        "RUNTIME_ROOT": str(tmp_path),
        "SPLIT_ROOT": str(split),
        "PYTHON_BIN": sys.executable,
        "TRAIN_LAUNCHER": str(fake_train),
        "CHECKPOINT_VALIDATOR": str(fake_validator),
        "CHECKPOINT_DIR": str(tmp_path / "checkpoints" / "run"),
        "RUN_NAME": "test-resume",
        "EXPORT_HF_AFTER_RESUME": "false",
        "CALLS_PATH": str(calls),
    }
    result = subprocess.run(
        ["bash", str(ROOT / "scripts" / "run_r2e128_checkpoint_resume.sh")],
        env=environment,
        text=True,
        capture_output=True,
        check=True,
    )
    recorded = [json.loads(line) for line in calls.read_text(encoding="utf-8").splitlines()]
    assert recorded == [
        {"mode": "disable", "budget": 32, "resume": ""},
        {
            "mode": "resume_path",
            "budget": 64,
            "resume": str(tmp_path / "checkpoints" / "run" / "global_step_1"),
        },
    ]
    assert "eval_model=not_exported" in result.stdout
