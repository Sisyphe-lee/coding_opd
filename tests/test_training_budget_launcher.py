from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_step_budget_is_not_truncated_by_epoch_guard(tmp_path: Path) -> None:
    calls = tmp_path / "python-args.txt"
    fake_python = tmp_path / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\nprintf '%s\\n' \"$@\" > \"$CALLS_PATH\"\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    train_file = tmp_path / "train.parquet"
    train_file.write_bytes(b"fixture")

    environment = os.environ | {
        "REPO_ROOT": str(ROOT),
        "RUNTIME_ROOT": str(tmp_path),
        "PYTHON_BIN": str(fake_python),
        "CALLS_PATH": str(calls),
        "TRAIN_FILE": str(train_file),
        "VAL_FILE": str(train_file),
        "CHECKPOINT_DIR": str(tmp_path / "checkpoints"),
        "LOG_DIR": str(tmp_path / "logs"),
        "IMAGE_PREFLIGHT": "false",
        "TRAINER_MODE": "sync",
        "TRAIN_BATCH_SIZE": "32",
        "ROLLOUT_BUDGET": "512",
        "TOTAL_EPOCHS": "1",
        "RUN_NAME": "budget-test",
    }
    subprocess.run(
        ["bash", str(ROOT / "scripts" / "run_r2e_opd_smoke.sh")],
        env=environment,
        text=True,
        capture_output=True,
        check=True,
    )

    arguments = calls.read_text(encoding="utf-8").splitlines()
    assert "trainer.total_training_steps=16" in arguments
    assert "trainer.total_epochs=16" in arguments
