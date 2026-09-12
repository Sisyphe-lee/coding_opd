from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("algorithm", ["vanilla", "adaptive"])
def test_step_budget_is_not_truncated_by_epoch_guard(tmp_path: Path, algorithm: str) -> None:
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
        "CODING_OPD_PODMAN_SERVICE": "false",
        "TRAINER_MODE": "separate_async",
        "OPD_ALGORITHM": algorithm,
        "ROLLOUT_GPUS": "3",
        "PPO_MINI_BATCH_SIZE": "16",
        "PARAMETER_SYNC_STEP": "2",
        "ASYNC_PREFETCH": "true",
        "ASYNC_WARMUP_BATCHES": "0",
        "MAX_OFF_POLICY_THRESHOLD": "2",
        "HYBRID_ROLLOUT_ENABLE_SWITCH": "true",
        "TRAIN_BATCH_SIZE": "32",
        "ROLLOUT_BUDGET": "512",
        "TOTAL_EPOCHS": "1",
        "RUN_NAME": "budget-test",
        "RUN_RECORD_DIR": str(tmp_path / "runs" / "budget-test"),
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
    af = "+actor_rollout_ref.rollout.custom.agent_framework"
    if algorithm == "adaptive":
        assert f"{af}.synchronous_rollouts=true" in arguments
        assert f"{af}.agent_runners.task.runner_kwargs.adaptive_threshold=0.1" in arguments
        assert "+trainer.v1.separate_async.checkpoint_safe_prefetch=false" in arguments
        assert "trainer.v1.separate_async.num_warmup_batches=0" in arguments
        assert "trainer.v1.separate_async.hybrid_rollout.enable_switch=false" in arguments
    else:
        assert f"{af}.synchronous_rollouts=false" in arguments
        assert "+trainer.v1.separate_async.checkpoint_safe_prefetch=true" in arguments
    assert (
        "+actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs.run_evaluation=false"
        in arguments
    )
    assert (
        f"+ray_kwargs.ray_init.runtime_env.env_vars.CODING_OPD_PROFILE_DIR='{tmp_path}/logs/budget-test.profile'"
        in arguments
    )


def test_resume_reuses_checkpoint_run_record(tmp_path: Path) -> None:
    fake_python = tmp_path / "python"
    fake_python.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    fake_python.chmod(0o755)
    train_file = tmp_path / "train.parquet"
    train_file.write_bytes(b"fixture")
    resume_root = tmp_path / "old-checkpoints"
    (resume_root / "global_step_16").mkdir(parents=True)
    (resume_root / "run_record_name").write_text("logical-run\n")
    checkpoint_dir = tmp_path / "new-checkpoints"

    environment = os.environ | {
        "REPO_ROOT": str(ROOT),
        "RUNTIME_ROOT": str(tmp_path),
        "PYTHON_BIN": str(fake_python),
        "TRAIN_FILE": str(train_file),
        "VAL_FILE": str(train_file),
        "CHECKPOINT_DIR": str(checkpoint_dir),
        "LOG_DIR": str(tmp_path / "logs"),
        "RUN_RECORD_DIR": str(tmp_path / "runs" / "logical-run"),
        "IMAGE_PREFLIGHT": "false",
        "CODING_OPD_PODMAN_SERVICE": "false",
        "RESUME_MODE": "resume_path",
        "RESUME_FROM_PATH": str(resume_root / "global_step_16"),
        "RUN_NAME": "resumed-process",
    }
    subprocess.run(
        ["bash", str(ROOT / "scripts" / "run_r2e_opd_smoke.sh")],
        env=environment,
        text=True,
        capture_output=True,
        check=True,
    )

    assert (checkpoint_dir / "run_record_name").read_text() == "logical-run\n"
