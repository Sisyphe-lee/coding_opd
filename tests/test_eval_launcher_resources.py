import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).parents[1]


def dry_run(**overrides):
    env = dict(os.environ)
    for key in ("TEMPERATURE", "SAMPLING_CONFIG_JSON"):
        env.pop(key, None)
    env.update(
        REPO_ROOT=str(ROOT), PYTHON_BIN=sys.executable, DRY_RUN="true",
        MODEL_PATH="/not/materialized/global_step_200/actor/huggingface",
        CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7", TENSOR_PARALLEL_SIZE="1",
        TASKS_PER_REPLICA="4", MAX_NUM_SEQS="4", SAMPLING_PROFILE="student",
    )
    env.update(overrides)
    return subprocess.run(
        ["bash", str(ROOT / "scripts/run_deepswe_codex_eval.sh")],
        env=env, text=True, capture_output=True, timeout=10,
    )


@pytest.mark.parametrize("tp,replicas,total", [("1", 8, 32), ("2", 4, 16), ("4", 2, 8)])
def test_gpu_groups_and_total_task_slots(tp, replicas, total):
    completed = dry_run(TENSOR_PARALLEL_SIZE=tp)
    assert completed.returncode == 0, completed.stderr
    layout = json.loads(completed.stdout)
    assert layout["replica_count"] == replicas
    assert layout["total_task_concurrency"] == total
    assert all(len(group) == int(tp) for group in layout["gpu_groups"])
    assert sum(layout["gpu_groups"], []) == list(map(str, range(8)))
    assert layout["sampling_config"]["temperature"] == 0.6
    assert layout["speculative_decoding"] is False


def test_teacher_profile_and_explicit_temperature():
    result = dry_run(SAMPLING_PROFILE="teacher")
    assert json.loads(result.stdout)["sampling_config"]["temperature"] == 1.0
    result = dry_run(SAMPLING_PROFILE="student", TEMPERATURE="0.7")
    assert json.loads(result.stdout)["sampling_config"]["temperature"] == 0.7


@pytest.mark.parametrize("overrides", [
    {"TENSOR_PARALLEL_SIZE": "3"},
    {"TENSOR_PARALLEL_SIZE": "0"},
    {"CUDA_VISIBLE_DEVICES": "0,0"},
    {"CUDA_VISIBLE_DEVICES": "0,"},
    {"TASKS_PER_REPLICA": "-1"},
    {"SAMPLING_PROFILE": "auto"},
])
def test_invalid_resource_layout_fails_before_launch(overrides):
    assert dry_run(**overrides).returncode != 0
