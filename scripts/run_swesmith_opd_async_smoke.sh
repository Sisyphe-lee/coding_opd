#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/personal/coding_opd}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/personal/coding_opd_runtime}"

export TRAIN_FILE="${TRAIN_FILE:-${RUNTIME_ROOT}/datasets/swesmith_opd_v1/debug_repeat2.parquet}"
export VAL_FILE="${VAL_FILE:-${TRAIN_FILE}}"
export TASK_CONFIG="${TASK_CONFIG:-${REPO_ROOT}/configs/swesmith_react.yaml}"
export TASK_RUNNER_FQN="coding_opd.swesmith_task.run_swesmith_task"
export DISTILLATION_KEY="swe_smith"
export RUN_TASK_EVALUATION="${RUN_TASK_EVALUATION:-false}"
export RUN_NAME="${RUN_NAME:-swesmith_opd_async_smoke_$(date +%Y%m%d_%H%M%S)}"

exec bash "${REPO_ROOT}/scripts/run_r2e_opd_async_smoke.sh" "$@"
