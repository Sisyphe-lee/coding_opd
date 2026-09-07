#!/usr/bin/env bash
set -euo pipefail

RUNTIME_ROOT="${RUNTIME_ROOT:-/personal/coding_opd_runtime}"
SPLIT_ROOT="${SPLIT_ROOT:-${RUNTIME_ROOT}/datasets/r2e_train_128}"

export TRAIN_FILE="${TRAIN_FILE:-${SPLIT_ROOT}/train.parquet}"
export VAL_FILE="${VAL_FILE:-${TRAIN_FILE}}"
export ROLLOUT_BUDGET="${ROLLOUT_BUDGET:-128}"
export RUN_NAME="${RUN_NAME:-r2e_opd_pilot_$(date +%Y%m%d_%H%M%S)}"

exec bash "$(dirname "$0")/run_r2e_opd_async_smoke.sh" "$@"
