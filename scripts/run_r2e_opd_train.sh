#!/usr/bin/env bash
set -euo pipefail

RUNTIME_ROOT="${RUNTIME_ROOT:-/personal/coding_opd_runtime}"
SPLIT_ROOT="${SPLIT_ROOT:-${RUNTIME_ROOT}/datasets/r2e_train_512}"

export TRAIN_FILE="${TRAIN_FILE:-${SPLIT_ROOT}/train.parquet}"
export VAL_FILE="${VAL_FILE:-${TRAIN_FILE}}"
export ROLLOUT_BUDGET="${ROLLOUT_BUDGET:-8192}"
export RUN_NAME="${RUN_NAME:-r2e_opd_train_$(date +%Y%m%d_%H%M%S)}"

# Research baseline: one batch, one optimizer update, then weight sync. Prefetch
# exactly one following batch while Actor updates; measured policy lag stays 1.
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
export PPO_MINI_BATCH_SIZE="${TRAIN_BATCH_SIZE}"
export PARAMETER_SYNC_STEP=1
export OPD_SYNC_ROLLOUTS=false
export MAX_OFF_POLICY_THRESHOLD=2
export ASYNC_WARMUP_BATCHES=0
export ASYNC_PREFETCH=true
export HYBRID_ROLLOUT_ENABLE_SWITCH=false
# 32K prefix OPD: keep the full sequence in one Actor microbatch.
export ACTOR_GPUS="${ACTOR_GPUS:-2}"
export ROLLOUT_GPUS="${ROLLOUT_GPUS:-4}"
export TEACHER_GPUS="${TEACHER_GPUS:-2}"
export ENABLE_GRADIENT_CHECKPOINTING="${ENABLE_GRADIENT_CHECKPOINTING:-true}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-32768}"
export PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-32768}"
export SAVE_FREQ="${SAVE_FREQ:-64}"

exec bash "$(dirname "$0")/run_r2e_opd_async_smoke.sh" "$@"
