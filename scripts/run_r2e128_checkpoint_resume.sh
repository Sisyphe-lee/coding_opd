#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/personal/coding_opd}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/personal/coding_opd_runtime}"
SPLIT_ROOT="${SPLIT_ROOT:-${RUNTIME_ROOT}/datasets/r2e_train_128}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
TRAIN_LAUNCHER="${TRAIN_LAUNCHER:-${REPO_ROOT}/scripts/run_r2e_opd_async_smoke.sh}"
CHECKPOINT_VALIDATOR="${CHECKPOINT_VALIDATOR:-${REPO_ROOT}/scripts/validate_training_checkpoint.py}"
RUN_NAME="${RUN_NAME:-r2e128_checkpoint_resume_$(date +%Y%m%d_%H%M%S)}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${RUNTIME_ROOT}/checkpoints/${RUN_NAME}}"
SAVE_AFTER_STEP="${SAVE_AFTER_STEP:-1}"
RESUME_TO_STEP="${RESUME_TO_STEP:-2}"
EXPORT_HF_AFTER_RESUME="${EXPORT_HF_AFTER_RESUME:-true}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"

if (( SAVE_AFTER_STEP < 1 || RESUME_TO_STEP <= SAVE_AFTER_STEP )); then
    echo "Require 1 <= SAVE_AFTER_STEP < RESUME_TO_STEP" >&2
    exit 2
fi
if [[ ! -f "${SPLIT_ROOT}/train.parquet" ]]; then
    echo "Missing frozen R2E-128 parquet: ${SPLIT_ROOT}/train.parquet" >&2
    exit 2
fi
if [[ -e "${CHECKPOINT_DIR}/latest_checkpointed_iteration.txt" || -d "${CHECKPOINT_DIR}/global_step_${SAVE_AFTER_STEP}" ]]; then
    echo "Refusing to overwrite an existing checkpoint run: ${CHECKPOINT_DIR}" >&2
    exit 2
fi

common_env=(
    "REPO_ROOT=${REPO_ROOT}"
    "RUNTIME_ROOT=${RUNTIME_ROOT}"
    "TRAIN_FILE=${SPLIT_ROOT}/train.parquet"
    "VAL_FILE=${SPLIT_ROOT}/train.parquet"
    "CHECKPOINT_DIR=${CHECKPOINT_DIR}"
    "SAVE_FREQ=${SAVE_AFTER_STEP}"
    "MAX_ACTOR_CKPT_TO_KEEP=2"
    "TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE}"
    # No look-ahead batch: TransferQueue 0.1.9 cannot persist in-flight work.
    "ASYNC_WARMUP_BATCHES=0"
)

echo "[phase 1/2] training from scratch through global step ${SAVE_AFTER_STEP}"
env "${common_env[@]}" \
    RUN_NAME="${RUN_NAME}_save" \
    TOTAL_TRAINING_STEPS= \
    ROLLOUT_BUDGET="$((SAVE_AFTER_STEP * TRAIN_BATCH_SIZE))" \
    RESUME_MODE=disable \
    ACTOR_CHECKPOINT_SAVE_CONTENTS='[model,optimizer,extra]' \
    ACTOR_CHECKPOINT_LOAD_CONTENTS='[model,optimizer,extra]' \
    bash "${TRAIN_LAUNCHER}"

"${PYTHON_BIN}" "${CHECKPOINT_VALIDATOR}" "${CHECKPOINT_DIR}" \
    --expected-step "${SAVE_AFTER_STEP}"

resume_checkpoint="${CHECKPOINT_DIR}/global_step_${SAVE_AFTER_STEP}"
resume_save_contents='[model,optimizer,extra]'
validator_hf_args=()
if [[ "${EXPORT_HF_AFTER_RESUME}" == "true" ]]; then
    resume_save_contents='[model,optimizer,extra,hf_model]'
    validator_hf_args+=(--require-hf-model)
elif [[ "${EXPORT_HF_AFTER_RESUME}" != "false" ]]; then
    echo "EXPORT_HF_AFTER_RESUME must be true or false" >&2
    exit 2
fi

echo "[phase 2/2] resuming ${resume_checkpoint} through global step ${RESUME_TO_STEP}"
env "${common_env[@]}" \
    RUN_NAME="${RUN_NAME}_resume" \
    TOTAL_TRAINING_STEPS= \
    ROLLOUT_BUDGET="$((RESUME_TO_STEP * TRAIN_BATCH_SIZE))" \
    RESUME_MODE=resume_path \
    RESUME_FROM_PATH="${resume_checkpoint}" \
    ACTOR_CHECKPOINT_SAVE_CONTENTS="${resume_save_contents}" \
    ACTOR_CHECKPOINT_LOAD_CONTENTS='[model,optimizer,extra]' \
    bash "${TRAIN_LAUNCHER}"

"${PYTHON_BIN}" "${CHECKPOINT_VALIDATOR}" "${CHECKPOINT_DIR}" \
    --expected-step "${RESUME_TO_STEP}" "${validator_hf_args[@]}"

if [[ "${EXPORT_HF_AFTER_RESUME}" == "true" ]]; then
    eval_model="${CHECKPOINT_DIR}/global_step_${RESUME_TO_STEP}/actor/huggingface"
else
    eval_model="not_exported"
fi
echo "CHECKPOINT_RESUME_OK root=${CHECKPOINT_DIR} eval_model=${eval_model}"
