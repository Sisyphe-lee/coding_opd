#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/personal/coding_opd}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/personal/coding_opd_runtime}"
STUDENT_MODEL="${STUDENT_MODEL:-${RUNTIME_ROOT}/models/Qwen3.5-9B}"
TEACHER_MODEL="${TEACHER_MODEL:-${RUNTIME_ROOT}/models/Qwen3.8-27B}"
TRAIN_FILE="${TRAIN_FILE:-${RUNTIME_ROOT}/datasets/r2e_opd_smoke.parquet}"
VAL_FILE="${VAL_FILE:-${TRAIN_FILE}}"
TASK_CONFIG="${TASK_CONFIG:-${REPO_ROOT}/configs/r2e_react.yaml}"
LOG_DIR="${LOG_DIR:-${RUNTIME_ROOT}/logs}"
RUN_NAME="${RUN_NAME:-r2e_opd_smoke_$(date +%Y%m%d_%H%M%S)}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-${RUN_NAME}}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${RUNTIME_ROOT}/checkpoints/${RUN_NAME}}"
SAVE_FREQ="${SAVE_FREQ:--1}"
RESUME_MODE="${RESUME_MODE:-disable}"
RESUME_FROM_PATH="${RESUME_FROM_PATH:-}"
MAX_ACTOR_CKPT_TO_KEEP="${MAX_ACTOR_CKPT_TO_KEEP:-null}"
ACTOR_CHECKPOINT_SAVE_CONTENTS="${ACTOR_CHECKPOINT_SAVE_CONTENTS:-[model,optimizer,extra]}"
ACTOR_CHECKPOINT_LOAD_CONTENTS="${ACTOR_CHECKPOINT_LOAD_CONTENTS:-[model,optimizer,extra]}"

ACTOR_GPUS="${ACTOR_GPUS:-2}"
TEACHER_GPUS="${TEACHER_GPUS:-2}"
ROLLOUT_GPUS="${ROLLOUT_GPUS:-0}"
ROLLOUT_TP="${ROLLOUT_TP:-1}"
TEACHER_TP="${TEACHER_TP:-2}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-16384}"
MAX_MODEL_LEN=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + 1))
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-${TRAIN_BATCH_SIZE}}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-}"
ROLLOUT_BUDGET="${ROLLOUT_BUDGET:-}"
DATA_SEED="${DATA_SEED:-42}"
TRAINER_MODE="${TRAINER_MODE:-sync}"
PARAMETER_SYNC_STEP="${PARAMETER_SYNC_STEP:-1}"
MAX_OFF_POLICY_THRESHOLD="${MAX_OFF_POLICY_THRESHOLD:-1}"
MAX_OFF_POLICY_STRATEGY="${MAX_OFF_POLICY_STRATEGY:-drop}"
ASYNC_WARMUP_BATCHES="${ASYNC_WARMUP_BATCHES:-1}"
HYBRID_ROLLOUT_ENABLE_SWITCH="${HYBRID_ROLLOUT_ENABLE_SWITCH:-false}"
ROLLOUT_CORRECTION_BYPASS="${ROLLOUT_CORRECTION_BYPASS:-false}"
AGENT_WORKERS="${AGENT_WORKERS:-1}"
MAX_CONCURRENT_SESSIONS="${MAX_CONCURRENT_SESSIONS:-1}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.35}"
STANDALONE_ROLLOUT_GPU_MEMORY_UTILIZATION="${STANDALONE_ROLLOUT_GPU_MEMORY_UTILIZATION:-${ROLLOUT_GPU_MEMORY_UTILIZATION}}"
UPDATE_WEIGHTS_BUCKET_MEGABYTES="${UPDATE_WEIGHTS_BUCKET_MEGABYTES:-2048}"
CHECKPOINT_ENGINE_BACKEND="${CHECKPOINT_ENGINE_BACKEND:-nccl}"
DELTA_SHARDED_ENCODING="${DELTA_SHARDED_ENCODING:-indices}"
NCCL_CHECKPOINT_MULTI_SENDER="${NCCL_CHECKPOINT_MULTI_SENDER:-true}"
TEACHER_GPU_MEMORY_UTILIZATION="${TEACHER_GPU_MEMORY_UTILIZATION:-0.35}"
TEACHER_ENFORCE_EAGER="${TEACHER_ENFORCE_EAGER:-true}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
TEACHER_MAX_NUM_BATCHED_TOKENS="${TEACHER_MAX_NUM_BATCHED_TOKENS:-${MAX_NUM_BATCHED_TOKENS}}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1024}"
TEACHER_MAX_NUM_SEQS="${TEACHER_MAX_NUM_SEQS:-${MAX_NUM_SEQS}}"
USE_DYNAMIC_BSZ="${USE_DYNAMIC_BSZ:-false}"
PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-${MAX_MODEL_LEN}}"
LOG_PROB_MAX_TOKEN_LEN_PER_GPU="${LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-${PPO_MAX_TOKEN_LEN_PER_GPU}}"
ENABLE_GRADIENT_CHECKPOINTING="${ENABLE_GRADIENT_CHECKPOINTING:-true}"
ACTOR_USE_TORCH_COMPILE="${ACTOR_USE_TORCH_COMPILE:-false}"
ACTOR_USE_LIGER="${ACTOR_USE_LIGER:-false}"
ACTOR_USE_FUSED_KERNELS="${ACTOR_USE_FUSED_KERNELS:-false}"
ACTOR_FUSED_KERNELS_BACKEND="${ACTOR_FUSED_KERNELS_BACKEND:-torch}"
ACTOR_FUSED_ADAMW="${ACTOR_FUSED_ADAMW:-false}"
ACTOR_RESHARD_AFTER_FORWARD="${ACTOR_RESHARD_AFTER_FORWARD:-true}"
ACTOR_FORWARD_PREFETCH="${ACTOR_FORWARD_PREFETCH:-false}"
ACTOR_NO_SYNC_GRAD_ACCUMULATION="${ACTOR_NO_SYNC_GRAD_ACCUMULATION:-false}"
ROLLOUT_CUDAGRAPH_MODE="${ROLLOUT_CUDAGRAPH_MODE:-FULL_AND_PIECEWISE}"
ROLLOUT_CUDAGRAPH_CAPTURE_SIZES="${ROLLOUT_CUDAGRAPH_CAPTURE_SIZES:-null}"
TEACHER_CUDAGRAPH_MODE="${TEACHER_CUDAGRAPH_MODE:-FULL_AND_PIECEWISE}"
TEACHER_CUDAGRAPH_CAPTURE_SIZES="${TEACHER_CUDAGRAPH_CAPTURE_SIZES:-null}"
ROLLOUT_CALCULATE_LOG_PROBS="${ROLLOUT_CALCULATE_LOG_PROBS:-true}"
VLLM_DISABLE_COMPILE_CACHE="${VLLM_DISABLE_COMPILE_CACHE:-1}"
CODING_OPD_VLLM_COMPILE_CACHE_ROOT="${CODING_OPD_VLLM_COMPILE_CACHE_ROOT:-}"
ACTOR_SEQUENCE_PARALLEL_SIZE="${ACTOR_SEQUENCE_PARALLEL_SIZE:-1}"
ACTOR_PAD_TO_LENGTH="${ACTOR_PAD_TO_LENGTH:-false}"
ACTOR_PAD_TO_LENGTH_BUCKET="${ACTOR_PAD_TO_LENGTH_BUCKET:-1024}"
SKIP_ROLLOUT_TQ="${SKIP_ROLLOUT_TQ:-false}"
ROLLOUT_CACHE_DIR="${ROLLOUT_CACHE_DIR:-${RUNTIME_ROOT}/rollout_cache}"
ROLLOUT_CACHE_ACTION="${ROLLOUT_CACHE_ACTION:-repeat}"
ROLLOUT_CACHE_STEPS="${ROLLOUT_CACHE_STEPS:-[1]}"
IMAGE_PREFLIGHT="${IMAGE_PREFLIGHT:-true}"
IMAGE_PREFLIGHT_SCRIPT="${IMAGE_PREFLIGHT_SCRIPT:-${REPO_ROOT}/scripts/check_r2e_images.py}"
TASK_RUNNER_FQN="${TASK_RUNNER_FQN:-coding_opd.r2e_task.run_r2e_task}"
DISTILLATION_KEY="${DISTILLATION_KEY:-r2e_gym}"
RUN_TASK_EVALUATION="${RUN_TASK_EVALUATION:-true}"

case "${RESUME_MODE}" in
    disable|auto)
        if [[ -n "${RESUME_FROM_PATH}" ]]; then
            echo "RESUME_FROM_PATH is only valid when RESUME_MODE=resume_path" >&2
            exit 2
        fi
        ;;
    resume_path)
        if [[ -z "${RESUME_FROM_PATH}" || ! -d "${RESUME_FROM_PATH}" ]]; then
            echo "RESUME_MODE=resume_path requires an existing RESUME_FROM_PATH directory" >&2
            exit 2
        fi
        if [[ "$(basename "${RESUME_FROM_PATH}")" != global_step_* ]]; then
            echo "RESUME_FROM_PATH must end in global_step_<N>" >&2
            exit 2
        fi
        ;;
    *)
        echo "Unsupported RESUME_MODE: ${RESUME_MODE}" >&2
        exit 2
        ;;
esac

if [[ ! "${SAVE_FREQ}" =~ ^-?[0-9]+$ ]] || (( SAVE_FREQ == 0 || SAVE_FREQ < -1 )); then
    echo "SAVE_FREQ must be -1 (disabled) or a positive integer" >&2
    exit 2
fi

# TransferQueue 0.1.9 has no queue checkpoint API. With async warmup/prefetch,
# its dataloader can advance past trajectories that have not reached an actor
# update, so a saved dataloader state would skip those rows after restart.
# A zero-warmup checkpoint run keeps exactly one submitted batch in flight.
if [[ "${TRAINER_MODE}" == "separate_async" && "${SAVE_FREQ}" -gt 0 && "${ASYNC_WARMUP_BATCHES}" -gt 0 ]]; then
    if ! "${PYTHON_BIN}" - <<'PY'
import transfer_queue as tq
raise SystemExit(0 if hasattr(tq, "save_checkpoint") and hasattr(tq, "load_checkpoint") else 1)
PY
    then
        echo "separate_async checkpointing with ASYNC_WARMUP_BATCHES>0 requires TransferQueue save/load support; use ASYNC_WARMUP_BATCHES=0 for the controlled resume run" >&2
        exit 2
    fi
fi

if [[ -n "${ROLLOUT_BUDGET}" ]]; then
    if [[ -n "${TOTAL_TRAINING_STEPS}" ]]; then
        echo "Set only one of ROLLOUT_BUDGET or TOTAL_TRAINING_STEPS" >&2
        exit 2
    fi
    if (( ROLLOUT_BUDGET < 1 || ROLLOUT_BUDGET % TRAIN_BATCH_SIZE != 0 )); then
        echo "ROLLOUT_BUDGET must be positive and divisible by TRAIN_BATCH_SIZE" >&2
        exit 2
    fi
    TOTAL_TRAINING_STEPS=$((ROLLOUT_BUDGET / TRAIN_BATCH_SIZE))
fi

if [[ -n "${TOTAL_TRAINING_STEPS}" && ! "${TOTAL_TRAINING_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "TOTAL_TRAINING_STEPS must be a positive integer" >&2
    exit 2
fi
if [[ -n "${TOTAL_EPOCHS}" && ! "${TOTAL_EPOCHS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "TOTAL_EPOCHS must be a positive integer" >&2
    exit 2
fi

# veRL V1 stops when either total_training_steps or total_epochs is reached.
# When a step budget is supplied, make the epoch guard conservatively large
# enough that a small frozen pool can cycle until the requested step. The
# explicit step budget remains the exact stopping condition.
if [[ -n "${TOTAL_TRAINING_STEPS}" ]]; then
    if [[ -z "${TOTAL_EPOCHS}" ]] || (( TOTAL_EPOCHS < TOTAL_TRAINING_STEPS )); then
        TOTAL_EPOCHS="${TOTAL_TRAINING_STEPS}"
    fi
elif [[ -z "${TOTAL_EPOCHS}" ]]; then
    TOTAL_EPOCHS=1
fi

TRAINING_BUDGET_ARGS=()
if [[ -n "${TOTAL_TRAINING_STEPS}" ]]; then
    TRAINING_BUDGET_ARGS+=("trainer.total_training_steps=${TOTAL_TRAINING_STEPS}")
fi

ACTOR_OPTIMIZER_ARGS=()
if [[ "${ACTOR_FUSED_ADAMW}" == "true" ]]; then
    ACTOR_OPTIMIZER_ARGS+=("actor_rollout_ref.actor.optim.override_optimizer_config={fused:true}")
fi

TRAINER_MODE_ARGS=()
CHECKPOINT_ENGINE_ARGS=()
case "${TRAINER_MODE}" in
    sync)
        ;;
    separate_async)
        if (( ROLLOUT_GPUS < 1 )); then
            echo "ROLLOUT_GPUS must be positive for separate_async" >&2
            exit 2
        fi
        if (( TRAIN_BATCH_SIZE != PARAMETER_SYNC_STEP * PPO_MINI_BATCH_SIZE )); then
            echo "separate_async requires TRAIN_BATCH_SIZE = PARAMETER_SYNC_STEP * PPO_MINI_BATCH_SIZE" >&2
            exit 2
        fi
        TRAINER_MODE_ARGS+=(
            "actor_rollout_ref.rollout.nnodes=1"
            "actor_rollout_ref.rollout.n_gpus_per_node=${ROLLOUT_GPUS}"
            "actor_rollout_ref.rollout.checkpoint_engine.backend=${CHECKPOINT_ENGINE_BACKEND}"
            "trainer.v1.separate_async.parameter_sync_step=${PARAMETER_SYNC_STEP}"
            "trainer.v1.separate_async.num_warmup_batches=${ASYNC_WARMUP_BATCHES}"
            "trainer.v1.separate_async.hybrid_rollout.enable_switch=${HYBRID_ROLLOUT_ENABLE_SWITCH}"
            "trainer.v1.sampler.max_off_policy_threshold=${MAX_OFF_POLICY_THRESHOLD}"
            "trainer.v1.sampler.max_off_policy_strategy=${MAX_OFF_POLICY_STRATEGY}"
        )
        if [[ "${CHECKPOINT_ENGINE_BACKEND}" == "delta_sharded" ]]; then
            CHECKPOINT_ENGINE_ARGS+=(
                "+actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs.delta_sharded.encoding=${DELTA_SHARDED_ENCODING}"
            )
        elif [[ "${CHECKPOINT_ENGINE_BACKEND}" == "nccl" ]]; then
            CHECKPOINT_ENGINE_ARGS+=(
                "actor_rollout_ref.rollout.checkpoint_engine.custom_backend_module=coding_opd.nccl_checkpoint_plugin"
                "+actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs.nccl.multi_sender=${NCCL_CHECKPOINT_MULTI_SENDER}"
            )
        fi
        ;;
    *)
        echo "Unsupported TRAINER_MODE: ${TRAINER_MODE}" >&2
        exit 2
        ;;
esac

mkdir -p "${LOG_DIR}" "${CHECKPOINT_DIR}"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}/third_party/verl:${REPO_ROOT}/third_party/uni-agent:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export RAY_DEDUP_LOGS=0
export TRANSFER_QUEUE_ENABLE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
# The B300 reports sm_103a, which the container's CUDA 12.8 nvcc cannot JIT.
# Keep vLLM on prebuilt FlashAttention/Triton paths instead of FlashInfer JIT.
export VLLM_USE_FLASHINFER_SAMPLER=0
# The node masks B300 as L20D. veRL's tables use dense BF16 Tensor Core peak;
# HGX B300 is 18 PFLOPS dense across eight GPUs, or 2250 TFLOPS per GPU.
export CODING_OPD_DEVICE_PEAK_TFLOPS="${CODING_OPD_DEVICE_PEAK_TFLOPS:-2250}"
# The baseline uses sampled-token reverse KL; top-k is intentionally not configured.

if [[ "${IMAGE_PREFLIGHT}" == "true" ]]; then
    "${PYTHON_BIN}" "${IMAGE_PREFLIGHT_SCRIPT}" "${TRAIN_FILE}" --min-rows "${TRAIN_BATCH_SIZE}"
fi

"${PYTHON_BIN}" -m coding_opd.train_entrypoint \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.rollout_correction.bypass_mode="${ROLLOUT_CORRECTION_BYPASS}" \
    data.train_files="['${TRAIN_FILE}']" \
    data.val_files="['${VAL_FILE}']" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.shuffle=True \
    data.seed="${DATA_SEED}" \
    data.validation_shuffle=False \
    data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH}" \
    data.filter_overlong_prompts=False \
    data.truncation=error \
    actor_rollout_ref.model.path="${STUDENT_MODEL}" \
    actor_rollout_ref.model.external_lib=coding_opd.qwen35_kernel_plugin \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing="${ENABLE_GRADIENT_CHECKPOINTING}" \
    actor_rollout_ref.model.use_liger="${ACTOR_USE_LIGER}" \
    actor_rollout_ref.model.use_fused_kernels="${ACTOR_USE_FUSED_KERNELS}" \
    actor_rollout_ref.model.fused_kernel_options.impl_backend="${ACTOR_FUSED_KERNELS_BACKEND}" \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    "${ACTOR_OPTIMIZER_ARGS[@]}" \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU}" \
    actor_rollout_ref.actor.use_dynamic_bsz="${USE_DYNAMIC_BSZ}" \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.calculate_entropy=False \
    actor_rollout_ref.actor.use_torch_compile="${ACTOR_USE_TORCH_COMPILE}" \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size="${ACTOR_SEQUENCE_PARALLEL_SIZE}" \
    actor_rollout_ref.actor.fsdp_config.pad_to_length="${ACTOR_PAD_TO_LENGTH}" \
    actor_rollout_ref.actor.fsdp_config.pad_to_length_bucket="${ACTOR_PAD_TO_LENGTH_BUCKET}" \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.reshard_after_forward="${ACTOR_RESHARD_AFTER_FORWARD}" \
    actor_rollout_ref.actor.fsdp_config.forward_prefetch="${ACTOR_FORWARD_PREFETCH}" \
    actor_rollout_ref.actor.fsdp_config.use_no_sync_for_gradient_accumulation="${ACTOR_NO_SYNC_GRAD_ACCUMULATION}" \
    actor_rollout_ref.actor.checkpoint.save_contents="${ACTOR_CHECKPOINT_SAVE_CONTENTS}" \
    actor_rollout_ref.actor.checkpoint.load_contents="${ACTOR_CHECKPOINT_LOAD_CONTENTS}" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz="${USE_DYNAMIC_BSZ}" \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="${LOG_PROB_MAX_TOKEN_LEN_PER_GPU}" \
    actor_rollout_ref.ref.use_torch_compile=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}" \
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION}" \
    +actor_rollout_ref.rollout.standalone_gpu_memory_utilization="${STANDALONE_ROLLOUT_GPU_MEMORY_UTILIZATION}" \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.max_model_len="${MAX_MODEL_LEN}" \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens="${MAX_NUM_BATCHED_TOKENS}" \
    actor_rollout_ref.rollout.max_num_seqs="${MAX_NUM_SEQS}" \
    actor_rollout_ref.rollout.disable_log_stats=False \
    actor_rollout_ref.rollout.calculate_log_probs="${ROLLOUT_CALCULATE_LOG_PROBS}" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz="${USE_DYNAMIC_BSZ}" \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${LOG_PROB_MAX_TOKEN_LEN_PER_GPU}" \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes="${UPDATE_WEIGHTS_BUCKET_MEGABYTES}" \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.attention_backend=FLASH_ATTN \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.gdn_prefill_backend=triton \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_mode="${ROLLOUT_CUDAGRAPH_MODE}" \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_capture_sizes="${ROLLOUT_CUDAGRAPH_CAPTURE_SIZES}" \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1 \
    actor_rollout_ref.rollout.multi_turn.format=qwen3_coder \
    actor_rollout_ref.rollout.agent.num_workers="${AGENT_WORKERS}" \
    +actor_rollout_ref.rollout.agent.agent_loop_manager_class=coding_opd.opd_framework.OPDAgentFrameworkRolloutAdapter \
    +actor_rollout_ref.rollout.custom.agent_framework.gateway_count=1 \
    +actor_rollout_ref.rollout.custom.agent_framework.log_dir="${LOG_DIR}/agents" \
    +actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_fqn="${TASK_RUNNER_FQN}" \
    +actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.dispatch_mode=ray_task \
    +actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.max_concurrent_sessions="${MAX_CONCURRENT_SESSIONS}" \
    +actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.session_timeout_seconds=1200 \
    +actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs.task_config_path="${TASK_CONFIG}" \
    +actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs.model_name=Qwen3.5-9B \
    +actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs.report_reward=True \
    +actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs.run_evaluation="${RUN_TASK_EVALUATION}" \
    +actor_rollout_ref.rollout.custom.agent_framework.mask_unfinished_episode=False \
    +actor_rollout_ref.rollout.custom.agent_framework.use_reward_loop_worker=False \
    distillation.enabled=True \
    distillation.n_gpus_per_node="${TEACHER_GPUS}" \
    distillation.nnodes=1 \
    distillation.teacher_models.teacher_model.key="${DISTILLATION_KEY}" \
    distillation.teacher_models.teacher_model.model_path="${TEACHER_MODEL}" \
    distillation.teacher_models.teacher_model.inference.name=vllm \
    distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size="${TEACHER_TP}" \
    distillation.teacher_models.teacher_model.inference.gpu_memory_utilization="${TEACHER_GPU_MEMORY_UTILIZATION}" \
    distillation.teacher_models.teacher_model.inference.enforce_eager="${TEACHER_ENFORCE_EAGER}" \
    distillation.teacher_models.teacher_model.inference.max_model_len="${MAX_MODEL_LEN}" \
    distillation.teacher_models.teacher_model.inference.enable_chunked_prefill=True \
    distillation.teacher_models.teacher_model.inference.max_num_batched_tokens="${TEACHER_MAX_NUM_BATCHED_TOKENS}" \
    distillation.teacher_models.teacher_model.inference.max_num_seqs="${TEACHER_MAX_NUM_SEQS}" \
    distillation.teacher_models.teacher_model.inference.disable_log_stats=False \
    +distillation.teacher_models.teacher_model.inference.engine_kwargs.vllm.attention_backend=FLASH_ATTN \
    +distillation.teacher_models.teacher_model.inference.engine_kwargs.vllm.gdn_prefill_backend=triton \
    +distillation.teacher_models.teacher_model.inference.engine_kwargs.vllm.compilation_config.cudagraph_mode="${TEACHER_CUDAGRAPH_MODE}" \
    +distillation.teacher_models.teacher_model.inference.engine_kwargs.vllm.compilation_config.cudagraph_capture_sizes="${TEACHER_CUDAGRAPH_CAPTURE_SIZES}" \
    +distillation.teacher_models.teacher_model.inference.engine_kwargs.vllm.disable_custom_all_reduce=True \
    distillation.distillation_loss.loss_mode=k3 \
    distillation.distillation_loss.use_task_rewards=False \
    distillation.distillation_loss.use_policy_gradient=False \
    skip.rollout_tq.enable="${SKIP_ROLLOUT_TQ}" \
    skip.rollout_tq.dump_dir="${ROLLOUT_CACHE_DIR}" \
    skip.rollout_tq.action="${ROLLOUT_CACHE_ACTION}" \
    skip.rollout_tq.steps="${ROLLOUT_CACHE_STEPS}" \
    trainer.n_gpus_per_node="${ACTOR_GPUS}" \
    trainer.nnodes=1 \
    trainer.logger="['console']" \
    trainer.project_name=coding_opd \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.default_local_dir="${CHECKPOINT_DIR}" \
    trainer.val_before_train=False \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.resume_mode="${RESUME_MODE}" \
    trainer.resume_from_path="${RESUME_FROM_PATH:-null}" \
    trainer.max_actor_ckpt_to_keep="${MAX_ACTOR_CKPT_TO_KEEP}" \
    trainer.test_freq=-1 \
    trainer.total_epochs="${TOTAL_EPOCHS}" \
    trainer.v1.trainer_mode="${TRAINER_MODE}" \
    +ray_kwargs.ray_init.runtime_env.env_vars.CODING_OPD_DEVICE_PEAK_TFLOPS="'${CODING_OPD_DEVICE_PEAK_TFLOPS}'" \
    +ray_kwargs.ray_init.runtime_env.env_vars.VLLM_USE_FLASHINFER_SAMPLER="'${VLLM_USE_FLASHINFER_SAMPLER}'" \
    +ray_kwargs.ray_init.runtime_env.env_vars.VLLM_DISABLE_COMPILE_CACHE="'${VLLM_DISABLE_COMPILE_CACHE}'" \
    +ray_kwargs.ray_init.runtime_env.env_vars.CODING_OPD_VLLM_COMPILE_CACHE_ROOT="'${CODING_OPD_VLLM_COMPILE_CACHE_ROOT}'" \
    "${TRAINER_MODE_ARGS[@]}" \
    "${CHECKPOINT_ENGINE_ARGS[@]}" \
    "${TRAINING_BUDGET_ARGS[@]}" \
    "$@" 2>&1 | tee "${LOG_DIR}/${RUN_NAME}.log"
