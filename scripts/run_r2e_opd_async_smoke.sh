#!/usr/bin/env bash
set -euo pipefail

RUNTIME_ROOT="${RUNTIME_ROOT:-/personal/coding_opd_runtime}"
SPLIT_ROOT="${SPLIT_ROOT:-${RUNTIME_ROOT}/datasets/r2e_opd_v1}"

# Validated eight-GPU asynchronous baseline:
# 4 FSDP trainer/hybrid-rollout GPUs + 2 standalone rollout GPUs + 2 teacher replicas.
# This wrapper is the shared optimized core for both R2E and SWE-Smith launchers.
export TRAIN_FILE="${TRAIN_FILE:-${SPLIT_ROOT}/debug.parquet}"
export VAL_FILE="${VAL_FILE:-${SPLIT_ROOT}/dev.parquet}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export TRAINER_MODE="${TRAINER_MODE:-separate_async}"
export ACTOR_GPUS="${ACTOR_GPUS:-4}"
export ROLLOUT_GPUS="${ROLLOUT_GPUS:-2}"
export TEACHER_GPUS="${TEACHER_GPUS:-2}"
export ROLLOUT_TP="${ROLLOUT_TP:-1}"
export TEACHER_TP="${TEACHER_TP:-1}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-16}"
export PARAMETER_SYNC_STEP="${PARAMETER_SYNC_STEP:-2}"
export MAX_OFF_POLICY_THRESHOLD="${MAX_OFF_POLICY_THRESHOLD:-2}"
export MAX_OFF_POLICY_STRATEGY="${MAX_OFF_POLICY_STRATEGY:-drop}"
export ASYNC_WARMUP_BATCHES="${ASYNC_WARMUP_BATCHES:-1}"
export HYBRID_ROLLOUT_ENABLE_SWITCH="${HYBRID_ROLLOUT_ENABLE_SWITCH:-true}"
export ROLLOUT_CORRECTION_BYPASS="${ROLLOUT_CORRECTION_BYPASS:-true}"
# Two batches are intentional: step 1 includes warm-up/JIT, while step 2 is the
# first useful steady-state throughput sample.
export ROLLOUT_BUDGET="${ROLLOUT_BUDGET:-64}"
export AGENT_WORKERS="${AGENT_WORKERS:-32}"
export MAX_CONCURRENT_SESSIONS="${MAX_CONCURRENT_SESSIONS:-32}"
export ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.8}"
# Keep all eight GPUs productive while leaving bounded headroom on the two
# standalone rollout GPUs for small, bursty colocated inference workloads.
export STANDALONE_ROLLOUT_GPU_MEMORY_UTILIZATION="${STANDALONE_ROLLOUT_GPU_MEMORY_UTILIZATION:-0.7}"
# An 8 GiB bucket reduced steady-state B300 weight synchronization from 30.07s
# to 18.75s versus 4 GiB with the same multi-sender topology.
export UPDATE_WEIGHTS_BUCKET_MEGABYTES="${UPDATE_WEIGHTS_BUCKET_MEGABYTES:-8192}"
# Keep NCCL as the compatibility baseline. delta_sharded provides a bit-exact
# sparse FSDP2 -> vLLM transfer after its initial dense seed sync.
export CHECKPOINT_ENGINE_BACKEND="${CHECKPOINT_ENGINE_BACKEND:-nccl}"
# Multi-sender is faster on the B300 NV18 domain: every actor rank relays the
# full-weight broadcast to the standalone rollout replicas in parallel.
export NCCL_CHECKPOINT_MULTI_SENDER="${NCCL_CHECKPOINT_MULTI_SENDER:-true}"
export TEACHER_GPU_MEMORY_UTILIZATION="${TEACHER_GPU_MEMORY_UTILIZATION:-0.5}"
export TEACHER_ENFORCE_EAGER="${TEACHER_ENFORCE_EAGER:-false}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-32768}"
export TEACHER_MAX_NUM_BATCHED_TOKENS="${TEACHER_MAX_NUM_BATCHED_TOKENS:-49152}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
export TEACHER_MAX_NUM_SEQS="${TEACHER_MAX_NUM_SEQS:-16}"
export USE_DYNAMIC_BSZ="${USE_DYNAMIC_BSZ:-true}"
# 32K exhausts a B300 during the Qwen3.5-9B MLP forward without gradient
# checkpointing; 24K keeps the faster no-checkpoint path with safe headroom.
export PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-24576}"
export LOG_PROB_MAX_TOKEN_LEN_PER_GPU="${LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-131072}"
export ENABLE_GRADIENT_CHECKPOINTING="${ENABLE_GRADIENT_CHECKPOINTING:-false}"
export ACTOR_USE_TORCH_COMPILE="${ACTOR_USE_TORCH_COMPILE:-true}"
# Keep the extra actor fast paths opt-in. On the B300 smoke, enabling Liger,
# fused AdamW, forward prefetch, and deferred gradient sync together regressed
# steady actor latency from 0.137 to 0.196 ms/token.
export ACTOR_USE_LIGER="${ACTOR_USE_LIGER:-false}"
export ACTOR_USE_FUSED_KERNELS="${ACTOR_USE_FUSED_KERNELS:-true}"
export ACTOR_FUSED_KERNELS_BACKEND="${ACTOR_FUSED_KERNELS_BACKEND:-triton}"
export ACTOR_FUSED_ADAMW="${ACTOR_FUSED_ADAMW:-false}"
# The 9B actor fits comfortably without post-forward resharding on B300. Keeping
# full parameters resident avoids repeated FSDP all-gathers across packed
# microbatches (7.7% faster in the identical-trajectory 24K actor A/B).
export ACTOR_RESHARD_AFTER_FORWARD="${ACTOR_RESHARD_AFTER_FORWARD:-false}"
export ACTOR_FORWARD_PREFETCH="${ACTOR_FORWARD_PREFETCH:-false}"
# PyTorch 2.11 crashes when deferred sync touches an unused FSDP2 parameter
# group. coding_opd's actor plugin limits deferral to modules that actually ran.
export ACTOR_NO_SYNC_GRAD_ACCUMULATION="${ACTOR_NO_SYNC_GRAD_ACCUMULATION:-false}"
export ROLLOUT_CUDAGRAPH_MODE="${ROLLOUT_CUDAGRAPH_MODE:-FULL_DECODE_ONLY}"
export ROLLOUT_CUDAGRAPH_CAPTURE_SIZES="${ROLLOUT_CUDAGRAPH_CAPTURE_SIZES:-[1,2,4,8,16,24,32,48,64]}"
export TEACHER_CUDAGRAPH_MODE="${TEACHER_CUDAGRAPH_MODE:-FULL_DECODE_ONLY}"
export TEACHER_CUDAGRAPH_CAPTURE_SIZES="${TEACHER_CUDAGRAPH_CAPTURE_SIZES:-[1,2,4,8,12,16]}"
export ROLLOUT_CALCULATE_LOG_PROBS="${ROLLOUT_CALCULATE_LOG_PROBS:-false}"
export VLLM_DISABLE_COMPILE_CACHE="${VLLM_DISABLE_COMPILE_CACHE:-0}"
export CODING_OPD_VLLM_COMPILE_CACHE_ROOT="${CODING_OPD_VLLM_COMPILE_CACHE_ROOT:-/var/lib/coding-opd-vllm-compile-cache}"
export ACTOR_PAD_TO_LENGTH="${ACTOR_PAD_TO_LENGTH:-true}"
export ACTOR_PAD_TO_LENGTH_BUCKET="${ACTOR_PAD_TO_LENGTH_BUCKET:-1024}"
export RUN_NAME="${RUN_NAME:-r2e_opd_async_smoke_$(date +%Y%m%d_%H%M%S)}"

exec bash "$(dirname "$0")/run_r2e_opd_smoke.sh" "$@"
