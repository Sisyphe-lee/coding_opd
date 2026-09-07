#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/personal/coding_opd}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/personal/coding_opd_runtime}"
ACTOR_PERF_VARIANT="${ACTOR_PERF_VARIANT:-gc32k}"

case "${ACTOR_PERF_VARIANT}" in
    gc32k)
        export PPO_MAX_TOKEN_LEN_PER_GPU=32768
        export ENABLE_GRADIENT_CHECKPOINTING=true
        ;;
    nogc32k)
        export PPO_MAX_TOKEN_LEN_PER_GPU=32768
        export ENABLE_GRADIENT_CHECKPOINTING=false
        ;;
    nogc28k)
        export PPO_MAX_TOKEN_LEN_PER_GPU=28672
        export ENABLE_GRADIENT_CHECKPOINTING=false
        ;;
    gc48k)
        export PPO_MAX_TOKEN_LEN_PER_GPU=49152
        export ENABLE_GRADIENT_CHECKPOINTING=true
        ;;
    gc64k)
        export PPO_MAX_TOKEN_LEN_PER_GPU=65536
        export ENABLE_GRADIENT_CHECKPOINTING=true
        ;;
    *)
        echo "Unknown ACTOR_PERF_VARIANT: ${ACTOR_PERF_VARIANT}" >&2
        exit 2
        ;;
esac

# veRL's V1 rollout cache stores the complete TransferQueue batch, including
# sampled responses and teacher logprobs. A missing cache is populated by the
# first run; later variants replay the identical batch for an actor-only A/B.
export SKIP_ROLLOUT_TQ=true
export ROLLOUT_CACHE_ACTION=repeat
export ROLLOUT_CACHE_STEPS='[1]'
export ROLLOUT_CACHE_DIR="${ROLLOUT_CACHE_DIR:-${RUNTIME_ROOT}/rollout_cache}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-swesmith_actor_perf_cache_v1}"
export RUN_NAME="${RUN_NAME:-swesmith_actor_perf_${ACTOR_PERF_VARIANT}_$(date +%Y%m%d_%H%M%S)}"

# Actor A/Bs do not need the separate-async standalone rollout replicas. Sync V1
# consumes the same cached TransferQueue batch and initializes only the colocated
# rollout needed to populate the cache on the first run.
export TRAINER_MODE=sync
export ROLLOUT_GPUS=0

exec bash "${REPO_ROOT}/scripts/run_swesmith_opd_async_smoke.sh" "$@"
