#!/usr/bin/env bash
set -euo pipefail

if (( $# > 1 )); then
    echo "usage: $0 [swebench_verified|deepswe]" >&2
    exit 2
fi

BENCHMARK="${1:-${BENCHMARK:-swebench_verified}}"
# An unqualified benchmark run means the complete benchmark.  Quick panels
# must always be requested explicitly.
EVAL_TIER="${EVAL_TIER:-full}"
REPO_ROOT="${REPO_ROOT:-/personal/coding_opd}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/personal/coding_opd_runtime}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/var/lib/coding-opd-vllm-eval-cache}"
MODEL_PATH="${MODEL_PATH:-${RUNTIME_ROOT}/models/Qwen3.5-9B}"
EVAL_ROOT="${EVAL_ROOT:-${RUNTIME_ROOT}/datasets/coding_opd_eval_v2/${BENCHMARK}}"
RESULT_ROOT="${RESULT_ROOT:-${RUNTIME_ROOT}/eval_results}"
RUN_NAME="${RUN_NAME:-${BENCHMARK}_${EVAL_TIER}_$(date +%Y%m%d_%H%M%S)}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-4}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}"
EVAL_CONCURRENCY="${EVAL_CONCURRENCY:-16}"
GATEWAY_COUNT="${GATEWAY_COUNT:-2}"
ROLLOUT_N="${ROLLOUT_N:-1}"
IMAGE_PREFLIGHT="${IMAGE_PREFLIGHT:-true}"

case "${EVAL_TIER}" in
    quick|full) ;;
    *) echo "EVAL_TIER must be quick or full" >&2; exit 2 ;;
esac
case "${BENCHMARK}" in
    swebench_verified)
        TASK_CONFIG="${TASK_CONFIG:-${REPO_ROOT}/configs/swebench_react.yaml}"
        ;;
    deepswe)
        TASK_CONFIG="${TASK_CONFIG:-${REPO_ROOT}/configs/deepswe_react.yaml}"
        ;;
    *)
        echo "Unsupported benchmark: ${BENCHMARK}" >&2
        exit 2
        ;;
esac

DATA_PATH="${DATA_PATH:-${EVAL_ROOT}/${EVAL_TIER}.parquet}"
RUNTIME_MANIFEST="${RUNTIME_MANIFEST:-${EVAL_ROOT}/manifest.json}"
if [[ -z "${RAY_ADDRESS:-}" ]]; then
    echo "RAY_ADDRESS must point to a dedicated Coding OPD Ray head" >&2
    exit 2
fi
case "${RAY_ADDRESS}" in
    127.0.0.1:*|localhost:*) ;;
    *) echo "RAY_ADDRESS must use the dedicated loopback Ray head, got ${RAY_ADDRESS}" >&2; exit 2 ;;
esac
for path in "${DATA_PATH}" "${RUNTIME_MANIFEST}" "${TASK_CONFIG}" "${MODEL_PATH}/config.json"; do
    if [[ ! -e "${path}" ]]; then
        echo "Missing evaluation input: ${path}" >&2
        exit 2
    fi
done
if ! find "${MODEL_PATH}" -maxdepth 1 -type f \( -name '*.safetensors' -o -name '*.bin' \) -print -quit | grep -q .; then
    echo "MODEL_PATH has no HF model weights: ${MODEL_PATH}" >&2
    exit 2
fi

IFS=',' read -r -a eval_devices <<<"${CUDA_VISIBLE_DEVICES}"
N_GPUS="${#eval_devices[@]}"
if (( N_GPUS < 1 || N_GPUS % TENSOR_PARALLEL_SIZE != 0 )); then
    echo "CUDA_VISIBLE_DEVICES count must be positive and divisible by TENSOR_PARALLEL_SIZE" >&2
    exit 2
fi

RESULT_DIR="${RESULT_ROOT}/${RUN_NAME}"
RESULT_PATH="${RESULT_DIR}/result.json"
mkdir -p "${RESULT_DIR}" "${RUNTIME_ROOT}/logs/eval/${RUN_NAME}" "${VLLM_CACHE_ROOT}"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}/third_party/verl:${REPO_ROOT}/third_party/uni-agent:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES
export VLLM_USE_FLASHINFER_SAMPLER=0
export RAY_DEDUP_LOGS=0

if [[ "${IMAGE_PREFLIGHT}" == "true" ]]; then
    "${PYTHON_BIN}" scripts/check_external_eval_images.py "${DATA_PATH}"
elif [[ "${IMAGE_PREFLIGHT}" != "false" ]]; then
    echo "IMAGE_PREFLIGHT must be true or false" >&2
    exit 2
fi

"${PYTHON_BIN}" -m coding_opd.eval_entrypoint \
    --data-path "${DATA_PATH}" \
    --runtime-manifest "${RUNTIME_MANIFEST}" \
    --split "${EVAL_TIER}" \
    --model-path "${MODEL_PATH}" \
    --vllm-cache-root "${VLLM_CACHE_ROOT}" \
    --task-config "${TASK_CONFIG}" \
    --result-path "${RESULT_PATH}" \
    --log-dir "${RUNTIME_ROOT}/logs/eval/${RUN_NAME}/agents" \
    --n "${ROLLOUT_N}" \
    --n-gpus "${N_GPUS}" \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --gateway-count "${GATEWAY_COUNT}" \
    --concurrency "${EVAL_CONCURRENCY}" \
    2>&1 | tee "${RUNTIME_ROOT}/logs/eval/${RUN_NAME}.log"

if [[ "${EVAL_TIER}" == "full" ]]; then
    "${PYTHON_BIN}" scripts/summarize_external_eval.py \
        "${RESULT_PATH}" \
        "${RUNTIME_MANIFEST}" \
        --output "${RESULT_DIR}/summary.json"
fi
