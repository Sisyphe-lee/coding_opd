#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/personal/coding_opd}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/personal/coding_opd_runtime}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
MODEL_PATH="${MODEL_PATH:-${RUNTIME_ROOT}/models/Qwen3.5-9B}"
BENCHMARK="${BENCHMARK:-deepswe}"
case "${BENCHMARK}" in
    deepswe) default_task_config=deepswe_codex.yaml; default_eval_bundle=coding_opd_eval_v2 ;;
    swebench_verified) default_task_config=swebench_codex.yaml; default_eval_bundle=coding_opd_eval_v3 ;;
    *) echo "Unsupported benchmark: ${BENCHMARK}" >&2; exit 2 ;;
esac
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$(basename "${MODEL_PATH}")}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
EVAL_TIER="${EVAL_TIER:-canary}"
CANARY_COUNT="${CANARY_COUNT:-5}"
SAMPLE_SEED="${SAMPLE_SEED:-0}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}"
TASKS_PER_REPLICA="${TASKS_PER_REPLICA:-2}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-${TASKS_PER_REPLICA}}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
SAMPLING_PROFILE="${SAMPLING_PROFILE:-student}"
DRY_RUN="${DRY_RUN:-false}"
VLLM_SEED="${VLLM_SEED:-0}"
CONTEXT_WINDOW="${CONTEXT_WINDOW:-262144}"
AUTO_COMPACT_TOKEN_LIMIT="${AUTO_COMPACT_TOKEN_LIMIT:-}"
REASONING_EFFORT="${REASONING_EFFORT:-xhigh}"
REASONING_SUMMARY="${REASONING_SUMMARY:-auto}"
VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/var/lib/coding-opd-vllm-eval-cache}"
MODEL_SOCKET_DIR="${MODEL_SOCKET_DIR:-/var/lib/coding-opd-codex-eval}"
RUN_NAME="${RUN_NAME:-${BENCHMARK}_codex_${SERVED_MODEL_NAME}_${EVAL_TIER}_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUNTIME_ROOT}/eval_results/${RUN_NAME}"
LOG_ROOT="${RUNTIME_ROOT}/logs/eval/${RUN_NAME}"
FULL_ROOT="${EVAL_ROOT:-${RUNTIME_ROOT}/datasets/${default_eval_bundle}/${BENCHMARK}}"
TASK_CONFIG="${TASK_CONFIG:-${REPO_ROOT}/configs/${default_task_config}}"
LANGUAGE_MODEL_ONLY="${LANGUAGE_MODEL_ONLY:-true}"

[[ "${CONTEXT_WINDOW}" =~ ^[1-9][0-9]*$ ]] && (( CONTEXT_WINDOW >= 4096 )) || {
    echo "CONTEXT_WINDOW must be an integer >= 4096" >&2; exit 2;
}
AUTO_COMPACT_TOKEN_LIMIT="${AUTO_COMPACT_TOKEN_LIMIT:-$((CONTEXT_WINDOW * 9 / 10))}"
[[ "${AUTO_COMPACT_TOKEN_LIMIT}" =~ ^[1-9][0-9]*$ ]] && \
    (( AUTO_COMPACT_TOKEN_LIMIT <= CONTEXT_WINDOW * 9 / 10 )) || {
    echo "AUTO_COMPACT_TOKEN_LIMIT must be positive and <= 90% of CONTEXT_WINDOW" >&2; exit 2;
}

# The role is explicit: exported OPD paths are often named "huggingface".
# Never infer sampling settings from the checkpoint directory or service alias.
case "${SAMPLING_PROFILE}" in
    student) profile_temperature=0.6 ;;
    teacher) profile_temperature=1.0 ;;
    *) echo "SAMPLING_PROFILE must be student or teacher" >&2; exit 2 ;;
esac
if [[ -z "${TEMPERATURE:-}" ]]; then
    TEMPERATURE="${profile_temperature}"
fi
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-20}"
MIN_P="${MIN_P:-0.0}"
PRESENCE_PENALTY="${PRESENCE_PENALTY:-0.0}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.0}"
if [[ -z "${SAMPLING_CONFIG_JSON:-}" ]]; then
    SAMPLING_CONFIG_JSON=$(printf \
        '{"temperature":%s,"top_p":%s,"top_k":%s,"min_p":%s,"presence_penalty":%s,"repetition_penalty":%s}' \
        "${TEMPERATURE}" "${TOP_P}" "${TOP_K}" "${MIN_P}" \
        "${PRESENCE_PENALTY}" "${REPETITION_PENALTY}")
fi

case "${EVAL_TIER}" in
    canary|quick|full) ;;
    *) echo "EVAL_TIER must be canary, quick or full" >&2; exit 2 ;;
esac
case "${LANGUAGE_MODEL_ONLY}" in
    true|false) ;;
    *) echo "LANGUAGE_MODEL_ONLY must be true or false" >&2; exit 2 ;;
esac
for count in "${CANARY_COUNT}" "${MAX_NUM_SEQS}" "${TASKS_PER_REPLICA}" \
    "${TENSOR_PARALLEL_SIZE}" "${MAX_NUM_BATCHED_TOKENS}"; do
    [[ "${count}" =~ ^[1-9][0-9]*$ ]] || { echo "Counts must be positive integers" >&2; exit 2; }
done
case "${DRY_RUN}" in
    true|false) ;;
    *) echo "DRY_RUN must be true or false" >&2; exit 2 ;;
esac

IFS=',' read -r -a eval_devices <<<"${CUDA_VISIBLE_DEVICES}"
GPU_COUNT="${#eval_devices[@]}"
if (( GPU_COUNT < 1 || GPU_COUNT % TENSOR_PARALLEL_SIZE != 0 )); then
    echo "GPU count must be positive and divisible by TENSOR_PARALLEL_SIZE" >&2
    exit 2
fi
REPLICA_COUNT=$((GPU_COUNT / TENSOR_PARALLEL_SIZE))
replica_devices=()
for ((offset=0; offset<GPU_COUNT; offset+=TENSOR_PARALLEL_SIZE)); do
    group=("${eval_devices[@]:offset:TENSOR_PARALLEL_SIZE}")
    replica_devices+=("$(IFS=,; echo "${group[*]}")")
done
SERVING_CONFIG_JSON=$("${PYTHON_BIN}" - "${CUDA_VISIBLE_DEVICES}" "${TENSOR_PARALLEL_SIZE}" \
    "${TASKS_PER_REPLICA}" "${MAX_NUM_SEQS}" "${MAX_NUM_BATCHED_TOKENS}" \
    "${GPU_MEMORY_UTILIZATION}" "${SAMPLING_PROFILE}" "${SAMPLING_CONFIG_JSON}" <<'PY'
import json, sys
devices = sys.argv[1].split(',')
if any(not device.strip() or device != device.strip() for device in devices) or len(set(devices)) != len(devices):
    raise SystemExit('GPU devices must be nonempty, unique, and have no whitespace')
tp, tasks, seqs, tokens = map(int, sys.argv[2:6])
memory = float(sys.argv[6])
if not 0 < memory < 1:
    raise SystemExit('GPU_MEMORY_UTILIZATION must be between 0 and 1')
print(json.dumps(dict(gpu_groups=[devices[i:i+tp] for i in range(0, len(devices), tp)],
    tensor_parallel_size=tp, replica_count=len(devices)//tp, tasks_per_replica=tasks,
    total_task_concurrency=len(devices)//tp*tasks, max_num_seqs=seqs,
    max_num_batched_tokens=tokens, gpu_memory_utilization=memory,
    sampling_profile=sys.argv[7], sampling_config=json.loads(sys.argv[8]),
    speculative_decoding=False)))
PY
)
echo "${SERVING_CONFIG_JSON}"
if [[ "${DRY_RUN}" == "true" ]]; then
    exit 0
fi
for path in \
    "${MODEL_PATH}/config.json" \
    "${FULL_ROOT}/full.parquet" \
    "${FULL_ROOT}/manifest.json" \
    "${TASK_CONFIG}" \
    "/opt/codex/vendor/x86_64-unknown-linux-musl/bin/codex"; do
    if [[ ! -e "${path}" ]]; then
        echo "Missing DeepSWE Codex input: ${path}" >&2
        exit 2
    fi
done

mkdir -p "${RUN_ROOT}/records" "${LOG_ROOT}/agents" "${MODEL_SOCKET_DIR}" \
    "${VLLM_CACHE_ROOT}"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}/third_party/verl:${REPO_ROOT}/third_party/uni-agent:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

if [[ "${EVAL_TIER}" == "canary" ]]; then
    CANARY_ROOT="${RUNTIME_ROOT}/datasets/coding_opd_eval_v2/canaries/${RUN_NAME}"
    mkdir -p "${CANARY_ROOT}"
    "${PYTHON_BIN}" scripts/materialize_external_eval_canary.py \
        "${FULL_ROOT}/full.parquet" \
        "${FULL_ROOT}/manifest.json" \
        "${CANARY_ROOT}" \
        --source-split full \
        --count "${CANARY_COUNT}" \
        --sample-seed "${SAMPLE_SEED}"
    DATA_PATH="${CANARY_ROOT}/quick.parquet"
    RUNTIME_MANIFEST="${CANARY_ROOT}/manifest.json"
    DATA_SPLIT=quick
elif [[ "${EVAL_TIER}" == "quick" ]]; then
    DATA_PATH="${FULL_ROOT}/quick.parquet"
    RUNTIME_MANIFEST="${FULL_ROOT}/manifest.json"
    DATA_SPLIT=quick
    if [[ "${BENCHMARK}" == "swebench_verified" ]]; then
        "${PYTHON_BIN}" - "${RUNTIME_MANIFEST}" <<'PY'
import json, sys
manifest = json.load(open(sys.argv[1]))
if manifest.get("frozen_manifest_name") != "swebench_verified_64" or manifest["splits"]["quick"]["count"] != 64:
    raise SystemExit("Verified quick requires a materialized frozen Verified-64 bundle; do not reuse legacy Verified-50")
PY
    fi
else
    DATA_PATH="${FULL_ROOT}/full.parquet"
    RUNTIME_MANIFEST="${FULL_ROOT}/manifest.json"
    DATA_SPLIT=full
fi
"${PYTHON_BIN}" scripts/check_external_eval_images.py "${DATA_PATH}"

socket_namespace=$(printf '%s' "${RUN_NAME}" | sha256sum | cut -c1-16)
server_pids=()
model_sockets=()
cleanup() {
    for pid in "${server_pids[@]}"; do
        if kill -0 "${pid}" 2>/dev/null; then
            kill -- "-${pid}" 2>/dev/null || kill "${pid}" 2>/dev/null || true
        fi
    done
    for pid in "${server_pids[@]}"; do
        wait "${pid}" 2>/dev/null || true
    done
}
trap cleanup EXIT INT TERM

for index in "${!replica_devices[@]}"; do
    device="${replica_devices[${index}]}"
    socket_host="${MODEL_SOCKET_DIR}/${socket_namespace}-r${index}.sock"
    socket_container="/opt/coding-opd/model/${socket_namespace}-r${index}.sock"
    server_log="${LOG_ROOT}/vllm-replica-${index}.log"
    if [[ -S "${socket_host}" ]] && curl --noproxy '*' --fail --silent --max-time 2 \
        --unix-socket "${socket_host}" http://localhost/health >/dev/null; then
        echo "A healthy server already owns ${socket_host}; refusing to reuse an untracked process" >&2
        exit 2
    fi
    rm -f "${socket_host}"
    language_model_args=()
    if [[ "${LANGUAGE_MODEL_ONLY}" == "true" ]]; then
        language_model_args+=(--language-model-only)
    fi
    CUDA_VISIBLE_DEVICES="${device}" \
    VLLM_USE_FLASHINFER_SAMPLER=0 \
    VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT}" \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    nohup setsid "${PYTHON_BIN}" -m coding_opd.vllm_codex_server \
        --model "${MODEL_PATH}" \
        --served-model-name "${SERVED_MODEL_NAME}" \
        --uds "${socket_host}" \
        --max-model-len "${CONTEXT_WINDOW}" \
        --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
        --max-num-seqs "${MAX_NUM_SEQS}" \
        --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
        --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
        --seed "${VLLM_SEED}" \
        --attention-backend FLASH_ATTN \
        --gdn-prefill-backend triton \
        --enable-prefix-caching \
        --mamba-cache-mode align \
        --generation-config vllm \
        --override-generation-config "${SAMPLING_CONFIG_JSON}" \
        --enable-auto-tool-choice \
        --tool-call-parser qwen3_coder \
        --reasoning-parser qwen3 \
        --no-enable-log-requests \
        "${language_model_args[@]}" \
        >"${server_log}" 2>&1 </dev/null &
    pid=$!
    server_pids+=("${pid}")
    model_sockets+=("${socket_container}")
    printf '%s\n' "${pid}" >"${RUN_ROOT}/vllm-replica-${index}.pid"
done

for index in "${!server_pids[@]}"; do
    pid="${server_pids[${index}]}"
    socket_host="${MODEL_SOCKET_DIR}/${socket_namespace}-r${index}.sock"
    for _ in $(seq 1 240); do
        if ! kill -0 "${pid}" 2>/dev/null; then
            tail -n 120 "${LOG_ROOT}/vllm-replica-${index}.log" >&2
            exit 1
        fi
        if [[ -S "${socket_host}" ]] && curl --noproxy '*' --fail --silent --max-time 2 \
            --unix-socket "${socket_host}" http://localhost/health >/dev/null; then
            break
        fi
        sleep 2
    done
    if ! curl --noproxy '*' --fail --silent --max-time 5 \
        --unix-socket "${socket_host}" http://localhost/health >/dev/null; then
        echo "vLLM replica ${index} did not become ready within 480 seconds" >&2
        exit 1
    fi
done

entrypoint_args=(
    --data-path "${DATA_PATH}"
    --runtime-manifest "${RUNTIME_MANIFEST}"
    --split "${DATA_SPLIT}"
    --task-config "${TASK_CONFIG}"
    --result-path "${RUN_ROOT}/result.json"
    --records-dir "${RUN_ROOT}/records"
    --log-dir "${LOG_ROOT}/agents"
    --model-path "${MODEL_PATH}"
    --served-model-name "${SERVED_MODEL_NAME}"
    --sampling-config-json "${SAMPLING_CONFIG_JSON}"
    --reasoning-effort "${REASONING_EFFORT}"
    --reasoning-summary "${REASONING_SUMMARY}"
    --context-window "${CONTEXT_WINDOW}"
    --task-cpu-threads "${TASK_CPU_THREADS:-2}"
    --auto-compact-token-limit "${AUTO_COMPACT_TOKEN_LIMIT}"
    --concurrency "$((REPLICA_COUNT * TASKS_PER_REPLICA))"
    --tasks-per-replica "${TASKS_PER_REPLICA}"
    --vllm-seed "${VLLM_SEED}"
    --serving-config-json "${SERVING_CONFIG_JSON}"
    --benchmark "${BENCHMARK}"
)
for socket in "${model_sockets[@]}"; do
    entrypoint_args+=(--model-socket "${socket}")
done
if [[ -n "${RESUME_FROM_RESULT:-}" ]]; then
    entrypoint_args+=(--resume-from-result "${RESUME_FROM_RESULT}")
fi
if [[ -n "${RESUME_SOURCE_GIT_SHA:-}" ]]; then
    entrypoint_args+=(--resume-source-git-sha "${RESUME_SOURCE_GIT_SHA}")
fi
if [[ -n "${RESUME_SOURCE_TASK_CONFIG_SHA256:-}" ]]; then
    entrypoint_args+=(--resume-source-task-config-sha256 "${RESUME_SOURCE_TASK_CONFIG_SHA256}")
fi
if [[ -n "${RETRY_TASK_IDS:-}" ]]; then
    IFS=',' read -r -a retry_task_ids <<<"${RETRY_TASK_IDS}"
    for task_id in "${retry_task_ids[@]}"; do
        entrypoint_args+=(--task-id "${task_id}")
    done
fi

"${PYTHON_BIN}" -m coding_opd.codex_eval_entrypoint "${entrypoint_args[@]}" \
    2>&1 | tee "${LOG_ROOT}.log"

if [[ "${EVAL_TIER}" == "full" ]]; then
    "${PYTHON_BIN}" scripts/summarize_external_eval.py \
        "${RUN_ROOT}/result.json" \
        "${RUNTIME_MANIFEST}" \
        --output "${RUN_ROOT}/summary.json"
fi
