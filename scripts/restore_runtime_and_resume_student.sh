#!/usr/bin/env bash
# Recovery of persistent bundles into disposable node-local Podman storage.
# No dataset/model downloads, deletion, or GPU-process termination is performed.
set -Eeuo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
root=${RUNTIME_ROOT:-/personal/coding_opd_runtime}
source_result=${RESUME_FROM_RESULT:?Set the stopped Student result.json path}
source_sha=${RESUME_SOURCE_GIT_SHA:?Set the reviewed source code SHA}
run=${RUN_NAME:?Set a unique resumed evaluation name}
expected_sha=${EXPECTED_GIT_SHA:?Require the deployed recovery code SHA}
[[ $(git -C "$repo" rev-parse HEAD) == "$expected_sha" ]]
[[ -z $(git -C "$repo" status --porcelain --untracked-files=no) ]]
cd "$repo"
mkdir -p "$root/logs/recovery" "$root/tmp" "$root/sources"
exec 9>"$root/logs/recovery/runtime-restore.lock"
flock -n 9 || { echo 'Recovery already running'; exit 7; }
export TMPDIR="$root/tmp"
export PYTHONPATH="$repo/src:$repo/third_party/verl:$repo/third_party/uni-agent${PYTHONPATH:+:$PYTHONPATH}"
export MIN_RUNTIME_FREE_KIB=104857600
runtime="$repo/scripts/podman_sandbox"
archives="$root/images/oci_archives"
layouts="$root/images/oci_layouts"
eval_root="$root/datasets/coding_opd_eval_v2"
"$runtime" info >/dev/null
.venv/bin/pytest -q

# Pinned DeepSWE sources reproduce independent verifier images without public access.
source_archive="$root/sources/deepswe-v1.1-tasks-0b9fabbb.tar.gz"
if [[ ! -f "$source_archive" ]]; then
    ossutil cp "${OPD_OSS_PREFIX:?Set your own OSS project prefix}/installers/deepswe-v1.1-tasks-0b9fabbb.tar.gz" "$source_archive.part" -f
    mv "$source_archive.part" "$source_archive"
fi
task_root="$root/sources/deepswe-v1.1/tasks"
if [[ ! -d "$task_root" ]]; then
    mkdir -p "$root/sources/deepswe-v1.1"
    tar -xzf "$source_archive" -C "$root/sources/deepswe-v1.1"
fi
.venv/bin/python - "$task_root" <<'PY'
from pathlib import Path
import sys
from coding_opd.eval_data import sha256_tree
assert sha256_tree(Path(sys.argv[1])) == "207e1309bfbaebdf9186123ea4e74732c9348c395687af9bd48e01193a8bd5d9"
print("DEEPSWE_SOURCE_VERIFIED", flush=True)
PY
echo 'STAGE DeepSWE agent images'
echo "e30977e1fddf9e5b43d8cbb85fd40be610b690f914cd2e063ca4d33a0317a793  $archives/deepswe113.oci.tar" | sha256sum -c
bash scripts/import_oci_manifest_images.sh "$archives/deepswe113.oci.tar" "$eval_root/deepswe/manifest.json" "$layouts/deepswe113"
echo 'STAGE DeepSWE independent verifier images'
.venv/bin/python scripts/prepare_deepswe_verifier_images.py "$task_root" "$eval_root/deepswe/full.parquet"
.venv/bin/python scripts/check_external_eval_images.py "$eval_root/deepswe/full.parquet"

echo 'STAGE R2E-512 (includes frozen R2E-128)'
bash scripts/import_oci_manifest_images.sh "$archives/r2e_train_512-dockerhub-proxy.oci.tar" configs/dataset_manifests/r2e_train_512.json "$layouts/r2e_train_512"
.venv/bin/python scripts/check_external_eval_images.py "$root/datasets/r2e_train_512/train.parquet"
.venv/bin/python scripts/check_external_eval_images.py "$root/datasets/r2e_train_128/train.parquet"

echo 'STAGE canonical Verified-500'
# The independent OSS downloader renames only after size and SHA256 verification.
deadline=$((SECONDS + 7200))
until [[ -f "$archives/swebench_verified_500-canonical.oci.tar" ]]; do
    (( SECONDS < deadline )) || { echo 'Verified download not ready after 2 hours'; exit 9; }
    sleep 30
done
bash scripts/import_oci_manifest_images.sh "$archives/swebench_verified_500-canonical.oci.tar" "$eval_root/swebench_verified/manifest.json" "$layouts/swebench_verified_500-canonical"
.venv/bin/python scripts/check_external_eval_images.py "$eval_root/swebench_verified/full.parquet"

# Exercise a real sandbox without loading a model or running benchmark tests.
image=$(.venv/bin/python - "$eval_root/swebench_verified/manifest.json" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))['agent_images'][0])
PY
)
"$runtime" run --rm --cgroups=disabled --network none --pull never --entrypoint /bin/sh "$image" -c 'test -d /testbed && echo SANDBOX_OK'
.venv/bin/pytest -q
[[ $(git rev-parse HEAD) == "$expected_sha" ]]
[[ -z $(git status --porcelain --untracked-files=no) ]]
.venv/bin/python - "$source_result" <<'PY'
import json,subprocess,sys
d=json.load(open(sys.argv[1]))
assert d['model_path'].endswith('/Qwen3.5-9B') and not d['complete']
for line in subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True).splitlines():
    gpu,used=map(int,line.split(','))
    if gpu in (0,1,2,3) and used > 20480:
        raise SystemExit(f'GPU {gpu} occupied ({used} MiB); refusing to start')
PY
echo 'RESTORE_COMPLETE; starting Student four-GPU resource-only resume'
# Release recovery lock; evaluator owns its independent run/output directories.
flock -u 9
export BENCHMARK=swebench_verified EVAL_TIER=full CUDA_VISIBLE_DEVICES=0,1,2,3
export TASKS_PER_REPLICA=4 MAX_NUM_SEQS=4 TENSOR_PARALLEL_SIZE=1
export GPU_MEMORY_UTILIZATION=0.8 MAX_NUM_BATCHED_TOKENS=8192 SAMPLING_PROFILE=student
export MODEL_PATH="$root/models/Qwen3.5-9B" SERVED_MODEL_NAME=Qwen3.5-9B
export CONTEXT_WINDOW=262144 REASONING_EFFORT=xhigh REASONING_SUMMARY=auto
export RESUME_FROM_RESULT="$source_result" RESUME_SOURCE_GIT_SHA="$source_sha" RUN_NAME="$run"
export VLLM_CACHE_ROOT="$root/cache/vllm-eval"
exec bash scripts/run_deepswe_codex_eval.sh
