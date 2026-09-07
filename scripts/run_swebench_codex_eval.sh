#!/usr/bin/env bash
set -euo pipefail
export BENCHMARK=swebench_verified
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/run_deepswe_codex_eval.sh" "$@"
