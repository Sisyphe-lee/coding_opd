#!/usr/bin/env bash
set -euo pipefail

export EVAL_TIER=canary
exec "$(dirname "$0")/run_deepswe_codex_eval.sh" "$@"
