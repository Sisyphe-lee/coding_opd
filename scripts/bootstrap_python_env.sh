#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
profile=${1:-dev}
uv_version=0.12.8
uv_default_index=${UV_DEFAULT_INDEX:-https://mirrors.aliyun.com/pypi/simple}
uv_http_timeout=${UV_HTTP_TIMEOUT:-300}
uv_concurrent_downloads=${UV_CONCURRENT_DOWNLOADS:-8}
uv_root="$repo_root/.tools/uv"
uv_bin="$uv_root/bin/uv"
venv="$repo_root/.venv"
flash_wheel_name=flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl
flash_wheel_sha=99f46231de527eb2ba7b59d7dfc8c045d76ec879b441915dfa244ad2d2fda1d5
swesmith_commit=9b74ac08118a85c39c356802f7961893af73e07f
swesmith_archive_sha=8ed05e7894624b6a9cfbbddd7c093516b2b841499b30e2957392a033c81f9538

if [[ ! -x "$uv_bin" ]]; then
  case "$(uname -m)" in
    x86_64) uv_target=x86_64-unknown-linux-gnu ;;
    aarch64|arm64) uv_target=aarch64-unknown-linux-gnu ;;
    *) echo "unsupported architecture: $(uname -m)" >&2; exit 1 ;;
  esac
  uv_tmp=$(mktemp -d)
  trap 'rm -rf "$uv_tmp"' EXIT
  mkdir -p "$uv_root/bin"

  if [[ "$uv_target" == x86_64-unknown-linux-gnu && -n "${OPD_OSS_PREFIX:-}" ]] && command -v ossutil >/dev/null; then
    uv_oss="${OPD_OSS_PREFIX%/}/installers/uv-0.12.8-x86_64-unknown-linux-gnu"
    if ossutil cp "$uv_oss" "$uv_tmp/uv" --force; then
      echo "fe60af3d6b1771ecea15fdfd3821959912009e1612b3767a4988f135bc3cdd9d  $uv_tmp/uv" \
        | sha256sum --check
      install -m 0755 "$uv_tmp/uv" "$uv_bin"
    fi
  fi

  if [[ ! -x "$uv_bin" ]]; then
    curl --fail --location --retry 3 \
      "https://github.com/astral-sh/uv/releases/download/$uv_version/uv-$uv_target.tar.gz" \
      --output "$uv_tmp/uv.tar.gz"
    tar -xzf "$uv_tmp/uv.tar.gz" -C "$uv_tmp"
    install -m 0755 "$uv_tmp/uv-$uv_target/uv" "$uv_bin"
  fi
fi

if [[ ! -x "$venv/bin/python" ]]; then
  "$uv_bin" venv --python 3.12 "$venv"
fi

export VIRTUAL_ENV="$venv"
export PATH="$venv/bin:$PATH"
export UV_HTTP_TIMEOUT="$uv_http_timeout"
export UV_CONCURRENT_DOWNLOADS="$uv_concurrent_downloads"

# Use veRL's checked-in lock while keeping the environment at the repository root.
case "$profile" in
  dev)
    "$uv_bin" sync \
      --project "$repo_root/third_party/verl" \
      --active \
      --frozen \
      --default-index "$uv_default_index" \
      --extra cpu \
      --no-dev
    ;;
  train)
    # uv.lock records the original registry URL. Export the selected lock fork and
    # install its exact versions from one source at a time so slow indexes are not scanned.
    "$uv_bin" export \
      --project "$repo_root/third_party/verl" \
      --frozen \
      --extra fsdp \
      --extra vllm \
      --no-dev \
      --no-emit-project \
      --no-hashes \
      --format requirements-txt \
      | grep -Ev '^(flash-attn|torch==|torchaudio==|torchvision==)' \
      | "$uv_bin" pip install \
        --python "$venv/bin/python" \
        --no-deps \
        --requirements - \
        --default-index "$uv_default_index"

    "$uv_bin" pip install --python "$venv/bin/python" --no-deps \
      --default-index https://download.pytorch.org/whl/cu130 \
      'torch==2.11.0+cu130' 'torchvision==0.26.0+cu130' 'torchaudio==2.11.0+cu130'

    flash_wheel="$repo_root/.tools/wheels/$flash_wheel_name"
    if [[ "$(uname -m)" == x86_64 && -n "${OPD_OSS_PREFIX:-}" ]] && command -v ossutil >/dev/null; then
      mkdir -p "$(dirname "$flash_wheel")"
      if ! echo "$flash_wheel_sha  $flash_wheel" | sha256sum --check --status 2>/dev/null; then
        ossutil cp \
          "${OPD_OSS_PREFIX%/}/wheels/$flash_wheel_name" \
          "$flash_wheel" \
          --force
      fi
      echo "$flash_wheel_sha  $flash_wheel" | sha256sum --check
      "$uv_bin" pip install --python "$venv/bin/python" --no-deps "$flash_wheel"
    else
      "$uv_bin" pip install --python "$venv/bin/python" --no-deps \
        --default-index https://verl-project.github.io/verl-wheelhouse/simple/ \
        'flash-attn==2.8.3'
    fi

    "$uv_bin" pip install --python "$venv/bin/python" --no-deps \
      -e "$repo_root/third_party/verl"

    # Qwen3.5 alternates full attention with Gated DeltaNet layers. flash-attn
    # accelerates the former; FLA supplies the Triton kernels for the latter.
    # Do not replace this with a native sm_103 causal-conv1d build until its
    # B300 numerical-correctness issue has been resolved and verified here.
    "$uv_bin" pip install --python "$venv/bin/python" --no-deps \
      --default-index "$uv_default_index" \
      'fla-core==0.5.2' 'flash-linear-attention==0.5.2'
    ;;
  *)
    echo "usage: $0 [dev|train]" >&2
    exit 2
    ;;
esac

# Uni-Agent imports the installed top-level veRL package. Its own nested veRL submodule is
# intentionally not initialized. R2E-Gym remains source-only until the slim adapter is ready.
"$uv_bin" pip install --python "$venv/bin/python" --no-deps \
  -e "$repo_root/third_party/uni-agent" \
  -e "$repo_root"

# SWE-smith is used only for its authoritative repository profiles and test-log
# parsers. Keep its source pinned and install the small verifier dependency set;
# do not install the broad `swesmith[all]` environment into veRL.
swesmith_archive="$repo_root/.tools/packages/swesmith-${swesmith_commit}.tar.gz"
mkdir -p "$(dirname "$swesmith_archive")"
if ! echo "$swesmith_archive_sha  $swesmith_archive" | sha256sum --check --status 2>/dev/null; then
  if [[ -n "${OPD_OSS_PREFIX:-}" ]] && command -v ossutil >/dev/null; then
    ossutil cp \
      "${OPD_OSS_PREFIX%/}/installers/swesmith-${swesmith_commit}.tar.gz" \
      "$swesmith_archive" \
      --force
  else
    curl --fail --location --retry 3 \
      "https://github.com/SWE-bench/SWE-smith/archive/${swesmith_commit}.tar.gz" \
      --output "$swesmith_archive"
  fi
fi
echo "$swesmith_archive_sha  $swesmith_archive" | sha256sum --check
"$uv_bin" pip install --python "$venv/bin/python" \
  --default-index "$uv_default_index" \
  'swebench==4.1.0' 'ghapi<2' 'python-dotenv' 'unidiff' 'GitPython'
"$uv_bin" pip install --python "$venv/bin/python" --no-deps "$swesmith_archive"

echo "Coding OPD environment profile: $profile"
"$venv/bin/python" "$repo_root/scripts/smoke_environment.py"
