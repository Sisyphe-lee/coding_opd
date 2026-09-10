#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
podman_runroot=${CODING_OPD_PODMAN_RUNROOT:-/run/coding-opd-podman-overlay}
if [[ "${1:-}" == --ensure && "${CODING_OPD_PODMAN_SERVICE:-true}" == false ]]; then
    exit 0
fi
mkdir -p "$podman_runroot"
umask 077
if [[ "${1:-}" == --ensure ]]; then
    socket="${podman_runroot}/podman.sock"
    log="${podman_runroot}/podman-service.log"
    exec 9>"${podman_runroot}/podman-service-start.lock"
    flock 9
    if [[ -S "$socket" ]]; then
        if timeout 10 podman --remote --url "unix://${socket}" info >/dev/null 2>&1; then
            exit 0
        fi
        # An outer-container restart can leave a socket with no listener.
        python3 - "$socket" <<'PY'
import os
import socket
import sys

with socket.socket(socket.AF_UNIX) as client:
    client.settimeout(2)
    try:
        client.connect(sys.argv[1])
    except ConnectionRefusedError:
        os.unlink(sys.argv[1])
    else:
        raise SystemExit("Podman service is listening but unhealthy; inspect its log")
PY
    fi
    nohup bash "${BASH_SOURCE[0]}" >"$log" 2>&1 </dev/null 9>&- &
    service_pid=$!
    echo "$service_pid" >"${podman_runroot}/podman-service.pid"
    for ((attempt = 0; attempt < 60; attempt++)); do
        if [[ -S "$socket" ]] && timeout 2 podman --remote --url "unix://${socket}" info >/dev/null 2>&1; then
            echo "PODMAN_SERVICE_READY socket=$socket"
            exit 0
        fi
        kill -0 "$service_pid" 2>/dev/null || break
        sleep 0.5
    done
    echo "Podman service failed to become ready; see $log" >&2
    exit 1
fi
exec bash "$script_dir/podman_sandbox" system service --time=0 \
    "unix://${podman_runroot}/podman.sock"
