#!/usr/bin/env bash
set -euo pipefail

data_root=${CODING_OPD_DOCKER_ROOT:-/var/lib/coding-opd-docker}
log_path=${CODING_OPD_DOCKER_LOG:-/var/log/coding-opd-dockerd.log}
pid_path=${CODING_OPD_DOCKER_PID:-/run/coding-opd-dockerd.pid}

if ! command -v docker >/dev/null || ! command -v dockerd >/dev/null; then
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y docker.io
fi

if docker info >/dev/null 2>&1; then
  docker info --format 'Docker already ready: driver={{.Driver}} root={{.DockerRootDir}}'
  exit 0
fi

mkdir -p "$data_root" "$(dirname "$log_path")" "$(dirname "$pid_path")"

# A failed startup can leave an unusable socket behind. Only remove the project daemon's
# exact PID/socket after confirming that no Docker server answers on it.
rm -f "$pid_path" /var/run/docker.sock

start_daemon() {
  local driver=$1
  nohup dockerd \
    --host=unix:///var/run/docker.sock \
    --data-root="$data_root" \
    --storage-driver="$driver" \
    --bridge=none \
    --iptables=false \
    --ip6tables=false \
    --ip-forward=false \
    --ip-masq=false \
    --pidfile="$pid_path" \
    >"$log_path" 2>&1 </dev/null &
}

wait_for_daemon() {
  local attempt
  for attempt in $(seq 1 30); do
    if docker info >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  return 1
}

start_daemon overlay2
if ! wait_for_daemon; then
  if daemon_pid=$(pgrep -f '[d]ockerd.*coding-opd-docker' | head -n 1); then
    kill "$daemon_pid"
    wait "$daemon_pid" 2>/dev/null || true
  fi
  rm -f "$pid_path"
  rm -f /var/run/docker.sock
  start_daemon vfs
  if ! wait_for_daemon; then
    tail -n 100 "$log_path"
    exit 1
  fi
fi

docker info --format 'Docker ready: version={{.ServerVersion}} driver={{.Driver}} root={{.DockerRootDir}}'
