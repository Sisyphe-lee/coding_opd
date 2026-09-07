#!/usr/bin/env bash
# Wait for a multipart OSS upload to become a complete object, then resume a
# download of that object into the canonical remote /personal archive area.
#
# Usage:
#   bash scripts/watch_oss_then_remote_download.sh <ssctl-port>
#
# The port is deliberately positional: it must be supplied for each session
# and is never persisted by this script.

set -Eeuo pipefail

usage() {
    printf 'Usage: %s <ssctl-port>\n' "$(basename "$0")" >&2
    printf 'Environment overrides: POLL_INTERVAL, REMOTE_RETRY_INTERVAL, MAX_REMOTE_ATTEMPTS, EXPECTED_SIZE, REMOTE_DEST_DIR\n' >&2
}

if [[ $# -ne 1 || ! "$1" =~ ^[0-9]+$ || "$1" -lt 1024 || "$1" -gt 65535 ]]; then
    usage
    exit 2
fi

PORT="$1"
CONNECT_SH="${CONNECT_SH:?Set the path to your remote Agent connection helper}"
OSS_OBJECT="${OSS_OBJECT:?Set the complete OSS object URI}"
LOCAL_ARCHIVE="${LOCAL_ARCHIVE:?Set the local archive path}"
REMOTE_DEST_DIR="${REMOTE_DEST_DIR:-/personal/coding_opd_runtime/images/oci_archives}"
REMOTE_FILENAME="${REMOTE_FILENAME:-r2e_train_512-dockerhub-proxy.oci.tar}"
POLL_INTERVAL="${POLL_INTERVAL:-30}"
REMOTE_RETRY_INTERVAL="${REMOTE_RETRY_INTERVAL:-60}"
MAX_REMOTE_ATTEMPTS="${MAX_REMOTE_ATTEMPTS:-10}"
WATCH_LOCK_FILE="${WATCH_LOCK_FILE:-${LOCAL_ARCHIVE}.watch.lock}"

if [[ -z "${EXPECTED_SIZE:-}" ]]; then
    if [[ -f "$LOCAL_ARCHIVE" ]]; then
        EXPECTED_SIZE="$(stat -c '%s' "$LOCAL_ARCHIVE")"
    else
        # Frozen archive size; override EXPECTED_SIZE when monitoring another object.
        EXPECTED_SIZE=322960742400
    fi
fi

if ! command -v ossutil64 >/dev/null 2>&1; then
    printf 'ERROR: ossutil64 is not available locally\n' >&2
    exit 1
fi
if [[ ! -x "$CONNECT_SH" ]]; then
    printf 'ERROR: connect helper is not executable: %s\n' "$CONNECT_SH" >&2
    exit 1
fi
if ! [[ "$EXPECTED_SIZE" =~ ^[0-9]+$ ]]; then
    printf 'ERROR: EXPECTED_SIZE must be an integer, got %s\n' "$EXPECTED_SIZE" >&2
    exit 2
fi

# Only one local watcher may own this OSS object at a time. The lock is held
# through the process lifetime and is released automatically on exit.
exec 9>"$WATCH_LOCK_FILE"
if ! flock -n 9; then
    printf '[%s] another watcher already owns %s; exiting\n' "$(date '+%Y-%m-%d %H:%M:%S%z')" "$WATCH_LOCK_FILE"
    exit 0
fi

timestamp() { date '+%Y-%m-%d %H:%M:%S%z'; }

oss_object_size() {
    local output
    output="$(ossutil64 stat "$OSS_OBJECT" 2>&1)" || return 1
    awk -F: '/^Content-Length[[:space:]]*:/ {gsub(/[[:space:]]/, "", $2); print $2; exit}' <<<"$output"
}

printf '[%s] waiting for complete OSS object: %s (expected %s bytes)\n' \
    "$(timestamp)" "$OSS_OBJECT" "$EXPECTED_SIZE"

while :; do
    current_size="$(oss_object_size || true)"
    if [[ -n "$current_size" ]]; then
        if [[ "$current_size" != "$EXPECTED_SIZE" ]]; then
            printf '[%s] ERROR: OSS object exists with size %s, expected %s; refusing download\n' \
                "$(timestamp)" "$current_size" "$EXPECTED_SIZE" >&2
            exit 3
        fi
        printf '[%s] OSS object is complete (%s bytes)\n' "$(timestamp)" "$current_size"
        break
    fi
    printf '[%s] OSS object is not complete yet; checking again in %ss\n' \
        "$(timestamp)" "$POLL_INTERVAL"
    sleep "$POLL_INTERVAL"
done

# Build the remote command with shell-quoted values. The command uses an OSS
# checkpoint directory and a .part destination, so a dropped connection can be
# retried without restarting the 323 GB download or exposing a partial final
# archive to consumers.
remote_command=$'set -Eeuo pipefail\n'
printf -v q_object '%q' "$OSS_OBJECT"
printf -v q_dest_dir '%q' "$REMOTE_DEST_DIR"
printf -v q_filename '%q' "$REMOTE_FILENAME"
remote_command+="object=$q_object"$'\n'
remote_command+="dest_dir=$q_dest_dir"$'\n'
remote_command+="filename=$q_filename"$'\n'
remote_command+="expected_size=$EXPECTED_SIZE"$'\n'
remote_command+=$'final_path="$dest_dir/$filename"\n'
remote_command+=$'part_path="$final_path.part"\n'
remote_command+=$'checkpoint_dir="$dest_dir/.oss-checkpoints/$filename"\n'
remote_command+=$'lock_path="$dest_dir/.oss-checkpoints/$filename.lock"\n'
remote_command+=$'mkdir -p "$dest_dir" "$checkpoint_dir"\n'
remote_command+=$'exec 9>"$lock_path"\n'
remote_command+=$'if ! flock -n 9; then\n'
remote_command+=$'    printf "another remote downloader owns %s\\n" "$lock_path" >&2\n'
remote_command+=$'    exit 7\n'
remote_command+=$'fi\n'
remote_command+=$'if [[ -f "$final_path" ]]; then\n'
remote_command+=$'    final_size="$(stat -c "%s" "$final_path")"\n'
remote_command+=$'    if [[ "$final_size" == "$expected_size" ]]; then\n'
remote_command+=$'        printf "remote archive already complete: %s bytes\\n" "$final_size"\n'
remote_command+=$'        printf "REMOTE_ARCHIVE_READY size=%s\\n" "$final_size"\n'
remote_command+=$'        exit 0\n'
remote_command+=$'    fi\n'
remote_command+=$'    printf "ERROR: remote final archive exists with size %s, expected %s\\n" "$final_size" "$expected_size" >&2\n'
remote_command+=$'    exit 4\n'
remote_command+=$'fi\n'
remote_command+=$'ossutil64 cp "$object" "$part_path" --parallel 16 --part-size 536870912 --bigfile-threshold 104857600 --checkpoint-dir "$checkpoint_dir" -f\n'
remote_command+=$'part_size="$(stat -c "%s" "$part_path")"\n'
remote_command+=$'if [[ "$part_size" != "$expected_size" ]]; then\n'
remote_command+=$'    printf "ERROR: remote download ended at %s bytes, expected %s\\n" "$part_size" "$expected_size" >&2\n'
remote_command+=$'    exit 5\n'
remote_command+=$'fi\n'
remote_command+=$'mv -- "$part_path" "$final_path"\n'
remote_command+=$'printf "remote archive ready: %s (%s bytes)\\n" "$final_path" "$part_size"\n'
remote_command+=$'printf "REMOTE_ARCHIVE_READY size=%s\\n" "$part_size"\n'

attempt=0
while (( attempt < MAX_REMOTE_ATTEMPTS )); do
    attempt=$((attempt + 1))
    printf '[%s] starting remote OSS download (attempt %s/%s) -> %s/%s\n' \
        "$(timestamp)" "$attempt" "$MAX_REMOTE_ATTEMPTS" "$REMOTE_DEST_DIR" "$REMOTE_FILENAME"
    set +e
    remote_output="$(REMOTE_TIMEOUT=3600 bash "$CONNECT_SH" "$PORT" "$remote_command" 2>&1)"
    rc=$?
    set -e
    printf '%s\n' "$remote_output"
    # connect.sh has historically printed an Agent/SSH error while returning
    # zero. Require an explicit remote sentinel as well as a zero exit code.
    if (( rc == 0 )) && grep -q '^REMOTE_ARCHIVE_READY size=' <<<"$remote_output"; then
        printf '[%s] remote download completed successfully\n' "$(timestamp)"
        exit 0
    fi
    # A previous Agent call can lose its SSH stream while the remote ossutil
    # child continues downloading. In that case the remote lock is the
    # expected state, not a failed attempt; wait without consuming the retry
    # budget until that owner finishes and the sentinel becomes visible.
    if (( rc == 7 )) && grep -q 'another remote downloader owns ' <<<"$remote_output"; then
        printf '[%s] remote downloader is still active; waiting in %ss without starting another copy\n' \
            "$(timestamp)" "$REMOTE_RETRY_INTERVAL"
        attempt=$((attempt - 1))
        sleep "$REMOTE_RETRY_INTERVAL"
        continue
    fi
    if (( rc == 0 )); then rc=7; fi
    printf '[%s] remote download attempt failed with rc=%s; checkpoint is retained, retrying in %ss\n' \
        "$(timestamp)" "$rc" "$REMOTE_RETRY_INTERVAL" >&2
    sleep "$REMOTE_RETRY_INTERVAL"
done

printf '[%s] ERROR: remote download did not complete after %s attempts\n' \
    "$(timestamp)" "$MAX_REMOTE_ATTEMPTS" >&2
exit 6
