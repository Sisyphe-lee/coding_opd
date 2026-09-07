#!/usr/bin/env bash
# Extract a verified multi-image OCI archive once and import exactly the images
# named by a frozen Coding OPD dataset manifest into the project Podman store.

set -Eeuo pipefail

if [[ $# -ne 3 ]]; then
    echo "Usage: $(basename "$0") <oci-archive.tar> <dataset-manifest.json> <extracted-layout-dir>" >&2
    exit 2
fi

archive="$(readlink -f "$1")"
manifest="$(readlink -f "$2")"
layout="$(readlink -m "$3")"
runtime="${PODMAN_RUNTIME:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/podman_sandbox}"
partial="${layout}.extracting"
lock="${layout}.import.lock"

[[ -f "$archive" ]] || { echo "OCI archive not found: $archive" >&2; exit 2; }
[[ -f "$manifest" ]] || { echo "dataset manifest not found: $manifest" >&2; exit 2; }
mkdir -p "$(dirname -- "$layout")"

exec 9>"$lock"
if ! flock -n 9; then
    echo "another importer owns $lock" >&2
    exit 7
fi

if [[ ! -f "$layout/index.json" ]]; then
    if [[ -e "$layout" || -e "$partial" ]]; then
        echo "refusing to overwrite incomplete OCI layout: $layout or $partial" >&2
        exit 3
    fi
    mkdir -p "$partial"
    echo "extracting $archive to $partial"
    tar -xf "$archive" -C "$partial"
    [[ -f "$partial/index.json" && -f "$partial/oci-layout" ]] || {
        echo "archive did not produce an OCI image layout" >&2
        exit 4
    }
    mv "$partial" "$layout"
fi

image_list="$(mktemp "${layout}.required-images.XXXXXX")"
trap 'rm -f "$image_list"' EXIT
python3 - "$layout/index.json" "$manifest" >"$image_list" <<'PY'
import json
import sys

index = json.load(open(sys.argv[1], encoding="utf-8"))
manifest = json.load(open(sys.argv[2], encoding="utf-8"))
archive_images = {
    item.get("annotations", {}).get("org.opencontainers.image.ref.name")
    for item in index.get("manifests", [])
}
archive_images.discard(None)
if "agent_images" in manifest:
    required = [str(image) for image in manifest["agent_images"]]
    expected_count = int(manifest["agent_image_count"])
else:
    required = [item["docker_image"] for item in manifest["tasks"]]
    expected_count = int(manifest["count"])
missing = sorted(set(required) - archive_images)
extra = sorted(archive_images - set(required))
if len(required) != len(set(required)):
    raise SystemExit("manifest image list contains duplicates")
if missing or extra or len(required) != expected_count:
    raise SystemExit(
        f"OCI/manifest mismatch: required={len(required)} archive={len(archive_images)} "
        f"missing={missing[:3]} extra={extra[:3]}"
    )
print("\n".join(required))
PY
mapfile -t images <"$image_list"

completed=0
for image in "${images[@]}"; do
    if "$runtime" image exists "$image"; then
        completed=$((completed + 1))
        echo "[$completed/${#images[@]}] already present: $image"
        continue
    fi
    # Stop before exhausting the disposable runtime filesystem during recovery.
    if [[ -n "${MIN_RUNTIME_FREE_KIB:-}" ]]; then
        available=$(df -Pk "${CODING_OPD_PODMAN_ROOT:-/workspaces/coding-opd-podman-overlay}" | awk 'END {print $4}')
        (( available >= MIN_RUNTIME_FREE_KIB )) || { echo "Runtime disk reserve reached" >&2; exit 8; }
    fi
    image_id="$("$runtime" pull --quiet "oci:${layout}:${image}" | tail -n 1)"
    [[ -n "$image_id" ]] || { echo "import returned no image id: $image" >&2; exit 5; }
    # Podman supplies docker.io for short names; preserve explicit ECR registries.
    "$runtime" tag "$image_id" "$image"
    "$runtime" image exists "$image" || {
        echo "imported image is not addressable as $image" >&2
        exit 6
    }
    completed=$((completed + 1))
    echo "[$completed/${#images[@]}] imported: $image"
done

echo "IMPORT_COMPLETE images=$completed layout=$layout"
