from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


OCI_MANIFEST = "application/vnd.oci.image.manifest.v1+json"
OCI_CONFIG = "application/vnd.oci.image.config.v1+json"
MEDIA_TYPE_MAP = {
    "application/vnd.docker.distribution.manifest.v2+json": OCI_MANIFEST,
    "application/vnd.docker.container.image.v1+json": OCI_CONFIG,
    "application/vnd.docker.image.rootfs.diff.tar.gzip": "application/vnd.oci.image.layer.v1.tar+gzip",
    "application/vnd.docker.image.rootfs.foreign.diff.tar.gzip": (
        "application/vnd.oci.image.layer.nondistributable.v1.tar+gzip"
    ),
}


def _digest(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def normalize_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(manifest)
    normalized["mediaType"] = MEDIA_TYPE_MAP.get(str(manifest.get("mediaType", "")), OCI_MANIFEST)
    normalized["config"] = dict(manifest["config"])
    normalized["config"]["mediaType"] = MEDIA_TYPE_MAP.get(
        str(normalized["config"].get("mediaType", "")), normalized["config"].get("mediaType")
    )
    normalized["layers"] = []
    for raw_layer in manifest["layers"]:
        layer = dict(raw_layer)
        layer["mediaType"] = MEDIA_TYPE_MAP.get(str(layer.get("mediaType", "")), layer.get("mediaType"))
        normalized["layers"].append(layer)
    return normalized


def normalize_layout(layout: Path) -> dict[str, int]:
    index_path = layout / "index.json"
    index = json.loads(index_path.read_text())
    converted = 0
    for descriptor in index["manifests"]:
        algorithm, value = descriptor["digest"].split(":", 1)
        source_path = layout / "blobs" / algorithm / value
        manifest = json.loads(source_path.read_bytes())
        normalized = normalize_manifest(manifest)
        data = json.dumps(normalized, separators=(",", ":"), sort_keys=True).encode()
        digest = _digest(data)
        target_path = layout / "blobs" / "sha256" / digest.split(":", 1)[1]
        target_path.write_bytes(data)
        descriptor.update({"mediaType": OCI_MANIFEST, "digest": digest, "size": len(data)})
        converted += 1
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    return {"manifests": converted}
