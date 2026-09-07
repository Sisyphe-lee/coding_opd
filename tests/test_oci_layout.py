from __future__ import annotations

import hashlib
import json
from pathlib import Path

from coding_opd.oci_layout import OCI_CONFIG, OCI_MANIFEST, normalize_layout


def test_normalize_layout_converts_docker_media_types(tmp_path: Path) -> None:
    manifest = {
        "schemaVersion": 2,
        "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
        "config": {
            "mediaType": "application/vnd.docker.container.image.v1+json",
            "digest": "sha256:config",
            "size": 1,
        },
        "layers": [
            {
                "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
                "digest": "sha256:layer",
                "size": 2,
            }
        ],
    }
    source = json.dumps(manifest).encode()
    source_digest = hashlib.sha256(source).hexdigest()
    blobs = tmp_path / "blobs" / "sha256"
    blobs.mkdir(parents=True)
    (blobs / source_digest).write_bytes(source)
    (tmp_path / "index.json").write_text(
        json.dumps(
            {
                "schemaVersion": 2,
                "manifests": [
                    {
                        "mediaType": manifest["mediaType"],
                        "digest": f"sha256:{source_digest}",
                        "size": len(source),
                    }
                ],
            }
        )
    )

    assert normalize_layout(tmp_path) == {"manifests": 1}
    descriptor = json.loads((tmp_path / "index.json").read_text())["manifests"][0]
    normalized = json.loads((blobs / descriptor["digest"].split(":", 1)[1]).read_text())
    assert descriptor["mediaType"] == OCI_MANIFEST
    assert normalized["mediaType"] == OCI_MANIFEST
    assert normalized["config"]["mediaType"] == OCI_CONFIG
    assert normalized["layers"][0]["mediaType"] == "application/vnd.oci.image.layer.v1.tar+gzip"
