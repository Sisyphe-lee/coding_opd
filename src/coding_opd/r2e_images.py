from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Iterable


def normalize_image_name(image: str) -> str:
    """Normalize optional Docker/Podman prefixes and an implicit latest tag."""
    image = image.strip()
    for prefix in ("docker.io/", "localhost/"):
        if image.startswith(prefix):
            image = image.removeprefix(prefix)
            break
    if "@" not in image and ":" not in image.rsplit("/", 1)[-1]:
        image += ":latest"
    return image


def images_from_extra_info(rows: Iterable[dict]) -> list[str]:
    images: set[str] = set()
    for index, extra_info in enumerate(rows):
        try:
            image = extra_info["tools_kwargs"]["task"]["sandbox"]["image"]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"row {index} has no R2E sandbox image") from exc
        if not isinstance(image, str) or not image.strip():
            raise ValueError(f"row {index} has an empty R2E sandbox image")
        images.add(normalize_image_name(image))
    return sorted(images)


def images_from_parquet(path: str | Path) -> list[str]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - bootstrap error path
        raise RuntimeError("pyarrow is required to inspect an R2E parquet") from exc

    table = pq.read_table(str(path), columns=["extra_info"])
    return images_from_extra_info(table["extra_info"].to_pylist())


def local_images(runtime: str | Path) -> set[str]:
    result = subprocess.run(
        [str(runtime), "images", "--format", "{{.Repository}}:{{.Tag}}"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"failed to query sandbox images: {detail}")
    return {normalize_image_name(line) for line in result.stdout.splitlines() if line.strip()}


def missing_images(required: Iterable[str], available: Iterable[str]) -> list[str]:
    normalized_available = {normalize_image_name(image) for image in available}
    return sorted(
        normalize_image_name(image)
        for image in required
        if normalize_image_name(image) not in normalized_available
    )
