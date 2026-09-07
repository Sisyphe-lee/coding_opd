#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import hashlib
import http.client
import json
import os
import shutil
import tarfile
import time
import urllib.parse
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TypeAlias

from coding_opd.oci_layout import normalize_layout


REGISTRY = "https://registry-1.docker.io"
AUTH = "https://auth.docker.io/token"
ACCEPT = ", ".join(
    [
        "application/vnd.oci.image.manifest.v1+json",
        "application/vnd.docker.distribution.manifest.v2+json",
        "application/vnd.oci.image.index.v1+json",
        "application/vnd.docker.distribution.manifest.list.v2+json",
    ]
)
_RETRYABLE_NETWORK_ERRORS = (
    TimeoutError,
    urllib.error.HTTPError,
    urllib.error.URLError,
    http.client.RemoteDisconnected,
    ConnectionResetError,
    BrokenPipeError,
)
_MANIFEST_ERRORS = _RETRYABLE_NETWORK_ERRORS + (ValueError, KeyError, json.JSONDecodeError)
Source: TypeAlias = tuple[str, str | None]


def _opener() -> urllib.request.OpenerDirector:
    if os.environ.get("CODING_OPD_OCI_DIRECT") == "1":
        return urllib.request.build_opener(urllib.request.ProxyHandler({}))
    # By default urllib honors the workstation's HTTP proxy without requiring
    # the optional httpx SOCKS transport used by huggingface_hub.
    return urllib.request.build_opener()


def _request(
    url: str,
    *,
    token: str | None = None,
    accept: str | None = None,
    offset: int | None = None,
    timeout: float = 300,
):
    headers = {"User-Agent": "docker/27.0 coding-opd-oci-fetch/2"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if accept:
        headers["Accept"] = accept
    if offset:
        headers["Range"] = f"bytes={offset}-"
    return _opener().open(urllib.request.Request(url, headers=headers), timeout=timeout)


def _token(repo: str, auth_url: str, *, retries: int = 8) -> str:
    auth_host = urllib.parse.urlparse(auth_url).hostname or ""
    service = "registry.docker.io" if auth_host == "auth.docker.io" else auth_host
    query = urllib.parse.urlencode({"service": service, "scope": f"repository:{repo}:pull"})
    for attempt in range(retries):
        try:
            with _request(f"{auth_url}?{query}", timeout=30) as response:
                return json.load(response)["token"]
        except _RETRYABLE_NETWORK_ERRORS:
            if attempt + 1 == retries:
                raise
            time.sleep(min(2**attempt, 30))
    raise AssertionError("token retry loop exhausted")


def _parse_source_spec(spec: str) -> Source:
    registry, separator, auth_url = spec.partition("=")
    if not registry:
        raise ValueError(f"invalid registry source: {spec!r}")
    return registry.rstrip("/"), auth_url if separator and auth_url else None


def _sources(
    registry: str,
    auth_url: str | None,
    specs: list[str] | None,
) -> list[Source]:
    if specs:
        return [_parse_source_spec(spec) for spec in specs]
    return [(registry.rstrip("/"), auth_url)]


def _manifest(
    registry: str,
    repo: str,
    reference: str,
    token: str | None,
    *,
    retries: int = 8,
) -> tuple[bytes, str]:
    url = f"{registry}/v2/{repo}/manifests/{reference}"
    for attempt in range(retries):
        try:
            with _request(url, token=token, accept=ACCEPT, timeout=30) as response:
                return response.read(), response.headers.get_content_type()
        except _RETRYABLE_NETWORK_ERRORS as error:
            if not isinstance(error, urllib.error.HTTPError):
                if attempt + 1 == retries:
                    raise
                time.sleep(min(2**attempt, 30))
                continue
            if error.code not in {429, 500, 502, 503, 504} or attempt + 1 == retries:
                raise
            retry_after = error.headers.get("Retry-After")
            try:
                delay = float(retry_after) if retry_after else min(2**attempt, 60)
            except ValueError:
                delay = min(2**attempt, 60)
            time.sleep(max(1.0, min(delay, 120.0)))
    raise AssertionError("manifest retry loop exhausted")


def _digest(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _blob_path(layout: Path, digest: str) -> Path:
    algorithm, value = digest.split(":", 1)
    return layout / "blobs" / algorithm / value


def _split_image_reference(image: str) -> tuple[str, str, str]:
    """Return registry API repo/tag plus the original ref for OCI annotations."""
    annotation = image if "/" in image else f"docker.io/{image}"
    normalized = image
    first, separator, remainder = image.partition("/")
    if separator and (first in {"docker.io", "ghcr.io", "public.ecr.aws"} or "." in first or ":" in first):
        normalized = remainder
    repo, separator, tag = normalized.rpartition(":")
    if not separator or "/" not in repo:
        repo, tag = normalized, "latest"
    return repo, tag, annotation


def _write_verified(path: Path, data: bytes, digest: str) -> None:
    if _digest(data) != digest:
        raise ValueError(f"digest mismatch for {digest}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _manifest_cache_path(cache_dir: Path, image: str) -> Path:
    key = hashlib.sha256(image.encode("utf-8")).hexdigest()
    return cache_dir / f"{key}.json"


def _load_manifest_cache(cache_dir: Path, image: str) -> tuple[bytes, str] | None:
    path = _manifest_cache_path(cache_dir, image)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("image") != image:
            return None
        data = base64.b64decode(payload["data"])
        if _digest(data) != payload["digest"]:
            return None
        return data, payload["media_type"]
    except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError):
        return None


def _write_manifest_cache(cache_dir: Path, image: str, data: bytes, media_type: str) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    digest = _digest(data)
    payload = {
        "image": image,
        "media_type": media_type,
        "digest": digest,
        "data": base64.b64encode(data).decode("ascii"),
    }
    path = _manifest_cache_path(cache_dir, image)
    temporary = path.with_suffix(".part")
    temporary.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def _download_blob(
    layout: Path,
    sources: list[Source],
    repo: str,
    descriptor: dict,
    *,
    retries: int = 8,
) -> tuple[str, int]:
    digest = descriptor["digest"]
    expected_size = int(descriptor["size"])
    target = _blob_path(layout, digest)
    if target.is_file() and target.stat().st_size == expected_size:
        return digest, expected_size
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".part")
    last_error: Exception | None = None
    completed = False
    for registry, auth_url in sources:
        for attempt in range(retries):
            size = temporary.stat().st_size if temporary.exists() else 0
            if size > expected_size:
                temporary.unlink()
                size = 0
            try:
                token = _token(repo, auth_url) if auth_url else None
                response = _request(
                    f"{registry}/v2/{repo}/blobs/{digest}", token=token, offset=size
                )
                if size and getattr(response, "status", None) != 206:
                    response.close()
                    # This source ignored Range; leave the partial for another
                    # source, which may support resumable requests.
                    last_error = RuntimeError(f"{registry} ignored Range for {digest}")
                    break
                with response, temporary.open("ab" if size else "wb") as output:
                    while chunk := response.read(8 * 1024 * 1024):
                        output.write(chunk)
                if temporary.stat().st_size == expected_size:
                    completed = True
                    break
            except urllib.error.HTTPError as error:
                last_error = error
                if error.code in {401, 403, 404, 416}:
                    break
                if attempt + 1 == retries:
                    break
                time.sleep(min(2**attempt, 30))
            except _RETRYABLE_NETWORK_ERRORS as error:
                last_error = error
                if attempt + 1 == retries:
                    break
                time.sleep(min(2**attempt, 30))
        if completed:
            break
    if not completed:
        if last_error:
            raise last_error
        raise RuntimeError(f"download remained incomplete after trying {len(sources)} registries: {digest}")

    hasher = hashlib.sha256()
    with temporary.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            hasher.update(chunk)
    size = temporary.stat().st_size
    actual = "sha256:" + hasher.hexdigest()
    if actual != digest or size != expected_size:
        temporary.unlink(missing_ok=True)
        raise ValueError(f"blob validation failed for {digest}: digest={actual}, size={size}, expected={expected_size}")
    os.replace(temporary, target)
    return digest, size


def build_layout(
    images: list[str],
    layout: Path,
    *,
    workers: int,
    registry: str = REGISTRY,
    auth_url: str | None = AUTH,
    manifest_registry: str | None = None,
    manifest_auth_url: str | None = None,
    registry_sources: list[str] | None = None,
    manifest_sources: list[str] | None = None,
    skip_unavailable: bool = False,
) -> dict:
    layout.mkdir(parents=True, exist_ok=True)
    (layout / "oci-layout").write_text('{"imageLayoutVersion":"1.0.0"}\n')
    index_descriptors: list[dict] = []
    downloads: dict[str, tuple[str, dict]] = {}
    layer_sources = _sources(registry, auth_url, registry_sources)
    manifest_sources_list = _sources(
        manifest_registry or registry,
        auth_url if manifest_auth_url is None else manifest_auth_url,
        manifest_sources,
    )
    manifest_cache = Path(f"{layout}.manifests")
    token_cache: dict[tuple[str, str], str] = {}

    def source_token(source: Source, repo: str) -> str | None:
        _, auth = source
        if not auth:
            return None
        key = (auth, repo)
        if key not in token_cache:
            token_cache[key] = _token(repo, auth)
        return token_cache[key]

    def fetch_manifest(repo: str, reference: str) -> tuple[bytes, str]:
        last_error: Exception | None = None
        for source in manifest_sources_list:
            registry_url, _ = source
            try:
                return _manifest(registry_url, repo, reference, source_token(source, repo))
            except _RETRYABLE_NETWORK_ERRORS as error:
                last_error = error
                continue
        if last_error:
            raise last_error
        raise RuntimeError(f"no manifest registries configured for {repo}:{reference}")

    for image in images:
        repo, tag, annotation = _split_image_reference(image)
        try:
            cached = _load_manifest_cache(manifest_cache, image)
            if cached:
                manifest_bytes, media_type = cached
            else:
                manifest_bytes, media_type = fetch_manifest(repo, tag)
                manifest = json.loads(manifest_bytes)
                if "manifests" in manifest:
                    candidates = [
                        item
                        for item in manifest["manifests"]
                        if item.get("platform", {}).get("os") == "linux"
                        and item.get("platform", {}).get("architecture") == "amd64"
                    ]
                    if not candidates:
                        raise ValueError(f"{image} has no linux/amd64 manifest")
                    manifest_bytes, media_type = fetch_manifest(repo, candidates[0]["digest"])
                _write_manifest_cache(manifest_cache, image, manifest_bytes, media_type)
        except _MANIFEST_ERRORS as error:
            if not skip_unavailable:
                raise
            print(f"skipping unavailable image {image}: {error}", flush=True)
            continue
        manifest = json.loads(manifest_bytes)

        manifest_digest = _digest(manifest_bytes)
        _write_verified(_blob_path(layout, manifest_digest), manifest_bytes, manifest_digest)
        index_descriptors.append(
            {
                "mediaType": media_type,
                "digest": manifest_digest,
                "size": len(manifest_bytes),
                "annotations": {"org.opencontainers.image.ref.name": annotation},
                "platform": {"architecture": "amd64", "os": "linux"},
            }
        )
        for descriptor in [manifest["config"], *manifest["layers"]]:
            downloads.setdefault(descriptor["digest"], (repo, descriptor))

    completed = 0
    total_bytes = sum(int(item[1]["size"]) for item in downloads.values())
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_download_blob, layout, layer_sources, repo, descriptor): digest
            for digest, (repo, descriptor) in downloads.items()
        }
        for future in as_completed(futures):
            _, size = future.result()
            completed += size
            print(f"downloaded {completed / 2**30:.2f}/{total_bytes / 2**30:.2f} GiB", flush=True)

    index = {"schemaVersion": 2, "manifests": index_descriptors}
    (layout / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    normalize_layout(layout)
    return {"images": len(images), "unique_blobs": len(downloads), "compressed_bytes": total_bytes}


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Docker images into a verified multi-image OCI archive")
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--registry", default=REGISTRY)
    parser.add_argument(
        "--manifest-registry",
        help="registry used for manifest metadata; defaults to --registry",
    )
    parser.add_argument(
        "--auth-url",
        default=AUTH,
        help="Bearer token endpoint; pass an empty string for an anonymous mirror",
    )
    parser.add_argument(
        "--manifest-auth-url",
        help="token endpoint for --manifest-registry; defaults to --auth-url",
    )
    parser.add_argument(
        "--registry-source",
        action="append",
        metavar="REGISTRY[=AUTH_URL]",
        help="layer source; may be repeated for fallback (empty AUTH_URL means anonymous)",
    )
    parser.add_argument(
        "--manifest-source",
        action="append",
        metavar="REGISTRY[=AUTH_URL]",
        help="manifest source; may be repeated for fallback (empty AUTH_URL means anonymous)",
    )
    parser.add_argument("--staging-dir", type=Path)
    parser.add_argument("--keep-staging", action="store_true")
    parser.add_argument(
        "--skip-unavailable",
        action="store_true",
        help="skip images whose manifests fail on every configured source",
    )
    parser.add_argument(
        "--direct",
        action="store_true",
        help="bypass HTTP(S) proxy settings (recommended for mainland registry mirrors)",
    )
    parser.add_argument(
        "--images-file",
        type=Path,
        action="append",
        help="read one image reference per line; may be repeated",
    )
    parser.add_argument("images", nargs="*")
    args = parser.parse_args()

    images = list(args.images)
    for image_file in args.images_file or []:
        images.extend(
            line.strip()
            for line in image_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    if not images:
        parser.error("provide at least one image reference or --images-file")

    if args.direct:
        os.environ["CODING_OPD_OCI_DIRECT"] = "1"

    temporary = args.staging_dir or args.archive.with_name(args.archive.name + ".staging")
    try:
        summary = build_layout(
            images,
            temporary,
            workers=args.workers,
            registry=args.registry.rstrip("/"),
            auth_url=args.auth_url or None,
            manifest_registry=args.manifest_registry,
            manifest_auth_url=args.manifest_auth_url,
            registry_sources=args.registry_source,
            manifest_sources=args.manifest_source,
            skip_unavailable=args.skip_unavailable,
        )
        archive_part = args.archive.with_suffix(args.archive.suffix + ".part")
        with tarfile.open(archive_part, "w") as output:
            for path in sorted(temporary.rglob("*")):
                output.add(path, arcname=path.relative_to(temporary), recursive=False)
        os.replace(archive_part, args.archive)
        summary["archive"] = str(args.archive)
        summary["archive_bytes"] = args.archive.stat().st_size
        print(json.dumps(summary, indent=2), flush=True)
    except Exception:
        print(f"download failed; resumable staging retained at {temporary}", flush=True)
        raise
    else:
        if not args.keep_staging:
            shutil.rmtree(temporary, ignore_errors=True)
            shutil.rmtree(Path(f"{temporary}.manifests"), ignore_errors=True)


if __name__ == "__main__":
    main()
