import pytest

from coding_opd.r2e_images import images_from_extra_info, missing_images, normalize_image_name


def _extra(image: str) -> dict:
    return {"tools_kwargs": {"task": {"sandbox": {"image": image}}}}


def test_images_are_normalized_deduplicated_and_sorted() -> None:
    assert images_from_extra_info(
        [_extra("docker.io/team/z:two"), _extra("team/a:one"), _extra("team/z:two")]
    ) == ["team/a:one", "team/z:two"]


def test_missing_images_accepts_docker_hub_prefix() -> None:
    assert normalize_image_name("docker.io/team/image:tag") == "team/image:tag"
    assert missing_images(["team/a:one", "team/b:two"], ["docker.io/team/a:one"]) == ["team/b:two"]


def test_missing_images_accepts_implicit_latest_tag() -> None:
    assert normalize_image_name("docker.io/team/image") == "team/image:latest"
    assert missing_images(["team/image"], ["docker.io/team/image:latest"]) == []


def test_missing_images_accepts_podman_localhost_prefix() -> None:
    assert normalize_image_name("localhost/coding-opd/verifier:v1") == "coding-opd/verifier:v1"
    assert missing_images(
        ["coding-opd/verifier:v1"], ["localhost/coding-opd/verifier:v1"]
    ) == []


def test_missing_image_metadata_is_rejected() -> None:
    with pytest.raises(ValueError, match="no R2E sandbox image"):
        images_from_extra_info([{}])
