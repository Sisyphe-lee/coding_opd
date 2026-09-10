import asyncio
import json
from pathlib import Path

import pytest

from coding_opd.r2e_task import RaySafeDockerSandbox, score_pytest_output
from uni_agent.sandbox.base import ExecResult


def test_score_pytest_output_supports_r2e_names() -> None:
    expected = {
        "TestContextHandler.test_read_defaults": "PASSED",
        "TestContextHandler.test_initialize": "FAILED",
    }
    output = """
================ short test summary info ================
PASSED r2e_tests/r2e_tests/__init__.py::TestContextHandler::test_read_defaults
FAILED r2e_tests/r2e_tests/__init__.py::TestContextHandler::test_initialize
"""
    result = score_pytest_output(json.dumps(expected), output, 1)
    assert result["reward"] == 1.0
    assert result["resolved"]
    assert result["matched_expected"] == [
        "TestContextHandler.test_initialize",
        "TestContextHandler.test_read_defaults",
    ]


def test_score_requires_the_complete_expected_status_map() -> None:
    expected = {"TestThing.test_ok": "PASSED", "test_known_failure": "FAILED"}
    output = """
================ short test summary info ================
PASSED tests/test_thing.py::TestThing::test_ok
FAILED tests/test_thing.py::test_known_failure - AssertionError
"""
    assert score_pytest_output(expected, output, 1)["resolved"]
    assert not score_pytest_output({"TestThing.test_ok": "PASSED"}, output, 0)["resolved"]


def test_score_supports_bare_and_ansi_decorated_test_names() -> None:
    expected = {
        "test_formatter": "PASSED",
        "\x1b[1mtest_version[10-deprecated \\(2023\\)\\.]\x1b[0m": "PASSED",
    }
    output = """
================ short test summary info ================
PASSED r2e_tests/test_1.py::test_formatter
PASSED r2e_tests/test_2.py::test_version[10-deprecated \\(2023\\)\\.]
"""

    result = score_pytest_output(expected, output, 0)

    assert result["reward"] == 1.0
    assert result["resolved"]
    assert result["missing_or_mismatched"] == []


def test_ray_safe_sandbox_write_file_uses_copy_data_plane() -> None:
    sandbox = RaySafeDockerSandbox(image="unused")
    captured: dict[str, object] = {}

    async def capture_upload(local_file: Path | str, remote_file: str) -> None:
        captured["content"] = Path(local_file).read_bytes()
        captured["remote_file"] = remote_file

    sandbox.upload_file = capture_upload  # type: ignore[method-assign]
    payload = "x" * 200_000
    asyncio.run(sandbox.write_file("/testbed/large.py", payload))

    assert captured == {
        "content": payload.encode(),
        "remote_file": "/testbed/large.py",
    }


def test_ray_safe_sandbox_stop_skips_podman_grace_period() -> None:
    sandbox = RaySafeDockerSandbox(image="unused", container_name="task-container")
    sandbox._container_name = "task-container"
    captured: list[tuple[str, ...]] = []

    async def capture_run(*args: str, timeout: float | None = None):
        del timeout
        captured.append(args)

    sandbox._run_docker = capture_run  # type: ignore[method-assign]
    asyncio.run(sandbox.stop())

    assert captured == [("rm", "-f", "--time", "0", "task-container")]
    assert sandbox._container_name is None


def test_missing_sandbox_image_fails_without_pulling_or_marking_started() -> None:
    sandbox = RaySafeDockerSandbox(image="missing:tag", pull_policy="never")
    captured: list[tuple[str, ...]] = []

    async def fail_run(*args: str, timeout: float | None = None) -> ExecResult:
        captured.append(args)
        return ExecResult(exit_code=125, stdout="", stderr="image not known")

    sandbox._run_docker = fail_run  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="image not known"):
        asyncio.run(sandbox.start())
    assert len(captured) == 1
    assert captured[0][0] == "run"
    assert captured[0][captured[0].index("--pull") + 1] == "never"
    assert sandbox._container_name is None
