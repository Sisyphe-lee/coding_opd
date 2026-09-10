"""Exercise actual kernel limits and timeout cleanup without requiring Docker."""

import asyncio
import subprocess
import sys
from pathlib import Path

import pytest

from coding_opd.r2e_task import RaySafeDockerSandbox
from uni_agent.sandbox.base import ExecResult


class LocalCommandSandbox(RaySafeDockerSandbox):
    async def _run_docker(self, *args, timeout=None):
        assert args[:2] == ("exec", "test-container")
        result = await asyncio.to_thread(
            subprocess.run, args[2:], capture_output=True, timeout=timeout,
        )
        # Podman exposes a container command's signal as shell status 128+sig.
        status = result.returncode if result.returncode >= 0 else 128 - result.returncode
        return ExecResult(exit_code=status, stdout=result.stdout.decode(), stderr=result.stderr.decode())


def test_memory_growth_is_stopped_by_kernel() -> None:
    sandbox = LocalCommandSandbox(process_memory_limit_mb=96)
    sandbox._container_name = "test-container"
    result = asyncio.run(sandbox.exec([sys.executable, "-c", "x = bytearray(256 * 1024 * 1024)"], timeout=10))
    assert result.exit_code != 0
    assert "MemoryError" in result.stderr


def test_timeout_kills_term_ignoring_child(tmp_path: Path) -> None:
    sandbox = LocalCommandSandbox()
    sandbox._container_name = "test-container"
    pidfile = tmp_path / "pid"
    # The Python child ignores SIGTERM; the whole process group must be killed.
    code = "import os,signal,time,pathlib; signal.signal(signal.SIGTERM,signal.SIG_IGN); pathlib.Path(os.sys.argv[1]).write_text(str(os.getpid())); time.sleep(60)"
    result = asyncio.run(sandbox.exec(
        ["bash", "-c", '"$@" & wait', "test", sys.executable, "-c", code, str(pidfile)], timeout=0.5,
    ))
    assert result.exit_code == -1
    pid = int(pidfile.read_text())
    stat = Path(f"/proc/{pid}/stat")
    assert not stat.exists() or stat.read_text().split()[2] == "Z"


def test_argument_boundaries_are_preserved() -> None:
    sandbox = LocalCommandSandbox()
    sandbox._container_name = "test-container"
    payload = 'a; $(echo unexpected) "two words"'
    result = asyncio.run(sandbox.exec(["printf", "%s", payload], timeout=3))
    assert result.exit_code == 0
    assert result.stdout == payload


def test_cancellation_preserves_sandbox_until_bounded_command_exits() -> None:
    sandbox = LocalCommandSandbox()
    sandbox._container_name = "test-container"

    async def run():
        task = asyncio.create_task(sandbox.exec(["sleep", "10"], timeout=0.2))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=2)
        assert sandbox._container_name == "test-container"
        result = await sandbox.exec(["printf", "patch remains available"], timeout=1)
        assert result.stdout == "patch remains available"

    asyncio.run(run())
