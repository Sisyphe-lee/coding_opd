from __future__ import annotations

import asyncio
import json
import logging
import re
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from uni_agent.framework.task_runner import run_task as _run_task
from uni_agent.sandbox.base import ExecResult
from uni_agent.sandbox.docker import DockerSandbox
from uni_agent.sandbox.registry import register_sandbox
from uni_agent.tasks.base import Task, TaskConfig, TaskResult
from uni_agent.tasks.registry import register_task
from uni_agent.tools.base import Tool, ToolResult, register_tool
from uni_agent.tools.edit_file import EditFileTool
from uni_agent.tools.shell import DESCRIPTION as SHELL_DESCRIPTION
from uni_agent.tools.shell import ShellArguments, ShellToolConfig

logger = logging.getLogger(__name__)

_CLIPPED_TOOL_OUTPUT = (
    "\n<response clipped><NOTE>Use `rg -n` or request a narrow `view_range` "
    "before viewing more of this file.</NOTE>"
)


class ConciseEditFileConfig(BaseModel):
    max_output_chars: int = Field(default=10_000, ge=2_000)


def cap_tool_result(result: ToolResult, max_output_chars: int) -> ToolResult:
    text = result.text or ""
    if len(text) <= max_output_chars:
        return result
    return ToolResult(
        text=text[:max_output_chars] + _CLIPPED_TOOL_OUTPUT,
        status=result.status,
    )


@register_tool("coding_opd_editor")
class ConciseEditFileTool(EditFileTool):
    """The Uni-Agent editor with a smaller observation cap for long rollouts."""

    name = "str_replace_editor"
    config_model = ConciseEditFileConfig

    async def run(self, args: dict[str, Any], *, timeout: float | None = None) -> ToolResult:
        result = await super().run(args, timeout=timeout)
        cfg: ConciseEditFileConfig = self.config  # type: ignore[assignment]
        return cap_tool_result(result, cfg.max_output_chars)


@register_sandbox("coding_opd_podman")
class RaySafeDockerSandbox(DockerSandbox):
    """Docker-compatible sandbox whose subprocess waits are safe in Ray workers."""

    async def _run_docker(self, *args: str, timeout: float | None = None) -> ExecResult:
        def run() -> subprocess.CompletedProcess[bytes]:
            return subprocess.run(
                [self.docker_binary, *args],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
                check=False,
            )

        try:
            completed = await asyncio.to_thread(run)
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(f"sandbox command timed out after {timeout}s") from exc
        return ExecResult(
            exit_code=completed.returncode,
            stdout=completed.stdout.decode("utf-8", errors="replace"),
            stderr=completed.stderr.decode("utf-8", errors="replace"),
        )

    async def stop(self) -> None:
        """Force-remove disposable Podman sandboxes without its 10s stop grace."""
        name, self._container_name = self._container_name, None
        if name is not None:
            await self._run_docker("rm", "-f", "--time", "0", name)

    async def write_file(self, path: str, content: bytes | str) -> None:
        """Copy file content through Podman's data plane instead of argv.

        The generic sandbox implementation embeds base64-encoded content in a
        ``bash -c`` argument. Replacing text in a large source file can exceed
        Linux's argv limit, so stage the bytes in a host temporary file and use
        the Docker-compatible ``cp`` path instead.
        """
        data = content.encode("utf-8") if isinstance(content, str) else content
        with tempfile.TemporaryDirectory(prefix="coding-opd-write-") as temp_dir:
            local_file = Path(temp_dir) / "payload"
            local_file.write_bytes(data)
            await self.upload_file(local_file, path)


@register_tool("coding_opd_shell")
class RaySafeShellTool(Tool):
    """A tmux-free shell for network-isolated R2E containers.

    Uni-Agent's built-in stateful shell installs tmux on first use. The R2E
    images intentionally have no network access, so execute each command with
    the sandbox data plane and carry the resulting working directory forward.
    """

    name = "shell"
    description = SHELL_DESCRIPTION
    args_model = ShellArguments
    config_model = ShellToolConfig

    def __init__(self, sandbox: RaySafeDockerSandbox, **kwargs: Any) -> None:
        super().__init__(sandbox, **kwargs)
        self._cwd = "/"

    async def run(self, args: dict[str, Any], *, timeout: float | None = None) -> ToolResult:
        command = str(args.get("command") or "")
        if not command.strip():
            return ToolResult(text="Error: Parameter `command` is required.", status="format_error")

        cfg: ShellToolConfig = self.config  # type: ignore[assignment]
        marker = f"__CODING_OPD_CWD_{uuid.uuid4().hex}__"
        script = (
            f"{command}\n"
            "__coding_opd_rc=$?\n"
            f"printf '\\n{marker}%s\\n' \"$PWD\"\n"
            "exit \"$__coding_opd_rc\""
        )
        result = await self.sandbox.exec_shell(
            script,
            timeout=timeout if timeout is not None else cfg.command_timeout,
            workdir=self._cwd,
            env=cfg.env_vars,
        )

        stdout = result.stdout
        prefix, found, suffix = stdout.rpartition(marker)
        if found:
            cwd, _, trailing = suffix.partition("\n")
            if cwd.startswith("/"):
                self._cwd = cwd
            stdout = prefix.rstrip("\n") + trailing

        parts: list[str] = []
        if stdout:
            parts.append(stdout)
        if result.stderr:
            parts.append(result.stderr)
        if result.exit_code != 0:
            parts.append(f"Command exited with code {result.exit_code}")
        return ToolResult(
            text="\n".join(parts),
            status="timeout" if result.exit_code == -1 else "ok",
        )


class R2EGymTaskConfig(TaskConfig):
    name: str = "r2e_gym"
    agent_timeout: float = Field(default=900.0, gt=0)
    eval_timeout: float = Field(default=300.0, gt=0)


def _parse_r2e_pytest_statuses(output: str) -> dict[str, str]:
    """Mirror R2E-Gym's status-map parser without importing its training dependencies."""
    ansi_escape = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
    plain_output = ansi_escape.sub("", output).replace("\r", "")
    marker = "short test summary info"
    if marker not in plain_output:
        return {}

    statuses: dict[str, str] = {}
    for line in plain_output.split(marker, 1)[1].splitlines():
        match = re.search(r"\b(PASSED|FAILED|ERROR)\s+(.+)", line)
        if not match:
            continue
        status, reference = match.groups()
        reference = reference.split(" - ", 1)[0].strip()
        if "::" in reference:
            test_name = ".".join(reference.split("::")[1:])
        else:
            test_name = reference
        statuses[test_name] = status
    return statuses


def score_pytest_output(expected_output_json: str | dict[str, str], output: str, exit_code: int) -> dict[str, Any]:
    expected = json.loads(expected_output_json) if isinstance(expected_output_json, str) else expected_output_json
    if not isinstance(expected, dict) or not expected:
        raise ValueError("expected_output_json must be a non-empty JSON object")

    ansi_escape = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
    normalized_expected = {
        ansi_escape.sub("", str(name)).split(" - ", 1)[0]: str(status)
        for name, status in expected.items()
    }
    actual = _parse_r2e_pytest_statuses(output)
    matched = sorted(
        name for name, status in normalized_expected.items() if actual.get(name) == status
    )
    missing_or_mismatched = sorted(
        name for name, status in normalized_expected.items() if actual.get(name) != status
    )
    unexpected = sorted(name for name in actual if name not in normalized_expected)
    score = float(actual == normalized_expected)
    return {
        "reward": score,
        "resolved": bool(score),
        "test_exit_code": exit_code,
        "matched_expected": matched,
        "missing_or_mismatched": missing_or_mismatched,
        "unexpected": unexpected,
        "actual_statuses": actual,
        "test_output_tail": output[-12000:],
    }


@register_task("r2e_gym")
class R2EGymTask(Task):
    config_model = R2EGymTaskConfig

    async def run(self) -> TaskResult:
        cfg: R2EGymTaskConfig = self.config  # type: ignore[assignment]
        metadata = cfg.metadata
        async with self.build_sandbox() as sandbox:
            setup = await sandbox.exec_shell(
                "set -eu; "
                "if [ -d /r2e_tests ]; then rm -rf /root/r2e_tests; mv /r2e_tests /root/r2e_tests; fi; "
                "test -d /root/r2e_tests; "
                "ln -sfn /root/r2e_tests /testbed/r2e_tests",
                timeout=60,
            )
            if setup.exit_code != 0:
                raise RuntimeError(f"failed to prepare R2E tests: {setup.stderr or setup.stdout}")

            agent = self.build_agent()
            try:
                agent_result = await asyncio.wait_for(
                    agent.run(sandbox=sandbox, messages=cfg.prompt, workdir="/testbed"),
                    timeout=cfg.agent_timeout,
                )
                finished = agent_result.finished
                agent_info = agent_result.info
            except TimeoutError:
                logger.warning("R2E agent timed out after %.0fs", cfg.agent_timeout)
                finished = False
                agent_info = {"timed_out": True}
            except Exception as exc:  # score the filesystem even when the agent fails
                logger.exception("R2E agent failed; continuing to evaluation")
                finished = False
                agent_info = {"error": f"{type(exc).__name__}: {exc}"}

            evaluation = await sandbox.exec(
                ["bash", "./run_tests.sh"],
                timeout=cfg.eval_timeout,
                workdir="/testbed",
            )
            output = evaluation.stdout + evaluation.stderr
            result = score_pytest_output(metadata["expected_output_json"], output, evaluation.exit_code)
            result["agent"] = agent_info

        score = float(result["reward"])
        return TaskResult(
            reward=score,
            accuracy=score,
            finished=finished,
            extra_info=result,
        )


async def run_r2e_task(**kwargs: Any) -> TaskResult:
    """Uni-Agent runner entry point; importing this module registers r2e_gym."""
    return await _run_task(**kwargs)
