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

from coding_opd.profiling import profile_span
from coding_opd.react_agent import CodingReActAgent as CodingReActAgent  # noqa: F401

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
        with profile_span("tool", tool="str_replace_editor"):
            result = await super().run(args, timeout=timeout)
        cfg: ConciseEditFileConfig = self.config  # type: ignore[assignment]
        return cap_tool_result(result, cfg.max_output_chars)


@register_sandbox("coding_opd_podman")
class RaySafeDockerSandbox(DockerSandbox):
    """Docker-compatible sandbox whose subprocess waits are safe in Ray workers."""

    def __init__(self, *, process_memory_limit_mb: int = 8192, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if (
            isinstance(process_memory_limit_mb, bool)
            or not isinstance(process_memory_limit_mb, int)
            or process_memory_limit_mb < 1
        ):
            raise ValueError("process_memory_limit_mb must be a positive integer")
        self.process_memory_limit_mb = process_memory_limit_mb

    async def start(self) -> None:
        """Let `run --pull never` check availability without a separate inspect."""
        if self._container_name is not None:
            return
        name = self.container_name or f"uni-agent-{uuid.uuid4().hex[:12]}"
        args = ["run", "--rm", "-d", "--name", name, "--pull", self.pull_policy]
        if self.entrypoint:
            args.extend(["--entrypoint", self.entrypoint])
        args.extend(self.run_args)
        args.append(self.image)
        args.extend(self.command)
        started = await self._run_docker(*args)
        if started.exit_code != 0:
            detail = started.stderr.strip() or started.stdout.strip()
            raise RuntimeError(f"Failed to start Docker sandbox from {self.image!r}: {detail}")
        self._container_name = name

    async def _exec(self, argv, *, timeout=None, workdir=None, env=None) -> ExecResult:
        # cgroups are disabled in the nested runtime. RLIMIT_AS bounds each
        # command process (and is inherited by children), including pytest.
        command = ["bash", "-c", 'ulimit -v "$1" || exit; shift; exec "$@"',
                   "opd-limited-command", str(self.process_memory_limit_mb * 1024)]
        if timeout is not None:
            # Run timeout INSIDE the container: killing only `podman exec`
            # leaves the actual test and its children alive. GNU timeout owns
            # a process group. Kill it directly: TERM can let the shell exit
            # before an ignoring child, cancelling timeout's escalation timer.
            command += ["timeout", "--signal=KILL", f"{timeout}s"]
        command += list(argv)
        try:
            pending = asyncio.create_task(super()._exec(
                command, timeout=None if timeout is None else timeout + 15,
                workdir=workdir, env=env,
            ))
            try:
                result = await asyncio.shield(pending)
            except asyncio.CancelledError:
                # The in-container timeout bounds tool execution. Let it finish
                # before patch collection instead of deleting the agent sandbox.
                await pending
                raise
        except TimeoutError:
            # Fallback for an unresponsive runtime or cancelled Ray task.
            await asyncio.shield(self.stop())
            raise
        if timeout is not None and result.exit_code in (124, 137):
            return ExecResult(exit_code=-1, stdout=result.stdout,
                              stderr=result.stderr + f"\ncommand timed out after {timeout}s; process group terminated")
        return result

    async def _run_docker(self, *args: str, timeout: float | None = None) -> ExecResult:
        def run() -> subprocess.CompletedProcess[bytes]:
            return subprocess.run(
                [self.docker_binary, *args],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
                check=False,
            )

        with profile_span("sandbox", operation=args[0], image=self.image) as span:
            try:
                completed = await asyncio.to_thread(run)
                span["exit_code"] = completed.returncode
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
        with profile_span("tool", tool="shell") as span:
            result = await self._run(args, timeout=timeout)
            span["tool_status"] = result.status
            return result

    async def _run(self, args: dict[str, Any], *, timeout: float | None = None) -> ToolResult:
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
    run_evaluation: bool = True
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

            if cfg.run_evaluation:
                evaluation = await sandbox.exec(
                    ["bash", "./run_tests.sh"],
                    timeout=cfg.eval_timeout,
                    workdir="/testbed",
                )
                output = evaluation.stdout + evaluation.stderr
                result = score_pytest_output(metadata["expected_output_json"], output, evaluation.exit_code)
            else:
                # Pure OPD does not consume executable rewards. Return as soon
                # as interaction ends; do not block Teacher scoring on pytest.
                result = {"reward": 0.0}
                logger.info("Post-agent evaluation skipped for pure OPD training")
            result["evaluation_ran"] = cfg.run_evaluation
            result["agent"] = agent_info

        score = float(result["reward"])
        return TaskResult(
            reward=score,
            accuracy=score if cfg.run_evaluation else None,
            finished=finished,
            extra_info=result,
        )


async def run_r2e_task(
    *, opd_algorithm: str = "vanilla", tcod_growth_interval: int = 2,
    adaptive_threshold: float = 0.1,
    run_evaluation: bool | None = None, **kwargs: Any
) -> TaskResult:
    """Uni-Agent runner entry point; importing this module registers r2e_gym."""
    from coding_opd.algorithms import get_algorithm
    from uni_agent.tasks.config import TaskConfigResolver

    algorithm = get_algorithm(opd_algorithm, growth_interval=tcod_growth_interval, threshold=adaptive_threshold)
    tools_kwargs = dict(kwargs.get("tools_kwargs") or {})
    progress = tools_kwargs.get("_opd_progress")
    if opd_algorithm != "vanilla":
        if progress is None:
            raise ValueError(f"{opd_algorithm} requires explicit _opd_progress from the OPD framework")
        if progress["training"]:
            config_path = kwargs.get("task_config_path")
            resolver = TaskConfigResolver.from_file(config_path) if config_path else TaskConfigResolver()
            task = resolver.resolve(tools_kwargs["task"])
            agent = dict(task.get("agent") or {})
            # Match the ReAct default when neither YAML nor the sample sets it.
            max_turns = agent.get("max_steps", 50)
            horizon = algorithm.rollout_max_turns(progress["step"], max_turns)
            agent["max_steps"] = horizon
            tools_kwargs["task"] = {**task, "agent": agent}
            kwargs["tools_kwargs"] = tools_kwargs
            logger.info(
                "OPD algorithm=%s step=%s horizon=%s max_turns=%s",
                opd_algorithm, progress["step"], horizon, max_turns,
            )
    if run_evaluation is not None:
        # Framework validation always measures task success. Standalone runs
        # retain the Task Config default unless explicitly overridden.
        evaluate = True if progress is not None and not progress["training"] else run_evaluation
        tools_kwargs["task"] = {**tools_kwargs["task"], "run_evaluation": evaluate}
        kwargs["tools_kwargs"] = tools_kwargs
    identity = tools_kwargs.get("_trace_identity") or {}
    with profile_span(
        "agent_task", algorithm=opd_algorithm,
        **{key: identity[key] for key in ("uid", "sample", "session", "session_id") if key in identity},
        step=identity.get("global_steps"),
    ):
        return await _run_task(**kwargs)
