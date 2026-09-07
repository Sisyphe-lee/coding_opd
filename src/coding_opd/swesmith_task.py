from __future__ import annotations

import asyncio
import logging
import re
import shlex
from typing import Any, Iterable

from pydantic import Field

from uni_agent.framework.task_runner import run_task as _run_task
from uni_agent.tasks.base import Task, TaskConfig, TaskResult
from uni_agent.tasks.registry import register_task

# Importing this module registers the Ray-safe Podman sandbox and shell tool used
# by the task config. Keep the implementations shared with the validated R2E path.
from .r2e_task import RaySafeDockerSandbox as RaySafeDockerSandbox  # noqa: F401
from .r2e_task import RaySafeShellTool as RaySafeShellTool  # noqa: F401

logger = logging.getLogger(__name__)

_INSTANCE_ID = re.compile(r"^[A-Za-z0-9_.-]+$")
_PASSING_STATUSES = {"PASSED", "XFAIL"}


class SWESmithTaskConfig(TaskConfig):
    name: str = "swe_smith"
    agent_timeout: float = Field(default=3600.0, gt=0)
    eval_timeout: float = Field(default=900.0, gt=0)
    run_evaluation: bool = True


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        import json

        parsed = json.loads(value)
        return [str(item) for item in parsed]
    return [str(item) for item in value]


def expected_test_files(metadata: dict[str, Any]) -> list[str]:
    selectors = [*_as_list(metadata.get("FAIL_TO_PASS")), *_as_list(metadata.get("PASS_TO_PASS"))]
    return sorted({selector.split("::", 1)[0] for selector in selectors if selector})


def score_test_statuses(metadata: dict[str, Any], statuses: dict[str, str], exit_code: int) -> dict[str, Any]:
    f2p = _as_list(metadata.get("FAIL_TO_PASS"))
    p2p = _as_list(metadata.get("PASS_TO_PASS"))

    def partition(expected: Iterable[str]) -> tuple[list[str], list[str]]:
        passed: list[str] = []
        failed: list[str] = []
        for test in expected:
            (passed if statuses.get(test) in _PASSING_STATUSES else failed).append(test)
        return passed, failed

    f2p_passed, f2p_failed = partition(f2p)
    p2p_passed, p2p_failed = partition(p2p)
    expected_count = len(f2p) + len(p2p)
    passed_count = len(f2p_passed) + len(p2p_passed)
    resolved = bool(f2p) and not f2p_failed and not p2p_failed
    return {
        "reward": float(resolved),
        "resolved": resolved,
        "test_exit_code": exit_code,
        "tests_passed": passed_count,
        "tests_expected": expected_count,
        "partial_score": passed_count / expected_count if expected_count else 0.0,
        "FAIL_TO_PASS": {"success": f2p_passed, "failure": f2p_failed},
        "PASS_TO_PASS": {"success": p2p_passed, "failure": p2p_failed},
    }


def _checkout_command(instance_id: str) -> str:
    if not _INSTANCE_ID.fullmatch(instance_id):
        raise ValueError(f"unsafe SWE-smith instance_id: {instance_id!r}")
    quoted = shlex.quote(instance_id)
    return (
        "set -eu; "
        f"git checkout --force {quoted}; "
        "git rev-parse HEAD > /tmp/coding-opd-swesmith-agent-base; "
        "git rev-parse HEAD~1 > /tmp/coding-opd-swesmith-bug-base"
    )


def _prepare_evaluation_command(metadata: dict[str, Any]) -> str:
    restore = " ".join(shlex.quote(path) for path in expected_test_files(metadata))
    restore_command = f"git checkout HEAD -- {restore}; " if restore else ""
    return (
        "set -eu; "
        "agent_base=$(cat /tmp/coding-opd-swesmith-agent-base); "
        "bug_base=$(cat /tmp/coding-opd-swesmith-bug-base); "
        "git add -A; "
        "git -c core.fileMode=false diff --cached --binary \"$agent_base\" > /tmp/coding-opd-swesmith-agent.patch; "
        "git reset --hard \"$bug_base\"; "
        "git clean -fd; "
        "if test -s /tmp/coding-opd-swesmith-agent.patch; then "
        "git apply --whitespace=nowarn /tmp/coding-opd-swesmith-agent.patch; fi; "
        + restore_command
        + "git status --short"
    )


@register_task("swe_smith")
class SWESmithTask(Task):
    config_model = SWESmithTaskConfig

    async def run(self) -> TaskResult:
        cfg: SWESmithTaskConfig = self.config  # type: ignore[assignment]
        metadata = dict(cfg.metadata)
        instance_id = str(metadata["instance_id"])

        # SWE-smith images contain one branch per task. The branch tip is the
        # agent-facing state with hidden F2P tests removed; HEAD~1 is the buggy
        # evaluation state with those tests restored.
        async with self.build_sandbox() as sandbox:
            setup = await sandbox.exec_shell(_checkout_command(instance_id), timeout=120, workdir="/testbed")
            if setup.exit_code != 0:
                raise RuntimeError(f"failed to initialize SWE-smith task {instance_id}: {setup.stderr or setup.stdout}")

            agent = self.build_agent()
            try:
                agent_result = await asyncio.wait_for(
                    agent.run(sandbox=sandbox, messages=cfg.prompt, workdir="/testbed"),
                    timeout=cfg.agent_timeout,
                )
                finished = agent_result.finished
                agent_info = agent_result.info
            except TimeoutError:
                logger.warning("SWE-smith agent timed out after %.0fs", cfg.agent_timeout)
                finished = False
                agent_info = {"timed_out": True}
            except Exception as exc:  # evaluate whatever remains in the workspace
                logger.exception("SWE-smith agent failed; continuing to evaluation")
                finished = False
                agent_info = {"error": f"{type(exc).__name__}: {exc}"}

            if cfg.run_evaluation:
                prepare = await sandbox.exec_shell(
                    _prepare_evaluation_command(metadata), timeout=120, workdir="/testbed"
                )
                if prepare.exit_code != 0:
                    raise RuntimeError(
                        f"failed to prepare SWE-smith evaluation: {prepare.stderr or prepare.stdout}"
                    )

                try:
                    from swesmith.profiles import registry
                except ImportError as exc:
                    raise RuntimeError(
                        "SWE-smith runtime package is not installed; rerun bootstrap_python_env.sh"
                    ) from exc

                profile = registry.get_from_inst(metadata)
                test_command, _ = profile.get_test_cmd(metadata)
                evaluation = await sandbox.exec_shell(test_command, timeout=cfg.eval_timeout, workdir="/testbed")
                output = evaluation.stdout + evaluation.stderr
                statuses = profile.log_parser(output)
                result = score_test_statuses(metadata, statuses, evaluation.exit_code)
                result.update(
                    {
                        "agent": agent_info,
                        "test_command": test_command,
                        "test_output_tail": output[-12000:],
                    }
                )
            else:
                # Direct OPD does not consume task rewards. Avoid running a full
                # repository test suite for every training trajectory while still
                # materializing the agent interaction and a deterministic reward.
                result = {
                    "reward": 0.0,
                    "resolved": False,
                    "evaluation_skipped": True,
                    "agent": agent_info,
                }

        return TaskResult(
            reward=float(result["reward"]),
            accuracy=float(result["reward"]),
            finished=finished,
            extra_info=result,
        )


def _backfill_problem_statement(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Make older prepared rows compatible with metadata prompt templates."""
    tools_kwargs = kwargs.get("tools_kwargs")
    if isinstance(tools_kwargs, dict) and isinstance(tools_kwargs.get("task"), dict):
        task = dict(tools_kwargs["task"])
        metadata = dict(task.get("metadata") or {})
        if not metadata.get("problem_statement"):
            for message in reversed(kwargs.get("raw_prompt") or []):
                if message.get("role") == "user" and isinstance(message.get("content"), str):
                    metadata["problem_statement"] = message["content"]
                    break
        task["metadata"] = metadata
        kwargs = dict(kwargs)
        kwargs["tools_kwargs"] = {**tools_kwargs, "task": task}
    return kwargs


async def run_swesmith_task(*, run_evaluation: bool = True, **kwargs: Any) -> TaskResult:
    """Uni-Agent runner entry point; importing this module registers SWE-smith.

    Older prepared parquets kept the issue only in ``raw_prompt``. Uni-Agent
    renders YAML prompt templates from task metadata before replacing the
    serialized prompt, so backfill the field for those fixtures as well.
    """
    kwargs = _backfill_problem_statement(kwargs)
    tools_kwargs = dict(kwargs.get("tools_kwargs") or {})
    task = dict(tools_kwargs.get("task") or {})
    task["run_evaluation"] = run_evaluation
    tools_kwargs["task"] = task
    return await _run_task(**{**kwargs, "tools_kwargs": tools_kwargs})
