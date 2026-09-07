from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

from pydantic import Field

from uni_agent.framework.task_runner import run_task as _run_task
from uni_agent.sandbox import build_sandbox
from uni_agent.tasks.base import Task, TaskConfig, TaskResult
from uni_agent.tasks.registry import register_task

# Register the project's Ray-safe Docker-compatible provider and concise tools.
from .r2e_task import RaySafeDockerSandbox as RaySafeDockerSandbox  # noqa: F401
from .r2e_task import RaySafeShellTool as RaySafeShellTool  # noqa: F401

logger = logging.getLogger(__name__)


class DeepSWETaskConfig(TaskConfig):
    name: str = "deep_swe"
    agent_timeout: float = Field(default=10800.0, gt=0)
    eval_timeout: float = Field(default=1800.0, gt=0)


def parse_deepswe_reward(payload: bytes | str) -> dict[str, Any]:
    text = payload.decode("utf-8", errors="replace") if isinstance(payload, bytes) else payload
    value = json.loads(text)
    if not isinstance(value, dict) or "reward" not in value:
        raise ValueError("DeepSWE reward.json must contain a reward field")
    reward = float(value["reward"])
    if reward not in (0.0, 1.0):
        raise ValueError(f"DeepSWE binary reward must be 0 or 1, got {reward}")
    return {**value, "reward": reward}


@register_task("deep_swe")
class DeepSWETask(Task):
    """Run a DeepSWE agent image, then grade its patch in a separate verifier image."""

    config_model = DeepSWETaskConfig

    async def run(self) -> TaskResult:
        cfg: DeepSWETaskConfig = self.config  # type: ignore[assignment]
        metadata = cfg.metadata
        task_id = str(metadata.get("task_id") or "")
        base_commit = str(metadata.get("base_commit_hash") or "")
        verifier_image = str(metadata.get("verifier_image") or "")
        if not task_id or re.fullmatch(r"[0-9a-f]{7,40}", base_commit) is None or not verifier_image:
            raise ValueError(f"invalid DeepSWE runtime metadata for {task_id or '<unknown>'}")

        finished = False
        agent_info: dict[str, Any] = {}
        patch = ""
        async with self.build_sandbox() as agent_sandbox:
            agent = self.build_agent()
            try:
                agent_result = await asyncio.wait_for(
                    agent.run(sandbox=agent_sandbox, messages=cfg.prompt, workdir="/app"),
                    timeout=cfg.agent_timeout,
                )
                finished = agent_result.finished
                agent_info = agent_result.info
            except TimeoutError:
                logger.warning("DeepSWE agent timed out for %s", task_id)
                agent_info = {"timed_out": True}
            except Exception as error:  # grade the resulting filesystem even after agent failure
                logger.exception("DeepSWE agent failed for %s; continuing to verifier", task_id)
                agent_info = {"error": f"{type(error).__name__}: {error}"}

            collected = await agent_sandbox.exec_shell(
                "set -eu; "
                "git config --global --add safe.directory /app; "
                f"git diff --binary {base_commit} HEAD",
                timeout=300,
                workdir="/app",
            )
            if collected.exit_code != 0:
                raise RuntimeError(
                    f"failed to collect DeepSWE patch for {task_id}: "
                    f"{collected.stderr or collected.stdout}"
                )
            patch = collected.stdout

        verifier_config = cfg.sandbox.model_copy(
            deep=True,
            update={"image": verifier_image, "runtime_timeout": cfg.eval_timeout + 300},
        )
        async with build_sandbox(verifier_config) as verifier_sandbox:
            await verifier_sandbox.write_file("/logs/artifacts/model.patch", patch)
            evaluation = await verifier_sandbox.exec(
                ["bash", "/tests/test.sh"], timeout=cfg.eval_timeout, workdir="/app"
            )
            try:
                reward_payload = await verifier_sandbox.read_file("/logs/verifier/reward.json")
                reward = parse_deepswe_reward(reward_payload)
            except Exception as error:
                raise RuntimeError(
                    f"DeepSWE verifier produced no valid reward.json for {task_id}; "
                    f"exit={evaluation.exit_code}, output={(evaluation.stdout + evaluation.stderr)[-4000:]}"
                ) from error

        extra_info = {
            **reward,
            "agent": agent_info,
            "eval_exit_code": evaluation.exit_code,
            "eval_output_tail": (evaluation.stdout + evaluation.stderr)[-12000:],
            "model_patch_bytes": len(patch.encode("utf-8")),
            "task_id": task_id,
        }
        score = float(reward["reward"])
        return TaskResult(reward=score, accuracy=score, finished=finished, extra_info=extra_info)


async def run_deepswe_task(**kwargs: Any) -> TaskResult:
    """Uni-Agent runner entry point; importing this module registers DeepSWE."""
    return await _run_task(**kwargs)
