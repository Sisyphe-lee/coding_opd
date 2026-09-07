"""Coding OPD registration shim for Uni-Agent's SWE-bench evaluator."""

from __future__ import annotations

from typing import Any

from uni_agent.framework.task_runner import run_task as _run_task

# Import side effects register the native SWE-bench task.  Importing the local
# R2E module first registers the Ray-safe Podman sandbox and shell used by the
# project; the benchmark task then reuses those implementations via YAML.
from .r2e_task import RaySafeDockerSandbox as RaySafeDockerSandbox  # noqa: F401
from .r2e_task import RaySafeShellTool as RaySafeShellTool  # noqa: F401
from uni_agent.tasks.swe_bench.task import SWEBenchTask as SWEBenchTask  # noqa: F401
from uni_agent.tasks.swe_bench.task import SWEBenchTaskConfig as SWEBenchTaskConfig  # noqa: F401


async def run_swe_bench_task(**kwargs: Any):
    """Uni-Agent runner entry point for inference-only Verified evaluation."""
    return await _run_task(**kwargs)
