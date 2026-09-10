"""Codex compatibility registration over the shared isolated SWE-bench grader."""

from pathlib import Path
from pydantic import Field
from uni_agent.tasks.registry import register_task

from coding_opd.swe_bench_task import (
    SWEBenchTask,
    SWEBenchTaskConfig,
    collect_patch_command as collect_patch_command,
    installed_image_test_spec as installed_image_test_spec,
)


class VerifiedCodexConfig(SWEBenchTaskConfig):
    name: str = "swe_bench_codex"
    agent_timeout: float = Field(default=10800, gt=0)


@register_task("swe_bench_codex")
class VerifiedCodexTask(SWEBenchTask):
    config_model = VerifiedCodexConfig

    def verification_directory(self) -> Path:
        return Path(self.config.agent.log_dir) / "verification"
