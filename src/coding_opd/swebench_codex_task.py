"""Isolated SWE-bench verification using the installed official harness APIs."""
from __future__ import annotations

import asyncio
import importlib.metadata
import json
import re
from pathlib import Path

from pydantic import Field
from swebench.harness.grading import get_eval_report, get_logs_eval
from swebench.harness.constants import MAP_REPO_TO_EXT, MAP_REPO_VERSION_TO_SPECS
from swebench.harness.test_spec.create_scripts import make_eval_script_list
from swebench.harness.test_spec.test_spec import TestSpec
from uni_agent.sandbox import build_sandbox
from uni_agent.tasks.base import Task, TaskConfig, TaskResult
from uni_agent.tasks.registry import register_task

from .r2e_task import RaySafeDockerSandbox as RaySafeDockerSandbox  # noqa: F401


def installed_image_test_spec(sample: dict) -> TestSpec:
    """Official eval commands without rebuilding/downloading an existing image.

    make_test_spec also generates environment build scripts and can fetch remote
    requirements. These canonical images are already built; only that unused
    build-script generation is omitted, not any evaluation commands.
    """
    specs = MAP_REPO_VERSION_TO_SPECS[sample["repo"]][sample["version"]]
    def tests(key):
        value = sample[key]
        return json.loads(value) if isinstance(value, str) else value
    return TestSpec(
        instance_id=sample["instance_id"], repo=sample["repo"], version=sample["version"],
        repo_script_list=[], env_script_list=[],
        eval_script_list=make_eval_script_list(
            sample, specs, "testbed", "/testbed", sample["base_commit"], sample["test_patch"]
        ), arch="x86_64", FAIL_TO_PASS=tests("FAIL_TO_PASS"), PASS_TO_PASS=tests("PASS_TO_PASS"),
        language=MAP_REPO_TO_EXT[sample["repo"]], docker_specs=specs.get("docker_specs", {}), namespace=None,
    )


def collect_patch_command(base_commit: str) -> str:
    if re.fullmatch(r"[0-9a-f]{40}", base_commit) is None:
        raise ValueError("invalid SWE-bench base commit")
    # Include committed and uncommitted changes, including new files. Unlike
    # DeepSWE, SWE-bench submits a patch and does not require an agent commit.
    return (
        "set -eu; git config --global --add safe.directory /testbed; "
        f"git add -A; git -c core.fileMode=false diff --binary --cached {base_commit}"
    )


class VerifiedCodexConfig(TaskConfig):
    name: str = "swe_bench_codex"
    agent_timeout: float = Field(default=10800, gt=0)
    eval_timeout: float = Field(default=1800, gt=0)


@register_task("swe_bench_codex")
class VerifiedCodexTask(Task):
    config_model = VerifiedCodexConfig

    async def run(self) -> TaskResult:
        cfg = self.config
        sample = cfg.metadata
        instance_id = sample["instance_id"]
        spec = installed_image_test_spec(sample)
        artifacts = Path(cfg.agent.log_dir) / "verification"
        artifacts.mkdir(parents=True, exist_ok=False)
        finished = False
        async with self.build_sandbox() as sandbox:
            initial = await sandbox.exec_shell("git rev-parse HEAD", timeout=30, workdir="/testbed")
            if initial.exit_code:
                raise RuntimeError("cannot identify canonical image checkout HEAD")
            image_head = initial.stdout.strip()
            # Canonical image setup commits chmod and other image preparation.
            # Do not misattribute those pre-existing changes to the model.
            collect_command = collect_patch_command(image_head)
            try:
                result = await asyncio.wait_for(
                    self.build_agent().run(sandbox=sandbox, messages=cfg.prompt, workdir="/testbed"),
                    timeout=cfg.agent_timeout,
                )
                finished, agent_info = result.finished, result.info
            except TimeoutError:
                agent_info = {"timed_out": True}
            except Exception as error:
                agent_info = {"error": f"{type(error).__name__}: {error}"}
            collected = await sandbox.exec_shell(
                collect_command, timeout=300, workdir="/testbed"
            )
            if collected.exit_code:
                raise RuntimeError(f"patch collection failed: {collected.stderr}")
            patch = collected.stdout

        prediction = {"instance_id": instance_id, "model_name_or_path": cfg.agent.model.model_name,
                      "model_patch": patch}
        (artifacts / "prediction.json").write_text(json.dumps(prediction) + "\n")
        (artifacts / "patch.diff").write_text(patch)
        (artifacts / "eval.sh").write_text(spec.eval_script)
        info = {"agent": agent_info, "model_patch_bytes": len(patch.encode()),
                "collection_base_commit": image_head, "dataset_base_commit": sample["base_commit"],
                "swebench_version": importlib.metadata.version("swebench"),
                "artifacts": str(artifacts), "instance_id": instance_id}
        if not patch.strip():
            info.update(resolved=False, failure_kind="empty_patch")
            (artifacts / "report.json").write_text(json.dumps({instance_id: info}, indent=2))
            return TaskResult(reward=0, accuracy=0, finished=finished, extra_info=info)

        # No agent code, socket, or logs are mounted into the independent grader.
        verifier_cfg = cfg.sandbox.model_copy(deep=True)
        verifier_cfg.runtime_timeout = cfg.eval_timeout + 300
        verifier_cfg.sandbox_kwargs["run_args"] = ["--cgroups=disabled", "--network", "none"]
        async with build_sandbox(verifier_cfg) as verifier:
            await verifier.write_file("/tmp/patch.diff", patch)
            applied = False
            apply_logs = []
            # Same ordered fallbacks as swebench.harness.run_evaluation.
            for command in ("git apply --verbose", "git apply --verbose --reject",
                            "patch --batch --fuzz=5 -p1 -i"):
                response = await verifier.exec_shell(
                    f"{command} /tmp/patch.diff", timeout=300, workdir="/testbed"
                )
                apply_logs.append(response.stdout + response.stderr)
                if response.exit_code == 0:
                    applied = True
                    break
            (artifacts / "patch_apply.log").write_text("\n".join(apply_logs))
            if not applied:
                raise RuntimeError(f"SWE-bench patch application failed; artifacts={artifacts}")
            await verifier.write_file("/eval.sh", spec.eval_script)
            response = await verifier.exec_shell(
                "bash /eval.sh > /tmp/test_output.txt 2>&1", timeout=cfg.eval_timeout, workdir="/testbed"
            )
            output = await verifier.read_file("/tmp/test_output.txt")
            if isinstance(output, bytes):
                output = output.decode("utf-8", errors="replace")
            log_path = artifacts / "test_output.txt"
            log_path.write_text(output)
            if response.exit_code == -1:
                raise RuntimeError(f"SWE-bench verifier timed out; artifacts={artifacts}")
            status_map, found = get_logs_eval(spec, str(log_path))
            report = get_eval_report(spec, prediction, str(log_path), include_tests_status=True)
            (artifacts / "report.json").write_text(json.dumps(report, indent=2) + "\n")
            if not found or not status_map:
                raise RuntimeError(f"SWE-bench produced no parseable test results; artifacts={artifacts}")
        info.update(report[instance_id], eval_exit_code=response.exit_code,
                    parsed_tests=len(status_map))
        score = float(info["resolved"])
        return TaskResult(reward=score, accuracy=score, finished=finished, extra_info=info)
