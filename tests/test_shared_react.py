import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
from uni_agent.tasks import TaskConfigResolver, get_task

from coding_opd import swe_bench_task
from coding_opd.opd_gateway import TerminalCodec
from coding_opd.react_agent import CodingReActAgent, CodingReActConfig

ROOT = Path(__file__).parents[1]


def test_reference_solver_matches_pinned_upstream():
    import uni_agent
    import yaml
    from omegaconf import OmegaConf
    from coding_opd.rollout_config import configure_coding_rollout

    upstream = Path(uni_agent.__file__).parents[1] / "examples/quickstart/inference/task_config_react.yaml"
    official = yaml.safe_load(upstream.read_text())[0]
    reference_path = ROOT / "configs/uni_agent_react_reference.yaml"
    reference = yaml.safe_load(reference_path.read_text())[0]
    assert reference["agent"] == official["agent"]
    assert reference["prompt_template"] == official["prompt_template"]
    config = OmegaConf.create({})
    configure_coding_rollout(config, reference_path, "swe_bench")
    policy = config.actor_rollout_ref.rollout.custom.coding_react
    assert policy.max_context_tokens == policy.max_tokens_per_turn == 65536
    assert not policy.repetition_detection


def test_training_and_verified_share_solver_and_keep_secrets_out():
    resolver = TaskConfigResolver.from_file(str(ROOT / "configs/coding_react.yaml"))
    tasks = [get_task(resolver.resolve({
        "name": name, "sandbox": {"image": "test-image"},
        "metadata": {"problem_statement": "Fix this public issue", "patch": "SECRET", "test_patch": "SECRET"},
    })) for name in ("r2e_gym", "swe_bench")]
    train, evaluation = [task.config for task in tasks]
    assert train.agent == evaluation.agent
    assert train.prompt == evaluation.prompt
    assert train.sandbox == evaluation.sandbox
    assert train.agent_timeout == evaluation.agent_timeout == 900
    assert "SECRET" not in str(train.prompt)
    assert train.agent.model.top_k == -1
    assert train.agent.model.max_total_tokens == 16384
    assert type(tasks[0].build_agent()) is type(tasks[1].build_agent()) is CodingReActAgent


@pytest.mark.parametrize("finish_reason", ["repetition", "length"])
def test_terminal_generation_never_executes_tools(finish_reason):
    calls = []

    class Model:
        async def query(self, transcript, **kwargs):
            calls.append("generate")
            return "partial response", [{"function": {"name": "shell", "arguments": "{}"}}], {
                "prompt_tokens": 20, "completion_tokens": 10, "finish_reason": finish_reason,
            }

    class Toolbox:
        async def call(self, *args, **kwargs):
            pytest.fail("a terminal response must not execute tools")

    info = {"steps": 1, "total_tokens": 0}
    transcript = [{"role": "user", "content": "original issue"}]
    agent = CodingReActAgent()
    reason = asyncio.run(agent.step(CodingReActConfig(), Model(), Toolbox(), transcript, info))
    assert reason == ("repetition" if finish_reason == "repetition" else "token_limit")
    assert info["termination_reason"] == reason
    assert calls == ["generate"]
    assert transcript[0]["content"] == "original issue"
    assert transcript[1] == {"role": "assistant", "content": "partial response"}


@pytest.mark.parametrize("reason", ["repetition", "length", "completed"])
def test_terminal_reason_takes_priority_over_tool_parser(reason):
    class Codec:
        async def decode_response(self, ids, *, tools=None, stop_reason=None):
            if tools:
                return {"tool_calls": ["parsed-tool"]}, "tool_calls"
            return {"content": "partial"}, stop_reason

    message, actual = asyncio.run(TerminalCodec(Codec()).decode_response(
        [1, 2], tools=[{"name": "shell"}], stop_reason=reason,
    ))
    assert actual == ("tool_calls" if reason == "completed" else reason)
    assert ("tool_calls" in message) is (reason == "completed")


@pytest.mark.parametrize("algorithm", ["vanilla", "tcod", "adaptive"])
def test_formal_launcher_disables_cross_batch_and_within_batch_lag(tmp_path, algorithm):
    capture = tmp_path / "capture"
    capture.write_text(f"#!{sys.executable}\nimport json, sys\nprint(json.dumps(sys.argv[1:]))\n")
    capture.chmod(0o755)
    # Deliberately supply the previous asynchronous settings: the formal entry
    # must replace them, including Adaptive's previous two minibatches.
    result = subprocess.run(["bash", str(ROOT / "scripts/run_r2e_opd_train.sh")], check=True,
                            capture_output=True, text=True, env={**os.environ,
        "REPO_ROOT": str(ROOT), "RUNTIME_ROOT": str(tmp_path), "PYTHON_BIN": str(capture),
        "IMAGE_PREFLIGHT": "false", "RUN_NAME": "test", "OPD_ALGORITHM": algorithm,
        "CODING_OPD_PODMAN_SERVICE": "false",
        "PPO_MINI_BATCH_SIZE": "16", "PARAMETER_SYNC_STEP": "2", "ASYNC_PREFETCH": "true",
        "HYBRID_ROLLOUT_ENABLE_SWITCH": "true", "TRAIN_BATCH_SIZE": "32",
    })
    args = json.loads(result.stdout)
    assert "data.train_batch_size=32" in args
    assert "actor_rollout_ref.actor.ppo_mini_batch_size=32" in args
    assert "actor_rollout_ref.actor.ppo_epochs=1" in args
    assert "trainer.v1.separate_async.parameter_sync_step=1" in args
    assert "trainer.v1.separate_async.num_warmup_batches=0" in args
    assert "+trainer.v1.separate_async.checkpoint_safe_prefetch=false" in args
    assert "trainer.v1.separate_async.hybrid_rollout.enable_switch=false" in args
    assert "+actor_rollout_ref.rollout.custom.agent_framework.synchronous_rollouts=true" in args


def test_verified_grades_only_patch_in_new_container(monkeypatch, tmp_path):
    from uni_agent.sandbox.base import ExecResult

    events = []
    patch = "diff --git a/file.py b/file.py\n"

    class Sandbox:
        def __init__(self, name):
            self.name = name
            self.files = {}

        async def __aenter__(self):
            events.append((self.name, "start"))
            return self

        async def __aexit__(self, *args):
            events.append((self.name, "stop"))

        async def exec_shell(self, command, **kwargs):
            events.append((self.name, command))
            stdout = "a" * 40 if command == "git rev-parse HEAD" else patch
            return ExecResult(exit_code=0, stdout=stdout, stderr="")

        async def write_file(self, path, content):
            self.files[path] = content

        async def read_file(self, path):
            return "test passed"

    agent_sandbox, verifier = Sandbox("agent"), Sandbox("verifier")

    class Agent:
        async def run(self, sandbox, **kwargs):
            sandbox.files["/tmp/agent-only"] = "contaminated state"
            return SimpleNamespace(finished=True, info={"steps": 2})

    def build_verifier(config):
        assert config.sandbox_kwargs["run_args"][:3] == ["--cgroups=disabled", "--network", "none"]
        return verifier

    monkeypatch.setattr(swe_bench_task, "installed_image_test_spec", lambda _: SimpleNamespace(eval_script="pytest"))
    monkeypatch.setattr(swe_bench_task, "build_sandbox", build_verifier)
    monkeypatch.setattr(swe_bench_task, "get_logs_eval", lambda *args: ({"test": "PASSED"}, True))
    monkeypatch.setattr(swe_bench_task, "get_eval_report", lambda *args, **kwargs: {"task": {"resolved": True}})
    config = swe_bench_task.SWEBenchTaskConfig(
        agent={"name": "coding_opd_react"}, sandbox={"provider": "coding_opd_podman"},
        metadata={"instance_id": "task", "base_commit": "a" * 40}, verification_dir=str(tmp_path),
    )
    task = swe_bench_task.SWEBenchTask(config)
    monkeypatch.setattr(task, "build_sandbox", lambda: agent_sandbox)
    monkeypatch.setattr(task, "build_agent", Agent)
    result = asyncio.run(task.run())
    assert events.index(("agent", "stop")) < events.index(("verifier", "start"))
    assert verifier.files == {"/tmp/patch.diff": patch, "/eval.sh": "pytest"}
    assert result.accuracy == 1
