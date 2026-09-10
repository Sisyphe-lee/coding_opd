import asyncio
from types import SimpleNamespace

import pytest

from coding_opd import r2e_task
from uni_agent.sandbox.base import ExecResult


@pytest.mark.parametrize("evaluate", [False, True])
def test_task_skips_only_post_agent_evaluation(monkeypatch, evaluate) -> None:
    calls = []

    class Sandbox:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            calls.append("closed")

        async def exec_shell(self, *args, **kwargs):
            return ExecResult(exit_code=0, stdout="", stderr="")

        async def exec(self, argv, **kwargs):
            calls.append(argv)
            return ExecResult(
                exit_code=0, stderr="",
                stdout="short test summary info\nPASSED test_example.py::test_ok\n",
            )

    class Agent:
        async def run(self, *, sandbox, **kwargs):
            # Tests explicitly requested by the Agent remain part of its trace.
            await sandbox.exec(["pytest", "agent_selected_test.py"])
            return SimpleNamespace(finished=True, info={"steps": 2})

    cfg = r2e_task.R2EGymTaskConfig(
        sandbox={"provider": "coding_opd_podman"}, agent={"name": "react"},
        run_evaluation=evaluate,
        metadata={"expected_output_json": {"test_ok": "PASSED"}} if evaluate else {},
    )
    task = r2e_task.R2EGymTask(cfg)
    monkeypatch.setattr(task, "build_sandbox", Sandbox)
    monkeypatch.setattr(task, "build_agent", Agent)
    result = asyncio.run(task.run())
    assert calls[0] == ["pytest", "agent_selected_test.py"]
    assert (["bash", "./run_tests.sh"] in calls) is evaluate
    assert calls[-1] == "closed"
    assert result.extra_info["evaluation_ran"] is evaluate
    assert result.accuracy == (1.0 if evaluate else None)
    assert result.finished
    if not evaluate:
        assert "resolved" not in result.extra_info


@pytest.mark.parametrize("algorithm", ["vanilla", "tcod"])
@pytest.mark.parametrize("training", [False, True])
def test_runner_routes_switch_and_preserves_validation(monkeypatch, algorithm, training) -> None:
    captured = {}
    tools = {"task": {"name": "r2e_gym", "agent": {"max_steps": 24}},
             "_opd_progress": {"step": 2, "training": training}}

    async def run(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(r2e_task, "_run_task", run)
    asyncio.run(r2e_task.run_r2e_task(
        opd_algorithm=algorithm, run_evaluation=False, tools_kwargs=tools,
    ))
    task = captured["tools_kwargs"]["task"]
    assert task["run_evaluation"] is (not training)
    assert task["agent"]["max_steps"] == (2 if algorithm == "tcod" and training else 24)
    assert "run_evaluation" not in tools["task"]


def test_standalone_task_evaluates_by_default() -> None:
    assert r2e_task.R2EGymTaskConfig(
        sandbox={"provider": "coding_opd_podman"}, agent={"name": "react"},
    ).run_evaluation is True
