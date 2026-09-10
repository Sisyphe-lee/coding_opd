import asyncio
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from coding_opd.algorithms import get_algorithm
from coding_opd.opd_framework import OPDGatewayAgentFramework
from coding_opd import r2e_task
from uni_agent.framework.framework import GatewayAgentFramework
from uni_agent.tasks.config import TaskConfigResolver


def test_horizon_schedule_boundaries_and_resume() -> None:
    vanilla = get_algorithm()
    tcod = get_algorithm("tcod", growth_interval=2)
    assert [tcod.rollout_max_turns(step, 24) for step in (0, 1, 2, 3, 16, 46, 256)] == [1, 1, 2, 2, 9, 24, 24]
    assert all(vanilla.rollout_max_turns(step, 24) == 24 for step in (0, 1, 16, 256))
    # A fresh instance at a restored step needs no separate curriculum state.
    assert get_algorithm("tcod").rollout_max_turns(16, 24) == tcod.rollout_max_turns(16, 24)


@pytest.mark.parametrize("interval", [0, -1, 1.5, True, "2"])
def test_invalid_growth_interval(interval) -> None:
    with pytest.raises(ValueError, match="growth_interval"):
        get_algorithm("tcod", growth_interval=interval)


def test_unknown_algorithm_and_invalid_progress() -> None:
    with pytest.raises(ValueError, match="Unknown OPD algorithm"):
        get_algorithm("unknown")
    with pytest.raises(ValueError, match="step"):
        get_algorithm("tcod").rollout_max_turns(None, 24)
    with pytest.raises(ValueError, match="max_turns"):
        get_algorithm("tcod").rollout_max_turns(0, 0)


@pytest.mark.parametrize("name,training,step,expected", [
    ("vanilla", True, 0, 24),
    ("tcod", True, 0, 1),
    ("tcod", True, 16, 9),
    ("tcod", True, 256, 24),
    ("tcod", False, 0, 24),
    ("tcod", False, None, 24),
])
def test_framework_to_runner_horizon(monkeypatch, name, training, step, expected) -> None:
    config_path = str(Path(__file__).parents[1] / "configs/coding_react.yaml")
    sample = {"tools_kwargs": {"task": {"name": "r2e_gym", "agent": {"action_timeout": 17}}}}
    original = deepcopy(sample)
    captured = {}

    async def capture_task(**kwargs):
        captured.update(TaskConfigResolver.from_file(config_path).resolve(kwargs["tools_kwargs"]["task"]))
        return "result"

    async def parent_rollouts(self, **kwargs):
        return await r2e_task.run_r2e_task(
            opd_algorithm=name, task_config_path=config_path,
            tools_kwargs=kwargs["sample_fields"]["tools_kwargs"],
        )

    monkeypatch.setattr(r2e_task, "_run_task", capture_task)
    monkeypatch.setattr(GatewayAgentFramework, "_run_prompt_rollouts", parent_rollouts)
    framework = object.__new__(OPDGatewayAgentFramework)
    result = asyncio.run(framework._run_prompt_rollouts(
        sample_fields=sample, sample_index=0, global_steps=np.int64(step) if step is not None else None,
        partition_id="train" if training else "val", num_sessions=1,
    ))
    assert result == "result"
    assert captured["agent"]["max_steps"] == expected
    assert captured["agent"]["action_timeout"] == 17
    assert captured["agent"]["model"]["max_total_tokens"] == 16384
    assert captured["sandbox"]["sandbox_kwargs"]["run_args"][-2:] == ["--network", "none"]
    assert sample == original


def test_tcod_requires_progress_and_respects_sample_cap(monkeypatch) -> None:
    with pytest.raises(ValueError, match="explicit _opd_progress"):
        asyncio.run(r2e_task.run_r2e_task(opd_algorithm="tcod"))

    captured = {}

    async def capture(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(r2e_task, "_run_task", capture)
    asyncio.run(r2e_task.run_r2e_task(
        opd_algorithm="tcod",
        task_config_path=str(Path(__file__).parents[1] / "configs/coding_react.yaml"),
        tools_kwargs={"task": {"name": "r2e_gym", "agent": {"max_steps": 3}},
                      "_opd_progress": {"training": True, "step": 16}},
    ))
    assert captured["tools_kwargs"]["task"]["agent"]["max_steps"] == 3


def test_default_vanilla_runner_is_passthrough(monkeypatch) -> None:
    captured = {}

    async def capture(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(r2e_task, "_run_task", capture)
    kwargs = {"tools_kwargs": {"task": {"name": "r2e_gym"}}, "raw_prompt": "original"}
    asyncio.run(r2e_task.run_r2e_task(**kwargs))
    assert captured == kwargs
