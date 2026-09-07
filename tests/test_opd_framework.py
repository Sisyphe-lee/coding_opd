from __future__ import annotations

import asyncio

import numpy as np
import torch

from coding_opd.opd_framework import OPDGatewayAgentFramework
from uni_agent.gateway.session import Trajectory


class _FakeTeacherManager:
    def __init__(self) -> None:
        self.calls = []

    async def compute_teacher_logprobs_single(self, **kwargs):
        self.calls.append(kwargs)
        size = len(kwargs["sequence_ids"])
        return torch.arange(size).unsqueeze(-1), torch.zeros((size, 1))


def test_teacher_scores_exact_gateway_trajectory() -> None:
    framework = object.__new__(OPDGatewayAgentFramework)
    framework._teacher_key = "data_source"
    framework._teacher_server_manager = _FakeTeacherManager()
    trajectory = Trajectory(
        prompt_ids=[11, 12],
        response_ids=[21, 22, 23],
        response_mask=[1, 0, 1],
    )

    asyncio.run(
        framework._attach_teacher_logprobs(
            trajectory,
            {"data_source": np.array("r2e", dtype=object)},
        )
    )

    call = framework._teacher_server_manager.calls[0]
    assert call["sequence_ids"] == [11, 12, 21, 22, 23]
    assert call["routing_key"] == "r2e"
    assert trajectory.extra_fields["teacher_ids"].shape == (5, 1)
    assert trajectory.extra_fields["teacher_logprobs"].shape == (5, 1)
