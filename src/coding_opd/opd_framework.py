from __future__ import annotations

import asyncio
from typing import Any

import ray

from uni_agent.framework.entry import build_gateway_manager
from uni_agent.framework.framework import GatewayAgentFramework
from uni_agent.gateway.session import Trajectory
from uni_agent.rlinsight_adapter import init_rollout_trace_config
from verl.experimental.teacher_loop.teacher_manager import AsyncTeacherLLMServerManager
from verl.trainer.distillation.losses import is_distillation_enabled
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.transferqueue_utils import tq
from verl.workers.config.model import HFModelConfig


def _as_python_scalar(value: object | None) -> object | None:
    if value is not None and hasattr(value, "item"):
        return value.item()
    return value


class OPDGatewayAgentFramework(GatewayAgentFramework):
    """Uni-Agent gateway framework with veRL teacher scoring before TQ writes."""

    def configure_teacher(self, *, config, teacher_client) -> None:
        self._distillation_enabled = is_distillation_enabled(config.distillation)
        if not self._distillation_enabled:
            self._teacher_key = None
            self._teacher_server_manager = None
            return
        if teacher_client is None:
            raise ValueError("distillation is enabled but teacher_client is missing")

        self._teacher_key = str(config.distillation.teacher_key)
        self._teacher_server_manager = AsyncTeacherLLMServerManager(
            config=config,
            teacher_client=teacher_client,
        )

    async def _attach_teacher_logprobs(
        self,
        trajectory: Trajectory,
        sample_fields: dict[str, object],
    ) -> None:
        routing_key = _as_python_scalar(sample_fields.get(self._teacher_key))
        teacher_ids, teacher_logprobs = await self._teacher_server_manager.compute_teacher_logprobs_single(
            sequence_ids=trajectory.prompt_ids + trajectory.response_ids,
            multi_modal_data=trajectory.multi_modal_data,
            mm_processor_kwargs=None,
            routing_key=routing_key,
        )
        trajectory.extra_fields["teacher_ids"] = teacher_ids
        trajectory.extra_fields["teacher_logprobs"] = teacher_logprobs

    async def _write_session_trajectories_to_tq(
        self,
        *,
        uid: str,
        session_index: int,
        trajectories: list[Trajectory],
        sample_fields: dict[str, object],
        global_steps: int | None,
        partition_id: str,
    ) -> None:
        if self._distillation_enabled and partition_id != "val":
            await asyncio.gather(
                *(self._attach_teacher_logprobs(trajectory, sample_fields) for trajectory in trajectories)
            )
        await super()._write_session_trajectories_to_tq(
            uid=uid,
            session_index=session_index,
            trajectories=trajectories,
            sample_fields=sample_fields,
            global_steps=global_steps,
            partition_id=partition_id,
        )


@ray.remote
class OPDAgentFrameworkWorker:
    def __init__(self, *, config, gateway_manager, teacher_client=None, reward_loop_worker_handles=None) -> None:
        tq.init()
        init_rollout_trace_config(config)
        model_config: HFModelConfig = omega_conf_to_dataclass(config.actor_rollout_ref.model)
        self.framework = OPDGatewayAgentFramework.from_config(
            config=config,
            gateway_manager=gateway_manager,
            processor=model_config.processor,
            reward_loop_worker_handles=reward_loop_worker_handles,
        )
        self.framework.configure_teacher(config=config, teacher_client=teacher_client)

    async def generate_sequences(self, prompts) -> None:
        await self.framework.generate_sequences(prompts)


class OPDAgentFrameworkRolloutAdapter:
    """Trainer adapter that preserves Uni-Agent rollouts and accepts veRL teachers."""

    def __init__(self) -> None:
        self.framework_worker = None
        self.gateway_manager = None

    @classmethod
    def create(
        cls,
        *,
        config,
        llm_client,
        teacher_client=None,
        reward_loop_worker_handles=None,
        **_: Any,
    ) -> OPDAgentFrameworkRolloutAdapter:
        gateway_manager = build_gateway_manager(config=config, llm_client=llm_client)
        framework_worker = OPDAgentFrameworkWorker.remote(
            config=config,
            gateway_manager=gateway_manager,
            teacher_client=teacher_client,
            reward_loop_worker_handles=reward_loop_worker_handles,
        )

        instance = cls()
        instance.framework_worker = framework_worker
        instance.gateway_manager = gateway_manager
        return instance

    def generate_sequences(self, prompts) -> None:
        if self.framework_worker is None:
            raise RuntimeError("framework must be initialized before generate_sequences")
        self.framework_worker.generate_sequences.remote(prompts)

    def generate_sequences_and_wait(self, prompts) -> None:
        if self.framework_worker is None:
            raise RuntimeError("framework must be initialized before generate_sequences")
        ray.get(self.framework_worker.generate_sequences.remote(prompts))
