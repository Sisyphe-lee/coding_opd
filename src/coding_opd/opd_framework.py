from __future__ import annotations

import asyncio
from typing import Any

import ray

from coding_opd.profiling import ProfiledLLMClient, profile_span

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

    async def _run_prompt_rollouts(
        self, *, sample_fields, sample_index, global_steps, partition_id, num_sessions
    ):
        # Per-prompt copies keep concurrent train/validation calls isolated. Pass
        # progress explicitly to the runner instead of reading mutable trainer state.
        sample_fields = dict(sample_fields)
        tools_kwargs = dict(sample_fields.get("tools_kwargs") or {})
        tools_kwargs["_opd_progress"] = {
            "step": _as_python_scalar(global_steps),
            "training": partition_id == "train",
        }
        sample_fields["tools_kwargs"] = tools_kwargs
        with profile_span(
            "prompt", uid=str(sample_fields.get("uid", "")), sample=sample_index,
            step=_as_python_scalar(global_steps), partition=partition_id,
        ):
            return await super()._run_prompt_rollouts(
                sample_fields=sample_fields,
                sample_index=sample_index,
                global_steps=global_steps,
                partition_id=partition_id,
                num_sessions=num_sessions,
            )

    async def _run_agent_episode_with_concurrency_limit(self, **kwargs):
        with profile_span("session_wait_and_run", session=kwargs["session_index"]):
            return await super()._run_agent_episode_with_concurrency_limit(**kwargs)

    async def _run_agent_episode(self, **kwargs):
        # This begins after acquiring the concurrency slot. Its start minus
        # session_wait_and_run.start measures slot waiting without copying the
        # upstream semaphore/dispatch implementation.
        runner = kwargs["runner_config"].runner_kwargs
        progress = kwargs["sample_fields"].get("tools_kwargs", {}).get("_opd_progress", {})
        if runner.get("opd_algorithm") == "adaptive" and progress.get("training"):
            kwargs["sampling_params"] = {
                **kwargs["sampling_params"],
                "_opd_adaptive": {
                    "threshold": runner.get("adaptive_threshold", 0.1),
                    "routing_key": _as_python_scalar(kwargs["sample_fields"].get(self._teacher_key)),
                },
            }
        with profile_span("agent_episode"):
            return await super()._run_agent_episode(**kwargs)

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
            teacher_client={key: ProfiledLLMClient(client, "teacher") for key, client in teacher_client.items()},
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
        with profile_span("teacher_and_enqueue", session=session_index, trajectories=len(trajectories)):
            if self._distillation_enabled and partition_id != "val":
                with profile_span("teacher_score"):
                    await asyncio.gather(
                        *(self._attach_teacher_logprobs(trajectory, sample_fields) for trajectory in trajectories
                          if "teacher_logprobs" not in trajectory.extra_fields)
                    )
            with profile_span("tq_write"):
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
        self.synchronous_rollouts = False

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
        af = config.actor_rollout_ref.rollout.custom.agent_framework
        synchronous = af.get("synchronous_rollouts", False)
        adaptive = af.agent_runners.task.runner_kwargs.get("opd_algorithm", "vanilla") == "adaptive"
        if adaptive and not synchronous:
            raise ValueError("Adaptive requires synchronous_rollouts=true")
        if synchronous and config.trainer.v1.trainer_mode == "separate_async":
            async_config = config.trainer.v1.separate_async
            if (async_config.num_warmup_batches or async_config.get("checkpoint_safe_prefetch", False)
                    or async_config.hybrid_rollout.enable_switch):
                raise ValueError("Synchronous rollouts require warmup=0, prefetch=false and hybrid switch=false")
        student = ProfiledLLMClient(llm_client, "rollout")
        from coding_opd.opd_gateway import OPDGatewayManager
        from coding_opd.adaptive_rollout import AdaptiveRolloutClient

        manager = None
        if adaptive:
            manager = AsyncTeacherLLMServerManager(
                config=config,
                teacher_client={key: ProfiledLLMClient(client, "teacher") for key, client in teacher_client.items()},
            )
        gateway_manager = OPDGatewayManager(config, AdaptiveRolloutClient(student, manager))
        framework_worker = OPDAgentFrameworkWorker.remote(
            config=config,
            gateway_manager=gateway_manager,
            teacher_client=teacher_client,
            reward_loop_worker_handles=reward_loop_worker_handles,
        )

        instance = cls()
        instance.framework_worker = framework_worker
        instance.gateway_manager = gateway_manager
        instance.synchronous_rollouts = synchronous
        return instance

    def generate_sequences(self, prompts) -> None:
        if self.framework_worker is None:
            raise RuntimeError("framework must be initialized before generate_sequences")
        result = self.framework_worker.generate_sequences.remote(prompts)
        if self.synchronous_rollouts:
            # One batch uses fixed Student weights. All sessions and online
            # Teacher scoring finish before the Actor update can start.
            ray.get(result)

    def generate_sequences_and_wait(self, prompts) -> None:
        if self.framework_worker is None:
            raise RuntimeError("framework must be initialized before generate_sequences")
        ray.get(self.framework_worker.generate_sequences.remote(prompts))
