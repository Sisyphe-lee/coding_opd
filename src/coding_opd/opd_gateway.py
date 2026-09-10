"""OPD stopping and Adaptive wiring over Uni-Agent's existing HTTP and session implementation."""

import ray
from omegaconf import OmegaConf

from uni_agent.gateway.config import GatewayActorConfig
from uni_agent.gateway.gateway import _GatewayActor
from uni_agent.gateway.manager import GatewayManager
from verl.utils.config import omega_conf_to_dataclass

from coding_opd.adaptive_rollout import AdaptiveSession
from coding_opd.algorithms.adaptive import AdaptiveOPD


class TerminalCodec:
    """Keep terminal engine reasons even when a partial reply parses as a tool call."""

    def __init__(self, codec):
        self.codec = codec

    def __getattr__(self, name):
        return getattr(self.codec, name)

    async def decode_response(self, response_ids, *, tools=None, stop_reason=None):
        if stop_reason in ("repetition", "length"):
            tools = None
        return await self.codec.decode_response(response_ids, tools=tools, stop_reason=stop_reason)


class OPDGatewayActor(_GatewayActor):
    def __init__(self, config, backend):
        super().__init__(config, backend)
        self._codec = TerminalCodec(self._codec)

    async def create_session(self, session_id, metadata=None, sampling_params=None):
        sampling_params = dict(sampling_params or {})
        adaptive = sampling_params.pop("_opd_adaptive", None)
        handle = await super().create_session(session_id, metadata, sampling_params)
        if adaptive is not None:
            self._backend.sessions[session_id] = AdaptiveSession(
                AdaptiveOPD(threshold=adaptive["threshold"]), adaptive["routing_key"],
            )
        return handle

    async def finalize_session(self, session_id):
        try:
            trajectories = await super().finalize_session(session_id)
            state = self._backend.sessions.get(session_id)
            return state.attach(trajectories) if state is not None else trajectories
        finally:
            self._backend.sessions.pop(session_id, None)

    async def abort_session(self, session_id):
        try:
            return await super().abort_session(session_id)
        finally:
            self._backend.sessions.pop(session_id, None)


class OPDGatewayManager(GatewayManager):
    def __init__(self, config, backend):
        # Upstream's manager hardcodes GatewayActor. Only pool construction is
        # specialized here; routing, cleanup and the complete session loop stay
        # inherited. These research runs use one node.
        af = config.actor_rollout_ref.rollout.custom.agent_framework
        rollout = config.actor_rollout_ref.rollout
        model = omega_conf_to_dataclass(config.actor_rollout_ref.model)
        template = OmegaConf.select(config, "data.apply_chat_template_kwargs", default={}) or {}
        if OmegaConf.is_config(template):
            template = OmegaConf.to_container(template, resolve=True)
        actor_config = GatewayActorConfig(
            tokenizer=model.tokenizer, processor=model.processor,
            tool_parser_name=rollout.get("multi_turn", {}).get("format"),
            rollout_backend=rollout.name,
            enable_tool_parser_cache=af.get("enable_tool_parser_cache", True),
            apply_chat_template_kwargs=template,
            prompt_length=rollout.prompt_length, response_length=rollout.response_length,
            enable_last_assistant_rollback=af.get("enable_last_assistant_rollback", True),
        )
        actor = ray.remote(OPDGatewayActor)
        self.gateways = [actor.remote(actor_config, backend=backend) for _ in range(af.gateway_count)]
        ray.get([gateway.start.remote() for gateway in self.gateways])
        self.gateway_count = len(self.gateways)
        self.active_sessions_per_gateway = [0] * self.gateway_count
        self._session_to_gateway_index = {}
