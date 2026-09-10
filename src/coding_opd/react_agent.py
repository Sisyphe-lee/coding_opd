"""The Uni-Agent ReAct loop with explicit engine termination handling."""

from pydantic import Field
from uni_agent.agents.react.agent import ReActAgent, ReActConfig
from uni_agent.agents.registry import register_agent


class CodingReActConfig(ReActConfig):
    name: str = "coding_opd_react"
    repetition_detection: dict[str, int] = Field(default_factory=dict)


class _TurnModel:
    def __init__(self, model):
        self.model = model
        self.stop_reason = None

    async def query(self, *args, **kwargs):
        content, tools, info = await self.model.query(*args, **kwargs)
        self.stop_reason = info.get("finish_reason")
        if self.stop_reason in ("repetition", "length"):
            # Let the existing loop record the response and stop before tools.
            return content, [], {**info, "finish_reason": "length"}
        return content, tools, info


@register_agent("coding_opd_react")
class CodingReActAgent(ReActAgent):
    config_model = CodingReActConfig

    async def run(self, **kwargs):
        result = await super().run(**kwargs)
        if "error" in result.info:
            result.info["termination_reason"] = "unknown_error"
        elif result.info.get("termination_reason") == "completed":
            result.info["termination_reason"] = "max_steps"
        return result

    async def step(self, cfg, model, toolbox, transcript, info):
        turn_model = _TurnModel(model)
        reason = await super().step(cfg, turn_model, toolbox, transcript, info)
        if turn_model.stop_reason == "repetition":
            reason = "repetition"
        info["termination_reason"] = reason
        return reason
