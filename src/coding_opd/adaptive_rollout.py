"""Per-session online stopping and reuse of the Teacher's sampled-token labels."""

from dataclasses import dataclass, field
import logging

import torch

from coding_opd.algorithms.adaptive import AdaptiveOPD
from coding_opd.profiling import profile_span

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class ScoredPrefix:
    tokens: list[int]
    logprobs: list[list[float]]


@dataclass
class AdaptiveSession:
    algorithm: AdaptiveOPD
    routing_key: str | None = None
    entropies: list[float] = field(default_factory=list)
    masses: list[float] = field(default_factory=list)
    prefixes: list[ScoredPrefix] = field(default_factory=list)
    stopped: bool = False

    def accept(self, tokens: list[int], extra: dict) -> bool:
        self.entropies.append(extra["opd_entropy"])
        self.masses.append(extra["opd_top16_mass"])
        self.stopped = self.algorithm.retained_turns(self.entropies) is not None
        if self.stopped:
            return False
        # Keep only the latest scoring result for each prefix chain. It already
        # includes all earlier labels; retain separate records across rewrites.
        self.prefixes = [p for p in self.prefixes if tokens[:len(p.tokens)] != p.tokens]
        self.prefixes.append(ScoredPrefix(tokens, extra["prompt_logprobs"]))
        return True

    def attach(self, trajectories):
        retained = []
        for trajectory in trajectories:
            if not any(trajectory.response_mask):
                continue
            tokens = trajectory.prompt_ids + trajectory.response_ids
            # Gateway may split long conversations or roll back a rewritten
            # assistant message. Match exact token prefixes, never turn-count
            # guesses or text retokenization.
            prefix = next((p for p in reversed(self.prefixes)
                           if tokens[:len(p.tokens)] == p.tokens or p.tokens[:len(tokens)] == tokens), None)
            if prefix is None:
                raise ValueError("Adaptive trajectory has no matching Teacher-scored prefix")
            length = min(len(tokens), len(prefix.tokens))
            response_length = length - len(trajectory.prompt_ids)
            if response_length <= 0:
                continue
            trajectory.response_ids = trajectory.response_ids[:response_length]
            trajectory.response_mask = trajectory.response_mask[:response_length]
            if trajectory.response_logprobs is not None:
                trajectory.response_logprobs = trajectory.response_logprobs[:response_length]
            if trajectory.routed_experts is not None:
                trajectory.routed_experts = trajectory.routed_experts[:length]
            # Final dummy row is masked out by veRL's next-token alignment.
            trajectory.extra_fields.update(
                teacher_ids=torch.tensor([[t] for t in tokens[1:length]] + [[0]], dtype=torch.int32),
                teacher_logprobs=torch.tensor(prefix.logprobs[:length - 1] + [[0.0]], dtype=torch.float32),
            )
            retained.append(trajectory)
        return retained


class AdaptiveRolloutClient:
    """Decorate generation in the Gateway actor; sessions still run concurrently."""

    def __init__(self, student, teacher_manager):
        self.student = student
        self.teacher_manager = teacher_manager
        self.sessions: dict[str, AdaptiveSession] = {}

    async def generate(self, request_id, *, prompt_ids, sampling_params, **kwargs):
        state = self.sessions.get(request_id)
        output = await self.student.generate(
            request_id=request_id, prompt_ids=prompt_ids, sampling_params=sampling_params, **kwargs,
        )
        if state is None or not output.token_ids:
            return output
        manager = self.teacher_manager
        key = manager._resolve_teacher_key(state.routing_key)
        sequence = prompt_ids + list(output.token_ids)
        with profile_span("adaptive_teacher_turn", session=request_id, turn=len(state.entropies) + 1):
            teacher = await manager.teacher_client[key].generate(
                request_id=request_id,
                prompt_ids=sequence,
                sampling_params={"max_tokens": 1, "temperature": 1.0, "prompt_logprobs": 16,
                                 "opd_entropy_start": len(prompt_ids) - 1},
                **kwargs,
            )
        accepted = state.accept(sequence, teacher.extra_fields)
        logger.info(
            "ADAPTIVE_TURN session=%s turn=%d entropy=%.6f top16_mass=%.6f retained_turns=%d stopped=%s",
            request_id, len(state.entropies), state.entropies[-1], state.masses[-1],
            len(state.entropies) - int(state.stopped), state.stopped,
        )
        if not accepted:
            # The detection turn is excluded in the original algorithm. An empty
            # length-terminated reply makes ReAct exit before executing its tools.
            # The Gateway keeps the preceding prefix, cropped at finalization.
            return output.model_copy(update={"token_ids": [], "log_probs": [], "routed_experts": None,
                                             "stop_reason": "length"})
        return output
