"""Native rollout stopping and Adaptive Teacher diagnostics for the pinned vLLM stack."""

from contextvars import ContextVar

import ray
from verl.workers.rollout.replica import TokenOutput
from verl.workers.rollout.vllm_rollout import vllm_async_server as upstream

from coding_opd.adaptive_teacher import extract_adaptive_logprobs

_suffix_start = ContextVar("opd_entropy_suffix_start", default=None)
_extract_prompt_logprobs = upstream.extract_prompt_logprobs


def _extract(output, num_prompt_logprobs, result_dict):
    result_dict["engine_finish_reason"] = output.outputs[0].finish_reason
    start = _suffix_start.get()
    if start is None:
        return _extract_prompt_logprobs(output, num_prompt_logprobs, result_dict)
    return extract_adaptive_logprobs(output, start, result_dict)


class OPDServer(upstream.vLLMHttpServer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # This hook is local to each server process. A ContextVar keeps
        # concurrent requests' suffix boundaries independent across awaits.
        upstream.extract_prompt_logprobs = _extract

    async def generate(self, prompt_ids, sampling_params, request_id, **kwargs):
        sampling_params = dict(sampling_params)
        policy = (self.config.custom or {}).get("coding_react", {})
        if policy:
            remaining = policy["max_context_tokens"] - len(prompt_ids)
            if remaining <= 0:
                return TokenOutput(token_ids=[], log_probs=[], stop_reason="length",
                                   extra_fields={"global_steps": self.global_steps})
            sampling_params["max_tokens"] = min(
                sampling_params.get("max_tokens", policy["max_tokens_per_turn"]),
                policy["max_tokens_per_turn"], remaining,
            )
            detection = policy.get("repetition_detection")
            if detection:
                from vllm.sampling_params import RepetitionDetectionParams
                sampling_params["repetition_detection"] = RepetitionDetectionParams(**dict(detection))
        token = _suffix_start.set(sampling_params.pop("opd_entropy_start", None))
        try:
            output = await super().generate(prompt_ids, sampling_params, request_id, **kwargs)
            reason = output.extra_fields.pop("engine_finish_reason", None)
            if reason in ("length", "repetition"):
                output.stop_reason = reason
            return output
        finally:
            _suffix_start.reset(token)


class OPDReplica(upstream.vLLMReplica):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.server_class = ray.remote(OPDServer)
