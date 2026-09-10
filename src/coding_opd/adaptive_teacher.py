"""Sampled-token K3 labels and Top-16 diagnostics from one Teacher forward."""

import math


def extract_adaptive_logprobs(output, start: int, result: dict) -> None:
    """Keep next-token alignment used by veRL; diagnose only this turn's suffix.

    vLLM includes the actual token even when its rank exceeds 16. Do not replace
    that label with a top-ranked token, or renormalize the diagnostic probabilities.
    """
    ids, logprobs = [], []
    entropy, mass, count = 0.0, 0.0, 0
    for position, (token, row) in enumerate(zip(output.prompt_token_ids[1:], output.prompt_logprobs[1:], strict=True)):
        ids.append([token])
        logprobs.append([row[token].logprob])
        if position >= start:
            count += 1
            for value in row.values():
                if value.rank <= 16 and math.isfinite(value.logprob):
                    probability = math.exp(value.logprob)
                    entropy -= probability * value.logprob
                    mass += probability
    result.update(
        prompt_ids=ids + [[0]], prompt_logprobs=logprobs + [[0.0]],
        opd_entropy=entropy / count if count else math.nan,
        opd_top16_mass=mass / count if count else math.nan,
    )
