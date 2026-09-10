"""Causal ALFWorld t0100 frontier, using unnormalized Teacher Top-16 entropy."""

from dataclasses import dataclass
import math

from .vanilla import VanillaOPD


@dataclass(frozen=True)
class AdaptiveOPD(VanillaOPD):
    threshold: float = 0.1
    baseline_turns: int = 3
    sustain_turns: int = 3
    min_retained_turns: int = 3

    def __post_init__(self):
        if not math.isfinite(self.threshold) or self.threshold <= 0:
            raise ValueError("adaptive threshold must be finite and positive")

    def retained_turns(self, entropies: list[float]) -> int | None:
        """Return the prefix length at the first frontier; None means continue.

        Match the original zero-based frontier: the detection turn is excluded.
        The window is an average drift, not three individual threshold crossings.
        """
        baseline = entropies[:self.baseline_turns]
        if len(baseline) < self.baseline_turns or not all(map(math.isfinite, baseline)):
            return None
        reference = sum(baseline) / self.baseline_turns
        for turn in range(max(self.baseline_turns, self.sustain_turns - 1), len(entropies)):
            window = entropies[turn - self.sustain_turns + 1:turn + 1]
            if all(map(math.isfinite, window)) and sum(v - reference for v in window) / self.sustain_turns >= self.threshold:
                return max(self.min_retained_turns, turn)
        return None
