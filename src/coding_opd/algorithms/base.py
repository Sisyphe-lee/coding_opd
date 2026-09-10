"""Small, framework-independent policies for OPD interaction horizons."""

from abc import ABC, abstractmethod


class OPDAlgorithm(ABC):
    @abstractmethod
    def rollout_max_turns(self, step: int, max_turns: int) -> int:
        """Return the training episode's turn limit, fixed before interaction."""

    def retained_turns(self, entropies: list[float]) -> int | None:
        """Optional online frontier; None keeps generating within the turn limit."""
        return None


def require_integer(name: str, value: int, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
