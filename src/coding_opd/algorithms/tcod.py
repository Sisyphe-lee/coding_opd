from dataclasses import dataclass

from .base import OPDAlgorithm, require_integer


@dataclass(frozen=True)
class TCOD(OPDAlgorithm):
    growth_interval: int = 2

    def __post_init__(self) -> None:
        require_integer("growth_interval", self.growth_interval, 1)

    def rollout_max_turns(self, step: int, max_turns: int) -> int:
        require_integer("step", step, 0)
        require_integer("max_turns", max_turns, 1)
        return min(max_turns, 1 + step // self.growth_interval)
