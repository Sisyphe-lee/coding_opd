from .base import OPDAlgorithm, require_integer


class VanillaOPD(OPDAlgorithm):
    def rollout_max_turns(self, step: int, max_turns: int) -> int:
        require_integer("step", step, 0)
        require_integer("max_turns", max_turns, 1)
        return max_turns
