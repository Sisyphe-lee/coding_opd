"""Explicit algorithm selection; all current algorithms share sampled-token K3."""

from .base import OPDAlgorithm
from .adaptive import AdaptiveOPD
from .tcod import TCOD
from .vanilla import VanillaOPD


def get_algorithm(name: str = "vanilla", *, growth_interval: int = 2, threshold: float = 0.1) -> OPDAlgorithm:
    if name == "vanilla":
        return VanillaOPD()
    if name == "tcod":
        return TCOD(growth_interval=growth_interval)
    if name == "adaptive":
        return AdaptiveOPD(threshold=threshold)
    raise ValueError(f"Unknown OPD algorithm {name!r}; choose vanilla, tcod or adaptive")
