from dataclasses import dataclass
from typing import Any

# Half-integer trellis rates with kernel instances (mul1 codebook only); above
# 3.5 bpw the steps are whole bits.
HALF_RATES: tuple[float, ...] = ...


def rate_floor(bpw: float, half_steps: bool) -> int | float: ...


def rate_next(bpw: float, half_steps: bool) -> int | float: ...


@dataclass
class QTarget:
    numel: int
    target_bpw: float
    min_bpw: float
    priority: int
    half_steps: bool = False

    def total_bits(self) -> float: ...

    def delta_1(self) -> float: ...

    def increase_1(self) -> None: ...

    def clamp_min(self) -> None: ...


def create_q_strategy(
    model: Any,
    mtp_model: Any,
    config: Any,
    bpw: float,
    head_bpw: float,
    mtp_bpw: float,
    hq: bool,
    vision_model: Any = None,
    vision_bpw: int | None = None,
    half_steps: bool = False,
) -> tuple[dict, float]: ...


def create_q_strategy_from_recipe(
    model: Any,
    mtp_model: Any,
    config: Any,
    recipe_tensors: dict,
    head_bpw: float,
    mtp_bpw: float,
    vision_model: Any = None,
    vision_bpw: int | None = None,
) -> tuple[dict, float]: ...


def print_strategy(strategy: dict) -> str: ...
