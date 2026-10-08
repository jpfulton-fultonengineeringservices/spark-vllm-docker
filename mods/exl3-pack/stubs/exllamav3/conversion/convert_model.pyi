import argparse
from collections.abc import Collection
from typing import Any

import torch

# Model / Tokenizer are not stubbed in PART 1; use Any for those positions.
# `config` resolves to the stubbed exllamav3.model.config.Config.


def prepare(
    args: argparse.Namespace,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, bool, str | None]: ...


def check_system() -> None: ...


def get_base_model(
    args: dict[str, Any],
) -> tuple[Any, Any, Any | None, Any | None, Any | None, bool]: ...


def prepare_state(
    args: dict[str, Any],
    job_state: dict[str, Any],
    config: Any,
    model: Any,
    tokenizer: Any | None,
) -> tuple[list[torch.Tensor], list[torch.Tensor] | None]: ...


def get_state_error(x: Any, ref: Any) -> tuple[float, float, float]: ...


def make_quant_args(
    args: dict[str, Any],
    idx: int,
    K: int,
    devices: list[int],
    device_ratios: list[Any] | None = None,
) -> dict[str, Any]: ...


def group_quant_linears(
    linears: list[Any],
    strategy: dict[str, Any],
    capture_H: dict[str, Any],
    max_cat_cols: int = 32768,
    max_stack: int = 16,
) -> list[Any]: ...


def check_bad_rows(
    bad_rows: Collection[int], num_rows: int, max_fraction: float = 0.10
) -> None: ...


def main(args: Any, job_state: Any) -> None: ...
