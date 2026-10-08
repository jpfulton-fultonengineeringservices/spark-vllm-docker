"""Strategy builder for the distributed EXL3 coordinator.

Provides ``_StrategyMixin`` with ``_build_strategy`` and ``_publish_strategy``.
"""

from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path
from typing import Any

from exllamav3.model.config import Config

from exl3pack.cli import CoordinatorArgs
from exl3pack.dist_types import write_atomic
from exl3pack.recipe import DEFAULT_HEAD_BITS, DEFAULT_MTP_BITS


class _StrategyMixin:
    """Strategy methods, mixed into ``Coordinator``."""

    # Declared by ``Coordinator.__init__``.
    # Union: pre-prepare CoordinatorArgs, post-prepare the merged in_args
    # superset (prepare() adds derived keys).
    args: CoordinatorArgs | dict[str, Any]
    work: Path
    _strategy: dict[str, float]

    # ------------------------------------------------------------------ #
    #  Strategy builder                                                   #
    # ------------------------------------------------------------------ #

    def _build_strategy(
        self,
        model: Any,
        mtp_model: Any,
        vision_model: Any,
        config: Config,
    ) -> dict[str, float]:
        """Build the model-global bitrate map (dict[linear.key → target_bpw]).

        Respects the recipe path (create_q_strategy_from_recipe) vs the
        simple path (create_q_strategy), exactly as ``convert_model.main``.
        """
        allocation = import_module("exllamav3.conversion.allocation")
        recipe_map = self.args.get("recipe_strategy")
        # head_bits/mtp_bits are float | None at the boundary; None means
        # "upstream default" (recipe/arg defaults apply), so coerce only
        # when actually set.
        head_bits = self.args.get("head_bits")
        head_bits_eff = DEFAULT_HEAD_BITS if head_bits is None else float(head_bits)
        mtp_bits = self.args.get("mtp_bits")
        mtp_bits_eff = DEFAULT_MTP_BITS if mtp_bits is None else float(mtp_bits)
        if isinstance(recipe_map, dict):
            (
                strategy,
                _bpw,
            ) = allocation.create_q_strategy_from_recipe(
                model,
                mtp_model,
                config,
                recipe_map,
                head_bits_eff,
                mtp_bits_eff,
                vision_model=vision_model,
                vision_bpw=int(self.args.get("vision_bits", 16)),
            )
        else:
            (
                strategy,
                _bpw,
            ) = allocation.create_q_strategy(
                model,
                mtp_model,
                config,
                float(self.args["bits"]),
                head_bits_eff,
                mtp_bits_eff,
                bool(self.args.get("hq", False)),
                vision_model=vision_model,
                vision_bpw=int(self.args.get("vision_bits", 16)),
                half_steps=self.args.get("codebook") == "mul1",
            )
        # Strategy may map keys to int or float bpw values.
        return {str(k): float(v) for k, v in strategy.items()}

    def _publish_strategy(self, strategy: dict[str, float]) -> str:
        """Publish ``{linear.key: K}`` as ``<work>/dist/strategy.json`` (atomic).

        Workers load this one-shot file to obtain K without reconstructing
        the model. It is written before any shard spec is dispatched and
        holds the exact dict fed to ``group_quant_linears``/``make_quant_args``.
        """
        self._strategy = dict(strategy)
        path = self.work / "dist" / "strategy.json"
        payload = json.dumps(strategy, sort_keys=True, separators=(",", ":"))

        def _write(tmp: Path) -> None:
            tmp.write_text(payload, encoding="utf-8")

        write_atomic(path, _write)
        return str(path.resolve())
