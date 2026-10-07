"""Commit / checkpoint logic for the distributed EXL3 coordinator.

Houses the module-level quant-suffix and MoE constants, the ``_base_key``
helper, the per-row preserve helpers, and the ``_CommitMixin`` providing
``_commit_module``, ``_checkpoint``, and ``_restore_checkpoint_backup``.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

from exl3pack.dist_types import coverage_assert, write_atomic

# Quantized keys emitted by ``LinearEXL3.get_tensors()`` (no ".weight").
# ``LinearFP16.get_tensors()`` emits "{key}.weight" and "{key}.bias".
# These suffixes are stripped in :func:`_base_key` before coverage_assert.
_QUANT_SUFFIXES = frozenset(
    {
        "weight",
        "bias",
        "su",
        "sv",
        "suh",
        "svh",
        "trellis",
        "mcg",
        "mul1",
    }
)

# MiMo v2.6 MoE shape: layers 1-47 carry 256 experts each, and every expert
# contributes exactly one down projection to the gathered module tensors.
_MOE_LAYER_LO = 1
_MOE_LAYER_HI = 47
_MOE_EXPERT_COUNT = 256

# Expert down projection key (bare, after suffix stripping):
# ``model.layers.<i>.mlp.experts.<j>.down_proj``
_EXPERT_DOWN_RE = re.compile(
    r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.down_proj$"
)

# Module key regex for identifying MiMo MoE layers (e.g. "model.layers.12").
_MODULE_KEY_RE = re.compile(r"^model\.layers\.(\d+)$")


def _base_key(key: str) -> str:
    """Strip one ``get_tensors()`` quant suffix from *key* (linear.key form).

    Returns the bare ``linear.key`` so that coverage can compare planned bare
    keys against suffixed gathered keys (plan §9).
    """
    head, dot, tail = key.rpartition(".")
    return head if dot and tail in _QUANT_SUFFIXES else key


def _get_preserve(
    quant_preserves: list[dict[str, Any]], si: int, params: dict[str, Any]
) -> None:
    """Merge per-row quant-preserve state into *params* (mirrors main)."""
    params.update(quant_preserves[si])
    params["quant_preserve"] = quant_preserves[si]


def _put_preserve(
    quant_preserves: list[dict[str, Any]], si: int, params: dict[str, Any]
) -> None:
    """Save per-row quant-preserve state back out of *params* (mirrors main)."""
    quant_preserves[si] = params["quant_preserve"]


class _CommitMixin:
    """Commit and checkpoint methods, mixed into ``Coordinator``."""

    work: Path
    _planned_keys: set[str]
    _strategy: dict[str, float]

    # ------------------------------------------------------------------ #
    #  Commit                                                             #
    # ------------------------------------------------------------------ #

    def _commit_module(
        self,
        module_key: str,
        q_tensors: dict[str, torch.Tensor],
    ) -> None:
        """Coverage-assert gathered tensors, then write the module file.

        P1-a: per-linear required-key assertion (plan §9a):
          - K == 16 → LinearEXL3 packed weight, require <key>.weight.
          - K < 16 → trellis mode, require <key>.trellis.
        Format-specific keys (su/sv/suh/svh) are NOT asserted here;
        upstream merge.py or tensor-level code determines the exact set.
        """
        coverage_assert(
            self._planned_keys,
            {_base_key(k) for k in q_tensors},
        )

        # MiMo-specific: every MoE layer must return exactly experts 0-255 of
        # each expert down projection (plan §9).
        layer_match = _MODULE_KEY_RE.fullmatch(module_key)
        if layer_match is not None:
            layer_i = int(layer_match.group(1))
            if _MOE_LAYER_LO <= layer_i <= _MOE_LAYER_HI:
                experts: set[int] = set()
                for key in q_tensors:
                    m = _EXPERT_DOWN_RE.fullmatch(_base_key(key))
                    if m is not None and int(m.group(1)) == layer_i:
                        experts.add(int(m.group(2)))
                expected = set(range(_MOE_EXPERT_COUNT))
                if experts != expected:
                    raise ValueError(
                        f"{module_key}: expected 256 expert down_proj keys "
                        f"(experts 0-255), got {len(experts)} distinct"
                    )

        # P1-a: per-linear required-key assertion
        strategy = getattr(self, '_strategy', {})
        for key in self._planned_keys:
            K = strategy.get(key)
            if K is None:
                continue
            required = f"{key}.weight" if K == 16 else f"{key}.trellis"
            if required not in q_tensors:
                raise ValueError(
                    f"{module_key}: missing required key '{required}' for linear {key}"
                )


        out_path = self.work / "qtensors" / f"{module_key}.safetensors"

        def _write(tmp: Path) -> None:
            save_file(q_tensors, str(tmp))

        write_atomic(out_path, _write)

    # ------------------------------------------------------------------ #
    #  Checkpoint                                                         #
    # ------------------------------------------------------------------ #

    def _checkpoint(
        self,
        job_state: dict[str, Any],
        state: list[torch.Tensor],
        original_input_ids: list[torch.Tensor],
    ) -> None:
        """Swap in a fresh ``ckpt/{job.json,state,original_input_ids}`` trio.

        Staged in ``ckpt_new/``, swapped via directory renames so a crash
        never leaves a half-written checkpoint live.
        """
        ckpt = self.work / "ckpt"
        stage = self.work / "ckpt_new"
        backup = self.work / "ckpt_old"

        if stage.exists():
            shutil.rmtree(stage)

        stage.mkdir(parents=True, exist_ok=True)
        (stage / "job.json").write_text(
            json.dumps(job_state, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        save_file(
            {f"tensor.{i}": t for i, t in enumerate(state)},
            str(stage / "state.safetensors"),
        )
        save_file(
            {
                f"tensor.{i}": t
                for i, t in enumerate(original_input_ids)
            },
            str(stage / "original_input_ids.safetensors"),
        )
        backup = self.work / "ckpt_old"
        if backup.exists():
            shutil.rmtree(backup)
        if ckpt.exists():
            os.replace(ckpt, backup)
        os.replace(stage, ckpt)
        if backup.exists():
            shutil.rmtree(backup)

    def _restore_checkpoint_backup(self) -> None:
        """Restore ``ckpt_old`` before prepare reads the checkpoint."""
        ckpt = self.work / "ckpt"
        backup = self.work / "ckpt_old"
        if not ckpt.exists() and backup.exists():
            os.replace(backup, ckpt)
