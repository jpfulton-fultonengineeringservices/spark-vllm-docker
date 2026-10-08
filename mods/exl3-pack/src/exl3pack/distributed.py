"""Coordinator for distributed per-layer EXL3 quantization.

Runs INSIDE the b12x GPU image (torch available). Drives the per-module
capture / quantize / reload / advance loop for a MiMo (or other) checkpoint;
only the quantization of each module's linears is distributed to worker nodes
via a shared NFS work directory. Capture and state-advance stay on the
coordinator (plan §3).

Transport contract (shared with ``worker.py``), all under ``<work>``:

- ``dist/mod<N>/h/<qmap-hash>.safetensors`` — immutable H payload, one file
  per distinct qmap (``qmap-hash`` = sha256 hex of the qmap string).
- ``dist/inbox/<node>/shard-<i>.json`` — ``ShardSpec.to_json()``.
- ``dist/mod<N>/out/<node>/shard-<i>.safetensors`` + ``shard-<i>.done`` —
  worker output (atomic write, then a done-marker carrying the sha256).
- ``qtensors/<module_key>.safetensors`` — coverage-asserted module merge.

``ShardSpec``, ``WorkerEndpoint`` and the atomic-IO helpers come from
:mod:`exl3pack.dist_types`; they are never redefined here. Upstream
(``exllamav3.conversion.convert_model``) is unannotated; its functions are
called with the exact positional signatures of the installed wheel.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from exllamav3.conversion import convert_model
from exllamav3.model.config import Config
from exllamav3.modules.linear import Linear
from safetensors import safe_open  # noqa: F401  — patched in tests via mock.patch.object

from exl3pack.dist_commit import _CommitMixin, _get_preserve, _put_preserve
from exl3pack.dist_strategy import _StrategyMixin
from exl3pack.dist_transport import _TransportMixin
from exl3pack.dist_types import WorkerEndpoint
from exl3pack.logconfig import get_logger, log_event

# Upstream constant for reference-state count (plan §3.1).
_NUM_REF_STATES = 5


class Coordinator(_StrategyMixin, _TransportMixin, _CommitMixin):
    """Distributed quantization coordinator (see module docstring)."""

    def __init__(
        self,
        args: dict[str, Any],
        endpoints: Sequence[WorkerEndpoint],
        work: Path,
        log_dir: Path | None = None,
    ) -> None:
        self.args = args
        self.endpoints = list(endpoints)
        self.work = work
        self.log = get_logger("coordinator", log_dir=log_dir)
        self._log_dir = log_dir
        self.gather_timeout = float(args.get("gather_timeout", 600.0))
        # Upstream default is 120s via --cpi (plan §3).
        self.checkpoint_interval = int(
            args.get("checkpoint_interval") or args.get("cpi") or 120
        )
        self._module_idx = 0
        self._h_dir = ""
        self._last_checkpoint_time = 0.0
        self._index_sha: str | None = None
        self._planned_keys: set[str] = set()
        self._strategy: dict[str, float] = {}
        (self.work / "dist").mkdir(parents=True, exist_ok=True)

    def run(self) -> None:
        """Drive the full loop; return on success, raise on hard failure."""
        import logging

        # P1-e: crash-window restore must happen before prepare() reads ckpt/job.json.
        self._restore_checkpoint_backup()
        log_event(
            self.log,
            logging.INFO,
            "coordinator.start",
            fields={
                "work": str(self.work),
                "nodes": [e.node for e in self.endpoints],
                "log_dir": str(self._log_dir) if self._log_dir is not None else None,
            },
        )

        in_args, job_state, ok, err = convert_model.prepare(
            argparse.Namespace(**self.args)
        )
        if not ok or in_args is None or job_state is None:
            raise RuntimeError(f"prepare failed: {err}")
        self.args = in_args  # merged superset (adds image_dump, verbose, ...)

        config: Config
        (
            config,
            model,
            _mtp_model,
            _vision_model,
            tokenizer,
            use_reference_state,
        ) = convert_model.get_base_model(in_args)

        state: list[torch.Tensor]
        original_input_ids: list[torch.Tensor]
        state, original_input_ids = convert_model.prepare_state(
            in_args, job_state, config, model, tokenizer
        )
        # B2: upstream guard: fresh run copies state; resumed run materializes None
        if original_input_ids is None:
            original_input_ids = (
                state.copy()
                if int(job_state.get("next_module_idx", 0) or 0) == 0
                else [None] * len(state)
            )

        # Build the model-global bitrate strategy (plan §8).
        strategy = self._build_strategy(model, _mtp_model, _vision_model, config)
        # Store strategy for later per-linear required-key checks (plan §9a)
        self._strategy = strategy

        # One-shot publish of the resolved per-linear K map for workers
        # (job.json q_strategy is dead/None). Must precede any shard dispatch.
        self._publish_strategy(strategy)

        # ``can_resume_quant`` gates checkpointing (plan §3).
        can_resume_quant = bool(
            model.caps.get("can_resume_quant", use_reference_state)
        )
        if not can_resume_quant:
            print(
                " !! Warning, resuming an interrupted quant is not possible "
                "for this architecture. Checkpoints will not be saved."
            )

        modules = list(model.modules)
        last_ckp_idx = self.args.get("last_checkpoint_index") or -1

        for idx in range(
            int(job_state.get("next_module_idx", 0) or 0), len(modules)
        ):
            self._module_idx = idx
            module = modules[idx]
            module_key = str(getattr(module, "key", f"module_{idx}"))
            log_event(
                self.log,
                logging.INFO,
                "coordinator.shard_assigned",
                fields={"module_idx": idx, "module": module_key},
            )
            device = torch.device(str(self.args.get("device", "cuda:0")))
            module.load(device)

            # -- capture online while fp16 (plan §3.1) -----------------------
            capture_H: dict[str, dict[str, Any]] = {}
            bad_rows: set[int] = set(job_state.get("bad_rows") or [])
            quant_preserves: list[dict[str, Any]] = [
                {} for _ in range(len(state))
            ]

            if int(getattr(module, "num_slices", 1)) > 1:
                raise NotImplementedError(
                    f"{module_key}: sliced modules (num_slices > 1) are not "
                    "supported by the distributed coordinator"
                )

            ref_states: dict[int, torch.Tensor | None] = {}

            for i in range(len(state)):
                if i in bad_rows:
                    continue
                # CAPTURE pass (accumulates H)
                params: dict[str, Any] = {
                    "attn_mode": "flash_attn_nc",
                    "capture": capture_H,
                    "activate_all_experts": bool(
                        getattr(model, "calibration_all_experts", False)
                    ),
                    "input_ids": original_input_ids[i],
                }
                _get_preserve(quant_preserves, i, params)
                rs = module.prepare_for_device(state[i], params)
                rs = module.forward(rs, params)
                _put_preserve(quant_preserves, i, params)

                # REFERENCE pass (first N rows, no capture). The second
                # forward only runs when calibration_all_experts is set (MiMo);
                # otherwise ``rs`` is the capture pass's own result.
                if i < _NUM_REF_STATES:
                    if getattr(model, "calibration_all_experts", False):
                        ref_params: dict[str, Any] = {
                            "attn_mode": "flash_attn_nc",
                            "input_ids": original_input_ids[i],
                        }
                        _get_preserve(quant_preserves, i, ref_params)
                        rs = module.forward(
                            module.prepare_for_device(state[i], ref_params),
                            ref_params,
                        )
                        _put_preserve(quant_preserves, i, ref_params)
                    # N1: parity-consistency with upstream: use .all().item()
                    if torch.isfinite(rs).all().item():
                        ref_states[i] = rs.cpu()
                    else:
                        bad_rows.add(i)
                        print(
                            f" !! Non-finite reference state in "
                            f"calibration row {i}, excluding row"
                        )

            # Swap H to CPU for NFS publication.
            for rec in capture_H.values():
                h = rec.get("H")
                if not isinstance(h, torch.Tensor) or bool(
                    getattr(h, "is_meta", False)
                ):
                    raise ValueError(
                        f"{module_key}: captured H is missing or meta"
                    )
                rec["H_swap_device"] = h.device
                rec["H"] = h.cpu()

            # -- distribute quantization (only when the module has linears) --
            linears: list[Linear] = [
                m
                for m in module
                if isinstance(m, Linear) and m.qmap and m.device is not None
            ]
            # M2: swap CPU weights pre-dispatch (upstream ~1360)
            for linear in linears:
               if getattr(linear, 'inner', None) is not None:
                   linear.inner.swap_cpu()
            if linears:
                groups = convert_model.group_quant_linears(
                    linears, strategy, capture_H
                )
                qmaps = {str(m.qmap) for m in linears if m.qmap}
                self._publish_H(module_key, capture_H, qmaps)
                specs = self._plan_shards(module, groups, strategy, self.endpoints)
                self._planned_keys = {k for s in specs for k in s.linear_keys}
                self._dispatch(specs)
                try:
                    q_tensors = self._gather(specs)
                except Exception:
                    log_event(
                        self.log,
                        logging.WARNING,
                        "coordinator.gather_retry",
                        fields={"module": module_key},
                    )
                    # M3: clear stale partial results before retry (plan §5)
                    for spec in specs:
                        Path(spec.result_uri).unlink(missing_ok=True)
                        Path(spec.result_uri).with_suffix('.done').unlink(missing_ok=True)
                    self._dispatch(specs)
                    q_tensors = self._gather(specs)
                log_event(
                    self.log,
                    logging.INFO,
                    "coordinator.shard_done",
                    fields={"module_idx": idx, "module": module_key},
                )
                self._commit_module(module_key, q_tensors)
                config.stc.set_new_tensors(q_tensors)
                try:
                    module.load(device, source=q_tensors, keep_source_weights=True)
                finally:
                    config.stc.set_new_tensors(None)
                # M2: unload module post-commit if not retaining during quant (upstream ~1500)
                if not getattr(module, 'caps', {}).get('retain_during_quant'):
                    module.unload()

            # -- advance state serially through the (quantized) module ---
            for i in range(len(state)):
                if i in bad_rows:
                    continue
                adv_params: dict[str, Any] = {
                    'attn_mode': 'flash_attn_nc',
                    'input_ids': original_input_ids[i],
                }
                state[i] = module.prepare_for_device(state[i], adv_params)
                if i < _NUM_REF_STATES or idx < len(modules) - 1:
                    _get_preserve(quant_preserves, i, adv_params)
                    rs = module.forward(state[i], adv_params)
                    # N1: use .all().item() for boolean context parity with upstream
                    if not torch.isfinite(rs).all().item():
                        bad_rows.add(i)
                        print(f" !! Non-finite hidden state in calibration row {i}, excluding row")
                    state[i] = rs.cpu()
                    _put_preserve(quant_preserves, i, adv_params)

            # Mirror main: measure state error against the reference states
            # (consumed once), then gate the job on the bad-row fraction.
            for i in range(len(state)):
                if i in bad_rows:
                    continue
                ref = ref_states.get(i)
                if ref is not None and linears:
                    ref = ref.to(state[i].device)
                    convert_model.get_state_error(state[i], ref)
                    ref_states[i] = None

            convert_model.check_bad_rows(bad_rows, len(state))

            job_state["next_module_idx"] = idx + 1
            job_state["bad_rows"] = sorted(bad_rows)
            # Time-based checkpointing (mirrors main L≈1486-1507).
            now = time.time()
            if (
                can_resume_quant
                and self._last_checkpoint_time + self.checkpoint_interval <= now
                and (last_ckp_idx < 0 or idx <= last_ckp_idx)
            ):
                self._checkpoint(job_state, state, original_input_ids)
                self._last_checkpoint_time = now
                log_event(
                    self.log,
                    logging.INFO,
                    "coordinator.checkpoint",
                    fields={"module_idx": idx, "module": module_key},
                )
        log_event(self.log, logging.INFO, "coordinator.stop_sentinel", fields={})
