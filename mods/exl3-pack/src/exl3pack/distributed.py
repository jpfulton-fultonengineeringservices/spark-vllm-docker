"""Coordinator for distributed per-layer EXL3 quantization.

Runs INSIDE the b12x GPU image (torch available). Drives the per-module
capture / quantize / reload / advance loop for a MiMo (or other) checkpoint;
only the quantization of each module's linears is distributed to worker nodes
via a shared NFS work directory. Capture and state-advance stay on the
coordinator (plan §3).

Transport contract (shared with ``worker.py``), all under ``<work>``:

- ``dist/mod<N>/h/<qmap-hash>.safetensors`` — immutable H payload, one file
  per distinct qmap (``qmap-hash`` = sha256 hex of the qmap string).
- ``dist/mod<N>/inbox/<node>/shard-<i>.json`` — ``ShardSpec.to_json()``.
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
import hashlib
import json
import os
import re
import shutil
import time
from collections.abc import Sequence
from importlib import import_module, metadata
from pathlib import Path
from typing import Any

import torch

# Upstream imports (unannotated; exact signatures per installed wheel)
from exllamav3.conversion import convert_model
from exllamav3.model.config import Config
from exllamav3.modules.linear import Linear
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from exl3pack.dist_types import (
    ShardSpec,
    WorkerEndpoint,
    cfg_hash,
    coverage_assert,
    read_done_marker,
    save_h_file,
    sha256_file,
    write_atomic,
)
from exl3pack.recipe import DEFAULT_HEAD_BITS, DEFAULT_MTP_BITS

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

# Upstream constant for reference-state count (plan §3.1).
_NUM_REF_STATES = 5


def _base_key(key: str) -> str:
    """Strip one ``get_tensors()`` quant suffix from *key* (linear.key form).

    Returns the bare ``linear.key`` so that coverage can compare planned bare
    keys against suffixed gathered keys (plan §9).
    """
    head, dot, tail = key.rpartition(".")
    return head if dot and tail in _QUANT_SUFFIXES else key


def _exllamav3_version() -> str:
    """Installed exllamav3 distribution version ("" when not installed)."""
    try:
        return metadata.version("exllamav3")
    except metadata.PackageNotFoundError:
        return ""


class Coordinator:
    """Distributed quantization coordinator (see module docstring)."""

    def __init__(
        self,
        args: dict[str, Any],
        endpoints: Sequence[WorkerEndpoint],
        work: Path,
    ) -> None:
        self.args = args
        self.endpoints = list(endpoints)
        self.work = Path(work)
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
        # P1-e: crash-window restore must happen before prepare() reads ckpt/job.json.
        self._restore_checkpoint_backup()

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
                    # M3: clear stale partial results before retry (plan §5)
                    for spec in specs:
                        Path(spec.result_uri).unlink(missing_ok=True)
                        Path(spec.result_uri).with_suffix('.done').unlink(missing_ok=True)
                    self._dispatch(specs)
                    q_tensors = self._gather(specs)
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
                    # N1: use .all().item() for boolean
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
        if isinstance(recipe_map, dict):
            (
                strategy,
                _bpw,
            ) = allocation.create_q_strategy_from_recipe(
                model,
                mtp_model,
                config,
                recipe_map,
                float(self.args.get("head_bits", DEFAULT_HEAD_BITS)),
                float(self.args.get("mtp_bits", DEFAULT_MTP_BITS)),
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
                float(self.args.get("bits", 4)),
                float(self.args.get("head_bits", DEFAULT_HEAD_BITS)),
                float(self.args.get("mtp_bits", DEFAULT_MTP_BITS)),
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

    # ------------------------------------------------------------------ #
    #  H publication                                                      #
    # ------------------------------------------------------------------ #

    def _publish_H(
        self,
        module_key: str,
        capture_H: dict[str, dict[str, Any]],
        qmaps: set[str],
    ) -> str:
        """Write one immutable H file per distinct qmap; return the H dir."""
        h_dir = self.work / "dist" / f"mod{self._module_idx}" / "h"
        h_dir.mkdir(parents=True, exist_ok=True)
        for qmap in sorted(qmaps):
            rec = capture_H.get(qmap)
            if rec is None:
                raise ValueError(
                    f"{module_key}: no captured H for qmap {qmap!r}"
                )
            h = rec.get("H")
            if (
                not isinstance(h, torch.Tensor)
                or bool(getattr(h, "is_meta", False))
            ):
                raise ValueError(
                    f"{module_key}: H for qmap {qmap!r} is missing or meta"
                )
            if h.dtype != torch.float32:
                raise ValueError(
                    f"{module_key}: H for qmap {qmap!r} is {h.dtype}"
                )
            inf_nan_raw = rec["inf_nan"]
            inf_nan: int | tuple[int, ...] = (
                int(inf_nan_raw)
                if isinstance(inf_nan_raw, int)
                else tuple(int(v) for v in inf_nan_raw)
            )
            dev = rec["device"]
            record: dict[str, Any] = {
                "H": h.cpu(),
                "count": int(rec["count"]),
                "num_total": int(rec["num_total"]),
                "inf_nan": inf_nan,
                "first_key": str(rec["first_key"]),
                "device": dev
                if isinstance(dev, (str, int))
                else str(dev),
                "finalized": bool(rec["finalized"]),
            }
            qmap_hash = hashlib.sha256(qmap.encode("utf-8")).hexdigest()
            save_h_file(h_dir / f"{qmap_hash}.safetensors", record)
        self._h_dir = str(h_dir.resolve())
        return self._h_dir

    # ------------------------------------------------------------------ #
    #  Shard planning                                                     #
    # ------------------------------------------------------------------ #

    def _plan_shards(
        self,
        module: Any,
        groups: list[list[Linear]],
        strategy: dict[str, float],
        endpoints: Sequence[WorkerEndpoint],
    ) -> list[ShardSpec]:
        """Greedily assign whole groups to workers by ``weights_numel``."""
        if not endpoints:
            raise ValueError("no worker endpoints available")

        weighted: list[tuple[int, list[Linear]]] = []
        for group in groups:
            if not group:
                raise ValueError("empty linear group")
            weighted.append(
                (
                    sum(int(lin.weights_numel()) for lin in group),
                    group,
                )
            )
        weighted.sort(key=lambda t: (-t[0], str(t[1][0].key)))
        loads = [0] * len(endpoints)
        assigned: list[list[list[Linear]]] = [[] for _ in endpoints]
        for weight, group in weighted:
            wi = min(range(len(endpoints)), key=lambda i: (loads[i], i))
            loads[wi] += weight
            assigned[wi].append(group)

        module_key = str(getattr(module, "key", ""))
        weights_source = self._weights_source()
        out_dir = self.work / "dist" / f"mod{self._module_idx}" / "out"
        specs: list[ShardSpec] = []

        for wi, ep in enumerate(endpoints):
            linear_keys: list[str] = []
            qmaps: set[str] = set()
            for group in assigned[wi]:
                for lin in group:
                    linear_keys.append(str(lin.key))
                    if lin.qmap:
                        qmaps.add(str(lin.qmap))

            result_uri = str(
                (out_dir / ep.node / f"shard-{wi}.safetensors").resolve()
            )

            # Compute a per-shard weights digest (plan §4).
            weights_digest = self._weights_digest(linear_keys, module)

            ch = cfg_hash(
                exllamav3_version=_exllamav3_version(),
                wheel_sha=str(self.args.get("wheel_sha", "")),
                bits=int(self.args["bits"]),
                codebook=str(self.args["codebook"]),
                recipe_strategy=json.dumps(
                    strategy, sort_keys=True, separators=(",", ":")
                ),
                seed_basis=str(self.args.get("seed_basis", "module_idx")),
                devices=(int(ep.device),),
                apply_out_scales=bool(
                    self.args.get("apply_out_scales") or False
                ),
                source_index_hash=self._source_index_hash(
                    weights_source
                ),
                weights_digest=weights_digest,
            )
            specs.append(
                ShardSpec(
                    job_id=str(self.args.get("job_id", "exl3-dist")),
                    module_idx=self._module_idx,
                    module_key=module_key,
                    shard_idx=wi,
                    linear_keys=tuple(linear_keys),
                    qmaps=tuple(sorted(qmaps)),
                    h_dir=self._h_dir,
                    weights_source=weights_source,
                    result_uri=result_uri,
                    cfg_hash=ch,
                )
            )
        return specs

    # ------------------------------------------------------------------ #
    #  Dispatch / Gather                                                  #
    # ------------------------------------------------------------------ #

    def _dispatch(self, specs: list[ShardSpec]) -> None:
        """Write one ShardSpec JSON per shard into its worker inbox (atomic)."""
        for spec in specs:
            if spec.shard_idx >= len(self.endpoints):
                raise ValueError(
                    f"shard_idx {spec.shard_idx} has no endpoint"
                )
            node = self.endpoints[spec.shard_idx].node
            inbox = (
                self.work
                / "dist"
                / f"mod{spec.module_idx}"
                / "inbox"
                / node
            )
            payload = spec.to_json()

            def _write(tmp: Path, p: str = payload) -> None:
                tmp.write_text(p, encoding="utf-8")

            write_atomic(inbox / f"shard-{spec.shard_idx}.json", _write)

    def _gather(
        self, specs: list[ShardSpec]
    ) -> dict[str, torch.Tensor]:
        """Wait for shard outputs (bounded), verify hashes, merge q_tensors.

        P1-d: per-shard gather timeout window (plan §5).
        """
        merged: dict[str, torch.Tensor] = {}
        for spec in specs:
            # P1-d: per-shard deadline
            deadline = time.monotonic() + self.gather_timeout
            data_path = Path(spec.result_uri)
            done_path = data_path.with_suffix(".done")
            while not done_path.exists():
                if time.monotonic() > deadline:
                    raise TimeoutError(
                        f"{spec.module_key} shard {spec.shard_idx}: "
                        f"timeout after {self.gather_timeout:.0f}s"
                    )
                time.sleep(1.0)
            expected = read_done_marker(done_path)
            actual = sha256_file(data_path)
            if actual != expected:
                raise ValueError(
                    f"{spec.module_key} shard {spec.shard_idx}: output hash "
                    f"mismatch (marker {expected}, file {actual})"
                )
            with safe_open(str(data_path), framework="pt") as f:
                echo = (f.metadata() or {}).get("cfg_hash")
            if echo is not None and echo != spec.cfg_hash:
                raise ValueError(
                    f"{spec.module_key} shard {spec.shard_idx}: cfg_hash "
                    f"echo {echo} != issued {spec.cfg_hash}"
                )
            for key, tensor in load_file(str(data_path)).items():
                if key in merged:
                    raise ValueError(
                        f"key {key!r} returned by more than one shard"
                    )
                merged[key] = tensor
        return merged

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

    # ------------------------------------------------------------------ #
    #  Weights digest                                                     #
    # ------------------------------------------------------------------ #

    def _weights_source(self) -> str:
        """Path to the shared fp16 ``model.safetensors.index.json``."""
        index = (
            Path(str(self.args["in_dir"]))
            / "model.safetensors.index.json"
        )
        return str(index.resolve())

    def _source_index_hash(self, weights_source: str) -> str:
        """sha256 of the source index.json (cached)."""
        if self._index_sha is None:
            self._index_sha = sha256_file(Path(weights_source))
        return self._index_sha

    def _weights_digest(
        self,
        linear_keys: Sequence[str],
        module: Any,
    ) -> str:
        """Digest binding each shard key to its consumed fp16 weight bytes.

        For each key the sha256 of ``linear.inner.get_weight_tensor()`` is
        bound (files hashed once and cached), so a source-data edit changes
        every affected shard's cfg_hash.
        """
        # Build a key→Linear lookup from the module for weight-digest access.
        linear_map: dict[str, Linear] = {}
        for lin in module:
            if isinstance(lin, Linear):
                linear_map[str(lin.key)] = lin

        pairs: list[list[str]] = []
        for key in sorted(linear_keys):
            lin = linear_map.get(key)
            if lin is None:
                raise ValueError(
                    f"linear key {key!r} not found in module"
                )
            wt = lin.inner.get_weight_tensor()
            pairs.append(
                [key, hashlib.sha256(wt.tobytes()).hexdigest()]
            )
        blob = json.dumps(pairs, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

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
