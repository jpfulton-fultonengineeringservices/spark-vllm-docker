"""Worker-side per-layer EXL3 quantization.

Runs INSIDE the b12x GPU image on each worker node.  Polls a shared inbox
directory for ``ShardSpec`` JSON files, loads fp16 weights for the assigned
linears, quantizes each to EXL3, and writes the result atomically.

All heavy (torch / exllamav3) imports are lazy so the module stays
importable without the GPU image, mirroring :mod:`exl3pack.recipe`.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import logging
import time
import traceback
from pathlib import Path
from typing import Any

from exl3pack.dist_types import (
    ShardSpec,
    load_h_file,
    sha256_file,
    write_atomic,
    write_done_marker,
)
from exl3pack.logconfig import get_logger, log_event

__all__ = ["serve"]

_POLL_INTERVAL_S = 1.0
_MAX_CONSECUTIVE_SPEC_FAILURES = 3

def _lazy(name: str) -> Any:
    """Import *name* lazily (torch / exllamav3 are image-only)."""
    return importlib.import_module(name)


# ---------------------------------------------------------------------------
# serve loop
# ---------------------------------------------------------------------------

def serve(
    inbox: Path,
    shared: Path,
    device: int,
    stop: Path,
    log_dir: Path | None = None,
) -> None:
    """Poll *inbox* for ``ShardSpec`` JSON files and process each one.

    The loop exits when the *stop* sentinel file appears.  Successfully
    processed spec files are removed from the inbox; failed specs are left
    in place for retry after a short backoff.
    """
    inbox = Path(inbox)
    stop = Path(stop)
    log = get_logger("worker", log_dir=log_dir)
    consecutive_failures: dict[str, int] = {}
    log_event(
        log,
        logging.INFO,
        "worker.start",
        fields={
            "inbox": str(inbox),
            "device": device,
            "stop": str(stop),
            "log_dir": str(log_dir) if log_dir is not None else None,
        },
    )
    while not stop.exists():
        try:
            spec_paths = sorted(inbox.glob("*.json"))
        except OSError as exc:
            log_event(
                log,
                logging.ERROR,
                "worker.inbox_list_error",
                fields={"inbox": str(inbox), "error": str(exc)},
            )
            time.sleep(_POLL_INTERVAL_S)
            continue
        if not spec_paths:
            time.sleep(_POLL_INTERVAL_S)
            continue
        for spec_path in spec_paths:
            if stop.exists():
                return
            try:
                spec = ShardSpec.from_json(spec_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                log_event(
                    log,
                    logging.WARNING,
                    "worker.bad_spec",
                    fields={"spec": str(spec_path), "error": str(exc)},
                )
                time.sleep(_POLL_INTERVAL_S)
                continue
            log_event(
                log,
                logging.INFO,
                "worker.spec_taken",
                fields={"spec": str(spec_path), "shard": spec.shard_idx},
            )
            try:
                _run_shard(spec, device, Path(shared))
            except Exception as exc:  # noqa: BLE001 — keep serving after a bad shard
                key = str(spec_path)
                consecutive_failures[key] = consecutive_failures.get(key, 0) + 1
                if consecutive_failures[key] >= _MAX_CONSECUTIVE_SPEC_FAILURES:
                    log_event(
                        log,
                        logging.CRITICAL,
                        "worker.shard_failed_fatal",
                        fields={
                            "shard": spec.shard_idx,
                            "module": spec.module_key,
                            "consecutive_failures": consecutive_failures[key],
                            "error": str(exc),
                        },
                    )
                    raise SystemExit(1) from exc
                log_event(
                    log,
                    logging.ERROR,
                    "worker.shard_failed",
                    fields={
                        "shard": spec.shard_idx,
                        "module": spec.module_key,
                        "error": str(exc),
                    },
                )
                traceback.print_exc()
                time.sleep(_POLL_INTERVAL_S)
                continue
            log_event(
                log,
                logging.INFO,
                "worker.shard_done",
                fields={"shard": spec.shard_idx, "module": spec.module_key},
            )
            consecutive_failures.pop(str(spec_path), None)
            try:
                spec_path.unlink()
            except OSError as exc:
                log_event(
                    log,
                    logging.WARNING,
                    "worker.spec_unlink_error",
                    fields={"spec": str(spec_path), "error": str(exc)},
                )
    log_event(
        log,
        logging.INFO,
        "worker.stop_sentinel",
        fields={"stop": str(stop)},
    )


# ---------------------------------------------------------------------------
# shard execution
# ---------------------------------------------------------------------------

def _run_shard(spec: ShardSpec, device: int, work: Path) -> None:
    """Quantize one shard's linears and write the output atomically."""
    torch = _lazy("torch")
    save_file: Any = _lazy("safetensors.torch").save_file
    convert_model: Any = _lazy("exllamav3.conversion.convert_model")
    Linear: Any = _lazy("exllamav3.modules.linear").Linear

    torch_device = torch.device(f"cuda:{device}")
    strategy = _load_strategy(work)
    args = _load_conversion_args(work, spec)

    # Build the model exactly as the coordinator does.
    in_dir = Path(spec.weights_source).parent
    in_args: dict[str, Any] = dict(args)
    in_args["in_dir"] = str(in_dir)
    config, model, _mtp, _vis, _tok, _ref = convert_model.get_base_model(in_args)

    modules = list(model.modules)
    if spec.module_idx >= len(modules):
        raise RuntimeError(
            f"module_idx {spec.module_idx} out of range (model has {len(modules)} modules)"
        )
    module = modules[spec.module_idx]
    if str(getattr(module, "key", "")) != spec.module_key:
        raise RuntimeError(
            f"module key mismatch: expected {spec.module_key!r}, "
            f"got {getattr(module, 'key', None)!r}"
        )

    # Load H records keyed by qmap (filenames are sha256(qmap) as the
    # coordinator writes them in ``_publish_H``).
    h_by_qmap = _load_h_records(spec)

    wanted = set(spec.linear_keys)
    q_tensors: dict[str, Any] = {}
    found: set[str] = set()
    for m in module:
        if not isinstance(m, Linear):
            continue
        key = str(getattr(m, "key", ""))
        if key not in wanted:
            continue
        if m.device is None:
            m.load(torch_device)
        qmap_key = str(m.qmap)
        h_data = h_by_qmap.get(qmap_key)
        if h_data is None:
            raise RuntimeError(f"no H record for qmap {qmap_key!r} (linear {key!r})")
        if _h_is_meta(h_data):
            raise RuntimeError(f"meta H for qmap {qmap_key!r} (linear {key!r})")
        k_bits = strategy.get(key)
        if k_bits is None:
            raise RuntimeError(f"no strategy K for linear {key!r}")
        # JSON round-trip keeps floats floats; the nanobind scratch ext requires a
        # real int for K (SupportsInt does not auto-coerce 3.0 -> 3). True half-rates
        # (2.5/3.5) must pass through untouched: quantize_tiles routes frac K to
        # quantize_tiles_frac and only integral K reaches get_temp_buffers.
        if isinstance(k_bits, float) and k_bits.is_integer():
            k_bits = int(k_bits)
        quant_args = convert_model.make_quant_args(
            args, spec.module_idx, k_bits, [device], None
        )
        # Set H_swap_device before convert_exl3 (plan §6 contract mechanism).
        h_data["H_swap_device"] = torch_device
        m.convert_exl3(h_data, quant_args, override_swap_device=torch_device)
        found.add(key)
        for tkey, tval in m.get_tensors().items():
            if tkey in q_tensors:
                raise RuntimeError(f"duplicate output tensor key {tkey!r}")
            q_tensors[tkey] = tval

    missing = wanted - found
    if missing:
        raise RuntimeError(f"shard {spec.shard_idx}: missing linears {sorted(missing)}")

    out_path = Path(spec.result_uri)

    def _write(tmp: Path) -> None:
        save_file({k: v.contiguous() for k, v in q_tensors.items()}, str(tmp))

    write_atomic(out_path, _write)
    done_path = out_path.with_suffix(".done")
    write_done_marker(done_path, sha256_file(out_path))


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _h_is_meta(h_data: dict[str, Any]) -> bool:
    """Return ``True`` when the H tensor in *h_data* is a meta (uncalibrated) tensor."""
    h = h_data.get("H")
    return h is not None and bool(getattr(h, "is_meta", False))


def _load_strategy(work: Path) -> dict[str, float]:
    """Read the q-strategy (linear key -> bits-per-weight) from ``<work>/dist/strategy.json``.

    The coordinator publishes the fully resolved model-global strategy as a
    shared JSON sidecar (a flat ``{"<linear.key>": <K float target_bpw>, ...}``
    mapping) before any shard is dispatched.  This is the single source of truth
    so every worker sees identical K per linear.  ``ckpt/job.json['q_strategy']``
    is dead/None upstream and must not be used.
    """
    strategy_path = Path(work) / "dist" / "strategy.json"
    data: dict[str, Any] = json.loads(strategy_path.read_text(encoding="utf-8"))
    return {str(k): float(v) for k, v in data.items()}


def _load_conversion_args(work: Path, spec: ShardSpec) -> dict[str, Any]:
    """Return the ``convert_model`` args dict for :func:`make_quant_args`.

    Prefers ``<work>/ckpt/args.json`` (written by ``convert_model.prepare()``);
    falls back to a minimal dict derived from ``job.json`` + spec geometry.
    """
    args_path = Path(work) / "ckpt" / "args.json"
    if args_path.exists():
        data: dict[str, Any] = json.loads(args_path.read_text(encoding="utf-8"))
        return data

    job_path = Path(work) / "ckpt" / "job.json"
    job: dict[str, Any] = {}
    if job_path.exists():
        job = json.loads(job_path.read_text(encoding="utf-8"))
    return {
        "in_dir": str(Path(spec.weights_source).parent),
        "work_dir": str(work),
        "bits": int(job.get("bits", 4)),
        "codebook": str(job.get("codebook", "mcg")),
        "apply_out_scales": bool(job.get("apply_out_scales", True)),
        "hq": bool(job.get("hq", False)),
        "head_bits": float(job.get("head_bits", 6)),
        "mtp_bits": float(job.get("mtp_bits", 4)),
        "vision_bits": int(job.get("vision_bits", 0)),
    }


def _load_h_records(spec: ShardSpec) -> dict[str, dict[str, Any]]:
    """Load H records for every qmap in *spec* keyed by their raw qmap string.

    Filenames follow the coordinator convention ``sha256(qmap_key) + ".safetensors"``
    inside ``spec.h_dir``.
    """
    h_dir = Path(spec.h_dir)
    records: dict[str, dict[str, Any]] = {}
    for qmap_key in spec.qmaps:
        h_name = hashlib.sha256(qmap_key.encode("utf-8")).hexdigest() + ".safetensors"
        h_path = h_dir / h_name
        rec = load_h_file(h_path)
        if _h_is_meta(rec):
            raise RuntimeError(f"meta H for qmap {qmap_key!r} in {h_path}")
        records[qmap_key] = rec
    return records
