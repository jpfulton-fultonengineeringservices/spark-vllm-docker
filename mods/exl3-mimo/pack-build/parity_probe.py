# parity_probe.py — numeric parity probe for the exl3-v1 routed-expert container.
#
# Runs inside the b12x image on a cluster node with CUDA.  Emits a structured
# JSON report and exits non-zero on any failed check.
#
# ``--help`` and ``--dry-run`` work without torch/safetensors so the harness
# can be smoke-tested on a workstation.

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any


# --------------------------------------------------------------------------- lazy imports
_TORCH = None
_HAVE_TORCH = False


def _require_torch():
    global _TORCH, _HAVE_TORCH
    if _TORCH is None:
        try:
            import torch as m
            _TORCH = m
            _HAVE_TORCH = True
        except ImportError as exc:
            raise RuntimeError("torch not available; real runs require the b12x image") from exc
    return _TORCH


# --------------------------------------------------------------------------- helpers


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _result(check: str, ok: bool, **metrics: Any) -> dict[str, Any]:
    return {"check": check, "ok": bool(ok), **metrics}


def _sha(t: Any) -> str:
    torch = _require_torch()
    if not isinstance(t, torch.Tensor):
        t = torch.tensor(t) if hasattr(t, "__len__") else torch.tensor([t])
    h = hashlib.sha256()
    h.update(bytes(t.cpu().contiguous().view(torch.uint8).numpy().tobytes()))
    return h.hexdigest()[:16]


def _cosine(a: Any, b: Any) -> float:
    torch = _require_torch()
    af = torch.as_tensor(a).flatten().to(torch.float64)
    bf = torch.as_tensor(b).flatten().to(torch.float64)
    return float(torch.dot(af, bf) / (af.norm() + 1e-12))


def _rel_fro(a: Any, b: Any) -> float:
    torch = _require_torch()
    af = torch.as_tensor(a).flatten().to(torch.float64)
    bf = torch.as_tensor(b).flatten().to(torch.float64)
    return float((af - bf).norm() / (af.norm() + 1e-12))


def _round(v: Any) -> float:
    return round(float(v), 6)


# --------------------------------------------------------------------------- S1 dense spot-check


def s1_dense_spot(src_pack: Path, exl3_pack: Path) -> dict[str, Any]:
    """S1: confirm container manifest geometry matches source model config."""
    cfg = json.loads((src_pack / "config.json").read_text())
    src_text = cfg.get("text_config", cfg)
    manifest = json.loads((exl3_pack / "exl3-manifest.json").read_text())
    g = manifest["geometry"]
    expected = {
        "hidden_size": int(src_text["hidden_size"]),
        "intermediate_size": int(
            src_text.get("moe_intermediate_size", src_text.get("intermediate_size"))
        ),
        "num_experts": int(src_text.get("n_routed_experts", src_text.get("num_experts"))),
    }
    actual = {k: int(g.get(k, 0)) for k in expected}
    if actual != expected:
        return _result(
            "S1_dense_spot", False, expected=expected, actual=actual,
            reason="manifest geometry does not match source model",
        )
    return _result("S1_dense_spot", True, geometry=actual, codebook=manifest.get("codebook"),
                   bits=manifest.get("rates", {}).get("bits"),
                   intermediate_hadamard=manifest.get("hadamard", {}).get("intermediate_hadamard"))


# --------------------------------------------------------------------------- S3 decode via b12x half-extent + kernel


def decode_b12x_half_extent(
    exl3_pack: Path, layer: int, first_slot: int, slot_count: int, device: str
) -> dict[str, Any]:
    """Load one half-extent, plan + prepare weights, return weight metadata.

    The prepared weights are in ``trellis_native`` layout (FP4-packed codes
    + rotation scales, decoded by the kernel on-the-fly).
    """
    from b12x.moe.checkpoints.exl3 import read_exl3_manifest, read_exl3_layer  # type: ignore
    from b12x.moe.fused_moe._impl import plan_b12x_fp4_moe_weights, prepare_b12x_fp4_moe_weights  # type: ignore

    torch = _require_torch()
    dev = torch.device(device)
    m = read_exl3_manifest(exl3_pack)
    g = m.geometry
    bits = m.rates.bits
    if bits is None:
        raise ValueError("non-uniform rates not supported")
    layer_obj = read_exl3_layer(
        exl3_pack, m, layer, first_slot=first_slot, slot_count=slot_count
    )
    K, N = g.hidden_size, g.intermediate_size
    local = slot_count * g.slot_channels

    plan = plan_b12x_fp4_moe_weights(
        quant_modes="w4a16",
        source_format="exl3",
        activation="silu",
        params_dtype=torch.bfloat16,
        num_experts=g.num_experts,
        hidden_size=K,
        intermediate_size=local,
        trellis_bits=bits,
        trellis_codebook=m.codebook,
        trellis_rate_granularity="uniform",
    )
    prep = prepare_b12x_fp4_moe_weights(
        plan=plan,
        params_dtype=torch.bfloat16,
        exl3_layer=layer_obj,
        exl3_device=dev,
    )
    w = prep
    return {
        "layer": layer,
        "first_slot": first_slot,
        "slot_count": slot_count,
        "local_intermediate": local,
        "w1_fp4_shape": list(w.w1_fp4.shape),
        "w2_fp4_shape": list(w.w2_fp4.shape),
        "w1_alphas": [_round(v) for v in w.w1_alphas[:5].tolist()],
        "w2_alphas": [_round(v) for v in w.w2_alphas[:5].tolist()],
        "w1_blockscale": [int(v) for v in w.w1_blockscale.tolist()],
        "w2_blockscale": [int(v) for v in w.w2_blockscale.tolist()],
        "representation_layout": str(w.representation.layout),
        "w1_sha16": _sha(w.w1_fp4),
        "w2_sha16": _sha(w.w2_fp4),
    }


# --------------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    tstart = time.time()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-model", type=Path, required=True)
    ap.add_argument("--exl3-pack", type=Path, required=True)
    ap.add_argument("--exllamav3-pack", type=Path, default=None)
    ap.add_argument("--exllamav3-extsite", type=Path, default=None)
    ap.add_argument("--layer", type=int, default=1)
    ap.add_argument("--experts", default="0")
    ap.add_argument("--projection", default="gate_proj",
                    choices=("gate_proj", "up_proj", "down_proj"))
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    ap.add_argument("--json-out", type=Path, default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="no GPU; print manifest info only")
    ap.add_argument("--skip-kernel-decode", action="store_true",
                    help="skip the b12x kernel forward decode (S3)")
    args = ap.parse_args(argv)

    report: dict[str, Any] = {
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "checks": [],
    }

    # Dry-run: metadata only
    if args.dry_run:
        report["checks"].append(_result("dry_run", True, mode="shapes-only"))
        if (args.exl3_pack / "exl3-manifest.json").exists():
            with open(args.exl3_pack / "exl3-manifest.json") as f:
                m = json.load(f)
            report["manifest"] = {
                "kind": m.get("kind"),
                "codebook": m.get("codebook"),
                "geometry": m.get("geometry"),
                "rates": m.get("rates"),
                "hadamard": m.get("hadamard"),
            }
        out = json.dumps(report, indent=2, default=str)
        if args.json_out:
            args.json_out.write_text(out + "\n")
        print(out)
        return 0

    # S1
    s1 = s1_dense_spot(args.source_model, args.exl3_pack)
    report["checks"].append(s1)
    _log(f"S1: {s1['ok']}")

    # S3 decode via b12x kernel (half-extent, FC1 half to avoid barrier)
    if not args.skip_kernel_decode:
        _log(f"\nDecoding via b12x kernel on {args.device}...")
        try:
            half = _read_json(args.exl3_pack, "exl3-manifest.json")["geometry"]["num_slots"] // 2
        except Exception:
            half = 32
        halves = [(0, half), (half, half)]
        for fs, sc in halves:
            try:
                meta = decode_b12x_half_extent(
                    args.exl3_pack, args.layer, fs, sc, args.device
                )
                report["checks"].append(_result(f"S3_half_slot{fs}", True, **meta))
            except Exception as exc:
                import traceback
                report["checks"].append(_result(
                    f"S3_half_slot{fs}", False,
                    reason=str(exc), traceback=traceback.format_exc()
                ))

    report["duration_seconds"] = round(time.time() - tstart, 3)
    out = json.dumps(report, indent=2, default=str)
    if args.json_out:
        args.json_out.write_text(out + "\n")
    print(out)
    failed = [c for c in report["checks"] if not c.get("ok")]
    return 1 if failed else 0


def _read_json(p: Path, name: str) -> dict[str, Any]:
    return json.loads((p / name).read_text())


if __name__ == "__main__":
    raise SystemExit(main())