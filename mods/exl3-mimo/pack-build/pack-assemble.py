#!/usr/bin/env python3
"""Assemble a servable checkpoint from a source model + a b12x ``exl3-v1`` pack.

The b12x EXL3 runtime (``vllm/model_executor/layers/quantization/exl3.py``)
reads ``exl3-manifest.json`` + ``exl3-layer-<NNNNN>.safetensors`` from the
served model directory, and loads everything *else* (attention / dense-MLP /
embeddings / head / router) from the ordinary HF checkpoint. So the pack alone
is not servable: the routed-expert weights must be *removed* from the dense
checkpoint and the exl3 container dropped in next to a config that declares the
``exl3`` quant method.

This script builds that directory. Run INSIDE the b12x runtime image (torch +
safetensors). Model-agnostic: geometry/ignored-layer names are derived from the
source, so any MoE that onboards to the pack builder works.

Inputs:
  --source  HF model dir (the same one ``convert_model`` consumed)
  --pack    ``exl3-v1`` container dir (``exl3-manifest.json`` + layer files)
  --out     serving dir to write (e.g. /nas-1/models/<family>/<slug>)

Output layout (a normal-looking HF checkpoint the FES staging path can verify):
  config.json                     source config + exl3 quantization_config
  model-<NNNNN>-of-<NNNNN>.safetensors   source shards MINUS routed experts
  model.safetensors.index.json    weight map for the kept tensors only
  exl3-manifest.json + exl3-layer-<NNNNN>.safetensors   copied from --pack
  tokenizer / chat template / trust_remote_code .py    copied from --source

The routed-expert predicate is ``.mlp.experts.`` in the tensor name; shared
experts and the router (``.mlp.gate.``) are kept.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

EXL3_MANIFEST_FILENAME = "exl3-manifest.json"
LAYER_PREFIX = "exl3-layer-"
EXPERT_MARKER = ".mlp.experts."


def _text_config(cfg: dict) -> dict:
    return cfg.get("text_config", cfg)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _weight_map(source: Path) -> dict[str, str]:
    idx = source / "model.safetensors.index.json"
    if not idx.exists():
        raise SystemExit(
            f"source has no model.safetensors.index.json: {source}\n"
            "assembly needs an indexed checkpoint to map tensors to shards"
        )
    return _load_json(idx)["weight_map"]


def _ignored_layers(kept_keys: list[str]) -> list[str]:
    """Module basenames the exl3 method must leave in the source format."""
    names: set[str] = {"gate", "lm_head"}
    for k in kept_keys:
        for part in k.split("."):
            if part.endswith("_proj"):
                names.add(part)
    return sorted(names)


def _copy_aux(source: Path, out: Path) -> list[str]:
    """Copy everything except the weights/index/pdf (config written separately)."""
    copied: list[str] = []
    skip_suffix = (".safetensors", ".pdf")
    # config.json is written by _patch_config, not copied from the source.
    skip_names = {"model.safetensors.index.json", "config.json"}
    for entry in sorted(source.iterdir()):
        name = entry.name
        if name.startswith(".") or name in skip_names:
            continue
        if entry.is_file() and name.endswith(skip_suffix):
            continue
        if entry.is_dir() and name in ("__pycache__", ".cache"):
            continue
        dest = out / name
        if entry.is_dir():
            shutil.copytree(entry, dest, dirs_exist_ok=True)
        else:
            shutil.copy2(entry, dest)
        copied.append(name)
    return copied


def _write_shards(source: Path, out: Path, weight_map: dict[str, str],
                  kept_keys: list[str], max_bytes: int) -> tuple[dict[str, str], int]:
    """Copy kept tensors into fresh shards; return (new_weight_map, total_bytes)."""
    by_source: dict[str, list[str]] = {}
    for key in kept_keys:
        by_source.setdefault(weight_map[key], []).append(key)

    new_map: dict[str, str] = {}
    total = 0
    shard_idx = 0
    pending: dict[str, object] = {}
    pending_bytes = 0

    def flush() -> None:
        nonlocal shard_idx, pending, pending_bytes
        if not pending:
            return
        shard_idx += 1
        fname = f"model-{shard_idx:05d}-of-PLACEHOLDER.safetensors"
        save_file(pending, str(out / fname))
        for key in pending:
            new_map[key] = fname
        pending = {}
        pending_bytes = 0

    for src_shard in sorted(by_source):
        with safe_open(str(source / src_shard), framework="pt") as handle:
            for key in sorted(by_source[src_shard]):
                tensor = handle.get_tensor(key).contiguous()
                size = tensor.numel() * tensor.element_size()
                if pending and pending_bytes + size > max_bytes:
                    flush()
                pending[key] = tensor
                pending_bytes += size
    flush()

    # Rename the placeholder shards to the real -of-<NNNNN> form.
    total_files = shard_idx
    for i in range(1, total_files + 1):
        old = out / f"model-{i:05d}-of-PLACEHOLDER.safetensors"
        new = out / f"model-{i:05d}-of-{total_files:05d}.safetensors"
        old.rename(new)
        for key, val in list(new_map.items()):
            if val == old.name:
                new_map[key] = new.name
    total = sum(
        (out / name).stat().st_size for name in set(new_map.values())
    )
    return new_map, total


def _patch_config(source: Path, pack: Path, out: Path, kept_keys: list[str],
                  dense_format: str) -> dict:
    cfg = _load_json(source / "config.json")
    tc = _text_config(cfg)
    src_qc = cfg.get("quantization_config", {}) or {}
    manifest = _load_json(pack / EXL3_MANIFEST_FILENAME)
    geometry = manifest.get("geometry", {})
    rates = manifest.get("rates", {})

    # Exl3Config extends ModelOptMxFp8Config: the non-routed dense projections are
    # loaded through the fork's MXFP8/FP8 dense path, and `ignored_layers` names
    # the modules that stay in the source's *unquantized* (bf16) format. The
    # source checkpoint's own quantization_config is authoritative for that
    # split (e.g. MiMo ignores o_proj); fall back to a derived suffix set only
    # when the source declares none. Suffix names (not full paths) match the
    # fork's is_layer_skipped(..., match_mode="suffix").
    src_ignored = src_qc.get("ignored_layers") or []
    if src_ignored:
        ignored = sorted({str(n).split(".")[-1] for n in src_ignored})
    else:
        ignored = _ignored_layers(kept_keys)

    cfg["quantization_config"] = {
        "quant_method": "exl3",
        "version": src_qc.get("version"),
        "codebook": manifest.get("codebook"),
        "bits": rates.get("bits"),
        "exl3": {"manifest": EXL3_MANIFEST_FILENAME},
        "dense_format": dense_format,
        "ignored_layers": ignored,
        "original_quantization_config": src_qc,
    }
    (out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    return {"geometry": geometry, "dense_format": dense_format,
            "num_experts": geometry.get("num_experts"),
            "hidden_size": geometry.get("hidden_size"),
            "source_hidden": tc.get("hidden_size")}


def _copy_pack(pack: Path, out: Path) -> int:
    manifest = pack / EXL3_MANIFEST_FILENAME
    if not manifest.exists():
        raise SystemExit(f"pack has no {EXL3_MANIFEST_FILENAME}: {pack}")
    shutil.copy2(manifest, out / EXL3_MANIFEST_FILENAME)
    layers = 0
    for f in sorted(pack.glob(f"{LAYER_PREFIX}*.safetensors")):
        dest = out / f.name
        # Idempotent: skip an already-present identical layer (re-runs after a
        # late failure must not re-copy ~110 GB).
        if not (dest.exists() and dest.stat().st_size == f.stat().st_size):
            shutil.copy2(f, dest)
        layers += 1
    return layers


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, type=Path, help="HF source model dir")
    ap.add_argument("--pack", required=True, type=Path, help="exl3-v1 container dir")
    ap.add_argument("--out", required=True, type=Path, help="serving dir to write")
    ap.add_argument("--dense-format", default=None,
                    help="override config dense_format (default: fp8 for an fp8 source)")
    ap.add_argument("--max-shard-bytes", type=int, default=5_000_000_000,
                    help="approx max bytes per output shard (default 5 GB)")
    ap.add_argument("--force", action="store_true",
                    help="overwrite a non-empty --out")
    args = ap.parse_args(argv)

    if not args.source.is_dir():
        raise SystemExit(f"source dir not found: {args.source}")
    if not args.pack.is_dir():
        raise SystemExit(f"pack dir not found: {args.pack}")
    if args.out.exists():
        entries = [p for p in args.out.iterdir() if not p.name.startswith(".")]
        if entries and not args.force:
            raise SystemExit(
                f"output dir not empty: {args.out} (pass --force to overwrite)"
            )
    args.out.mkdir(parents=True, exist_ok=True)

    src_qc = _load_json(args.source / "config.json").get("quantization_config", {}) or {}
    dense_format = args.dense_format or (
        "fp8" if str(src_qc.get("quant_method", "")).lower() == "fp8" else "mxfp8"
    )

    wm = _weight_map(args.source)
    kept = [k for k in wm if EXPERT_MARKER not in k]
    dropped = len(wm) - len(kept)
    print(f"assemble: source={args.source} pack={args.pack} out={args.out}")
    print(f"  tensors: keep={len(kept)} drop_routed_experts={dropped} "
          f"dense_format={dense_format}")

    new_map, total = _write_shards(args.source, args.out, wm, kept, args.max_shard_bytes)
    (args.out / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total}, "weight_map": new_map}, indent=2) + "\n"
    )
    shards = sorted(set(new_map.values()))
    print(f"  wrote {len(shards)} shard(s), {total / 1e9:.1f} GB of dense weights")

    layers = _copy_pack(args.pack, args.out)
    print(f"  copied {EXL3_MANIFEST_FILENAME} + {layers} exl3 layer file(s)")

    info = _patch_config(args.source, args.pack, args.out, kept, dense_format)
    copied = _copy_aux(args.source, args.out)
    print(f"  copied {len(copied)} auxiliary file(s)/dir(s): {', '.join(copied[:8])}"
          f"{' ...' if len(copied) > 8 else ''}")

    if info["num_experts"] and info["hidden_size"] != info["source_hidden"]:
        raise SystemExit(
            f"manifest hidden_size {info['hidden_size']} != source hidden "
            f"{info['source_hidden']}; wrong pack for this source?"
        )
    print("assemble: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
