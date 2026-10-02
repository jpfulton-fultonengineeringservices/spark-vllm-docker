#!/usr/bin/env python3
"""Assemble a servable checkpoint from a source model + a b12x ``exl3-v1`` pack.

The b12x EXL3 runtime (``vllm/model_executor/layers/quantization/exl3.py``)
reads ``exl3-manifest.json`` + ``exl3-layer-<NNNNN>.safetensors`` from the
served model directory, and loads everything *else* (attention / dense-MLP /
embeddings / head / router) from the ordinary HF checkpoint. Its dense path
(``Exl3Config`` extends ``ModelOptMxFp8Config``) only accepts MXFP8-serialized
weights — a ModelOpt-FP8 block checkpoint (``weight_scale_inv``) cannot pass
through it. So the assembly:

  1. strips the routed-expert tensors (they live in the exl3 container),
  2. dequantizes dense FP8-block weights to BF16 (exact: every e4m3 value is
     representable in bf16; one rounding on the scale product),
  3. copies the exl3 container next to the dense checkpoint,
  4. writes a config whose ``quantization_config`` declares the exl3 method with
     every non-routed linear module in ``ignored_layers`` (they load as plain
     BF16 through UnquantizedLinearMethod).

Run INSIDE the b12x runtime image (torch + safetensors). Model-agnostic: the
expert predicate and the ignored-layer set are derived from the source.

Inputs:
  --source  HF model dir (the same one ``convert_model`` consumed)
  --pack    ``exl3-v1`` container dir (``exl3-manifest.json`` + layer files)
  --out     serving dir to write (e.g. /nas-1/models/<family>/<slug>)

Output layout (a normal-looking HF checkpoint the FES staging path can verify):
  config.json                     source config + exl3 quantization_config
  model-<NNNNN>-of-<NNNNN>.safetensors   source shards MINUS routed experts,
                                          dense FP8-block weights dequantized
  model.safetensors.index.json    weight map for the kept tensors only
  exl3-manifest.json + exl3-layer-<NNNNN>.safetensors   copied from --pack
  tokenizer / chat template / trust_remote_code .py    copied from --source
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

EXL3_MANIFEST_FILENAME = "exl3-manifest.json"
LAYER_PREFIX = "exl3-layer-"
EXPERT_MARKER = ".mlp.experts."
WEIGHT_SUFFIX = ".weight"
SCALE_INV_SUFFIX = ".weight_scale_inv"


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
    """Module basenames the exl3 method must leave unquantized (bf16 dense)."""
    names: set[str] = {"gate", "lm_head"}
    for k in kept_keys:
        for part in k.split("."):
            if part.endswith("_proj"):
                names.add(part)
    return sorted(names)


def _dequant_fp8_block(weight: torch.Tensor, scale_inv: torch.Tensor,
                       block: tuple[int, int]) -> torch.Tensor:
    """Dequant a DeepSeek-style FP8-block weight (weight_scale_inv) to BF16.

    The scale grid covers the block-padded weight (ceil shape/dim per block);
    expand with repeat_interleave over the padded grid, then slice to the real
    weight shape.
    """
    m, n = weight.shape
    so, si = scale_inv.shape
    bs_out, bs_in = block
    if so * bs_out < m or si * bs_in < n:
        raise SystemExit(
            f"fp8 block dequant: scale grid {(so, si)} x block {block} does not "
            f"cover weight {(m, n)}"
        )
    scale = scale_inv.to(torch.float32)
    scale = scale.repeat_interleave(bs_out, dim=0).repeat_interleave(bs_in, dim=1)
    return (weight.to(torch.float32) * scale[:m, :n]).to(torch.bfloat16).contiguous()


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
                  kept_keys: list[str], fp8_bases: set[str],
                  fp8_block: tuple[int, int],
                  max_bytes: int) -> tuple[dict[str, str], int]:
    """Write kept tensors (fp8 pairs dequantized) to fresh shards."""
    out_keys = [k for k in kept_keys if not k.endswith(SCALE_INV_SUFFIX)]

    def load(key: str) -> torch.Tensor:
        with safe_open(str(source / weight_map[key]), framework="pt") as handle:
            return handle.get_tensor(key)

    new_map: dict[str, str] = {}
    shard_idx = 0
    pending: dict[str, torch.Tensor] = {}
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

    by_source: dict[str, list[str]] = {}
    for key in out_keys:
        by_source.setdefault(weight_map[key], []).append(key)

    dequanted = 0
    for src_shard in sorted(by_source):
        with safe_open(str(source / src_shard), framework="pt") as handle:
            for key in sorted(by_source[src_shard]):
                if key.endswith(WEIGHT_SUFFIX) and key[:-len(WEIGHT_SUFFIX)] in fp8_bases:
                    scale_key = key[:-len(WEIGHT_SUFFIX)] + SCALE_INV_SUFFIX
                    tensor = _dequant_fp8_block(
                        handle.get_tensor(key),
                        load(scale_key),
                        fp8_block,
                    )
                    dequanted += 1
                else:
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
    total = sum((out / name).stat().st_size for name in set(new_map.values()))
    print(f"  dense: {len(out_keys)} tensor(s), {dequanted} fp8-block pair(s) "
          f"dequanted to bf16, {total / 1e9:.1f} GB across {total_files} shard(s)")
    return new_map, total


def _patch_config(source: Path, pack: Path, out: Path, kept_keys: list[str],
                  dense_format: str) -> None:
    cfg = _load_json(source / "config.json")
    src_qc = cfg.get("quantization_config", {}) or {}
    manifest = _load_json(pack / EXL3_MANIFEST_FILENAME)
    rates = manifest.get("rates", {})

    cfg["quantization_config"] = {
        "quant_method": "exl3",
        "codebook": manifest.get("codebook"),
        "bits": rates.get("bits"),
        "exl3": {"manifest": EXL3_MANIFEST_FILENAME},
        "dense_format": dense_format,
        "ignored_layers": _ignored_layers(kept_keys),
        "original_quantization_config": src_qc,
    }
    (out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")


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
                    help="config dense_format label (default: bf16 — dense is "
                         "dequanted; override only with a matching layout)")
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

    wm = _weight_map(args.source)
    kept = [k for k in wm if EXPERT_MARKER not in k]
    dropped = len(wm) - len(kept)
    fp8_bases = (
        {k[:-len(WEIGHT_SUFFIX)] for k in kept if k.endswith(WEIGHT_SUFFIX)}
        & {k[:-len(SCALE_INV_SUFFIX)] for k in kept if k.endswith(SCALE_INV_SUFFIX)}
    )
    src_qc = _load_json(args.source / "config.json").get("quantization_config", {}) or {}
    block_raw = src_qc.get("weight_block_size") or [128, 128]
    fp8_block = (int(block_raw[0]), int(block_raw[1]))
    dense_format = args.dense_format or ("bf16" if fp8_bases else "fp8")

    print(f"assemble: source={args.source} pack={args.pack} out={args.out}")
    print(f"  tensors: keep={len(kept)} drop_routed_experts={dropped} "
          f"fp8_block_pairs={len(fp8_bases)} block={list(fp8_block)} "
          f"dense_format={dense_format}")

    new_map, total = _write_shards(args.source, args.out, wm, kept, fp8_bases,
                                   fp8_block, args.max_shard_bytes)
    (args.out / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total}, "weight_map": new_map}, indent=2) + "\n"
    )

    layers = _copy_pack(args.pack, args.out)
    print(f"  copied {EXL3_MANIFEST_FILENAME} + {layers} exl3 layer file(s)")

    _patch_config(args.source, args.pack, args.out, kept, dense_format)
    copied = _copy_aux(args.source, args.out)
    print(f"  copied {len(copied)} auxiliary file(s)/dir(s): {', '.join(copied[:8])}"
          f"{' ...' if len(copied) > 8 else ''}")

    cfg = _load_json(args.out / "config.json")
    tc = cfg.get("text_config", cfg)
    manifest = _load_json(args.out / EXL3_MANIFEST_FILENAME)
    geometry = manifest.get("geometry", {})
    if geometry.get("hidden_size") and geometry["hidden_size"] != tc.get("hidden_size"):
        raise SystemExit(
            f"manifest hidden_size {geometry['hidden_size']} != source hidden "
            f"{tc.get('hidden_size')}; wrong pack for this source?"
        )
    print("assemble: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
