#!/usr/bin/env python3
"""Verify a staged model weights directory.

Port of FES cluster-config scripts/model-weights.sh verify_location: shard
count vs model.safetensors.index.json, total size parity, and
hf_quant_config.json presence for nvfp4 checkpoints.

Exit 0 with a JSON summary on stdout; exit 1 with a single reason line on
stderr on any failure.
"""

import json
import os
import sys

SIZE_TOLERANCE = 1.02
EXL3_MANIFEST = "exl3-manifest.json"


def fail(reason):
    print(reason, file=sys.stderr)
    sys.exit(1)


def exl3_layer_files(path):
    """Layer file names declared by an exl3-v1 manifest, or None if not an exl3 pack."""
    manifest_path = os.path.join(path, EXL3_MANIFEST)
    if not os.path.isfile(manifest_path):
        return None
    try:
        with open(manifest_path) as f:
            manifest = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        fail(f"unreadable {EXL3_MANIFEST} in {path}: {exc}")
    if manifest.get("kind") != "exl3-manifest":
        fail(f"{EXL3_MANIFEST} in {path} is not kind=exl3-manifest")
    layers = manifest.get("layers") or {}
    if not layers:
        fail(f"{EXL3_MANIFEST} in {path} declares no layers")
    return {ref["file"] for ref in layers.values() if isinstance(ref, dict) and ref.get("file")}


def verify(path):
    if not os.path.isdir(path):
        fail(f"weights dir not found: {path}")
    if not os.listdir(path):
        fail(f"weights dir is empty: {path}")

    config_path = os.path.join(path, "config.json")
    if not os.path.isfile(config_path):
        fail(f"missing config.json in {path}")
    try:
        with open(config_path) as f:
            config = json.load(f)
        quant_format = str((config.get("quantization_config") or {}).get("format") or "").lower()
    except (json.JSONDecodeError, OSError) as exc:
        fail(f"unreadable config.json in {path}: {exc}")

    if quant_format == "nvfp4" and not os.path.isfile(os.path.join(path, "hf_quant_config.json")):
        fail("nvfp4 checkpoint missing hf_quant_config.json")

    exl3_files = exl3_layer_files(path)
    if exl3_files is not None:
        missing_layers = sorted(
            f for f in exl3_files if not os.path.isfile(os.path.join(path, f))
        )
        if missing_layers:
            fail(
                f"exl3 manifest references {len(missing_layers)} missing layer "
                f"file(s), e.g. {missing_layers[0]}"
            )
    exl3_layers = len(exl3_files) if exl3_files is not None else 0

    index_path = os.path.join(path, "model.safetensors.index.json")
    if os.path.isfile(index_path):
        try:
            with open(index_path) as f:
                index = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            fail(f"unreadable model.safetensors.index.json in {path}: {exc}")
        weight_map = index.get("weight_map") or {}
        total_size = index.get("total_size") or 0
        shards = sorted(set(weight_map.values()))
        if not shards:
            fail("empty weight_map in model.safetensors.index.json")
        missing = [s for s in shards if not os.path.isfile(os.path.join(path, s))]
        if missing:
            fail(f"missing {len(missing)}/{len(shards)} shards, e.g. {missing[0]}")
        staged = sum(os.path.getsize(os.path.join(path, s)) for s in shards)
        if total_size and not total_size <= staged <= total_size * SIZE_TOLERANCE:
            fail(f"size mismatch: index total_size={total_size} staged={staged}")
        return {"shards": len(shards), "bytes": staged, "quant_format": quant_format,
                "exl3_layers": exl3_layers}

    single = os.path.join(path, "model.safetensors")
    if not os.path.isfile(single):
        fail("neither model.safetensors.index.json nor model.safetensors present")
    return {"shards": 1, "bytes": os.path.getsize(single), "quant_format": quant_format,
            "exl3_layers": exl3_layers}


def main():
    if len(sys.argv) != 2:
        print(f"usage: {os.path.basename(sys.argv[0])} <weights-dir>", file=sys.stderr)
        sys.exit(2)
    return verify(sys.argv[1])


if __name__ == "__main__":
    print(json.dumps(main()))