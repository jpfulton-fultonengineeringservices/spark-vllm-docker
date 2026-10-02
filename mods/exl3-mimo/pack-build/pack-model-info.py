#!/usr/bin/env python3
"""Auto-detect model geometry from a HuggingFace model directory.

Reads ``config.json`` and the weight map, emits a JSON geometry record on
stdout. Works for any MoE transformer: MiMo-V2, DeepSeek-V4, Kimi-K3, Qwen-MoE,
etc. No per-model config needed for standard architectures.

Usage::

    python3 pack-model-info.py /models/mimo-v2.6-flash-rl
    python3 pack-model-info.py /models/deepseek-v4.1 --json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _load_config(source: Path) -> dict:
    cfg_path = source / "config.json"
    if not cfg_path.exists():
        raise SystemExit(f"config.json not found in {source}")
    return json.loads(cfg_path.read_text())


def _text_config(cfg: dict) -> dict:
    return cfg.get("text_config", cfg)


def _weight_map(source: Path) -> dict[str, str] | None:
    idx = source / "model.safetensors.index.json"
    if idx.exists():
        return json.loads(idx.read_text()).get("weight_map")
    return None


def _list_keys_from_safetensors(source: Path) -> list[str]:
    try:
        from safetensors import safe_open
    except ImportError:
        print("safetensors not available; install safetensors or use an indexed model", file=sys.stderr)
        raise SystemExit(1)

    shards = sorted(source.glob("*.safetensors"))
    if not shards:
        return []
    keys: list[str] = []
    for shard in shards:
        with safe_open(str(shard), framework="pt") as f:
            keys.extend(f.keys())
    return keys


def _all_keys(source: Path) -> list[str]:
    wm = _weight_map(source)
    if wm is not None:
        return list(wm.keys())
    return _list_keys_from_safetensors(source)


def _find_moe_layers(keys: list[str]) -> list[int]:
    layers: set[int] = set()
    for k in keys:
        parts = k.split(".")
        for i, p in enumerate(parts):
            if p == "mlp" and i + 1 < len(parts) and parts[i + 1] == "experts":
                for j in range(i):
                    if parts[j] == "layers" and j + 1 < len(parts):
                        try:
                            layers.add(int(parts[j + 1]))
                        except ValueError:
                            pass
                break
    return sorted(layers)


def _find_dense_layers(keys: list[str], moe_indices: set[int]) -> list[int]:
    all_layers: set[int] = set()
    for k in keys:
        parts = k.split(".")
        for i, part in enumerate(parts):
            if part == "layers" and i + 1 < len(parts):
                try:
                    all_layers.add(int(parts[i + 1]))
                except ValueError:
                    pass
    return sorted(all_layers - moe_indices)


def detect(source: Path) -> dict:
    cfg = _load_config(source)
    tc = _text_config(cfg)

    architectures = cfg.get("architectures", [])
    hidden_size = int(tc.get("hidden_size", 0))
    inter = int(tc.get("moe_intermediate_size",
                       tc.get("intermediate_size", 0)))
    num_experts = int(tc.get("n_routed_experts",
                             tc.get("num_experts",
                                    tc.get("num_local_experts", 0))))
    num_slots = inter // 32 if inter % 32 == 0 else 0

    keys = _all_keys(source)
    moe_layers = _find_moe_layers(keys)
    dense_layers = _find_dense_layers(keys, set(moe_layers))

    result: dict = {
        "source": str(source),
        "architecture": architectures[0] if architectures else None,
        "hidden_size": hidden_size,
        "intermediate_size": inter,
        "num_experts": num_experts,
        "num_slots": num_slots,
        "moe_layer_count": len(moe_layers),
        "moe_layers": moe_layers,
        "dense_layers": dense_layers,
        "text_config": {
            k: v for k, v in tc.items()
            if isinstance(v, (str, int, float, bool, type(None)))
        },
    }

    if num_slots == 0 and inter > 0:
        result["_warning"] = (
            f"intermediate_size {inter} not cleanly divisible by 32; "
            "b12x exl3-v1 requires slot_channels=32"
        )

    if not moe_layers:
        result["_warning"] = (
            "no MoE expert layers detected in weight map; "
            "model may be dense or use non-standard key naming"
        )

    return result


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    json_flag = False
    args: list[str] = []
    for a in argv:
        if a in ("--json", "-j"):
            json_flag = True
        else:
            args.append(a)

    if len(args) != 1:
        print(f"usage: {sys.argv[0]} [--json] <source-model-dir>", file=sys.stderr)
        return 2

    source = Path(args[0])
    if not source.is_dir():
        print(f"source is not a directory: {source}", file=sys.stderr)
        return 1

    info = detect(source)
    print(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())