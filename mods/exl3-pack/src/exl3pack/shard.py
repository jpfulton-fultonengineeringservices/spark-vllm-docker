"""Torch-free module-range planning for parallel EXL3 conversion."""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

MOE_EXPERT_MARKER = ".mlp.experts."
_LAYER_FILE_RE = re.compile(r"^model\.layers\.(\d+)\.safetensors$")


@dataclass(frozen=True)
class Shard:
    """One inclusive absolute ``model.modules`` range."""

    index: int
    module_start: int
    module_end: int
    num_hidden_layers: int

    @property
    def layers(self) -> tuple[int, ...]:
        return tuple(
            module - 1
            for module in range(self.module_start, self.module_end + 1)
            if 1 <= module <= self.num_hidden_layers
        )


def plan_shards(
    num_modules: int, num_shards: int, *, num_hidden_layers: int | None = None
) -> list[Shard]:
    """Partition all absolute module indices into contiguous inclusive ranges."""
    if num_modules < 1:
        raise ValueError("num_modules must be positive")
    if num_shards < 1:
        raise ValueError("num_shards must be positive")
    if num_shards > num_modules:
        raise ValueError("num_shards cannot exceed num_modules")
    layers = num_hidden_layers if num_hidden_layers is not None else max(0, num_modules - 3)
    if layers > num_modules - 1:
        raise ValueError("num_hidden_layers does not fit module count")
    base, remainder = divmod(num_modules, num_shards)
    result: list[Shard] = []
    start = 0
    for index in range(num_shards):
        size = base + (1 if index < remainder else 0)
        end = start + size - 1
        result.append(Shard(index, start, end, layers))
        start = end + 1
    return result


def module_args(shard_or_start: Shard | int, end: int | None = None) -> list[str]:
    """Return convert_model's absolute inclusive module-range arguments."""
    if isinstance(shard_or_start, Shard):
        start, last = shard_or_start.module_start, shard_or_start.module_end
    else:
        if end is None:
            raise TypeError("end is required when start is an integer")
        start, last = shard_or_start, end
    if start < 0 or last < start:
        raise ValueError("invalid module range")
    return ["--module-start", str(start), "--max_module", str(last)]


def discover_moe_layers(source: Path) -> list[int]:
    """Read MoE layer indices from a safetensors index, without torch."""
    index_path = source / "model.safetensors.index.json"
    if not index_path.is_file():
        raise SystemExit(f"model.safetensors.index.json not found in {source}")
    data = json.loads(index_path.read_text())
    weight_map = data.get("weight_map")
    if not isinstance(weight_map, dict):
        raise SystemExit(f"{index_path}: missing weight_map")
    layers: set[int] = set()
    for key in weight_map:
        if MOE_EXPERT_MARKER not in key:
            continue
        parts = key.split(".")
        try:
            pos = parts.index("layers")
            layers.add(int(parts[pos + 1]))
        except (ValueError, IndexError):
            continue
    if not layers:
        raise SystemExit(f"no MoE layers found in {index_path}")
    return sorted(layers)


def _config(source: Path) -> dict[str, object]:
    path = source / "config.json"
    if not path.is_file():
        raise SystemExit(f"config.json not found in {source}")
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise SystemExit(f"{path}: expected JSON object")
    text_cfg = data.get("text_config", data)
    return text_cfg if isinstance(text_cfg, dict) else data


def _as_int(value: object, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str) and value.strip():
        try:
            return int(value)
        except ValueError:
            return default
    return default


def discover_num_modules(source: Path) -> tuple[int, int]:
    """Return ``(total_modules, num_hidden_layers)`` for MiMo-style models."""
    config = _config(source)
    if "num_hidden_layers" not in config:
        raise SystemExit(f"{source}/config.json: num_hidden_layers missing")
    layers = _as_int(config["num_hidden_layers"])
    return layers + 3, layers


def plan_for_source(source: Path, num_shards: int) -> list[Shard]:
    modules, layers = discover_num_modules(source)
    return plan_shards(modules, num_shards, num_hidden_layers=layers)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", "--source", type=Path)
    parser.add_argument("--nodes", default=None, help="comma-separated node names")
    parser.add_argument("--shards", required=True, type=int)
    parser.add_argument("--num-modules", type=int, default=None)
    parser.add_argument("--num-hidden-layers", type=int, default=None)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--tsv", action="store_true")
    args = parser.parse_args(argv)
    if args.model:
        modules, layers = discover_num_modules(args.model)
        moe = discover_moe_layers(args.model)
    elif args.num_modules is not None and args.num_hidden_layers is not None:
        modules, layers = args.num_modules, args.num_hidden_layers
        moe = []
    else:
        raise SystemExit("specify --model or --num-modules/--num-hidden-layers")
    shards = plan_shards(
        modules, args.shards, num_hidden_layers=layers
    )
    if args.tsv:
        for s in shards:
            layers_csv = ",".join(str(n) for n in s.layers)
            print(f"{s.index}\t{s.module_start}\t{s.module_end}\t{layers_csv}")
        return 0
    payload = {
        "source": str(args.model) if args.model else None,
        "num_modules": modules,
        "num_hidden_layers": shards[0].num_hidden_layers,
        "moe_layer_indices": moe,
        "nodes": args.nodes.split(",") if args.nodes else [],
        "shards": [asdict(shard) | {"layers": list(shard.layers)} for shard in shards],
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
