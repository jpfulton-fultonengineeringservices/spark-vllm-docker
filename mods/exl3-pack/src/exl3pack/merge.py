"""Merge per-shard qtensors with disjointness/coverage validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import shard as shard_mod

_LAYER_FILE_RE = re.compile(r"^model\.layers\.(\d+)\.safetensors$")


@dataclass
class FileRecord:
    path: Path
    shard_index: int
    sha256: str


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def collect(work_dirs: list[Path]) -> tuple[dict[int, FileRecord], dict[str, FileRecord]]:
    """Walk qtensors/ in each work dir; classify layer vs shared."""
    layers: dict[int, FileRecord] = {}
    shared: dict[str, FileRecord] = {}
    for idx, base in enumerate(work_dirs):
        qtensors = base / "qtensors" if (base / "qtensors").is_dir() else base
        if not qtensors.is_dir():
            raise SystemExit(f"{base}: no qtensors directory")
        for p in qtensors.iterdir():
            if not p.is_file():
                continue
            name = p.name
            m = _LAYER_FILE_RE.match(name)
            if m:
                layer_num = int(m.group(1))
                if layer_num in layers:
                    existing = layers[layer_num]
                    raise SystemExit(
                        f"overlap: model.layers.{layer_num}.safetensors in "
                        f"shard {existing.shard_index} and {idx}"
                    )
                layers[layer_num] = FileRecord(p, idx, _sha256(p))
            else:
                if name in shared:
                    existing = shared[name]
                    raise SystemExit(f"overlap: {name} in shard {existing.shard_index} and {idx}")
                shared[name] = FileRecord(p, idx, _sha256(p))
    return layers, shared


def validate(
    layers: dict[int, FileRecord],
    shared: dict[str, FileRecord],
    *,
    expected_layers: set[int],
    required_shared: Sequence[str] = (),
) -> None:
    """Fail loudly on gap/extra/missing-shared."""
    observed_layers = set(layers.keys())
    missing = expected_layers - observed_layers
    extra = observed_layers - expected_layers
    if missing:
        raise SystemExit(f"gap: missing layers {sorted(missing)}")
    if extra:
        raise SystemExit(f"unexpected layers {sorted(extra)}")
    missing_shared = [name for name in required_shared if name not in shared]
    if missing_shared:
        raise SystemExit(f"missing shared files: {missing_shared}")


def merge(
    work_dirs: list[Path],
    out: Path,
    *,
    expected_layers: set[int],
    required_shared: Sequence[str] = (),
) -> dict[str, Any]:
    """Validate, copy, and write merge-receipt.json."""
    layers, shared = collect(work_dirs)
    validate(layers, shared, expected_layers=expected_layers, required_shared=required_shared)
    qtensors = out / "qtensors"
    qtensors.mkdir(parents=True, exist_ok=True)
    receipt_layers: dict[str, str] = {}
    receipt_shared: dict[str, str] = {}
    for layer_num, rec in sorted(layers.items()):
        dest = qtensors / f"model.layers.{layer_num}.safetensors"
        shutil.copy2(rec.path, dest)
        receipt_layers[f"model.layers.{layer_num}.safetensors"] = rec.sha256
    for name, rec in sorted(shared.items()):
        dest = qtensors / name
        shutil.copy2(rec.path, dest)
        receipt_shared[name] = rec.sha256
    receipt: dict[str, Any] = {
        "schema": "merge-receipt/v1",
        "out": str(out),
        "shard_count": len(work_dirs),
        "layers": receipt_layers,
        "shared": receipt_shared,
        "received_from": [str(d) for d in work_dirs],
    }
    (out / "merge-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def parse_layer_spec(spec: str) -> set[int]:
    """Parse ``N1-N2,N3,...`` into a set of layer indices."""
    layers: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            layers.update(range(int(lo), int(hi) + 1))
        else:
            layers.add(int(part))
    return layers


def expected_layers_from_source(source: Path) -> set[int]:
    """Derive the full layer set (every MoE layer + the single dense layer)."""
    moe = set(shard_mod.discover_moe_layers(source))
    _, total = shard_mod.discover_num_modules(source)
    dense = [i for i in range(total) if i not in moe]
    if len(dense) != 1:
        raise SystemExit(f"expected exactly one dense layer, got {dense}")
    return moe | set(dense)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", required=True, nargs="+", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--layers", default=None, help="explicit layer range (N1-N2,N3,...)")
    parser.add_argument("--source", type=Path, help="source model dir to derive expected layers")
    parser.add_argument(
        "--required-shared",
        default="model.embed_tokens.safetensors",
        help="comma-separated required shared basenames",
    )
    args = parser.parse_args(argv)
    if args.layers:
        expected = parse_layer_spec(args.layers)
    elif args.source:
        expected = expected_layers_from_source(args.source)
    else:
        raise SystemExit("specify --layers or --source to derive expected layer set")
    required = [s.strip() for s in args.required_shared.split(",") if s.strip()]
    receipt = merge(args.work, args.out, expected_layers=expected, required_shared=required)
    print(f"merged {len(receipt['layers'])} layers + {len(receipt['shared'])} shared -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
