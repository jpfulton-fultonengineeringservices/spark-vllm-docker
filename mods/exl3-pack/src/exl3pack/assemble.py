"""Assemble a servable checkpoint from a source model + a b12x ``exl3-v1`` pack.

Torch-dependent (runs INSIDE the b12x image). Faithful port of the original
``pack-assemble.py`` with two additions driven by the "output on the NAS,
tensor-by-tensor, clean up node disk" requirement:

- **Streamed output**: kept dense tensors are written to the output directory
  (on the NAS) in incremental shards; nothing large is buffered in memory.
- **Cleanup**: with ``--cleanup pack``/``all``, each source/bespoke input is
  released as soon as it has been fully copied into the NAS output, so node
  local disk usage stays minimal.

The FP8-block dequant semantics (flat vs per-group interleaved scale grid) are
preserved exactly from the original — do not "simplify".
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from safetensors import safe_open
from safetensors.torch import save_file

EXL3_MANIFEST_FILENAME = "exl3-manifest.json"
LAYER_PREFIX = "exl3-layer-"
EXPERT_MARKER = ".mlp.experts."
WEIGHT_SUFFIX = ".weight"
SCALE_INV_SUFFIX = ".weight_scale_inv"


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise SystemExit(f"expected a JSON object in {path}")
    return data


def _weight_map(source: Path) -> dict[str, str]:
    idx = source / "model.safetensors.index.json"
    if not idx.exists():
        raise SystemExit(
            f"source has no model.safetensors.index.json: {source}\n"
            "assembly needs an indexed checkpoint to map tensors to shards"
        )
    wm = _load_json(idx)["weight_map"]
    if not isinstance(wm, dict):
        raise SystemExit(f"{idx}: weight_map is not an object")
    return {str(k): str(v) for k, v in wm.items()}


def _ignored_layers(kept_keys: list[str]) -> list[str]:
    """Module basenames the exl3 method must leave unquantized (bf16 dense)."""
    names: set[str] = {"gate", "lm_head"}
    for k in kept_keys:
        for part in k.split("."):
            if part.endswith("_proj"):
                names.add(part)
    return sorted(names)


def _dequant_fp8_block(
    weight: torch.Tensor, scale_inv: torch.Tensor, block: tuple[int, int]
) -> torch.Tensor:
    m, n = weight.shape
    so, si = scale_inv.shape
    bs_out, bs_in = block
    if so * bs_out < m or si * bs_in < n:
        raise SystemExit(
            f"fp8 block dequant: scale grid {(so, si)} x block {block} does not "
            f"cover weight {(m, n)}"
        )
    scale = scale_inv.to(torch.float32).repeat_interleave(bs_out, dim=0).repeat_interleave(
        bs_in, dim=1
    )
    return (weight.to(torch.float32) * scale[:m, :n]).to(torch.bfloat16).contiguous()


def _dequant_fp8_block_per_group(
    weight: torch.Tensor, scale_inv: torch.Tensor, block: tuple[int, int], num_groups: int
) -> torch.Tensor:
    m, n = weight.shape
    so, _si = scale_inv.shape
    bs_out, bs_in = block
    if m % num_groups:
        raise SystemExit(
            f"fp8 per-group dequant: weight rows {m} not divisible by {num_groups} groups"
        )
    rows_per_group = m // num_groups
    scale_rows_per_group = -(-rows_per_group // bs_out)
    if so != scale_rows_per_group * num_groups:
        raise SystemExit(
            f"fp8 per-group dequant: scale grid rows {so} != "
            f"{scale_rows_per_group} * {num_groups} groups "
            f"(rows_per_group={rows_per_group}, block={bs_out})"
        )
    out = torch.empty((m, n), dtype=torch.float32)
    wf = weight.to(torch.float32)
    sf = scale_inv.to(torch.float32)
    for g in range(num_groups):
        row_start = g * rows_per_group
        scale_start = g * scale_rows_per_group
        w_g = wf[row_start : row_start + rows_per_group]
        s_g = (
            sf[scale_start : scale_start + scale_rows_per_group]
            .repeat_interleave(bs_out, dim=0)
            .repeat_interleave(bs_in, dim=1)[:rows_per_group, :n]
        )
        out[row_start : row_start + rows_per_group] = w_g * s_g
    return out.to(torch.bfloat16).contiguous()


def _dequant_fp8_block_dispatch(
    base: str,
    weight: torch.Tensor,
    scale_inv: torch.Tensor,
    block: tuple[int, int],
    num_groups: int,
) -> tuple[torch.Tensor, bool]:
    """Pick flat vs per-group FP8 dequant from the scale-grid geometry."""
    m = weight.shape[0]
    bs_out = block[0]
    so = scale_inv.shape[0]
    flat_rows = -(-m // bs_out)
    if so == flat_rows or num_groups <= 1:
        return _dequant_fp8_block(weight, scale_inv, block), False
    group_rows = -(-(m // num_groups) // bs_out) * num_groups
    if m % num_groups == 0 and so == group_rows:
        return _dequant_fp8_block_per_group(weight, scale_inv, block, num_groups), True
    raise SystemExit(
        f"{base}: fp8 scale grid rows {so} matches neither flat ({flat_rows}) nor "
        f"per-group ({group_rows} for {num_groups} groups); weight rows={m} block={bs_out}"
    )


@dataclass
class CleanupPolicy:
    """What local inputs to release after they are copied to the NAS output."""

    delete_pack_after_copy: bool = False
    delete_source_after_read: bool = False


def _copy_aux(source: Path, out: Path) -> list[str]:
    """Copy everything except weights/index/pdf (config written separately)."""
    copied: list[str] = []
    skip_suffix = (".safetensors", ".pdf")
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


def _write_shards(
    source: Path,
    out: Path,
    weight_map: dict[str, str],
    kept_keys: list[str],
    fp8_bases: set[str],
    fp8_block: tuple[int, int],
    ckpt_groups: int,
    max_bytes: int,
    progress: Callable[[str], None],
) -> tuple[dict[str, str], int]:
    """Write kept tensors (fp8 pairs dequantized) to fresh shards in ``out``."""
    out_keys = [k for k in kept_keys if not k.endswith(SCALE_INV_SUFFIX)]

    def load(key: str) -> torch.Tensor:
        with safe_open(str(source / weight_map[key]), framework="pt") as handle:
            return cast(torch.Tensor, handle.get_tensor(key))

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
        progress(f"dense shard {shard_idx}: {len(pending)} tensor(s) -> {fname}")
        pending = {}
        pending_bytes = 0

    by_source: dict[str, list[str]] = {}
    for key in out_keys:
        by_source.setdefault(weight_map[key], []).append(key)

    dequanted = 0
    per_group = 0
    for src_shard in sorted(by_source):
        with safe_open(str(source / src_shard), framework="pt") as handle:
            for key in sorted(by_source[src_shard]):
                if key.endswith(WEIGHT_SUFFIX) and key[: -len(WEIGHT_SUFFIX)] in fp8_bases:
                    base = key[: -len(WEIGHT_SUFFIX)]
                    weight = handle.get_tensor(key)
                    scale = load(base + SCALE_INV_SUFFIX)
                    tensor, used_per_group = _dequant_fp8_block_dispatch(
                        base, weight, scale, fp8_block, ckpt_groups
                    )
                    if used_per_group:
                        per_group += 1
                    dequanted += 1
                else:
                    tensor = handle.get_tensor(key).contiguous()
                size = tensor.numel() * tensor.element_size()
                if pending and pending_bytes + size > max_bytes:
                    flush()
                pending[key] = tensor
                pending_bytes += size
    flush()

    total_files = shard_idx
    for i in range(1, total_files + 1):
        old = out / f"model-{i:05d}-of-PLACEHOLDER.safetensors"
        new = out / f"model-{i:05d}-of-{total_files:05d}.safetensors"
        old.rename(new)
        for key, val in list(new_map.items()):
            if val == old.name:
                new_map[key] = new.name
    total = sum((out / name).stat().st_size for name in set(new_map.values()))
    progress(
        f"dense: {len(out_keys)} tensor(s), {dequanted} fp8-block pair(s) dequanted "
        f"to bf16 ({per_group} per-group interleaved), {total / 1e9:.1f} GB "
        f"across {total_files} shard(s)"
    )
    return new_map, total


def _copy_pack(
    pack: Path, out: Path, progress: Callable[[str], None], cleanup: CleanupPolicy
) -> int:
    manifest = pack / EXL3_MANIFEST_FILENAME
    if not manifest.exists():
        raise SystemExit(f"pack missing {EXL3_MANIFEST_FILENAME}: {pack}")
    shutil.copy2(manifest, out / EXL3_MANIFEST_FILENAME)
    copied = 0
    for f in sorted(pack.glob(f"{LAYER_PREFIX}*.safetensors")):
        dest = out / f.name
        # Idempotent: skip an already-present identical layer.
        if not (dest.exists() and dest.stat().st_size == f.stat().st_size):
            shutil.copy2(f, dest)
        copied += 1
        progress(f"pack layer {copied}: {f.name}")
        if cleanup.delete_pack_after_copy:
            f.unlink(missing_ok=True)
    return copied


def _patch_config(
    source: Path, pack: Path, out: Path, kept_keys: list[str], dense_format: str
) -> None:
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


def run(
    source: Path,
    pack: Path,
    out: Path,
    *,
    dense_format: str | None = None,
    max_shard_bytes: int = 5_000_000_000,
    force: bool = False,
    cleanup: CleanupPolicy | None = None,
    progress: Callable[[str], None] = print,
) -> None:
    cleanup = cleanup or CleanupPolicy()
    if not source.is_dir():
        raise SystemExit(f"source dir not found: {source}")
    if not pack.is_dir():
        raise SystemExit(f"pack dir not found: {pack}")
    if out.exists():
        entries = [p for p in out.iterdir() if not p.name.startswith(".")]
        if entries and not force:
            raise SystemExit(f"output dir not empty: {out} (pass --force to overwrite)")
    out.mkdir(parents=True, exist_ok=True)

    wm = _weight_map(source)
    kept = [k for k in wm if EXPERT_MARKER not in k]
    fp8_bases = (
        {k[: -len(WEIGHT_SUFFIX)] for k in kept if k.endswith(WEIGHT_SUFFIX)}
        & {k[: -len(SCALE_INV_SUFFIX)] for k in kept if k.endswith(SCALE_INV_SUFFIX)}
    )
    src_cfg = _load_json(source / "config.json")
    src_qc = src_cfg.get("quantization_config", {}) or {}
    block_raw = src_qc.get("weight_block_size") or [128, 128]
    fp8_block = (int(block_raw[0]), int(block_raw[1]))
    dense_format = dense_format or ("bf16" if fp8_bases else "fp8")
    src_tc = src_cfg.get("text_config", src_cfg)
    ckpt_groups = int(src_tc.get("num_key_value_heads") or 1)

    progress(f"assemble: source={source} pack={pack} out={out}")
    new_map, _total = _write_shards(
        source, out, wm, kept, fp8_bases, fp8_block, ckpt_groups, max_shard_bytes, progress
    )
    (out / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": _total}, "weight_map": new_map}, indent=2) + "\n"
    )
    layers = _copy_pack(pack, out, progress, cleanup)
    progress(f"copied {layers} exl3 layer file(s) + {EXL3_MANIFEST_FILENAME}")
    _patch_config(source, pack, out, kept, dense_format)
    copied = _copy_aux(source, out)
    progress(f"copied auxiliary file(s)/dir(s): {', '.join(copied[:8])}")

    cfg = _load_json(out / "config.json")
    tc = cfg.get("text_config", cfg)
    geometry = _load_json(out / EXL3_MANIFEST_FILENAME).get("geometry", {})
    if geometry.get("hidden_size") and geometry["hidden_size"] != tc.get("hidden_size"):
        raise SystemExit(
            f"manifest hidden_size {geometry['hidden_size']} != source hidden "
            f"{tc.get('hidden_size')}; wrong pack for this source?"
        )
    progress("assemble: OK")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True, type=Path)
    ap.add_argument("--pack", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--dense-format", default=None)
    ap.add_argument("--max-shard-bytes", type=int, default=5_000_000_000)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--cleanup", choices=("none", "pack", "all"), default="none")
    args = ap.parse_args(argv)
    cleanup = CleanupPolicy(delete_pack_after_copy=args.cleanup in ("pack", "all"))
    try:
        run(args.source, args.pack, args.out, dense_format=args.dense_format,
            max_shard_bytes=args.max_shard_bytes, force=args.force, cleanup=cleanup)
    except SystemExit as e:
        print(str(e), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
