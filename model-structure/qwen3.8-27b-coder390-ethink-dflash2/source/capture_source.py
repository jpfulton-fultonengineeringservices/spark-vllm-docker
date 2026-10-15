#!/usr/bin/env python3
"""Capture source-checkpoint structural analysis (pack-free).

Mirrors `mods/exl3-mimo/*-source/` and
`mods/mimo-v2.6-flash-rl-nvfp4/nvfp4-source/capture_nvfp4.py` capture convention:
geometry, summary, structure dump, weight map, per-tensor inventory, and a generic
drafter-subdir capture. Reads only safetensors *headers* (8-byte length + JSON), never
materializes tensors.

Generalized over `capture_nvfp4.py` in four ways, driven by config data rather than a
hard-coded model name:

1. `layer_of()` scans for any `layers.<N>` segment, so `model.layers.<N>`,
   `model.language_model.layers.<N>`, `mtp.layers.<N>` and drafter-side `layers.<N>` all
   resolve (mirrors `exl3pack.geometry._layer_index`).
2. Geometry reads the `text_config` sub-object when present (`_text_config`), which
   Qwen3.8/Qwen4 text-in-VLM checkpoints nest under the top level.
3. The expert->shard map counts expert *tensors* per shard instead of parsing a per-expert
   index, because fused experts are single stacked keys (no `experts.<N>.` segment). The
   map and its `structure.txt` section are omitted entirely for dense models.
4. Drafter subdirectories are discovered generically (any child dir holding a
   `*.safetensors`), replacing the hard-coded `dflash/dflash_draft_model.safetensors` path.

Usage: capture_source.py <checkpoint-dir> <out-dir>
"""
from __future__ import annotations

import gzip
import hashlib
import json
import pathlib
import shutil
import struct
import sys
from collections import Counter


def read_st_header(path: pathlib.Path) -> dict:
    with path.open("rb") as fh:
        (n,) = struct.unpack("<Q", fh.read(8))
        return json.loads(fh.read(n))


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def layer_of(name: str) -> int | None:
    """Return the index of the first `layers.<N>` segment in ``name``."""
    parts = name.split(".")
    for i, part in enumerate(parts):
        if part == "layers" and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                return None
    return None


def text_config(cfg: dict) -> dict:
    tc = cfg.get("text_config", cfg)
    return tc if isinstance(tc, dict) else cfg


def compact_quant(qc: dict | None) -> dict | None:
    """Summarize a ModelOpt quantization_config without the 36k-entry map."""
    if not isinstance(qc, dict):
        return qc
    out = {k: v for k, v in qc.items() if k != "quantized_layers"}
    ql = qc.get("quantized_layers")
    if isinstance(ql, dict):
        by_algo: Counter[str] = Counter()
        by_suffix: Counter[str] = Counter()
        for key, info in ql.items():
            algo = (info or {}).get("quant_algo", "?") if isinstance(info, dict) else "?"
            by_algo[algo] += 1
            by_suffix[key.rsplit(".", 1)[-1]] += 1
        out["quantized_layers_count"] = len(ql)
        out["quantized_layers_by_algo"] = dict(sorted(by_algo.items()))
        out["quantized_layers_by_module"] = dict(sorted(by_suffix.items(), key=lambda kv: -kv[1]))
        out["quantized_layers_sample"] = dict(list(ql.items())[:4])
    return out


# Subdirectories that are never drafter artifacts (Synology metadata, model cache).
SKIP_DIRS = {".cache", "@eaDir"}


def capture_drafters(src_p: pathlib.Path, out_p: pathlib.Path) -> list[str]:
    """Capture every immediate child dir holding a safetensors file as a drafter.

    Emits ``<out>/drafter/<dirname>/structure.json`` plus verbatim copies of the
    subdir's ``manifest.json`` / ``SHA256SUMS`` when present. Returns the captured
    subdir names (empty list for checkpoints that bundle no drafter).
    """
    captured: list[str] = []
    base = out_p / "drafter"
    for child in sorted(p for p in src_p.iterdir() if p.is_dir()):
        if child.name in SKIP_DIRS:
            continue
        shards = sorted(child.glob("*.safetensors"))
        if not shards:
            continue
        dest = base / child.name
        dest.mkdir(parents=True, exist_ok=True)
        capture: dict = {"source": str(child)}
        cc = child / "config.json"
        if cc.exists():
            raw = cc.read_text()
            try:
                capture["config"] = json.loads(raw)
            except json.JSONDecodeError:
                capture["config_raw"] = raw
        tensors: list[dict] = []
        for sh in shards:
            header = read_st_header(sh)
            tensors.extend(
                {"name": k, "dtype": v["dtype"], "shape": v["shape"]}
                for k, v in sorted(header.items())
                if k != "__metadata__"
            )
        capture["tensor_count"] = len(tensors)
        capture["tensors"] = tensors
        capture["files"] = [
            {"name": sh.name, "bytes": sh.stat().st_size, "sha256": sha256_file(sh)} for sh in shards
        ]
        (dest / "structure.json").write_text(json.dumps(capture, indent=2) + "\n")
        for extra in ("manifest.json", "SHA256SUMS"):
            src_extra = child / extra
            if src_extra.exists():
                shutil.copyfile(src_extra, dest / extra)
        captured.append(child.name)
    return captured


def main(src: str, out: str) -> None:
    src_p = pathlib.Path(src)
    out_p = pathlib.Path(out)
    out_p.mkdir(parents=True, exist_ok=True)

    cfg = json.loads((src_p / "config.json").read_text())
    idx = json.loads((src_p / "model.safetensors.index.json").read_text())
    wm: dict[str, str] = idx["weight_map"]
    meta = idx.get("metadata", {})
    tc = text_config(cfg)

    moe_layers: set[int] = set()
    for name in wm:
        if ".mlp.experts." in name:
            li = layer_of(name)
            if li is not None:
                moe_layers.add(li)
    all_layers = {li for name in wm if (li := layer_of(name)) is not None}
    dense_layers = sorted(all_layers - moe_layers)

    moe_intermediate = tc.get("moe_intermediate_size") or 0
    num_experts = tc.get("n_routed_experts", tc.get("num_experts", tc.get("num_local_experts", 0)))
    warnings: list[str] = []
    # Only meaningful for MoE checkpoints: dense models legitimately have 0 experts.
    if moe_layers and moe_intermediate and moe_intermediate % 32 != 0:
        warnings.append(f"moe_intermediate_size {moe_intermediate} not a multiple of 32")
    if not wm:
        warnings.append("empty weight_map")

    geometry: dict = {
        "source": str(src_p),
        "architecture": cfg.get("architectures"),
        "model_type": cfg.get("model_type"),
        "hidden_size": tc.get("hidden_size"),
        "intermediate_size": tc.get("intermediate_size"),
        "moe_intermediate_size": tc.get("moe_intermediate_size"),
        "num_experts": num_experts,
        "n_shared_experts": tc.get("n_shared_experts"),
        "num_experts_per_tok": tc.get("num_experts_per_tok"),
        "shared_expert_intermediate_size": tc.get("shared_expert_intermediate_size"),
        "num_slots": moe_intermediate // 32 if moe_intermediate else 0,
        "moe_layer_count": len(moe_layers),
        "dense_layers": dense_layers,
        "num_hidden_layers": tc.get("num_hidden_layers"),
        "num_nextn_predict_layers": tc.get("num_nextn_predict_layers"),
        "num_attention_heads": tc.get("num_attention_heads"),
        "num_key_value_heads": tc.get("num_key_value_heads"),
        "head_dim": tc.get("head_dim"),
        "v_head_dim": tc.get("v_head_dim"),
        "swa_num_key_value_heads": tc.get("swa_num_key_value_heads"),
        "sliding_window": tc.get("sliding_window"),
        "layer_types": tc.get("layer_types"),
        "full_attention_interval": tc.get("full_attention_interval"),
        "mtp_num_hidden_layers": tc.get("mtp_num_hidden_layers"),
        "vocab_size": tc.get("vocab_size"),
        "max_position_embeddings": tc.get("max_position_embeddings"),
        "hidden_act": tc.get("hidden_act"),
        "dtype": cfg.get("dtype"),
        "quantization_config": compact_quant(cfg.get("quantization_config")),
        "index_metadata": meta,
        "weight_map_tensor_count": len(wm),
        "weight_map_shard_count": len(set(wm.values())),
        "text_config": tc,
        "vision_config": cfg.get("vision_config"),
    }
    if warnings:
        geometry["_warnings"] = warnings

    # --- per-shard summary + per-tensor inventory ---
    shard_tensors: dict[str, list[str]] = {}
    for name, shard in wm.items():
        shard_tensors.setdefault(shard, []).append(name)

    inv_rows: list[dict] = []
    for shard in sorted(shard_tensors):
        sp = src_p / shard
        if not sp.exists():
            warnings.append(f"missing shard {shard}")
            continue
        header = read_st_header(sp)
        for name in sorted(shard_tensors[shard]):
            info = header.get(name)
            if not info:
                continue
            inv_rows.append(
                {"name": name, "dtype": info["dtype"], "shape": info["shape"], "shard": shard}
            )
    if warnings:
        geometry["_warnings"] = sorted(set(warnings))
    else:
        geometry.pop("_warnings", None)

    with gzip.open(out_p / "tensor_inventory.jsonl.gz", "wt") as fh:
        for row in inv_rows:
            fh.write(json.dumps(row) + "\n")

    summary: dict = {
        "source": str(src_p),
        "shard_count": len(shard_tensors),
        "total_tensors": len(wm),
        "index_metadata": meta,
        "shards": {
            shard: {"tensors": len(names), "bytes": (src_p / shard).stat().st_size}
            for shard, names in sorted(shard_tensors.items())
            if (src_p / shard).exists()
        },
        "dtype_counts": dict(Counter(r["dtype"] for r in inv_rows)),
    }

    # --- expert -> shard map (omitted for dense models) ---
    expert_map: dict[str, dict] = {}
    for shard, names in shard_tensors.items():
        ex_tensors = 0
        moe_ls: set[int] = set()
        for name in names:
            if ".mlp.experts." not in name:
                continue
            ex_tensors += 1
            li = layer_of(name)
            if li is not None:
                moe_ls.add(li)
        if ex_tensors:
            expert_map[shard] = {"expert_tensors": ex_tensors, "layers": len(moe_ls)}
    if expert_map:
        summary["expert_shard_map"] = {
            shard: info["expert_tensors"] for shard, info in sorted(expert_map.items())
        }
        summary["expert_tensor_total"] = sum(i["expert_tensors"] for i in expert_map.values())
        summary["expert_layer_total"] = len({li for li in moe_layers})

    (out_p / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    # --- key templates (numeric segments -> <N>) ---
    templates: Counter[str] = Counter()
    for name in wm:
        templates[".".join("<N>" if p.isdigit() else p for p in name.split("."))] += 1

    # --- structure.txt ---
    lines: list[str] = []
    lines.append(f"SOURCE: {src_p}")
    lines.append("")
    lines.append("=== GEOMETRY (capture_source.py detect-equivalent) ===")
    lines.append(json.dumps(geometry, indent=2))
    lines.append("")
    lines.append("=== SHARD SUMMARY ===")
    for shard, info in summary["shards"].items():
        lines.append(f"{shard:50s} tensors={info['tensors']:6d} bytes={info['bytes']}")
    lines.append("")
    if expert_map:
        lines.append("=== EXPERT SHARD MAP ===")
        for shard, info in sorted(expert_map.items()):
            lines.append(f"{shard}: expert_tensors={info['expert_tensors']} layers={info['layers']}")
        lines.append("")
    lines.append("=== KEY TEMPLATES ===")
    for t, c in sorted(templates.items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"{c:6d}  {t}")
    (out_p / "structure.txt").write_text("\n".join(lines) + "\n")

    (out_p / "geometry.json").write_text(json.dumps(geometry, indent=2) + "\n")

    with gzip.open(out_p / "weight_map.json.gz", "wt") as fh:
        fh.write(json.dumps(idx, indent=2))

    # --- drafter subdirs (generic) ---
    drafters = capture_drafters(src_p, out_p)
    if drafters:
        summary["drafter_subdirs"] = drafters
        (out_p / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(f"OK captured -> {out_p}")
    for p in sorted(out_p.rglob("*")):
        if p.is_file():
            print(f"  {p.relative_to(out_p)}: {p.stat().st_size} bytes")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
