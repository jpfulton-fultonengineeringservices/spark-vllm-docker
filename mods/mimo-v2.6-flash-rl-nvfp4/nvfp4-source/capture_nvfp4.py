#!/usr/bin/env python3
"""Capture a source-checkpoint structural analysis (pack-free).

Mirrors the `mods/exl3-mimo/*-source/` capture convention: geometry, summary,
structure dump, weight map, per-tensor inventory, DFlash structure. Reads only
safetensors *headers* (8-byte length + JSON) — never materializes tensors.

Usage: capture_nvfp4.py <checkpoint-dir> <out-dir>
"""
from __future__ import annotations

import gzip
import json
import pathlib
import struct
import sys
from collections import Counter, defaultdict


def read_st_header(path: pathlib.Path) -> dict:
    with path.open("rb") as fh:
        (n,) = struct.unpack("<Q", fh.read(8))
        return json.loads(fh.read(n))


def layer_of(name: str) -> int | None:
    parts = name.split(".")
    if len(parts) >= 3 and parts[0] == "model" and parts[1] == "layers":
        return int(parts[2]) if parts[2].isdigit() else None
    return None


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


def main(src: str, out: str) -> None:
    src_p = pathlib.Path(src)
    out_p = pathlib.Path(out)
    out_p.mkdir(parents=True, exist_ok=True)

    cfg = json.loads((src_p / "config.json").read_text())
    idx = json.loads((src_p / "model.safetensors.index.json").read_text())
    wm: dict[str, str] = idx["weight_map"]
    meta = idx.get("metadata", {})

    moe_layers: set[int] = set()
    for name in wm:
        if ".mlp.experts." in name:
            li = layer_of(name)
            if li is not None:
                moe_layers.add(li)
    all_layers = {li for name in wm if (li := layer_of(name)) is not None}
    dense_layers = sorted(all_layers - moe_layers)

    geometry = {
        "source": str(src_p),
        "architecture": cfg.get("architectures"),
        "hidden_size": cfg.get("hidden_size"),
        "intermediate_size": cfg.get("intermediate_size"),
        "moe_intermediate_size": cfg.get("moe_intermediate_size"),
        "num_experts": cfg.get("n_routed_experts"),
        "num_slots": (cfg.get("moe_intermediate_size") or 0) // 32,
        "moe_layer_count": len(moe_layers),
        "moe_layers": sorted(moe_layers),
        "dense_layers": dense_layers,
        "num_hidden_layers": cfg.get("num_hidden_layers"),
        "num_nextn_predict_layers": cfg.get("num_nextn_predict_layers"),
        "n_shared_experts": cfg.get("n_shared_experts"),
        "num_experts_per_tok": cfg.get("num_experts_per_tok"),
        "num_attention_heads": cfg.get("num_attention_heads"),
        "num_key_value_heads": cfg.get("num_key_value_heads"),
        "head_dim": cfg.get("head_dim"),
        "v_head_dim": cfg.get("v_head_dim"),
        "swa_num_key_value_heads": cfg.get("swa_num_key_value_heads"),
        "sliding_window": cfg.get("sliding_window"),
        "vocab_size": cfg.get("vocab_size"),
        "max_position_embeddings": cfg.get("max_position_embeddings"),
        "hidden_act": cfg.get("hidden_act"),
        "dtype": cfg.get("dtype"),
        "quantization_config": compact_quant(cfg.get("quantization_config")),
        "index_metadata": meta,
        "weight_map_tensor_count": len(wm),
        "weight_map_shard_count": len(set(wm.values())),
    }

    # --- per-shard summary + per-tensor inventory ---
    shard_tensors: dict[str, list[str]] = defaultdict(list)
    for name, shard in wm.items():
        shard_tensors[shard].append(name)

    inv_rows: list[dict] = []
    for shard in sorted(shard_tensors):
        sp = src_p / shard
        if not sp.exists():
            geometry.setdefault("_warnings", []).append(f"missing shard {shard}")
            continue
        header = read_st_header(sp)
        for name in sorted(shard_tensors[shard]):
            info = header.get(name)
            if not info:
                continue
            inv_rows.append(
                {"name": name, "dtype": info["dtype"], "shape": info["shape"], "shard": shard}
            )

    with gzip.open(out_p / "tensor_inventory.jsonl.gz", "wt") as fh:
        for row in inv_rows:
            fh.write(json.dumps(row) + "\n")

    summary = {
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
    (out_p / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    # --- expert -> shard map ---
    expert_map: dict[str, dict] = {}
    for shard, names in shard_tensors.items():
        exps: set[int] = set()
        moe_ls: set[int] = set()
        for name in names:
            if ".mlp.experts." not in name:
                continue
            parts = name.split(".")
            try:
                ei = parts.index("experts")
                li = parts.index("layers")
                exps.add(int(parts[ei + 1]))
                moe_ls.add(int(parts[li + 1]))
            except (ValueError, IndexError):
                continue
        if exps:
            expert_map[shard] = {
                "experts": len(exps),
                "range": [min(exps), max(exps)],
                "layers": len(moe_ls),
            }

    # --- key templates (numeric segments -> <N>) ---
    templates: Counter[str] = Counter()
    for name in wm:
        templates[".".join("<N>" if p.isdigit() else p for p in name.split("."))] += 1

    # --- structure.txt ---
    lines: list[str] = []
    lines.append(f"SOURCE: {src_p}")
    lines.append("")
    lines.append("=== GEOMETRY (pack-model-info.py detect-equivalent) ===")
    lines.append(json.dumps(geometry, indent=2))
    lines.append("")
    lines.append("=== SHARD SUMMARY ===")
    for shard, info in summary["shards"].items():
        lines.append(f"{shard:50s} tensors={info['tensors']:6d} bytes={info['bytes']}")
    lines.append("")
    lines.append("=== EXPERT SHARD MAP ===")
    for shard, info in sorted(expert_map.items()):
        lines.append(
            f"{shard}: experts={info['experts']} range={info['range'][0]}..{info['range'][1]} "
            f"layers={info['layers']}"
        )
    lines.append("")
    lines.append("=== KEY TEMPLATES ===")
    for t, c in sorted(templates.items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"{c:6d}  {t}")
    (out_p / "structure.txt").write_text("\n".join(lines) + "\n")

    (out_p / "geometry.json").write_text(json.dumps(geometry, indent=2) + "\n")

    with gzip.open(out_p / "weight_map.json.gz", "wt") as fh:
        fh.write(json.dumps(idx, indent=2))

    # --- DFlash structure ---
    dflash = src_p / "dflash"
    if dflash.exists():
        (out_p / "dflash").mkdir(exist_ok=True)
        capture: dict = {}
        dc = dflash / "config.json"
        if dc.exists():
            raw = dc.read_text()
            try:
                capture["config"] = json.loads(raw)
            except json.JSONDecodeError:
                capture["config_raw"] = raw
        dr = dflash / "dflash_structure.json"
        if dr.exists():
            capture["structure"] = json.loads(dr.read_text())
        else:
            st = dflash / "dflash_draft_model.safetensors"
            if st.exists():
                h = read_st_header(st)
                capture["tensors"] = sorted(
                    (
                        {"name": k, "dtype": v["dtype"], "shape": v["shape"]}
                        for k, v in h.items()
                        if k != "__metadata__"
                    ),
                    key=lambda r: r["name"],
                )
        (out_p / "dflash" / "dflash_structure.json").write_text(
            json.dumps(capture, indent=2) + "\n"
        )

    # --- vllm_patch diff (verbatim) ---
    patch = src_p / "vllm_patch" / "mimo_v2.diff"
    if patch.exists():
        (out_p / "vllm_patch").mkdir(exist_ok=True)
        (out_p / "vllm_patch" / "mimo_v2.diff").write_bytes(patch.read_bytes())

    print(f"OK captured -> {out_p}")
    for p in sorted(out_p.rglob("*")):
        if p.is_file():
            print(f"  {p.relative_to(out_p)}: {p.stat().st_size} bytes")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
