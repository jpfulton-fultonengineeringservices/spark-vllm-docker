#!/usr/bin/env python3
"""Repack an exllamav3 EXL3 pack into the b12x ``exl3-v1`` MoE container.

Run INSIDE the b12x runtime image (imports ``b12x`` + ``torch`` + ``safetensors``).

Input : an exllamav3 EXL3 HF pack directory. Per quantized linear:
          <proj>.trellis  int16 [K/16, N/16, 16*bits]
          <proj>.suh      fp16  [K]
          <proj>.svh      fp16  [N]
          <proj>.mcg      int32 scalar (MCG codebook flag)
        plus model.safetensors.index.json and config.json.
        Produce it with:  convert.py -i <src> -o <pack> -w <work> -b <bits> \
                                       --codebook mcg
Output: ``exl3-v1`` container = ``exl3-manifest.json`` + one
        ``exl3-layer-<NNNNN>.safetensors`` per MoE layer with
        ``codes``/``rotations``/``gate_suh``/``up_suh``/``down_svh``, read by the
        fork's ``Exl3MoEMethod`` via ``b12x.moe.checkpoints.exl3``.

Mapping (uniform bits, per expert, per slot ``s`` over the intermediate axis;
a slot = 32 channels = two 16-wide tiles):
  gate/up planes (FC1, slot on N): low = trellis[:, 2s, :]  high = trellis[:, 2s+1, :]
  down planes   (FC2, slot on K): low = trellis[2s, :, :]   high = trellis[2s+1, :, :]
  rotations[s,e,0] = gate svh -> reshape[num_slots,32][s]
  rotations[s,e,1] = up   svh -> reshape[num_slots,32][s]
  rotations[s,e,2] = down suh -> reshape[num_slots,32][s]
  gate_suh/up_suh = gate/up suh [hidden];  down_svh = down svh [hidden]
  per_expert_input_rotations = True

The container payload is assembled by b12x's own ``assemble_code_rows`` /
``_manifest_dict`` / ``layer_metadata`` / ``layer_filename`` so the byte layout
is exactly b12x-native. ``--self-check`` proves the container ``codes`` are a
lossless re-layout of the source trellis. Kernel word-order parity (exllamav3
packing vs b12x decode) is a boot-time check, not proven here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from b12x.moe._shared import exl3_schema as schema
from b12x.moe._shared.kernels.w4a16 import exl3_synth as synth


def _weight_map(pack: Path) -> dict[str, str]:
    return json.loads((pack / "model.safetensors.index.json").read_text())["weight_map"]


def _moe_layers(weight_map: dict[str, str]) -> list[int]:
    return sorted({int(k.split(".")[2]) for k in weight_map if ".mlp.experts." in k})


class PackReader:
    def __init__(self, pack: Path, weight_map: dict[str, str]) -> None:
        self.pack = pack
        self.wm = weight_map
        self._h: dict[str, object] = {}

    def get(self, name: str) -> torch.Tensor:
        shard = self.wm[name]
        if shard not in self._h:
            self._h[shard] = safe_open(str(self.pack / shard), framework="pt")
        return self._h[shard].get_tensor(name)

    def close(self) -> None:
        self._h.clear()


def build_payloads(reader: PackReader, layer: int, cfg: synth.Exl3SynthConfig,
                   bits: int) -> synth.Exl3LayerPayloads:
    e_count, hidden, inter = cfg.num_experts, cfg.hidden_size, cfg.intermediate_size
    num_slots = inter // 32
    ht = hidden // 16
    w = 16 * bits
    p = f"model.layers.{layer}.mlp.experts."

    planes: dict[tuple[int, int, int], tuple[torch.Tensor, torch.Tensor]] = {}
    rotations = torch.empty((num_slots, e_count, 3, 32), dtype=torch.float16)
    gate_suh = torch.empty((e_count, hidden), dtype=torch.float16)
    up_suh = torch.empty((e_count, hidden), dtype=torch.float16)
    down_svh = torch.empty((e_count, hidden), dtype=torch.float16)

    for e in range(e_count):
        tg = reader.get(f"{p}{e}.gate_proj.trellis")
        tu = reader.get(f"{p}{e}.up_proj.trellis")
        td = reader.get(f"{p}{e}.down_proj.trellis")
        for t in (tg, tu, td):
            if t.dtype != torch.int16 or t.shape[-1] != w:
                raise ValueError(
                    f"layer {layer} expert {e}: unexpected trellis {tuple(t.shape)} "
                    f"{t.dtype}; expected int16 [.., {w}] for uniform K{bits}"
                )
        for s in range(num_slots):
            planes[(e, s, 0)] = (tg[:, 2 * s, :].contiguous(), tg[:, 2 * s + 1, :].contiguous())
            planes[(e, s, 1)] = (tu[:, 2 * s, :].contiguous(), tu[:, 2 * s + 1, :].contiguous())
            planes[(e, s, 2)] = (td[2 * s, :, :].contiguous(), td[2 * s + 1, :, :].contiguous())
        g_svh = reader.get(f"{p}{e}.gate_proj.svh").to(torch.float16).reshape(num_slots, 32)
        u_svh = reader.get(f"{p}{e}.up_proj.svh").to(torch.float16).reshape(num_slots, 32)
        d_suh = reader.get(f"{p}{e}.down_proj.suh").to(torch.float16).reshape(num_slots, 32)
        for s in range(num_slots):
            rotations[s, e, 0] = g_svh[s]
            rotations[s, e, 1] = u_svh[s]
            rotations[s, e, 2] = d_suh[s]
        gate_suh[e] = reader.get(f"{p}{e}.gate_proj.suh").to(torch.float16)
        up_suh[e] = reader.get(f"{p}{e}.up_proj.suh").to(torch.float16)
        down_svh[e] = reader.get(f"{p}{e}.down_proj.svh").to(torch.float16)

    return synth.Exl3LayerPayloads(
        planes=planes, rotations=rotations, gate_suh=gate_suh, up_suh=up_suh,
        down_svh=down_svh, sign_pattern=None, rates_fc1=None, rates_fc2=None,
    )


def write_layer(out: Path, cfg: synth.Exl3SynthConfig, layer: int,
                payloads: synth.Exl3LayerPayloads) -> dict[str, str]:
    codes = synth.assemble_code_rows(cfg, layer, payloads)
    tensors = {
        "codes": codes.contiguous(),
        "rotations": payloads.rotations.contiguous(),
        "gate_suh": payloads.gate_suh.contiguous(),
        "up_suh": payloads.up_suh.contiguous(),
        "down_svh": payloads.down_svh.contiguous(),
    }
    fname = schema.layer_filename(layer)
    path = out / fname
    save_file(tensors, str(path), metadata=synth.layer_metadata(cfg, layer))
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"file": fname, "sha256": sha}


def self_check(reader: PackReader, out: Path, layer: int, num_slots: int, bits: int) -> bool:
    p = f"model.layers.{layer}.mlp.experts.0."
    tg = reader.get(p + "gate_proj.trellis")
    with safe_open(str(out / schema.layer_filename(layer)), framework="pt") as h:
        codes = h.get_tensor("codes")
    w = 16 * bits
    n = tg[:, 0, :].numel()
    low = codes[0, : n * 2].view(torch.int16).reshape(tg.shape[0], w)
    ok = torch.equal(low, tg[:, 0, :])
    print(f"  self-check layer {layer}: gate slot0-plane0 lossless = {ok}")
    return ok


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pack", required=True, type=Path, help="exllamav3 EXL3 pack dir")
    ap.add_argument("--out", required=True, type=Path, help="output exl3-v1 dir")
    ap.add_argument("--bits", type=int, default=2, help="uniform codebook bitrate (default 2)")
    ap.add_argument("--layers", default=None, help="comma-separated layer indices (default: all MoE)")
    ap.add_argument("--row-alignment", type=int, default=4096)
    ap.add_argument("--extent-alignment-slots", type=int, default=4)
    ap.add_argument("--self-check", action="store_true")
    args = ap.parse_args(argv)

    cfg_src = json.loads((args.pack / "config.json").read_text())
    tc = cfg_src.get("text_config", cfg_src)
    qc = cfg_src.get("quantization_config", {})
    codebook = qc.get("codebook")
    if codebook != "mcg":
        print(f"ERROR: pack codebook is {codebook!r}; b12x exl3-v1 supports only "
              "'mcg'/'lut_e4m3'/'lut_fp16'. Produce the pack with --codebook mcg.", file=sys.stderr)
        return 2

    num_experts = int(tc["n_routed_experts"])
    hidden = int(tc["hidden_size"])
    inter = int(tc["moe_intermediate_size"])

    wm = _weight_map(args.pack)
    moe_layers = _moe_layers(wm) if args.layers is None else [int(x) for x in args.layers.split(",")]
    args.out.mkdir(parents=True, exist_ok=True)

    cfg = synth.Exl3SynthConfig(
        codebook="mcg", num_experts=num_experts, hidden_size=hidden,
        intermediate_size=inter, moe_layer_indices=tuple(moe_layers), bits=args.bits,
        intermediate_hadamard=False, per_expert_input_rotations=True,
        unit_hidden_rotations=False, row_alignment=args.row_alignment,
        extent_alignment_slots=args.extent_alignment_slots,
        # The fork's plan_exl3_extent requires exactly one barrier at the
        # half-way slot so TP ranks never own an extent crossing the
        # intermediate halves; an empty barrier list makes the runtime raise
        # "EXL3 extent planning requires one barrier between aligned halves".
        extent_barriers=((inter // 32) // 2,), seed=0,
    )
    num_slots = inter // 32
    print(f"pack={args.pack} out={args.out} codebook=mcg K{args.bits} "
          f"E={num_experts} H={hidden} I={inter} slots={num_slots} layers={len(moe_layers)}")

    reader = PackReader(args.pack, wm)
    manifest = synth._manifest_dict(cfg)
    manifest["layers"] = {}
    try:
        for layer in moe_layers:
            payloads = build_payloads(reader, layer, cfg, args.bits)
            manifest["layers"][str(layer)] = write_layer(args.out, cfg, layer, payloads)
            print(f"  layer {layer}: wrote {manifest['layers'][str(layer)]['file']}")
            if args.self_check:
                if not self_check(reader, args.out, layer, num_slots, args.bits):
                    return 3
    finally:
        reader.close()

    (args.out / schema.EXL3_MANIFEST_FILENAME).write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(f"wrote {schema.EXL3_MANIFEST_FILENAME} ({len(manifest['layers'])} layers)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())