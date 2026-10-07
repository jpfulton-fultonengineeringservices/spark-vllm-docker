"""Repack an exllamav3 EXL3 pack into a b12x ``exl3-v1`` MoE container.

Torch-dependent (runs INSIDE the b12x image; imports ``b12x``). Faithful port of
the original ``repack_exl3_to_b12x.py`` — the trellis re-layout word-order
mapping is lossless and must not change:

Mapping (uniform bits, per expert, per slot ``s`` over the intermediate axis;
each slot is 32 channels = two 16-wide tiles):
  gate/up planes (FC1, slot on N): low trellis[:, 2s, :] high trellis[:, 2s+1, :]
  down planes (FC2, slot on K):    low trellis[2s, :, :] high trellis[2s+1, :, :]

Layers are written to ``--out`` (on the NAS) one at a time; nothing large is
buffered across layers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import cast

import torch
from b12x.moe._shared import exl3_schema as schema
from b12x.moe._shared.kernels.w4a16 import exl3_synth as synth
from safetensors import safe_open
from safetensors.torch import save_file

MOE_EXPERT_MARKER = ".mlp.experts."


def _weight_map(pack: Path) -> dict[str, str]:
    return json.loads((pack / "model.safetensors.index.json").read_text())["weight_map"]  # type: ignore[no-any-return]


def _moe_layers(weight_map: dict[str, str]) -> list[int]:
    return sorted({int(k.split(".")[2]) for k in weight_map if MOE_EXPERT_MARKER in k})


def _manifest_layers(
    manifest: dict[str, object], cfg: synth.Exl3SynthConfig
) -> dict[str, dict[str, str]]:
    layers = manifest.get("layers")
    if not isinstance(layers, dict):
        layers = {}
        manifest["layers"] = layers
    return layers


class PackReader:
    def __init__(self, pack: Path, weight_map: dict[str, str]) -> None:
        self.pack = pack
        self.wm = weight_map
        self._h: dict[str, object] = {}

    def get(self, name: str) -> torch.Tensor:
        shard = self.wm[name]
        if shard not in self._h:
            self._h[shard] = safe_open(str(self.pack / shard), framework="pt")
        handle = self._h[shard]
        return cast(torch.Tensor, handle.get_tensor(name))  # type: ignore[attr-defined]

    def close(self) -> None:
        self._h.clear()


def build_payloads(
    reader: PackReader, layer: int, cfg: synth.Exl3SynthConfig, bits: int
) -> synth.Exl3LayerPayloads:
    e_count, hidden, inter = cfg.num_experts, cfg.hidden_size, cfg.intermediate_size
    num_slots = inter // 32
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
                    f"{t.dtype}; expected int16 [.., {w}]"
                )
        for s in range(num_slots):
            planes[(e, s, 0)] = (
                tg[:, 2 * s, :].contiguous(),
                tg[:, 2 * s + 1, :].contiguous(),
            )
            planes[(e, s, 1)] = (
                tu[:, 2 * s, :].contiguous(),
                tu[:, 2 * s + 1, :].contiguous(),
            )
            planes[(e, s, 2)] = (
                td[2 * s, :, :].contiguous(),
                td[2 * s + 1, :, :].contiguous(),
            )
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
        planes=planes,
        rotations=rotations,
        gate_suh=gate_suh,
        up_suh=up_suh,
        down_svh=down_svh,
        sign_pattern=None,
        rates_fc1=None,
        rates_fc2=None,
    )


def write_layer(
    out: Path, cfg: synth.Exl3SynthConfig, layer: int, payloads: synth.Exl3LayerPayloads
) -> dict[str, str]:
    codes = synth.assemble_code_rows(cfg, layer, payloads)
    tensors = {
        "codes": codes.contiguous(),
        "rotations": payloads.rotations.contiguous(),
        "gate_suh": payloads.gate_suh.contiguous(),
        "up_suh": payloads.up_suh.contiguous(),
        "down_svh": payloads.down_svh.contiguous(),
    }
    fname = schema.layer_filename(layer)
    save_file(tensors, str(out / fname), metadata=synth.layer_metadata(cfg, layer))
    sha = hashlib.sha256((out / fname).read_bytes()).hexdigest()
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
    print(f"  self-check layer {layer}: gate slot0-plane0 lossless {ok}")
    return bool(ok)


def run(
    pack: Path,
    out: Path,
    *,
    bits: int = 2,
    layers: list[int] | None = None,
    row_alignment: int = 4096,
    extent_alignment_slots: int = 4,
    self_check_on: bool = False,
    progress: Callable[[str], None] = print,
) -> dict[str, object]:
    cfg_src = json.loads((pack / "config.json").read_text())
    tc = cfg_src.get("text_config", cfg_src)
    qc = cfg_src.get("quantization_config", {})
    codebook = qc.get("codebook")
    if codebook not in ("mcg", "lut_e4m3", "lut_fp16"):
        raise SystemExit(
            f"ERROR: exl3-v1 only accepts mcg/lut_e4m3/lut_fp16; pack codebook={codebook!r}. "
            "Convert with --codebook mcg."
        )

    weight_map = _weight_map(pack)
    moe_layers = layers if layers is not None else _moe_layers(weight_map)
    out.mkdir(parents=True, exist_ok=True)

    num_experts = int(tc["n_routed_experts"])
    hidden = int(tc["hidden_size"])
    inter = int(tc["moe_intermediate_size"])
    cfg = synth.Exl3SynthConfig(
        codebook="mcg",
        num_experts=num_experts,
        hidden_size=hidden,
        intermediate_size=inter,
        moe_layer_indices=tuple(moe_layers),
        bits=bits,
        intermediate_hadamard=False,
        per_expert_input_rotations=True,
        unit_hidden_rotations=False,
        row_alignment=row_alignment,
        extent_alignment_slots=extent_alignment_slots,
        # One barrier at the half-way slot so TP ranks never own an extent
        # crossing the intermediate halves (fork's plan_exl3_extent requires it).
        extent_barriers=((inter // 32) // 2,),
        seed=0,
    )
    num_slots = cfg.intermediate_size // 32
    manifest: dict[str, object] = dict(synth._manifest_dict(cfg))
    manifest_layers = _manifest_layers(manifest, cfg)

    reader = PackReader(pack, weight_map)
    try:
        for layer in moe_layers:
            payloads = build_payloads(reader, layer, cfg, bits)
            manifest_layers[str(layer)] = write_layer(out, cfg, layer, payloads)
            progress(f" layer {layer}: wrote {manifest_layers[str(layer)]['file']}")
            if self_check_on and not self_check(reader, out, layer, num_slots, bits):
                raise SystemExit(f"self-check failed at layer {layer}")
    finally:
        reader.close()

    (out / schema.EXL3_MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2) + "\n")
    progress(f"wrote {schema.EXL3_MANIFEST_FILENAME} ({len(manifest_layers)} layers)")
    return manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pack", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--bits", type=int, default=2)
    ap.add_argument("--layers", default=None)
    ap.add_argument("--row-alignment", type=int, default=4096)
    ap.add_argument("--extent-alignment-slots", type=int, default=4)
    ap.add_argument("--self-check", action="store_true")
    ap.add_argument("--cleanup", choices=("none", "pack", "all"), default="none")
    args = ap.parse_args(argv)
    layers = [int(x) for x in args.layers.split(",")] if args.layers else None
    try:
        run(args.pack, args.out, bits=args.bits, layers=layers,
            row_alignment=args.row_alignment,
            extent_alignment_slots=args.extent_alignment_slots,
            self_check_on=args.self_check)
    except SystemExit as e:
        print(str(e), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
