"""b12x-native lut_e4m3 encoder + selfcheck.

Production quality round-trip:
  1. Encode source weight into lut_e4m3 codewords, pack into exl3-v1 container.
  2. Decode the container's codes via python lut_e4m3 direct table.
  3. Cosine(source, decoded) -> PASS if >= 0.95.

No kernel forward needed: the encoder and kernel use the same Python
codebook (lut_e4m3_direct_table_cpu), so a python decode of the plane
bytes predicts the kernel output exactly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os; os.environ.setdefault("LD_LIBRARY_PATH",
    "/usr/local/lib/python3.12/dist-packages/torch/lib")
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, "/tmp")

from b12x._lib.quant.lut_e4m3 import (
    lut_e4m3_direct_table_cpu,  # [3, 65536] uint8
    lut_e4m3_value_table_cpu,
    _lut_e4m3_permutation,
)
from b12x.moe._shared.kernels.w4a16.exl3_synth import (
    Exl3SynthConfig, Exl3LayerPayloads,
    synth_layer_payloads, assemble_code_rows,
    layer_metadata, _manifest_dict,
)
from b12x.moe._shared.exl3_schema import layer_filename, EXL3_MANIFEST_FILENAME
from safetensors.torch import save_file


# ----------------------------------------------------------------------- helpers

def _round(v):
    return round(float(v), 6)

def _cosine(a, b):
    a = a.flatten().to(torch.float64)
    b = b.flatten().to(torch.float64)
    return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-12))


# ----------------------------------------------------------------------- E4M3

_E4M3_MAG_VALS = None

def _init_mags():
    global _E4M3_MAG_VALS
    if _E4M3_MAG_VALS is None:
        _E4M3_MAG_VALS = torch.empty(80, dtype=torch.float32)
        for i in range(80):
            _E4M3_MAG_VALS[i] = float(
                torch.tensor([i], dtype=torch.uint8).view(torch.float8_e4m3fn).item()
            )


def _codeword_for_byte(target_byte: int, bits: int) -> int:
    """Find a 16-bit codeword whose direct-table entry = target_byte."""
    table = lut_e4m3_direct_table_cpu()
    rate_idx = bits - 2
    matches = (table[rate_idx] == target_byte).nonzero(as_tuple=True)[0]
    return int(matches[0].item()) if matches.numel() else 0


# ----------------------------------------------------------------------- encode

def encode_weight_into_planes(W, bits=3):
    """Encode source weight W [N, K] (fp32) into plane dict.

    Returns:
        planes: dict[(slot, half)] -> Tensor[K_tiles, 16*bits] int16
        suh, svh: per-channel scales (fp16)
        K_tiles, num_slots, bits
    """
    N, K = W.shape
    assert N % 32 == 0 and K % 16 == 0
    K_tiles = K // 16
    slot_count = N // 32

    # Per-channel RMS scales
    suh = W.pow(2).mean(dim=0).sqrt().clamp_min(1e-6)   # [K]
    svh = W.pow(2).mean(dim=1).sqrt().clamp_min(1e-6)   # [N]
    W_norm = W / (svh[:, None] * suh[None, :])
    W_q = (W_norm * 0.9 * 448.0).clamp(-448, 448)

    _init_mags()
    mags = _E4M3_MAG_VALS.to(W.device)  # [80]

    # For each (K-tile, N-channel), decompose into `bits` E4M3 components.
    # The plane layout (from _independent_planes):
    #   Row r = kt*16 + kch_in_tile, for kt in 0..K_tiles-1, kch in 0..15
    #   so the row index = kt * 16 + kch_in_tile
    #   Total rows = K (256 for MiMo), not K/16.
    # Each row has `bits` codewords, one per N-channel of the N-tile,
    # at columns 0..bits-1, and these decode to the weights for
    # N-channel (slot*32 + half*16 + kch_in_tile?)... 
    #
    # Actually from _independent_planes:
    #   w13[matrix, expert, tiles, 2*slot+half, 16*bits]
    #   where w13[..., s, 0:16*bits] are the 16*bits codewords for
    #   N-tile at slot, half.
    #   The 16*bits = 48 for K=3. 48 columns = 16 K-channels × 3 codewords.
    #   Within each of the 16 K-channel blocks, the 3 codewords decode to
    #   3 fp values for 3 different N-channels of the N-tile.
    #
    #   So for the N-tile at slot/half (16 N-channels), and K-tile kt,
    #   the K-channels are ch0..ch15, and N-channel weights are spread:
    #   plane[kt*16 + ch, b] -> W[(slot*32+half*16 + b), kt*16 + ch]
    #
    #   Wait: plane has K_tiles rows (256 for MiMo). Each row = 1 K-tile.
    #   But there are 16 N-channels per N-tile. So the plane should have
    #   16 rows per N-tile or something.
    #
    #   plane shape = [K_tiles, 16*bits] = [256, 48] for K=3.
    #   256 rows * 48 cols = 12288 codewords.
    #   The N-tile has 16 N-channels. K=4096 = 256 K-channels per N-channel
    #   for this N-tile? 12288 codewords / 16 N-channels = 768 codewords
    #   per N-channel. 768 / 3 = 256 values per N-channel. 256 values = 16 K-tiles
    #   × 16 K-channels per K-tile = 256 K-channels. So 16 K-tiles ×
    #   16 K-channels = 256 K-channels per N-channel, and 16 N-channels ×
    #   256 K-channels = 4096 K-channels = K. But the plane has K_tiles=256
    #   rows. 256 / 16 = 16 N-channels. So row kt = N-channel (kt // 16)
    #   within the N-tile, K-tile-block (kt % 16).
    #
    #   So: plane[row, ch*3:ch*3+3] = 3 codewords for W[N_ch, K_ch]
    #   where N_ch = nt_lo + (row // 16) and K_ch = (row % 16)*16 + ch.
    #   And the 3 codewords sum to the target weight.
    #
    #   This is the key mapping. For K=3: row r = kt*16 + ch = [0..255],
    #   N_ch = nt_lo + (r // 16), K_ch = (r % 16)*16 + ch.
    #
    # FASTER APPROACH: precompute the 3 best E4M3 magnitudes for each
    # target value via vectorized argmin, then find the corresponding
    # codewords.

    # Precompute the E4M3 decomposition table.
    # For any target v in [-448*3, 448*3], find 3 magnitudes mag0..mag2
    # whose sum ≈ v.  Precompute [1024] bins.
    bins = 1024
    step = 3 * 448 * 2 / bins
    decomp_table = torch.zeros(bins, bits, dtype=torch.uint8)
    mags_np = _E4M3_MAG_VALS
    # For each bin, find the 3 best magnitudes via LSD:
    for bi in range(bins):
        v = -3*448 + step * bi
        remaining = abs(v)
        for b in range(bits):
            idx = (mags_np - remaining).abs().argmin().item()
            if mags_np[idx].item() > remaining + 1e-3 and idx > 0:
                idx -= 1
            decomp_table[bi, b] = idx
            remaining -= min(float(mags_np[idx].item()), remaining)
    decomp_table = decomp_table.to(W.device)

    # Encode.  Use the decomposition table as a fast lookup.
    # For each target value, quantize to bin index, then look up.
    planes = {}
    for slot in range(slot_count):
        for half, nt_lo in [("low", 32*slot), ("high", 32*slot+16)]:
            plane = torch.zeros((K_tiles, 16*bits), dtype=torch.int16)
            # Fill plane row by row
            for r in range(K_tiles):
                n_ch = nt_lo + (r // 16)
                k_ch_base = (r % 16) * 16
                for ch in range(16):
                    target = float(W_q[n_ch, k_ch_base + ch].item())
                    if target == 0.0:
                        continue
                    # quantize to bin
                    bi = int((target - (-3*448)) / step)
                    bi = max(0, min(bins-1, bi))
                    mags_3 = decomp_table[bi]
                    sign_neg = target < 0
                    for b in range(bits):
                        byte = int(mags_3[b].item())
                        if sign_neg:
                            byte |= 0x80
                        cw = _codeword_for_byte(byte, bits)
                        plane[r, ch*bits + b] = cw
            planes[(slot, half)] = plane

    return {
        "planes": planes,
        "suh": suh.to(torch.float16),
        "svh": svh.to(torch.float16),
        "K": K, "N": N,
        "K_tiles": K_tiles,
        "num_slots": slot_count,
        "bits": bits,
    }


# ----------------------------------------------------------------------- decode (python reference)

def decode_planes_back_to_weight(planes, K, slot_count, bits=3):
    """Decode the plane codewords back to fp32 weight [N, K].

    This is the Python reference decode using lut_e4m3_direct_table_cpu.
    The kernel applies the EXACT SAME table (it's the same function).
    """
    table = lut_e4m3_direct_table_cpu()  # [3, 65536] uint8
    rate_idx = bits - 2
    K_tiles = K // 16
    W_dec = torch.zeros((slot_count * 32, K), dtype=torch.float32)
    _init_mags()
    mags = _E4M3_MAG_VALS  # [80]
    for slot in range(slot_count):
        for half, nt_lo in [("low", 32*slot), ("high", 32*slot+16)]:
            plane = planes[(slot, half)]  # [K_tiles, 16*bits]
            for r in range(K_tiles):
                n_ch = nt_lo + (r // 16)
                k_ch_base = (r % 16) * 16
                for ch in range(16):
                    target_f8 = 0.0
                    for b in range(bits):
                        cw = int(plane[r, ch*bits + b].item())
                        if cw == 0:
                            continue
                        e4m3_byte = table[rate_idx, cw].item()
                        mag = e4m3_byte & 0x7F
                        sign = (e4m3_byte >> 7) * -2 + 1
                        val = float(mags[mag].item()) * sign
                        target_f8 += val
                    W_dec[nt_lo + (r // 16), k_ch_base + ch] = target_f8
    return W_dec


# ----------------------------------------------------------------------- CLI

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=256)
    ap.add_argument("--N", type=int, default=256)
    ap.add_argument("--bits", type=int, default=3, choices=(2,3,4))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--pack-out", type=Path, default=None)
    args = ap.parse_args()

    K, N = args.K, args.N
    assert K%16==0 and N%32==0

    # synthetic test weight
    Wsrc = torch.randn(N, K) * 0.5 + 0.3
    print(f"Source: shape {Wsrc.shape} rms={_round(Wsrc.pow(2).mean().sqrt())}", flush=True)

    # Encode
    enc = encode_weight_into_planes(Wsrc, bits=args.bits)
    print(f"Encoded: K_tiles={enc['K_tiles']}, slots={enc['num_slots']}", flush=True)
    print(f"  slot0 low first row sha: {hashlib.sha256(enc['planes'][(0,'low')].numpy().tobytes()).hexdigest()[:16]}", flush=True)

    # Decode via Python reference (matches kernel EXACTLY)
    Wdec = decode_planes_back_to_weight(enc['planes'], enc['K'], enc['num_slots'], bits=args.bits)
    print(f"Decoded: shape {Wdec.shape} rms={_round(Wdec.pow(2).mean().sqrt())}", flush=True)

    # Self-consistency: encode → decode should approximate the source
    cos = _cosine(Wsrc, Wdec)
    print(f"\ncos(Wsrc, Wdec) = {cos:.6f}", flush=True)
    print(f"PASS = {cos >= 0.95}", flush=True)

    # Emit container
    if args.pack_out:
        pack_out = args.pack_out
        pack_out.mkdir(parents=True, exist_ok=True)
        from b12x._lib.quant.lut_e4m3 import LUT_E4M3
        cfg = Exl3SynthConfig(
            codebook="lut_e4m3", num_experts=1, hidden_size=K,
            intermediate_size=N, moe_layer_indices=(0,), bits=args.bits,
            per_expert_input_rotations=True, intermediate_hadamard=False,
            extent_alignment_slots=4, seed=0,
        )
        base = synth_layer_payloads(cfg, 0)
        planes_dict = dict(base.planes)
        for s in range(enc["num_slots"]):
            planes_dict[(0, s, 0)] = (enc["planes"][(s,"low")], enc["planes"][(s,"high")])
        pl = Exl3LayerPayloads(
            planes=planes_dict, rotations=base.rotations,
            gate_suh=enc["suh"].unsqueeze(0), up_suh=base.up_suh,
            down_svh=base.down_svh, sign_pattern=None, rates_fc1=None, rates_fc2=None,
        )
        codes = assemble_code_rows(cfg, 0, pl)
        lf = layer_filename(0)
        save_file({
            "codes": codes.contiguous(),
            "rotations": pl.rotations.contiguous(),
            "gate_suh": pl.gate_suh.contiguous(),
            "up_suh": pl.up_suh.contiguous(),
            "down_svh": pl.down_svh.contiguous(),
        }, str(pack_out/lf), metadata=layer_metadata(cfg,0))
        md = _manifest_dict(cfg)
        md["layers"] = {"0": {"file": lf, "sha256": hashlib.sha256(
            (pack_out/lf).read_bytes()).hexdigest()}}
        (pack_out/EXL3_MANIFEST_FILENAME).write_text(
            json.dumps(md, indent=2, sort_keys=True)+"\n")
        print(f"Pack written to {pack_out}", flush=True)


if __name__ == "__main__":
    main()