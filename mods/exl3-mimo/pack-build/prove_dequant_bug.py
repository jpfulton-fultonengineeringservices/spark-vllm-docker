#!/usr/bin/env python3
# prove_dequant_bug.py — prove the root cause of EXL3 MiMo garbage serving.
#
# HYPOTHESIS: pack-assemble._dequant_fp8_block dequantizes the interleaved fused
# qkv_proj with ONE FLAT 128x128 scale grid, but the fork's
# _shard_fp8_qkv_proj dequantizes each interleaved group with its OWN scale rows:
#
#     scale_rows_per_group = ceil(rows_per_group / block)
#     assert s_full.shape[0] == scale_rows_per_group * checkpoint_tp_size
#
# The two agree only when rows_per_group is an exact multiple of block:
#     kv=8 (39 SWA layers): 3072+384+256 = 3712 = 29*128  -> aligned, correct
#     kv=4 ( 9 full-attn):  3072+192+128 = 3392 = 26.5*128 -> MISALIGNED, wrong
#                           scale_rows_per_group = 27, and 27*4 = 108 == grid rows
# Layer 0 is kv=4, and the A/B per-layer residual dump shows layer 0 diverging
# 5.41x (control 2.6070 vs EXL3 14.1083) while every dense tensor is otherwise
# byte-identical. This script reports, per layer, cos(flat, per_group) and the
# cos of each against the BF16 tensor actually stored in the assembled pack.
import json

import torch
from safetensors import safe_open

S = "/nas-1/models/mimo/mimo-v2.6-flash-rl"
C = "/nas-1/models/mimo/mimo-v2.6-flash-rl-exl3-v1"
si = json.load(open(f"{S}/model.safetensors.index.json"))["weight_map"]
ci = json.load(open(f"{C}/model.safetensors.index.json"))["weight_map"]
BLOCK = 128


def rd(root, idx, k):
    with safe_open(f"{root}/{idx[k]}", framework="pt") as h:
        return h.get_tensor(k)


def cos(a, b):
    a = a.double().flatten()
    b = b.double().flatten()
    return float((a @ b) / (a.norm() * b.norm() + 1e-30))


def flat_dequant(w, sc):
    """Exactly pack-assemble._dequant_fp8_block: expand the scale grid by the
    config block size then slice to the real weight shape. The grid is PADDED
    (108 rows for kv=4 where only 106 blocks are needed), so this treats the
    tensor as one flat run of 128-row blocks and ignores group boundaries."""
    m, n = w.shape
    s = sc.float().repeat_interleave(BLOCK, 0).repeat_interleave(BLOCK, 1)
    return (w.float() * s[:m, :n]).to(torch.bfloat16)


def per_group_dequant(w, sc, rows_per_group):
    """The fork's convention: each interleaved group owns ceil(rpg/128) scale
    rows, dequantized independently (padding absorbed per group)."""
    m, n = w.shape
    srg = (rows_per_group + BLOCK - 1) // BLOCK
    n_groups = m // rows_per_group
    assert sc.shape[0] == srg * n_groups, (sc.shape[0], srg, n_groups)
    out = torch.empty((m, n), dtype=torch.float32)
    for g in range(n_groups):
        rs, ss = g * rows_per_group, g * srg
        wg = w[rs:rs + rows_per_group].float()
        sg = sc[ss:ss + srg].float()
        sg = sg.repeat_interleave(BLOCK, 0).repeat_interleave(BLOCK, 1)
        out[rs:rs + rows_per_group] = wg * sg[:rows_per_group, :n]
    return out.to(torch.bfloat16)


cfg = json.load(open(f"{C}/config.json"))
KV, NH = int(cfg["num_key_value_heads"]), int(cfg["num_attention_heads"])
HD, VHD = int(cfg["head_dim"]), int(cfg["v_head_dim"])

print(f"config: NH={NH} HD={HD} VHD={VHD} ckpt_tp={KV}")
print(f"{'layer':>6} {'kv':>3} {'rpg':>6} {'rpg/128':>9} {'grid':>5} "
      f"{'cos(flat,pergrp)':>17} {'cos(pack,flat)':>15} {'cos(pack,pergrp)':>17}")
n_bad = 0
for L in range(48):
    base = f"model.layers.{L}.self_attn.qkv_proj"
    if f"{base}.weight" not in si or f"{base}.weight" not in ci:
        continue
    w = rd(S, si, f"{base}.weight")
    sc = rd(S, si, f"{base}.weight_scale_inv")
    pk = rd(C, ci, f"{base}.weight")
    total_kv = 4 if pk.shape[0] == 13568 else 8
    rpg = (NH // KV) * HD + (total_kv // KV) * HD + (total_kv // KV) * VHD
    fl = flat_dequant(w, sc)
    pg = per_group_dequant(w, sc, rpg)
    cf_pg, c_pk_fl, c_pk_pg = cos(fl, pg), cos(pk, fl), cos(pk, pg)
    if cf_pg < 0.999:
        n_bad += 1
    print(f"{L:>6} {total_kv:>3} {rpg:>6} {rpg/128:>9.3f} {sc.shape[0]:>5} "
          f"{cf_pg:>17.6f} {c_pk_fl:>15.6f} {c_pk_pg:>17.6f}")
print(f"\nlayers where flat != per-group (cos<0.999): {n_bad}")
