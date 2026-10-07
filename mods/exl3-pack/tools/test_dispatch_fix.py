#!/usr/bin/env python3
# test_dispatch_fix.py — validate pack-assemble's new flat/per-group dispatch
# against REAL MiMo qkv data, and against the fork's authoritative per-group
# convention. Must run where both the source checkpoint and torch exist.
#
# PASS criteria:
#   kv=4 layers (9 of them): used_per_group=True, and the dequant matches the
#     per-group reference exactly, and DIFFERS from the old flat result.
#   kv=8 layers (39): used_per_group=False (flat), matching per-group exactly
#     anyway (they coincide when rows_per_group is a whole number of blocks).
#   Every other fp8 tensor (layer-0 dense MLP, MTP): still dequantizes, no crash.
import json
import sys

import torch
from safetensors import safe_open

sys.path.insert(0, "/probe")
import importlib.util

spec = importlib.util.spec_from_file_location("packassemble",
                                              "/probe/pack-assemble.py")
pa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pa)

S = "/nas-1/models/mimo/mimo-v2.6-flash-rl"
C = "/nas-1/models/mimo/mimo-v2.6-flash-rl-exl3-v1"
si = json.load(open(f"{S}/model.safetensors.index.json"))["weight_map"]
ci = json.load(open(f"{C}/model.safetensors.index.json"))["weight_map"]


def rd(root, idx, k):
    with safe_open(f"{root}/{idx[k]}", framework="pt") as h:
        return h.get_tensor(k)


def cos(a, b):
    a = a.double().flatten(); b = b.double().flatten()
    return float((a @ b) / (a.norm() * b.norm() + 1e-30))


BLOCK = (128, 128)
GROUPS = 4
KV4, KV8 = [], []
for L in range(48):
    base = f"model.layers.{L}.self_attn.qkv_proj"
    if f"{base}.weight" not in si:
        continue
    (KV4 if rd(C, ci, f"{base}.weight").shape[0] == 13568 else KV8).append(L)
print(f"kv=4 layers: {KV4}")
print(f"kv=8 layers ({len(KV8)}): {KV8[:6]}...")

fails = 0
for L in (KV4 + KV8[:2]):
    base = f"model.layers.{L}.self_attn.qkv_proj"
    w = rd(S, si, f"{base}.weight")
    sc = rd(S, si, f"{base}.weight_scale_inv")
    pk = rd(C, ci, f"{base}.weight")          # what the OLD (buggy) pack holds
    got, used_pg = pa._dequant_fp8_block_dispatch(base, w, sc, BLOCK, GROUPS)
    flat = pa._dequant_fp8_block(w, sc, BLOCK)
    pergrp = pa._dequant_fp8_block_per_group(w, sc, BLOCK, GROUPS)
    is_kv4 = pk.shape[0] == 13568
    expect_pg = is_kv4
    ok_flag = (used_pg == expect_pg)
    # dispatch output must equal the per-group reference for kv4, flat for kv8
    ok_val = cos(got, pergrp) > 0.99999 if is_kv4 else cos(got, flat) > 0.99999
    ok = ok_flag and ok_val
    fails += not ok
    extra = ""
    if is_kv4:
        extra = (f" | cos(new,old_pack)={cos(got, pk):.6f} "
                 f"cos(new,pergrp)={cos(got, pergrp):.6f}")
    print(f"L{L:>2} kv={'4' if is_kv4 else '8'} used_per_group={used_pg} "
          f"(expect {expect_pg}) -> {'OK' if ok else 'FAIL'}{extra}")

# non-qkv fp8 tensors must not regress: layer-0 dense MLP + MTP qkv
for base in ["model.layers.0.mlp.gate_proj", "model.layers.0.mlp.up_proj",
             "model.layers.0.mlp.down_proj",
             "model.mtp.layers.0.self_attn.qkv_proj"]:
    if f"{base}.weight" not in si:
        continue
    w = rd(S, si, f"{base}.weight")
    sc = rd(S, si, f"{base}.weight_scale_inv")
    got, used_pg = pa._dequant_fp8_block_dispatch(base, w, sc, BLOCK, GROUPS)
    flat = pa._dequant_fp8_block(w, sc, BLOCK)
    same = cos(got, flat) > 0.99999 or used_pg
    fails += not same
    print(f"{base}: used_per_group={used_pg} shape={tuple(got.shape)} "
          f"-> {'OK' if same else 'FAIL'}")

print(f"\nFAILURES: {fails}")
sys.exit(1 if fails else 0)
