#!/bin/bash
# [qwen3.8-coder390-dflash] Stage a vLLM-loadable copy of the bundled
# DFlash2-FP8 drafter.
#
# The checkpoint at /model/DFlash2-FP8 ships a config.json written for SGLang:
# its quantization_config.modules_to_not_convert lists layer-relative names
# ("q_proj", "layers.0.self_attn.q_proj", "fc", ...). vLLM's Fp8Config reads
# that same key but matches it with match_mode="exact" against the module
# prefixes it actually builds, which are rooted at "model." (e.g.
# "model.layers.0.self_attn.qkv_proj"). Nothing in the checkpoint's list
# matches, so every linear layer — including qkv_proj, whose q/k/v weights are
# BF16 in this checkpoint — is quantized to FP8 with no scale to load.
# DFlash's fused context K/V projection then concatenates the (now FP8) qkv
# slices and runs a plain F.linear against BF16 activations:
#   RuntimeError: expected mat1 and mat2 to have the same dtype,
#                 but got: c10::BFloat16 != c10::Float8_e4m3fn
#
# This mod symlinks the drafter's weights and writes a config.json whose
# ignored-layer list is expressed with vLLM's own prefixes, so the 20 genuinely
# FP8 tensors (self_attn.o_proj + mlp.{gate,up,down}_proj per layer) stay
# quantized and every BF16 tensor stays BF16.
#
# Staged at /workspace/Qwen3.8-27B-Coder390-dflash (the MiMo drafter-staging
# convention); the recipe's --speculative-config model= points there. Note the
# image sets WORKSPACE_DIR=/workspace/vllm, which is NOT the serve-visible
# scratch dir the recipes reference, so this is deliberately a fixed path with
# the same override name the sibling mod uses.

set -euo pipefail

PREFIX="[qwen3.8-coder390-dflash]"
LINK_DIR="${CODER390_DFLASH_LINK_DIR:-/workspace}"
SRC="${CODER390_DFLASH_SRC:-/model/DFlash2-FP8}"
DEST="$LINK_DIR/Qwen3.8-27B-Coder390-dflash"
# The target checkpoint's config sits one level up from the drafter subdir and
# supplies the layer offset the drafter's own prefixes are built with.
CODER390_TARGET_CONFIG="${CODER390_TARGET_CONFIG:-$(dirname "$SRC")/config.json}"

if [ ! -f "$SRC/config.json" ]; then
    echo "$PREFIX FAIL: no drafter config at $SRC" >&2
    exit 1
fi

echo "=== qwen3.8-coder390-dflash mod ==="

rm -rf "$DEST"
mkdir -p "$DEST"

for f in "$SRC"/*; do
    base="$(basename "$f")"
    case "$base" in
        config.json|@eaDir) continue ;;
    esac
    ln -sfn "$f" "$DEST/$base"
done

python3 - "$SRC/config.json" "$DEST/config.json" "$CODER390_TARGET_CONFIG" <<'PY'
import json
import os
import sys

src, dst, target_cfg = sys.argv[1], sys.argv[2], sys.argv[3]
with open(src) as fh:
    cfg = json.load(fh)

quant = cfg.get("quantization_config") or {}
original = quant.get("modules_to_not_convert") or []

# Rebuild the ignore list in vLLM's naming space.
#
# Two transforms matter:
#  1. Every nested module lives under "model." (DFlashQwen3ForCausalLM builds
#     its inner model with prefix "model"), and the checkpoint's list has no
#     such root.
#  2. The draft layers are NOT namespaced 0..N-1. DFlashQwen3ForCausalLM
#     instantiates its model with start_layer_id=<target's total hidden layers>
#     (qwen3_dflash.py: `start_layer_id=target_layer_num`), so the draft's
#     layer i is `model.layers.<target_num_layers + i>`. For this checkpoint
#     that is 64..68, not 0..4. vLLM's Fp8Config matches in exact mode, so
#     an un-offset name silently never matches and every linear layer is
#     quantized to FP8 — including qkv_proj, whose q/k/v weights are BF16 —
#     after which DFlash's fused context K/V projection runs F.linear over
#     FP8 weights with BF16 activations and dies with
#     "expected mat1 and mat2 to have the same dtype ... BFloat16 != FP8".
#
# The target layer count is read from the sibling target config.json (mounted
# alongside the drafter under the checkpoint root); it defaults to the model's
# own num_target_layers, then to 0.
rebuilt = []


def add(name):
    if name not in rebuilt:
        rebuilt.append(name)


OFFSET = 0
num_target_layers = cfg.get("num_target_layers")
if isinstance(num_target_layers, int):
    OFFSET = num_target_layers
if target_cfg and os.path.isfile(target_cfg):
    with open(target_cfg) as fh:
        parent = json.load(fh)
    text = parent.get("text_config") or {}
    offset = text.get("num_hidden_layers", parent.get("num_hidden_layers"))
    if isinstance(offset, int):
        OFFSET = offset


def shift(index_str):
    try:
        return str(int(index_str) + OFFSET)
    except ValueError:
        return index_str


for entry in original:
    add(entry)
    add(f"model.{entry}")
    parts = entry.split(".")
    if len(parts) >= 4 and parts[0] == "layers":
        module, leaf = parts[2], parts[3]
        index = shift(parts[1])
        add(f"model.layers.{index}.{module}.{leaf}")
        add(f"layers.{index}.{module}.{leaf}")
        if module == "self_attn" and leaf in ("q_proj", "k_proj", "v_proj", "qkv_proj"):
            add(f"model.layers.{index}.self_attn.qkv_proj")

# Every draft layer exposes the same attention shape, so make the fused name
# unconditional across the (offset) layer range even if the source list
# omitted one.
for i in range(int(cfg.get("num_hidden_layers") or 0)):
    add(f"model.layers.{shift(str(i))}.self_attn.qkv_proj")

print(f"[qwen3.8-coder390-dflash] layer offset: {OFFSET}")

quant["modules_to_not_convert"] = rebuilt
cfg["quantization_config"] = quant

with open(dst, "w") as fh:
    json.dump(cfg, fh, indent=2)
    fh.write("\n")

print(f"[qwen3.8-coder390-dflash] ignored layers: {len(original)} -> {len(rebuilt)}")
PY

echo "$PREFIX staged drafter at $DEST (from $SRC)"
echo "=== OK ==="
