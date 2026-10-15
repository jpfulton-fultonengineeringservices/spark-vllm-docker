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

python3 - "$SRC/config.json" "$DEST/config.json" <<'PY'
import json
import sys

src, dst = sys.argv[1], sys.argv[2]
with open(src) as fh:
    cfg = json.load(fh)

quant = cfg.get("quantization_config") or {}
original = quant.get("modules_to_not_convert") or []

# Rebuild the ignore list in vLLM's naming space. Bare names are kept (they
# match top-level modules such as lm_head exactly), and each entry also gets a
# "model."-rooted twin because every nested module lives under model. The
# q/k/v projection entries additionally collapse onto the fused qkv_proj name
# that vLLM instantiates, which is what the fused context K/V projection reads.
rebuilt = []


def add(name):
    if name not in rebuilt:
        rebuilt.append(name)


for entry in original:
    add(entry)
    add(f"model.{entry}")
    parts = entry.split(".")
    if len(parts) >= 4 and parts[0] == "layers":
        index, module, leaf = parts[1], parts[2], parts[3]
        if module == "self_attn" and leaf in ("q_proj", "k_proj", "v_proj", "qkv_proj"):
            add(f"model.layers.{index}.self_attn.qkv_proj")

# Every draft layer exposes the same attention shape, so make the fused name
# unconditional across the layer range even if the source list omitted one.
num_layers = int(cfg.get("num_hidden_layers") or 0)
for i in range(num_layers):
    add(f"model.layers.{i}.self_attn.qkv_proj")

quant["modules_to_not_convert"] = rebuilt
cfg["quantization_config"] = quant

with open(dst, "w") as fh:
    json.dump(cfg, fh, indent=2)
    fh.write("\n")

print(f"[qwen3.8-coder390-dflash] ignored layers: {len(original)} -> {len(rebuilt)}")
PY

echo "$PREFIX staged drafter at $DEST (from $SRC)"
echo "=== OK ==="
