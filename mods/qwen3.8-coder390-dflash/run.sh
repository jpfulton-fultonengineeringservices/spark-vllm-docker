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

# Long-context YaRN: the drafter must scale positions identically to the
# target. Its rotary embedding is built from THIS config.json
# (qwen3_dflash.py passes config.rope_parameters / config.max_position_embeddings),
# so leaving rope_type "default" sizes the cos/sin cache to exactly
# max_position_embeddings (262144) and the first request beyond that window
# trips the Triton device_assert in the cache lookup:
#   Assertion `index out of bounds: 0 <= ... < 262144` failed
# --hf-overrides cannot fix this: dict (non-callable) overrides stay
# target-only in SpeculativeConfig, so the draft ModelConfig never sees them.
#
# YaRN's cache length is original_max_position_embeddings * factor
# (YaRNScalingRotaryEmbedding._compute_cos_sin_cache), so max_position_embeddings
# stays at the checkpoint's native 262144 and the cache still spans 1048576.
# Values MUST match the recipe's --hf-overrides or the two models disagree on
# position phase. mscale/beta defaults are identical on both sides because
# neither config passes them.
drafter_rope = cfg.get("rope_parameters") or {}
if drafter_rope.get("rope_type", "default") == "default":
    drafter_rope.update(
        {
            "rope_type": "yarn",
            "factor": 4.0,
            "original_max_position_embeddings": 262144,
        }
    )
    cfg["rope_parameters"] = drafter_rope
    print(
        f"[qwen3.8-coder390-dflash] drafter rope -> yarn "
        f"factor={drafter_rope['factor']} "
        f"orig={drafter_rope['original_max_position_embeddings']} "
        f"(cache {drafter_rope['original_max_position_embeddings'] * drafter_rope['factor']:.0f})"
    )

with open(dst, "w") as fh:
    json.dump(cfg, fh, indent=2)
    fh.write("\n")

print(f"[qwen3.8-coder390-dflash] ignored layers: {len(original)} -> {len(rebuilt)}")
PY

# --- Force publisher sampling envelope at the sampling-param layer ---
# This checkpoint ships generation_config.json with do_sample=true,
# temperature=1.0, top_p=0.95, top_k=20 and NO repetition penalty.
# Agent/tool-calling clients send temperature=0 (greedy), which is out of the
# validated envelope and triggers the "fails to stop" reasoning degeneracy
# documented in docs/qwen38-coder390-degenerate-repetition.md.
#
# Force the publisher's envelope at the sampling-param layer
# (to_sampling_params) so it applies to ALL callers - LiteLLM, direct :8000,
# internal tool calls - always, regardless of client inputs. The sampling-param
# layer is the single point where client request fields meet server defaults
# before the Sampler sees them; clamping at the business/tool-call layer is
# partial and leaves other entry paths unprotected.
#
# Idempotent: re-running on an already-patched wheel is a no-op (MARK check).
python3 - <<'PY'
import importlib.util
import sys

MARK = "[qwen3.8-coder390-dflash] Force publisher sampling envelope"
ANCHOR = (
    "            for name, default in self._DEFAULT_SAMPLING_PARAMS.items()\n"
    "        }\n"
    "\n"
    "        # Merge server-default stop_token_ids"
)
INJECT = (
    "            for name, default in self._DEFAULT_SAMPLING_PARAMS.items()\n"
    "        }\n"
    "\n"
    "        # [qwen3.8-coder390-dflash] Force publisher sampling envelope:\n"
    "        # temperature=1.0, top_p=0.95, top_k=20 always, regardless of\n"
    "        # client inputs. The checkpoint ships do_sample=true with no\n"
    "        # repetition penalty; greedy (temperature=0) from agent/tool-calling\n"
    "        # clients is outside the validated envelope and triggers the\n"
    "        # 'fails to stop' reasoning degeneracy\n"
    "        # (docs/qwen38-coder390-degenerate-repetition.md).\n"
    '        sampling_params["temperature"] = 1.0\n'
    '        sampling_params["top_p"] = 0.95\n'
    '        sampling_params["top_k"] = 20\n'
    "        # [qwen3.8-coder390-dflash] Anti-repetition: 1.10 is the floor\n"
    "        # proven (commit 87c40e8fbd) to break the reasoning-channel loop\n"
    "        # deterministically while leaving prose length intact (1.15\n"
    "        # measurably shortens it). Set in the dict: repetition_penalty\n"
    "        # reaches from_optional ONLY via **sampling_params (no explicit\n"
    "        # kwarg), and the comprehension has already captured the request\n"
    "        # value (None -> 1.0), so a self-assignment here would never flow\n"
    "        # through. A dict set is safe: the key exists in\n"
    "        # _DEFAULT_SAMPLING_PARAMS, so no duplicate-kwarg collision.\n"
    '        sampling_params["repetition_penalty"] = 1.10\n'
    "        # [qwen3.8-coder390-dflash] Reasoning-budget gate: this checkpoint's\n"
    "        # reasoning channel degenerates into unbounded repetition on\n"
    "        # near-duplicate context (repeated file reads, license headers,\n"
    "        # similar tool results) -- content=0 with the entire generation\n"
    "        # budget billed as reasoning_tokens. thinking_token_budget forces\n"
    "        # the </think> close when the budget exhausts, converting every\n"
    "        # loop into a completed answer (verified live 2026-11-04: baseline\n"
    "        # looped 512/512 reasoning with content=0; budget=64 produced\n"
    "        # finish=stop with 1041 chars of content).\n"
    "        # Ratio: reasoning-heavy -- 75% of the generation budget goes to\n"
    "        # reasoning, 25% reserved for output; a default budget of 32768\n"
    "        # applies when the caller sends no generation cap. Never below\n"
    "        # max_gen - 256 so output always retains at least 256 tokens.\n"
    "        # NOTE: assigned on `self`, NOT into sampling_params -- the return\n"
    "        # statement passes thinking_token_budget=self.thinking_token_budget\n"
    "        # explicitly, so a dict entry would raise 'multiple values for\n"
    "        # keyword argument'. `max_tokens` is the method parameter (the\n"
    "        # effective generation budget), not a sampling_params key.\n"
    "        _gen_budget = max_tokens\n"
    "        if _gen_budget is None:\n"
    "            self.thinking_token_budget = 32768\n"
    "        elif _gen_budget <= 340:\n"
    "            self.thinking_token_budget = 0\n"
    "        else:\n"
    "            self.thinking_token_budget = min(32768, max(int(_gen_budget) * 75 // 100, int(_gen_budget) - 256))\n"
    "\n"
    "        # Merge server-default stop_token_ids"
)

targets = []
for modname in (
    "vllm.entrypoints.openai.chat_completion.protocol",
    "vllm.entrypoints.openai.completion.protocol",
):
    spec = importlib.util.find_spec(modname)
    if spec is not None and spec.origin is not None:
        targets.append(spec.origin)

if not targets:
    print(
        "[qwen3.8-coder390-dflash] sampling-envelope: cannot locate protocol files",
        file=sys.stderr,
    )
    sys.exit(0)

for path in targets:
    with open(path) as f:
        src = f.read()
    if MARK in src:
        print(f"[qwen3.8-coder390-dflash] sampling envelope already forced: {path}")
        continue
    count = src.count(ANCHOR)
    if count != 1:
        print(
            f"[qwen3.8-coder390-dflash] sampling-envelope: anchor count={count} != 1 in {path}; skipping",
            file=sys.stderr,
        )
        continue
    src = src.replace(ANCHOR, INJECT, 1)
    with open(path, "w") as f:
        f.write(src)
    print(f"[qwen3.8-coder390-dflash] sampling envelope forced: {path}")
PY

echo "$PREFIX staged drafter at $DEST (from $SRC)"
echo "=== OK ==="
