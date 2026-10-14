#!/bin/bash
set -euo pipefail

# MiMo-V2.6-Flash-RL NVFP4 (ModelOpt MIXED_PRECISION) load-bridge mod.
#
# The XiaomiMiMo/MiMo-V2.6-Flash-RL-NVFP4 checkpoint is a ModelOpt
# MIXED_PRECISION export: routed experts are NVFP4 and the dense/attention
# linears are FP8_PB_WO. The b12x fork's vLLM already parses the ModelOpt
# quantization_config (MIXED_PRECISION -> modelopt_mixed) and drives NVFP4
# routed experts through ModelOptNvFp4FusedMoE, but two ModelOpt naming
# mismatches drop weights on the floor:
#
# 1. ModelOpt exports the FP8_PB_WO per-block scale as `weight_scale`
#    (4-D, shape [out_blk, 1, in_blk, 1]) while the MiMo fused-QKV loader
#    (`mimo_v2.py::_try_load_fp8_qkv_proj`) and the generic path only
#    recognise the FP8 `weight_scale_inv` suffix and the 2-D block shape.
#    Left unbridged, the fused qkv_proj weight is silently dropped and the
#    dense block scales are loaded at the wrong shape -> garbage output.
# 2. The generic per-tensor tail looks the param up by its on-disk name;
#    ModelOpt's `weight_scale` never resolves against a `weight_scale_inv`
#    param name.
#
# The checkpoint ships this bridge as `vllm_patch/mimo_v2.diff` (21 lines,
# copied verbatim under `nvfp4-source/vllm_patch/`). This mod ports those
# hunks into the IMAGE's vLLM tree via an anchor-gated, idempotent, fail-loud
# inline python3 patcher:
#
#   (a) detect the ModelOpt layout once (`_mopt_ws`) in MiMoV2Model.load_weights;
#   (b) strip a stray FP8 `_inv` suffix when the ModelOpt layout is active;
#   (c) reshape the 2-D on-disk block scale to the 4-D param shape at the two
#       weight_loader call sites (stacked fused + generic tail);
#   (d) in the fused-qkv sharded path, remap `weight_scale_inv` -> `weight_scale`
#       and reshape the consumed scale tensor.
#
# ADDITIVE ONLY: the existing FP8 `weight_scale_inv` path is untouched (every
# ModelOpt-only branch is gated on `_mopt_ws` or `param.dim() == 4`), so
# EXL3 / pro-FP8 checkpoints keep loading exactly as before. This mod does NOT
# touch `exl3.py`; the EXL3 mod (`mods/exl3-mimo`) is NOT applied to NVFP4.
#
# The patcher locates every anchor by exact substring, asserts the expected
# count, and fail-loud exits on a mismatch (image drift). It is idempotent:
# a second run detects the `# mimo-nvfp4:` marker and skips. The patched file
# is validated with `ast.parse` before it is written.

PREFIX="[mimo-v2.6-flash-rl-nvfp4]"
PYTHON_ROOT="${VLLM_SITE_PACKAGES:-${PYTHON_ROOT:-/usr/local/lib/python3.12/dist-packages}}"
MIMO_V2="$PYTHON_ROOT/vllm/model_executor/models/mimo_v2.py"

echo "=== MiMo-V2.6-Flash-RL NVFP4 ModelOpt load bridge ==="

if [ ! -f "$MIMO_V2" ]; then
  echo "$PREFIX Missing $MIMO_V2; a vLLM runtime with the MiMo-V2 model is required." >&2
  exit 1
fi

python3 - "$MIMO_V2" <<'PY'
from pathlib import Path
import ast
import sys

path = Path(sys.argv[1])
text = path.read_text()

MARKER = "# mimo-nvfp4:"

if MARKER in text:
    print("[mimo-nvfp4] already patched; skipping.")
    raise SystemExit(0)

# (a) ModelOpt layout detection: no non-expert `weight_scale_inv`, at least one
# non-expert `weight_scale`. Inserted once, right after `params_dict` is built;
# `_mopt_ws` is then in scope for every later branch in the same method.
EDIT_A_OLD = (
    "        params_dict = dict(self.named_parameters(remove_duplicate=False))\n"
)
EDIT_A_NEW = EDIT_A_OLD + (
    "        # mimo-nvfp4: ModelOpt FP8_PB_WO dense layers name the block scale\n"
    "        # `weight_scale` (4-D). Detect that layout once for the whole load.\n"
    "        _mopt_ws = not any(\n"
    "            k.endswith(\"weight_scale_inv\") and \".experts\" not in k\n"
    "            for k in params_dict\n"
    "        ) and any(\n"
    "            k.endswith(\".weight_scale\") and \".experts\" not in k\n"
    "            for k in params_dict\n"
    "        )\n"
)

# (b) ModelOpt names the block scale `weight_scale`; strip the FP8 `_inv`
# suffix when the ModelOpt layout is active so the lookup resolves.
EDIT_B_OLD = (
    "        name = mapped_name if mapped_name is not None else orig_name\n"
)
EDIT_B_NEW = EDIT_B_OLD + (
    "        # mimo-nvfp4: ModelOpt block scale is named `weight_scale`.\n"
    "        if name.endswith(\".weight_scale_inv\") and _mopt_ws:\n"
    "            name = name[: -len(\"_inv\")]\n"
)

# (c1) Stacked fused-expert loader: reshape the 2-D on-disk block scale to the
# 4-D ModelOpt param shape before handing it to the weight loader.
EDIT_C1_OLD = (
    "                weight_loader(param, loaded_weight, stacked_shard_id)\n"
)
EDIT_C1_NEW = (
    "                # mimo-nvfp4: ModelOpt 4-D [out_blk, 1, in_blk, 1] block scale.\n"
    "                if param.dim() == 4 and loaded_weight.dim() == 2 and name_rewritten.endswith(\"weight_scale\"):\n"
    "                    loaded_weight = loaded_weight.reshape(\n"
    "                        loaded_weight.shape[0], 1, loaded_weight.shape[1], 1\n"
    "                    )\n"
) + EDIT_C1_OLD

# (c2) Generic tail loader: same reshape at the plain weight_loader call.
EDIT_C2_OLD = (
    "                weight_loader(param, loaded_weight)\n"
)
EDIT_C2_NEW = (
    "                # mimo-nvfp4: ModelOpt 4-D block scale (see the stacked site).\n"
    "                if param.dim() == 4 and loaded_weight.dim() == 2 and name.endswith(\"weight_scale\"):\n"
    "                    loaded_weight = loaded_weight.reshape(\n"
    "                        loaded_weight.shape[0], 1, loaded_weight.shape[1], 1\n"
    "                    )\n"
) + EDIT_C2_OLD

# (d1) Fused-qkv sharded path: ModelOpt names the param `weight_scale`, not
# `weight_scale_inv`; remap the assembled param name before lookup.
EDIT_D1_OLD = (
    "            param_name = f\"{prefix}.{kind}\"\n"
)
EDIT_D1_NEW = EDIT_D1_OLD + (
    "            # mimo-nvfp4: ModelOpt names this param `weight_scale`.\n"
    "            if param_name not in params_dict and kind == \"weight_scale_inv\":\n"
    "                param_name = f\"{prefix}.weight_scale\"\n"
)

# (d2) Fused-qkv sharded path: reshape the 2-D consumed scale to the 4-D param.
EDIT_D2_OLD = (
    "            param = params_dict[param_name]\n"
)
EDIT_D2_NEW = EDIT_D2_OLD + (
    "            # mimo-nvfp4: reshape the 2-D loaded block scale to the 4-D param.\n"
    "            if param.dim() == 4 and tensor.dim() == 2:\n"
    "                tensor = tensor.reshape(tensor.shape[0], 1, tensor.shape[1], 1)\n"
)

edits = [
    ("(a) ModelOpt layout detection", EDIT_A_OLD, EDIT_A_NEW),
    ("(b) weight_scale_inv suffix strip", EDIT_B_OLD, EDIT_B_NEW),
    ("(c1) stacked loader 4-D reshape", EDIT_C1_OLD, EDIT_C1_NEW),
    ("(c2) generic loader 4-D reshape", EDIT_C2_OLD, EDIT_C2_NEW),
    ("(d1) fused-qkv param_name remap", EDIT_D1_OLD, EDIT_D1_NEW),
    ("(d2) fused-qkv scale reshape", EDIT_D2_OLD, EDIT_D2_NEW),
]

for label, old, new in edits:
    count = text.count(old)
    if count != 1:
        raise SystemExit(
            f"[mimo-nvfp4] {label}: expected exactly 1 anchor, found {count}; "
            "the installed vLLM differs from the ModelOpt bridge layout this mod knows."
        )
    text = text.replace(old, new, 1)

# Fail loud if the patched module no longer parses (anchor hit the wrong place).
ast.parse(text)

path.write_text(text)
print("[mimo-nvfp4] Patched mimo_v2.py (ModelOpt weight_scale bridge).")
PY

echo "$PREFIX ModelOpt FP8_PB_WO block scale is named weight_scale (4-D)"
echo "$PREFIX fused-qkv loader accepts the ModelOpt weight_scale param name"
echo "OK"
