#!/bin/bash
set -euo pipefail

# The PP=3 hybrid splitter (_get_kv_cache_groups_uniform_page_size) picks the
# per-group layer count from the SMALLEST attention-type bucket on the stage
# (3 full-attention layers per 16-layer stage) and shards the 13 SWA-128
# layers into ~10 same-spec groups. Every group carries its own per-request
# block table and its own retention floor of (sliding_window + one in-flight
# prefill chunk), so the fragmentation multiplies each request's prefill-time
# KV footprint ~11x (measured: a 63K prompt transiently consumed ~53% of the
# 847K-token pool at 16384-token chunks and ~20% at 4096).
#
# Groups whose KVCacheSpec is value-identical can share one block table
# losslessly (same page geometry, same window): this mod merges them back
# into a single group at the single return point of get_kv_cache_groups'
# parallel-draft branch, so the EngineCore, the model runner, and the CUDA
# graph capture all observe the merged layout consistently. After the merge
# each stage has 3 groups: full-attention, SWA-128, and the SWA-1024 draft.

PREFIX="[merge-swa-groups]"
PYTHON_ROOT="${VLLM_SITE_PACKAGES:-${PYTHON_ROOT:-/usr/local/lib/python3.12/dist-packages}}"
KVUTILS="$PYTHON_ROOT/vllm/v1/core/kv_cache_utils.py"

echo "=== merge-swa-groups mod ==="

if [ ! -f "$KVUTILS" ]; then
  echo "$PREFIX Missing $KVUTILS; a newer image is required." >&2
  exit 1
fi

python3 - "$KVUTILS" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text()

helper = '''def _merge_identical_spec_groups(
    kv_cache_groups: list[KVCacheGroupSpec],
) -> list[KVCacheGroupSpec]:
    # spark-vllm-docker/mods/merge-swa-groups: undo the PP-stage splitter's
    # fragmentation of value-identical specs (13 SWA-128 layers -> ~10
    # groups). Same-spec groups share a block table losslessly; each extra
    # group instead multiplied per-request KV retention. UniformType groups
    # are left untouched (their per-layer specs differ inside one wrapper).
    merged: list[KVCacheGroupSpec] = []
    for group in kv_cache_groups:
        spec = group.kv_cache_spec
        if isinstance(spec, UniformTypeKVCacheSpecs):
            merged.append(group)
            continue
        for existing in merged:
            existing_spec = existing.kv_cache_spec
            if isinstance(existing_spec, UniformTypeKVCacheSpecs):
                continue
            if existing_spec == spec:
                existing.layer_names.extend(group.layer_names)
                break
        else:
            merged.append(group)
    if len(merged) != len(kv_cache_groups):
        logger.info(
            "Merged %d same-spec KV cache groups into %d",
            len(kv_cache_groups),
            len(merged),
        )
    return merged


'''

def_anchor = "def get_kv_cache_groups(\n"
return_anchor = "        return [*target_groups, *draft_groups]"
return_patched = (
    "        return _merge_identical_spec_groups([*target_groups, *draft_groups])"
)
# The per-stage worker path (no draft layers on PP ranks 0..N-2) skips the
# parallel-draft branch entirely and falls through to this return; it is the
# path whose 11-group view the KVCacheManager actually uses.
tail_anchor = (
    "    _annotate_eagle_groups(vllm_config, kv_cache_spec, groups)\n"
    "    _warn_if_unannotated_eagle_mamba(vllm_config, groups)\n"
    "    return groups\n"
)
tail_patched = (
    "    _annotate_eagle_groups(vllm_config, kv_cache_spec, groups)\n"
    "    _warn_if_unannotated_eagle_mamba(vllm_config, groups)\n"
    "    return _merge_identical_spec_groups(groups)\n"
)

if helper in text and return_patched in text and tail_patched in text:
    print("[merge-swa-groups] already patched; skipping.")
    raise SystemExit(0)

for anchor, label in (
    (def_anchor, "get_kv_cache_groups definition"),
    (return_anchor, "parallel-draft return"),
    (tail_anchor, "fall-through return"),
):
    if anchor not in text:
        raise SystemExit(
            f"[merge-swa-groups] {label} not found; "
            "the installed vLLM differs from the layout this mod knows."
        )

text = text.replace(def_anchor, helper + def_anchor, 1)
text = text.replace(return_anchor, return_patched, 1)
text = text.replace(tail_anchor, tail_patched, 1)
path.write_text(text)
print("[merge-swa-groups] Patched get_kv_cache_groups with same-spec merge.")
PY

echo "=====> same-spec KV cache groups are merged; expect 3 groups per stage"
