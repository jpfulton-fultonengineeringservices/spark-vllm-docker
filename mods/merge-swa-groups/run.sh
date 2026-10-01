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
# Groups whose block-table semantics are value-identical can share one
# block table losslessly (same page geometry, same window): this mod merges
# them back into a single group at both get_kv_cache_groups return points
# (the parallel-draft branch used by the coordinator/global view, and the
# fall-through return used by the per-stage worker path), so the EngineCore,
# the model runner, and CUDA graph capture all observe the merged layout
# consistently. Groups are plain-spec or UniformTypeKVCacheSpecs-wrapped;
# wrappers merge only when every inner per-layer spec is value-identical
# (verified on node: the fragmented SWA-128 groups are all wrapper-wrapped
# with identical inner specs). After the merge each stage has 3 groups:
# full-attention, SWA-128, and the SWA-1024 draft.

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

helper = '''def _uniform_merge_key(spec: KVCacheSpec):
    # Groups may merge when their block-table semantics are identical: the
    # same block_size and either one plain spec or a UniformType wrapper
    # whose inner per-layer specs are all value-identical.
    block_size = getattr(spec, "block_size", None)
    if isinstance(spec, UniformTypeKVCacheSpecs):
        inner = list(spec.kv_cache_specs.values())
        if not inner:
            return None
        first = inner[0]
        if any(s != first for s in inner[1:]):
            return None
        return (block_size, first)
    return (block_size, spec)


def _merge_identical_spec_groups(
    kv_cache_groups: list[KVCacheGroupSpec],
) -> list[KVCacheGroupSpec]:
    # spark-vllm-docker/mods/merge-swa-groups: undo the PP-stage splitter's
    # fragmentation of value-identical specs (13 SWA-128 layers -> ~10
    # groups). Same-spec groups share a block table losslessly; each extra
    # group instead multiplied per-request KV retention.
    merged: list[KVCacheGroupSpec] = []
    keys: list[object] = []
    for group in kv_cache_groups:
        key = _uniform_merge_key(group.kv_cache_spec)
        target = None
        if key is not None:
            for existing, existing_key in zip(merged, keys):
                if existing_key is not None and existing_key == key:
                    target = existing
                    break
        if target is None:
            merged.append(group)
            keys.append(key)
            continue
        target.layer_names.extend(group.layer_names)
        target_spec = target.kv_cache_spec
        if isinstance(target_spec, UniformTypeKVCacheSpecs):
            target_spec.kv_cache_specs.update(group.kv_cache_spec.kv_cache_specs)
    if len(merged) != len(kv_cache_groups):
        logger.info(
            "Merged %d same-spec KV cache groups into %d",
            len(kv_cache_groups),
            len(merged),
        )
    else:
        for group in kv_cache_groups:
            logger.info(
                "merge-swa-groups diag: nlayers=%d spec=%r",
                len(group.layer_names),
                group.kv_cache_spec,
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
