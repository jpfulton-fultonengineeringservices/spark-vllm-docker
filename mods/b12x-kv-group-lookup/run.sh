#!/bin/bash
set -euo pipefail

# b12x KV-cache group-lookup mod.
#
# Fixes a StopIteration crash in allocate_kv_cache (vllm/v1/worker/utils.py):
#
#     group_id, group = next(
#         (group_id, group)
#         for group_id, group in enumerate(kv_cache_config.kv_cache_groups)
#         if layer_name in group.layer_names
#     )
#     StopIteration
#
# The lookup resolves each KVCacheTensor to the cache group that owns its
# first layer. Tensor assembly and group construction disagree on where a
# group's layer names live: get_kv_cache_config_from_groups builds tensors
# from UniformTypeKVCacheSpecs.kv_cache_specs keys, while several group
# construction sites leave layer_names inconsistent with those keys. When the
# two diverge, next() finds nothing and the worker dies with a bare
# StopIteration from initialize_b12x_tuning_cache -> init_kv_cache ->
# allocate_kv_cache (observed on the 3x PP=3 DFlash launch, 2026-10-01; the
# same lookup runs again for the real cache, so this blocks the engine).
#
# The patch accepts either source of truth and fails with the full
# group/tensor structure when neither owns the layer, so any residual
# mismatch is self-explanatory instead of a bare StopIteration.
#
# Each file is patched independently and skipped when already fixed.
# Idempotent within the same fresh container.

PREFIX="[b12x-kv-group-lookup]"
PYTHON_ROOT="${VLLM_SITE_PACKAGES:-${PYTHON_ROOT:-/usr/local/lib/python3.12/dist-packages}}"
UTILS="$PYTHON_ROOT/vllm/v1/worker/utils.py"

echo "=== b12x KV-cache group-lookup mod ==="

if [ ! -f "$UTILS" ]; then
  echo "$PREFIX Missing $UTILS; a newer image is required." >&2
  exit 1
fi

python3 - "$UTILS" <<'PY'
import sys
from pathlib import Path

target = Path(sys.argv[1])
text = target.read_text()

OLD = '''    kv_caches: dict[str, torch.Tensor] = {}
    for tensor in kv_cache_config.kv_cache_tensors:
        layer_name = tensor.layers[0]
        group_id, group = next(
            (group_id, group)
            for group_id, group in enumerate(kv_cache_config.kv_cache_groups)
            if layer_name in group.layer_names
        )
        spec = group.kv_cache_spec'''

NEW = '''    kv_caches: dict[str, torch.Tensor] = {}
    for tensor in kv_cache_config.kv_cache_tensors:
        layer_name = tensor.layers[0]
        # b12x-kv-group-lookup: match what the config builders actually emit.
        # Tensor assembly keys off UniformTypeKVCacheSpecs.kv_cache_specs while
        # some group construction sites leave layer_names inconsistent with
        # those keys; the plain next() then dies with a bare StopIteration.
        # Accept either source and fail with the full structure otherwise.
        match = next(
            (
                (group_id, group)
                for group_id, group in enumerate(kv_cache_config.kv_cache_groups)
                if layer_name in group.layer_names
                or layer_name in getattr(group.kv_cache_spec, "kv_cache_specs", {})
            ),
            None,
        )
        if match is None:
            structure = [
                {
                    "layer_names": list(group.layer_names),
                    "spec_keys": sorted(
                        getattr(group.kv_cache_spec, "kv_cache_specs", {})
                    ),
                    "spec_type": type(group.kv_cache_spec).__name__,
                }
                for group in kv_cache_config.kv_cache_groups
            ]
            raise RuntimeError(
                f"allocate_kv_cache: no KV cache group owns layer {layer_name!r}; "
                f"tensors={[list(t.layers) for t in kv_cache_config.kv_cache_tensors]}; "
                f"groups={structure}"
            )
        group_id, group = match
        spec = group.kv_cache_spec'''

if "b12x-kv-group-lookup: match what the config builders actually emit" in text:
    print("already patched; skipping")
    sys.exit(0)
if OLD not in text:
    print(
        "allocate_kv_cache group lookup not found; file shape changed",
        file=sys.stderr,
    )
    sys.exit(1)
target.write_text(text.replace(OLD, NEW, 1))
print("patched", target)
PY
