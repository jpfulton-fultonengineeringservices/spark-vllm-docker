#!/bin/bash
set -euo pipefail

# Temporary diagnostic mod: dump the built KV cache group composition
# (layer names and spec summary per group) at engine startup. The 3x PP=3
# MiMo launch builds 12 groups where 3 were expected (full-attn, SWA-128,
# draft), and per-request KV usage grows ~5.5x context instead of ~1x -
# this log identifies what the groups actually are.
#
# Remove from the recipe once the grouping question is answered.

PREFIX="[diag-kv-groups]"
PYTHON_ROOT="${VLLM_SITE_PACKAGES:-${PYTHON_ROOT:-/usr/local/lib/python3.12/dist-packages}}"
KVUTILS="$PYTHON_ROOT/vllm/v1/core/kv_cache_utils.py"

echo "=== diag-kv-groups mod ==="

if [ ! -f "$KVUTILS" ]; then
  echo "$PREFIX Missing $KVUTILS; a newer image is required." >&2
  exit 1
fi

python3 - "$KVUTILS" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text()

old = (
    "    groups = kv_cache_config.kv_cache_groups\n"
    "\n"
    "    if len(groups) <= 1:\n"
)
new = (
    "    groups = kv_cache_config.kv_cache_groups\n"
    "\n"
    "    # spark-vllm-docker/mods/diag-kv-groups: dump group composition.\n"
    "    for _gi, _g in enumerate(groups):\n"
    "        _spec = _g.kv_cache_spec\n"
    "        _inner = getattr(_spec, \"kv_cache_specs\", None)\n"
    "        if _inner:\n"
    "            _details = sorted(\n"
    "                {\n"
    "                    (\n"
    "                        type(_s).__name__,\n"
    "                        str(getattr(_s, \"sliding_window\", None)),\n"
    "                        str(getattr(_s, \"page_size_bytes\", None)),\n"
    "                        str(getattr(_s, \"extra_retained_tokens\", None)),\n"
    "                        str(getattr(_s, \"prefill_replay_tokens\", None)),\n"
    "                    )\n"
    "                    for _s in _inner.values()\n"
    "                }\n"
    "            )\n"
    "        else:\n"
    "            _details = [\n"
    "                (\n"
    "                    type(_spec).__name__,\n"
    "                    str(getattr(_spec, \"sliding_window\", None)),\n"
    "                    str(getattr(_spec, \"page_size_bytes\", None)),\n"
    "                    str(getattr(_spec, \"extra_retained_tokens\", None)),\n"
    "                    str(getattr(_spec, \"prefill_replay_tokens\", None)),\n"
    "                )\n"
    "            ]\n"
    "        logger.info(\n"
    "            \"KV GROUP %d: nlayers=%d layers=%s specs=%s\",\n"
    "            _gi,\n"
    "            len(_g.layer_names),\n"
    "            list(_g.layer_names),\n"
    "            _details,\n"
    "        )\n"
    "\n"
    "    if len(groups) <= 1:\n"
)

if new in text:
    print("[diag-kv-groups] already patched; skipping.")
elif old not in text:
    raise SystemExit(
        "[diag-kv-groups] expected anchor not found; "
        "the installed vLLM differs from the layout this mod knows."
    )
else:
    path.write_text(text.replace(old, new, 1))
    print("[diag-kv-groups] Patched group dump into _resolve_block_size.")
PY

echo "=====> diag-kv-groups will log every KV cache group's layers and spec"
