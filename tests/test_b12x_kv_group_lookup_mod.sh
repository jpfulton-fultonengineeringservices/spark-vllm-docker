#!/bin/bash
set -euo pipefail

# b12x KV-cache group-lookup mod test: bash 3.2 + 5.x portable.

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

fail() {
  echo "FAIL: $*" >&2
  exit 1
}

UTILS="$WORK/python_root/vllm/v1/worker/utils.py"
mkdir -p "$(dirname "$UTILS")"
cp "$ROOT/tests/fixtures/b12x-kv-group-lookup/python_root/vllm/v1/worker/utils.py" "$UTILS"

echo "=== b12x KV-cache group-lookup mod ==="

PYTHON_ROOT="$WORK/python_root" bash "$ROOT/mods/b12x-kv-group-lookup/run.sh" \
  | grep -q "patched" || fail "mod did not report a patch"

grep -qF "match = next(" "$UTILS" || fail "robust lookup missing"
grep -qF "if match is None:" "$UTILS" || fail "informative failure missing"
grep -qF "no KV cache group owns layer" "$UTILS" || fail "error text missing"
! grep -qF "        group_id, group = next(" "$UTILS" \
  || fail "original StopIteration lookup still present"

python3 -m py_compile "$UTILS" || fail "patched fragment does not compile"

python3 - "$UTILS" <<'PY' || fail "patched lookup does not behave"
import importlib.util
import sys

path = sys.argv[1]
spec = importlib.util.spec_from_file_location("utils_stub", path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class UniformSpec:
    def __init__(self, keys):
        self.kv_cache_specs = {k: object() for k in keys}


# Case 1: the bug -- uniform-type group with stale layer_names. The tensors
# were built from kv_cache_specs keys; the old lookup died with StopIteration.
cfg = m.KVCacheConfig(
    [m.KVCacheTensor(["draft.0", "draft.1"])],
    [m.KVCacheGroupSpec([], UniformSpec(["draft.0", "draft.1"]))],
)
out = m.allocate_kv_cache(cfg, None, None)
assert out["draft.0"] == ("buf", 0) and out["draft.1"] == ("buf", 0), out

# Case 2: legacy path -- layer_names populated; still resolves.
cfg = m.KVCacheConfig(
    [m.KVCacheTensor(["layer.0"])],
    [m.KVCacheGroupSpec(["layer.0"], object())],
)
assert m.allocate_kv_cache(cfg, None, None)["layer.0"] == ("buf", 0)

# Case 3: unknown layer fails loudly with the structure dump.
cfg = m.KVCacheConfig(
    [m.KVCacheTensor(["ghost.9"])],
    [m.KVCacheGroupSpec(["layer.0"], UniformSpec(["layer.0"]))],
)
try:
    m.allocate_kv_cache(cfg, None, None)
except RuntimeError as e:
    msg = str(e)
    assert "no KV cache group owns layer" in msg, msg
    assert "ghost.9" in msg and "layer.0" in msg, msg
else:
    raise AssertionError("expected RuntimeError for unknown layer")
PY

PYTHON_ROOT="$WORK/python_root" bash "$ROOT/mods/b12x-kv-group-lookup/run.sh" \
  | grep -q "already patched" || fail "second run is not idempotent"
[ "$(grep -c "match = next(" "$UTILS")" = 1 ] || fail "double patch"

echo "PASS: b12x-kv-group-lookup mod"
