#!/bin/bash
set -euo pipefail

# KV-cache memory-guard override mod test: bash 3.2 + 5.x portable.

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

fail() {
  echo "FAIL: $*" >&2
  exit 1
}

KVUTILS="$WORK/python_root/vllm/v1/core/kv_cache_utils.py"
mkdir -p "$(dirname "$KVUTILS")"
cp "$ROOT/tests/fixtures/kv-cache-guard-override/python_root/vllm/v1/core/kv_cache_utils.py" "$KVUTILS"

echo "=== KV-cache memory-guard override mod ==="

PYTHON_ROOT="$WORK/python_root" bash "$ROOT/mods/kv-cache-guard-override/run.sh" \
  | grep -q "patched" || fail "mod did not report a patch"

grep -qF "spark-vllm-docker/mods/kv-cache-guard-override" "$KVUTILS" \
  || fail "override marker missing"
grep -qF "VLLM_SKIP_KV_CACHE_MEMORY_CHECK" "$KVUTILS" \
  || fail "env gate missing"
grep -qF "KV memory guard skipped" "$KVUTILS" \
  || fail "warning path missing"

python3 -m py_compile "$KVUTILS" || fail "patched fragment does not compile"

python3 - "$KVUTILS" <<'PY' || fail "patched guard does not behave"
import importlib.util
import os
import sys

path = sys.argv[1]
spec = importlib.util.spec_from_file_location("kv_utils", path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

BIG = 1024**3, (lambda: 2 * 1024**3), 1048576, (lambda avail: avail // 2)


def run():
    m._check_enough_kv_cache_memory(*BIG)


os.environ.pop("VLLM_SKIP_KV_CACHE_MEMORY_CHECK", None)
try:
    run()
except ValueError as e:
    assert "To serve at least one request" in str(e), e
else:
    raise AssertionError("guard must fail fast by default")

for value in ("0", "x", ""):
    os.environ["VLLM_SKIP_KV_CACHE_MEMORY_CHECK"] = value
    try:
        run()
    except ValueError:
        pass
    else:
        raise AssertionError("guard must fail fast for %r" % value)

os.environ["VLLM_SKIP_KV_CACHE_MEMORY_CHECK"] = "1"
m.logger.warnings.clear()
run()
assert any("guard skipped" in w for w in m.logger.warnings), m.logger.warnings

os.environ["VLLM_SKIP_KV_CACHE_MEMORY_CHECK"] = "true"
m.logger.warnings.clear()
run()
assert any("guard skipped" in w for w in m.logger.warnings), m.logger.warnings

# Enough memory: silent pass regardless of the override.
os.environ.pop("VLLM_SKIP_KV_CACHE_MEMORY_CHECK", None)
m.logger.warnings.clear()
m._check_enough_kv_cache_memory(
    4 * 1024**3, lambda: 2 * 1024**3, 1048576, lambda avail: avail // 2
)
assert not m.logger.warnings, m.logger.warnings

# available_memory <= 0 still fails fast even with the override.
os.environ["VLLM_SKIP_KV_CACHE_MEMORY_CHECK"] = "1"
try:
    m._check_enough_kv_cache_memory(0, lambda: 0, 1, lambda avail: 0)
except ValueError as e:
    assert "No available memory" in str(e), e
else:
    raise AssertionError("out-of-memory guard must stay fail-fast")
PY

PYTHON_ROOT="$WORK/python_root" bash "$ROOT/mods/kv-cache-guard-override/run.sh" \
  | grep -q "already patched" || fail "second run is not idempotent"
[ "$(grep -c "spark-vllm-docker/mods/kv-cache-guard-override" "$KVUTILS")" = 1 ] \
  || fail "double patch"

echo "PASS: kv-cache-guard-override mod"
