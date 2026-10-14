#!/usr/bin/env bash
# test_patcher.sh — validate a spark-vllm-docker mod patch script against a
# captured clean vLLM source fixture, WITHOUT a container or GPU.
#
# Usage: test_patcher.sh <mod-run.sh> <fixture-file> [<fixture-dir-name>]
#
# How it works: the mod's run.sh patches files at absolute paths under
# /usr/local/lib/python3.12/dist-packages (PYTHON_ROOT). To stay hermetic, we
# create a temp PYTHON_ROOT, copy the fixture in at the exact in-image relative
# path, point VLLM_SITE_PACKAGES at the temp root, run the mod, then `compile()`
# every .py file the mod touched. Non-zero exit + a printed SyntaxError means
# the patch produced an unparseable module (the class of bug that silently
# landed a broken mimo_v2.py at container boot).
#
# The container smoke test (docker run ... bash run.sh) is still the strongest
# check; this harness is the fast local gate for the anchor/indent logic.
set -euo pipefail

MOD="${1:?usage: test_patcher.sh <mod-run.sh> <fixture.py> [relpath]}"
FIXTURE="${2:?fixture file required}"
RELPATH="${3:-vllm/model_executor/models/mimo_v2.py}"

[ -f "$MOD" ] || { echo "FATAL: mod not found: $MOD" >&2; exit 2; }
[ -f "$FIXTURE" ] || { echo "FATAL: fixture not found: $FIXTURE" >&2; exit 2; }

ROOT="$(mktemp -d /tmp/modtest.XXXXXX)"
trap 'rm -rf "$ROOT"' EXIT

# Scaffold the minimal tree the mods under test touch.
for rel in \
  "vllm/model_executor/models/mimo_v2.py" \
  "vllm/model_executor/models/qwen3_dflash.py" \
  "vllm/model_executor/layers/attention/attention.py" \
  "vllm/v1/attention/backends/triton_attn_diffkv.py" \
  "vllm/v1/core/kv_cache_utils.py" \
  "vllm/envs.py" ; do
  mkdir -p "$ROOT/$(dirname "$rel")"
done
cp "$FIXTURE" "$ROOT/$RELPATH"

before="$(sha256sum "$ROOT/$RELPATH" | awk '{print $1}')"

echo "=== running mod: $MOD ==="
PYTHON_ROOT="$ROOT" VLLM_SITE_PACKAGES="$ROOT" \
  bash "$MOD" || { echo "FATAL: mod run.sh exited non-zero" >&2; exit 1; }

after="$(sha256sum "$ROOT/$RELPATH" | awk '{print $1}')"
if [ "$before" = "$after" ]; then
  echo "FATAL: mod did not modify $RELPATH (anchor missed?)" >&2; exit 1
fi
echo "mod modified $RELPATH (sha ${before:0:12} -> ${after:0:12})"

echo "=== compiling every touched .py ==="
fail=0
while IFS= read -r f; do
  if ! python3 -c "import sys; compile(open(sys.argv[1]).read(), sys.argv[1], 'exec')" "$f" 2>/tmp/compile_err; then
    echo "SYNTAX FAIL: $f"
    cat /tmp/compile_err
    fail=1
  else
    echo "ok: ${f#"$ROOT"/}"
  fi
done < <(find "$ROOT" -name '*.py' | sort)

# Idempotence: a second run must be a no-op and must not re-break the file.
echo "=== idempotence: running mod a second time ==="
PYTHON_ROOT="$ROOT" VLLM_SITE_PACKAGES="$ROOT" bash "$MOD" >/dev/null 2>&1 || true
python3 -c "import sys; compile(open(sys.argv[1]).read(), sys.argv[1], 'exec')" "$ROOT/$RELPATH" \
  || { echo "SYNTAX FAIL after second run"; fail=1; }

if [ "$fail" -ne 0 ]; then
  echo "=== HARNESS RESULT: FAIL ==="
  exit 1
fi
echo "=== HARNESS RESULT: PASS ==="
