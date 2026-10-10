#!/bin/bash
set -euo pipefail

# b12x reclamation-sweep gate mod.
#
# b12x/preparation/session.py:_reclaim_programs runs evict_unretained(keep)
# after a warmup/profiling batch release. evict_unretained ->
# retained_program_keys (b12x/_lib/compile_plan.py) walks every live retained
# owner via program_keys() recursion, which has NO cycle guard and NO
# memoization: DAG-shaped owner graphs re-materialize values per path and the
# walk blows up exponentially. On the pd-disagg 4x recipe (NIXL-registered KV
# buffers join the retained set) the sweep spins at 100% CPU (GIL held)
# indefinitely: py-spy pins every sample in retained_program_keys.__iter__ via
# evict_unretained, the worker stops answering EngineCore RPCs, and engine init
# never completes (no API bind, no KV-cache init).
#
# The sweep is a stale-cache eviction optimization: `keep` protects live
# programs, and a skipped eviction only leaves extra memoized entries in
# per-process WeakValueDictionary-backed caches (reaped by GC). Patch: skip the
# sweep when B12X_SKIP_RECLAIM=1, keep gc.collect().
#
# Idempotent: skipped when already gated.

PREFIX="[b12x-reclaim-gate]"
PYTHON_ROOT="${VLLM_SITE_PACKAGES:-${PYTHON_ROOT:-/usr/local/lib/python3.12/dist-packages}}"
SESSION="$PYTHON_ROOT/b12x/preparation/session.py"

echo "=== b12x reclamation-sweep gate mod ==="

if [ ! -f "$SESSION" ]; then
  echo "$PREFIX Missing $SESSION; b12x wheel not installed at PYTHON_ROOT." >&2
  exit 1
fi

python3 - "$SESSION" <<'PY'
import sys
from pathlib import Path

target = Path(sys.argv[1])
text = target.read_text()

OLD = '''        with timing.span("program_reclamation"):
            removed = evict_unretained(keep)
            gc.collect()
        timing.record("complete", evicted_entries=removed)'''

NEW = '''        with timing.span("program_reclamation"):
            import os

            if os.environ.get("B12X_SKIP_RECLAIM") == "1":
                # b12x-reclaim-gate mod: evict_unretained's retained-owner walk
                # (retained_program_keys -> program_keys recursion) has no cycle
                # guard or memoization and spins on the pd-disagg 4x retained
                # set. Skipping the sweep only leaves extra memoized cache
                # entries, which are WeakValueDictionary-backed and reaped by
                # the gc.collect() below.
                removed = -1
                logger.warning(
                    "b12x-reclaim-gate: B12X_SKIP_RECLAIM=1; skipped "
                    "evict_unretained reclamation sweep"
                )
            else:
                removed = evict_unretained(keep)
            gc.collect()
        timing.record("complete", evicted_entries=removed)'''

if "b12x-reclaim-gate mod" in text:
    print("already gated:", target)
    sys.exit(0)
if OLD not in text:
    print(
        "anchor missing; b12x wheel layout changed:",
        target,
        file=sys.stderr,
    )
    sys.exit(1)

target.write_text(text.replace(OLD, NEW, 1))
print("patched", target)
PY
echo "$PREFIX done"