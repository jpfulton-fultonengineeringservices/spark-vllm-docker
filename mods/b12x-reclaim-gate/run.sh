#!/bin/bash
set -euo pipefail

# b12x reclamation-sweep gate mod.
#
# b12x/preparation/session.py:_reclaim_programs runs evict_unretained(keep)
# (stale-cache eviction) and an eager gc.collect() after each warmup/profiling
# batch release. evict_unretained -> retained_program_keys
# (b12x/_lib/compile_plan.py) walks every live retained owner via program_keys()
# recursion with NO cycle guard and NO memoization: DAG-shaped owner graphs
# re-materialize values per path, so the walk is exponential. The eager
# gc.collect() grinds the same owner graph. On the pd-disagg 4x recipe
# (NIXL-registered KV buffers join the retained set) both grind at 100% CPU
# (GIL held) indefinitely: py-spy pins samples in retained_program_keys.__iter__
# via evict_unretained, the worker stops answering EngineCore RPCs
# (determine_available_memory never returns), and engine init never completes.
#
# Patch: B12X_SKIP_RECLAIM=1 skips BOTH the sweep and the eager gc.collect()
# (CPython's threshold-triggered automatic collection still runs; a skipped
# eviction only leaves extra memoized entries in per-process
# WeakValueDictionary-backed caches). Default path byte-identical to upstream.
#
# Idempotent and upgrade-safe: v1 (sweep gated, gc.collect() eager) containers
# are upgraded in place.

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
                # CPython's automatic threshold collection.
                removed = -1
                logger.warning(
                    "b12x-reclaim-gate: B12X_SKIP_RECLAIM=1; skipped "
                    "evict_unretained reclamation sweep"
                )
            else:
                removed = evict_unretained(keep)
            if os.environ.get("B12X_SKIP_RECLAIM") != "1":
                gc.collect()
            # b12x-reclaim-gate v2: else skip the eager generational
            # gc.collect() -- it grinds the same retained-owner graph for
            # minutes per release on the pd-disagg 4x profile batch; CPython's
            # threshold-triggered automatic collection still runs.
        timing.record("complete", evicted_entries=removed)'''

V2_MARK = 'if os.environ.get("B12X_SKIP_RECLAIM") != "1":'
V1_MARK = 'b12x-reclaim-gate: B12X_SKIP_RECLAIM=1; skipped'
V1_GC = '''            else:
                removed = evict_unretained(keep)
            gc.collect()
        timing.record("complete", evicted_entries=removed)'''
V2_GC = '''            else:
                removed = evict_unretained(keep)
            if os.environ.get("B12X_SKIP_RECLAIM") != "1":
                gc.collect()
            # b12x-reclaim-gate v2: skip the eager generational gc.collect() --
            # it grinds the same retained-owner graph for minutes per release
            # on the pd-disagg 4x profile batch; CPython's threshold-triggered
            # automatic collection still runs.
        timing.record("complete", evicted_entries=removed)'''

if V2_MARK in text:
    print("already gated (v2):", target)
    sys.exit(0)

if V1_MARK in text:
    # v1->v2 upgrade: gate the eager gc.collect() the v1 patch left ungated.
    if V1_GC not in text:
        print(
            "v1 marker present but gc.collect() anchor missing:",
            target,
            file=sys.stderr,
        )
        sys.exit(1)
    target.write_text(text.replace(V1_GC, V2_GC, 1))
    print("upgraded v1->v2:", target)
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