#!/bin/bash
set -euo pipefail

# b12x reclamation-sweep gate mod (v3).
#
# b12x/preparation/session.py:_reclaim_programs does three things after every
# warmup/profiling batch release:
#
#   keep = frozenset(program for plan in self._plans ...)   <-- unguarded walk
#   evict_unretained(keep)                                  <-- stale-cache sweep
#   gc.collect()                                            <-- eager gen GC
#
# `keep` and `evict_unretained` both recurse the live retained-owner graph via
# retained_program_keys -> program_keys (b12x/_lib/compile_plan.py), which has
# NO cycle guard and NO memoization: a DAG-shaped owner graph re-materializes
# tuple(mapping.values()) per path, so the walk is exponential. On the
# pd-disagg / NVFP4 profiles (NIXL-registered KV buffers / large retained sets
# join the graph) all three phases grind at 100% CPU (GIL held) indefinitely:
# the worker stops answering EngineCore RPCs, determine_available_memory never
# returns, and engine init never completes.
#
# Patch: B12X_SKIP_RECLAIM=1 skips the whole reclamation block. CPython's
# threshold-triggered automatic collection still runs; skipping only leaves
# extra memoized entries in per-process WeakValueDictionary-backed caches.
# The default path is byte-identical to upstream.
#
# v3 supersedes v1 (gated evict_unretained only) and v2 (also gated the eager
# gc.collect()): both still executed the `keep` frozenset graph walk above the
# `with timing.span(...)` block, which spins by itself. All prior forms are
# upgraded to v3 in place. Idempotent and upgrade-safe.

PREFIX="[b12x-reclaim-gate]"
PYTHON_ROOT="${VLLM_SITE_PACKAGES:-${PYTHON_ROOT:-/usr/local/lib/python3.12/dist-packages}}"
SESSION="$PYTHON_ROOT/b12x/preparation/session.py"

echo "=== b12x reclamation-sweep gate mod (v3) ==="

if [ ! -f "$SESSION" ]; then
  echo "$PREFIX Missing $SESSION; b12x wheel not installed at PYTHON_ROOT." >&2
  exit 1
fi

python3 - "$SESSION" <<'PY'
import re
import sys
from pathlib import Path

target = Path(sys.argv[1])
text = target.read_text()

V3_MARK = "b12x-reclaim-gate v3"

NEW_BODY = '''    def _reclaim_programs(self):
        import os

        if os.environ.get("B12X_SKIP_RECLAIM") == "1":
            # b12x-reclaim-gate v3: skip the entire reclamation sweep.
            # `keep` materializes a frozenset over every prepared plan's
            # programs; with b12x's retained-owner DAG that walk (and the
            # evict_unretained sweep and eager gc.collect() that follow) has no
            # cycle guard or memoization and spins at 100% CPU holding the GIL
            # forever, so determine_available_memory never returns. Skipping
            # leaves only WeakValueDictionary-backed memoized entries, reaped
            # by CPython's automatic threshold collection.
            logger.warning_once(
                "b12x-reclaim-gate: B12X_SKIP_RECLAIM=1; skipped the whole "
                "reclamation sweep (keep-walk + evict_unretained + gc.collect)"
            )
            return
        timing = PreparationTiming("cleanup", rank=self._tuning_rank)
        keep = frozenset(program for plan in self._plans if plan.prepared is not None
                         for program in plan.prepared.programs)
        with timing.span("program_reclamation"):
            removed = evict_unretained(keep)
            gc.collect()
        timing.record("complete", evicted_entries=removed)
'''

if V3_MARK in text:
    print("already gated (v3):", target)
    sys.exit(0)

# Replace the whole _reclaim_programs body regardless of prior v0/v1/v2 shape.
pattern = re.compile(
    r"    def _reclaim_programs\(self\):\n.*?\n(?=    @contextmanager\n)",
    re.DOTALL,
)
matches = pattern.findall(text)
if len(matches) != 1:
    print(
        "expected exactly one _reclaim_programs body before @contextmanager "
        f"capture; found {len(matches)}; b12x wheel layout changed:",
        target,
        file=sys.stderr,
    )
    sys.exit(1)

text = pattern.sub(lambda _m: NEW_BODY, text, count=1)
target.write_text(text)
print("patched v3:", target)
PY

echo "$PREFIX done"
