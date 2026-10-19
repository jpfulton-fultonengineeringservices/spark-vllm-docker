#!/bin/bash
set -euo pipefail

# b12x reclamation-sweep gate mod (v4).
#
# b12x's preparation session does two unguarded reclamation sweeps:
#
#   PreparationSession._reclaim_programs:
#     keep = frozenset(program for plan in self._plans ...) <-- unguarded walk
#     evict_unretained(keep) <-- stale-cache sweep
#     gc.collect() <-- eager gen GC
#
#   PreparationJob._run (tail):
#     keep = frozenset(program for plan in self.session._plans ...)
#     self.session._synchronize()
#     self._program_cache.clear()   <-- O(1) dict drop, kept unconditional
#     evict_unretained(keep) <-- a second, previously un-gated sweep
#
# `keep` and `evict_unretained` recurse the live retained-owner graph via
# retained_program_keys -> program_keys (b12x/_lib/compile_plan.py), with NO
# cycle guard and NO memoization: a DAG-shaped owner graph re-materializes
# tuple(mapping.values()) per path, so the walk is exponential. On
# pd-disagg / NVFP4 profiles (NIXL-registered KV buffers / large retained sets
# joining the graph) the sweep grinds at 100% CPU (GIL held) indefinitely: the
# worker stops answering EngineCore RPCs, determine_available_memory never
# returns, and engine init never completes.
#
# Patch: B12X_SKIP_RECLAIM=1 skips both reclamation sweeps. In _run only the
# expensive part (keep walk + _synchronize + evict_unretained) is gated; the
# job-local PreparationProgramCache.clear() stays unconditional because it is an
# O(1) dict drop that unloads losing-candidate references. CPython's
# threshold-triggered automatic collection still runs; skipping only leaves
# extra memoized entries in per-process WeakValueDictionary-backed caches.
# In _run the skip branch binds `keep = frozenset()` because
# reclaim_device_memory(keep, ...) consumes it later, and rewrites that call to
# `stack_limit=None if not keep else self._stack_limit`: vendored
# reclaim_device_memory(keep, stack_limit=<baseline>) resets and re-sets the
# per-thread CUDA stack limit even for an empty keep, which would shrink it
# below retained-kernel needs. `stack_limit=None` takes its documented
# host-device path (gc.collect() only), preserving the reclaim of losing
# candidate libraries without touching the CUDA stack. Default path (env
# unset) is byte-identical to upstream.
#
# v4 (2026-10-19) adapts discovery to the flashinfer-vendored b12x layout:
# current images ship a shim-only `b12x/__init__.py` (a sys.meta_path aliaser)
# and the real code at `flashinfer/experimental/b12x/`. Discovery order is
# B12X_SESSION_PATH, then the importlib probe (portable across both layouts),
# then the historical `$PYTHON_ROOT/b12x/preparation/session.py`.
#
# v4 also fixes v3's `logger.warning_once(...)`: the vendored module's logger is
# a plain logging.Logger (vllm's warning_once patch only applies to loggers
# created through vllm.logger.init_logger), so v3 raised AttributeError the
# moment the gate fired. v4 upgrades v0/v1/v2/v3 containers in place.

PREFIX="[b12x-reclaim-gate]"
PYTHON_ROOT="${VLLM_SITE_PACKAGES:-${PYTHON_ROOT:-/usr/local/lib/python3.12/dist-packages}}"
LEGACY_SESSION="$PYTHON_ROOT/b12x/preparation/session.py"

probe_session() {
  python3 -c 'import importlib.util as u
try:
    spec = u.find_spec("b12x.preparation.session")
except Exception:
    spec = None
origin = getattr(spec, "origin", None)
print(origin if isinstance(origin, str) else "")' 2>/dev/null || true
}

SESSION=""
LAYOUT=""

if [ -n "${B12X_SESSION_PATH:-}" ]; then
  SESSION="$B12X_SESSION_PATH"
  LAYOUT="B12X_SESSION_PATH override"
  if [ ! -f "$SESSION" ]; then
    echo "$PREFIX B12X_SESSION_PATH=$SESSION does not exist." >&2
    exit 1
  fi
else
  PROBED="$(probe_session | awk 'NR == 1 { print; exit }')"
  if [ -n "$PROBED" ] && [ "$PROBED" != "None" ] && [ -f "$PROBED" ]; then
    SESSION="$PROBED"
    LAYOUT="vendored flashinfer.experimental.b12x"
  elif [ -f "$LEGACY_SESSION" ]; then
    SESSION="$LEGACY_SESSION"
    LAYOUT="legacy b12x"
  fi
fi

if [ -z "$SESSION" ]; then
  {
    echo "$PREFIX missing b12x preparation session; tried:"
    echo "  - B12X_SESSION_PATH (unset)"
    echo "  - importlib probe: python3 -c 'import importlib.util as u; u.find_spec(\"b12x.preparation.session\").origin'"
    echo "      vendored layout: \$PYTHON_ROOT/flashinfer/experimental/b12x/preparation/session.py"
    echo "  - legacy layout:  $LEGACY_SESSION"
    echo "Set B12X_SESSION_PATH to override discovery."
  } >&2
  exit 1
fi

echo "$PREFIX using $LAYOUT: $SESSION"

python3 - "$SESSION" <<'PY'
import ast
import re
import sys
import textwrap
from pathlib import Path

target = Path(sys.argv[1])
text = target.read_text()

V4_MARK = "b12x-reclaim-gate v4"

if V4_MARK in text:
    print("already gated (v4):", target)
    sys.exit(0)

if not re.search(r"^import os\b", text, re.M):
    print(
        "session.py has no module-level 'import os'; the v4 gate needs it:",
        target,
        file=sys.stderr,
    )
    sys.exit(1)

SITE1_RE = re.compile(
    r"    def _reclaim_programs\(self\):\n.*?\n(?=    @contextmanager\n)",
    re.DOTALL,
)
SITE1 = '''    def _reclaim_programs(self):
        if os.environ.get("B12X_SKIP_RECLAIM") == "1":
            # b12x-reclaim-gate v4: skip the entire reclamation sweep.
            # `keep` materializes a frozenset over every prepared plan's
            # programs; the retained-owner walk, the evict_unretained sweep and
            # the eager gc.collect() that follow have no cycle guard or
            # memoization and can spin at 100% CPU holding the GIL, so
            # determine_available_memory never returns. A skipped sweep only
            # leaves WeakValueDictionary-backed memoized entries behind, reaped
            # by CPython's automatic threshold-triggered collection.
            logger.warning(
                "b12x-reclaim-gate: B12X_SKIP_RECLAIM=1; skipped the whole "
                "reclamation sweep (keep-walk + evict_unretained + gc.collect)"
            )
            return
        timing = PreparationTiming("cleanup", rank=self._tuning_rank)
        keep = frozenset(
            program
            for plan in self._plans
            if plan.prepared is not None
            for program in plan.prepared.programs
        )
        with timing.span("program_reclamation"):
            removed = evict_unretained(keep)
            gc.collect()
        timing.record("complete", evicted_entries=removed)

'''

SITE2_RE = re.compile(
    r"(?P<head>        from b12x\._lib\.program_cache import evict_unretained\n"
    r"\n"
    r"        keep = frozenset\(\n"
    r".*?\n"
    r"        \)\n)"
    r"(?P<mid>        self\.session\._synchronize\(\)\n)"
    r"        self\._program_cache\.clear\(\)\n"
    r"(?P<tail>        evict_unretained\(keep\)\n)",
    re.DOTALL,
)

site1_matches = SITE1_RE.findall(text)
if len(site1_matches) != 1:
    print(
        "expected exactly one _reclaim_programs body before @contextmanager; "
        f"found {len(site1_matches)}; b12x wheel layout changed:",
        target,
        file=sys.stderr,
    )
    sys.exit(1)

site2_match = SITE2_RE.search(text)
site2_count = len(SITE2_RE.findall(text))
if site2_count != 1:
    print(
        "expected exactly one PreparationJob._run reclamation block "
        f"(keep + synchronize + clear + evict_unretained); found "
        f"{site2_count}; b12x wheel layout changed:",
        target,
        file=sys.stderr,
    )
    sys.exit(1)

site2_block = site2_match.group(0)
if "warning_once" in site2_block:
    print("upstream reclamation block references warning_once:", target,
          file=sys.stderr)
    sys.exit(1)

CALL_SITE_OLD = "reclaim_device_memory(keep, stack_limit=self._stack_limit)"
CALL_SITE_NEW = (
    "reclaim_device_memory("
    "keep, stack_limit=None if not keep else self._stack_limit)"
)
if text.count(CALL_SITE_OLD) != 1:
    print(
        "expected exactly one "
        f"{CALL_SITE_OLD!r}; found {text.count(CALL_SITE_OLD)}; "
        "b12x wheel layout changed:",
        target,
        file=sys.stderr,
    )
    sys.exit(1)

SITE2 = (
    "        self._program_cache.clear()\n"
    '        if os.environ.get("B12X_SKIP_RECLAIM") == "1":\n'
    "            # b12x-reclaim-gate v4: skip the expensive job-end reclamation.\n"
    "            # The job-local cache clear above stays unconditional (O(1)\n"
    "            # dict drop); the retained-owner walk behind `keep`, the eager\n"
    "            # _synchronize and the recursive evict_unretained sweep have\n"
    "            # no cycle guard or memoization and can spin at 100% CPU\n"
    "            # holding the GIL. `keep` is bound empty because\n"
    "            # reclaim_device_memory(keep, ...) consumes it later in _run.\n"
    "            logger.warning(\n"
    '                "b12x-reclaim-gate: B12X_SKIP_RECLAIM=1; skipped the "\n'
    '                "job-end reclamation sweep (keep-walk + synchronize + "\n'
    '                "evict_unretained)"\n'
    "            )\n"
    "            keep = frozenset()\n"
    "        else:\n"
    + textwrap.indent(
        SITE2_RE.search(site2_block).group("head")
        + SITE2_RE.search(site2_block).group("mid")
        + SITE2_RE.search(site2_block).group("tail"),
        "    ",
    )
)
if "warning_once" in SITE2:
    print("generated site 2 gate references warning_once:", target,
          file=sys.stderr)
    sys.exit(1)

clear_before = text.count("self._program_cache.clear()")

text = SITE1_RE.sub(lambda _m: SITE1, text, count=1)
text = SITE2_RE.sub(lambda _m: SITE2, text, count=1)
text = text.replace(CALL_SITE_OLD, CALL_SITE_NEW, 1)

problems = []
if text.count(V4_MARK) != 2:
    problems.append(f"expected 2 v4 markers, found {text.count(V4_MARK)}")
GATE = 'os.environ.get("B12X_SKIP_RECLAIM") == "1"'
if text.count(GATE) != 2:
    problems.append(
        f"expected 2 B12X_SKIP_RECLAIM gates, found {text.count(GATE)}"
    )
if "warning_once" in text:
    problems.append("patched file still references logger.warning_once")
if text.count("self._program_cache.clear()") != clear_before:
    problems.append(
        "self._program_cache.clear() count changed "
        f"({clear_before} -> {text.count('self._program_cache.clear()')})"
    )
site2_prefix = (
    "        self._program_cache.clear()\n"
    "        if " + GATE + ":\n"
)
if text.count(site2_prefix) != 1:
    problems.append(
        "job-local cache clear is not unconditional immediately above the gate "
        f"({text.count(site2_prefix)})"
    )
else:
    site2_gate = text.index(site2_prefix) + len(site2_prefix)
    site2_tail = text[site2_gate : text.index("evict_unretained(keep)\n", site2_gate)]
    if "self._program_cache.clear()" in site2_tail:
        problems.append("cache clear is still gated instead of unconditional")
if "keep = frozenset()" not in SITE2.split("else:\n", 1)[0]:
    problems.append("skip branch does not bind keep = frozenset()")
if text.count(CALL_SITE_NEW) != 1:
    problems.append(
        f"stack-limit call site not rewritten exactly once "
        f"({text.count(CALL_SITE_NEW)})"
    )
if CALL_SITE_OLD in text:
    problems.append("unguarded reclaim_device_memory call site survived")
try:
    ast.parse(text, filename=str(target))
except SyntaxError as error:
    problems.append(f"patched source does not compile: {error}")
if problems:
    for problem in problems:
        print(problem + ":", target, file=sys.stderr)
    sys.exit(1)

tmp = target.with_name(target.name + ".b12x-reclaim-gate.tmp")
tmp.write_text(text)
tmp.replace(target)
print("patched (v4):", target)
PY

echo "$PREFIX done"
