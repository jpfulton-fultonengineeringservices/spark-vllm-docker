#!/bin/bash
set -euo pipefail

# b12x reclamation-gate mod test: bash 3.2 + 5.x portable, no GPU/network.

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REAL_PYTHON="$(command -v python3)"
MOD="$ROOT/mods/b12x-reclaim-gate/run.sh"
FIX="$ROOT/tests/fixtures/b12x-reclaim-gate"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

fail() {
  echo "FAIL: $*" >&2
  exit 1
}

run_mod() {
  bash "$MOD" 2>&1
}

checksum() {
  cksum "$1" | awk '{ print $1 ":" $2 }'
}

echo "=== b12x reclaim-gate mod ==="

# ---------------------------------------------------------------- case 1
# Probe-based discovery in the vendored flashinfer layout: b12x/ is shim-only
# (no preparation/), the real code lives under flashinfer/experimental/b12x/.
VENDORED="$WORK/vendored"
cp -R "$FIX/vendored" "$VENDORED"
mkdir -p "$WORK/empty_root"
VENDORED_SESSION="$VENDORED/flashinfer/experimental/b12x/preparation/session.py"
cp "$FIX/pristine-session.py" "$VENDORED_SESSION"

OUT="$(PYTHONPATH="$VENDORED" PYTHON_ROOT="$WORK/empty_root" run_mod)" \
  || fail "probe discovery run failed: $OUT"
case "$OUT" in
  *"using vendored flashinfer.experimental.b12x: $VENDORED_SESSION"*) ;;
  *) fail "probe discovery did not select the vendored file: $OUT" ;;
esac
case "$OUT" in
  *"patched (v4)"*) ;;
  *) fail "probe discovery did not report a v4 patch: $OUT" ;;
esac
[ -f "$VENDORED_SESSION" ] || fail "vendored session.py disappeared"
[ ! -e "$VENDORED/b12x/preparation/session.py" ] \
  || fail "fixture b12x/ is not shim-only; probe test is vacuous"
[ "$(grep -c "b12x-reclaim-gate v4" "$VENDORED_SESSION")" = 2 ] \
  || fail "expected two v4 markers"
[ "$(grep -c 'os.environ.get("B12X_SKIP_RECLAIM") == "1"' "$VENDORED_SESSION")" = 2 ] \
  || fail "expected two env gates"
python3 -m py_compile "$VENDORED_SESSION" || fail "patched session.py does not compile"

# ---------------------------------------------------------------- case 2
# B12X_SESSION_PATH overrides both the probe and the legacy fallback.
OVERRIDE_ROOT="$WORK/override_root"
cp -R "$FIX/vendored" "$OVERRIDE_ROOT"
cp "$FIX/pristine-session.py" \
  "$OVERRIDE_ROOT/flashinfer/experimental/b12x/preparation/session.py"
OVERRIDE="$WORK/override_session.py"
cp "$FIX/pristine-session.py" "$OVERRIDE"
OVERRIDE_BEFORE="$(checksum "$OVERRIDE_ROOT/flashinfer/experimental/b12x/preparation/session.py")"

OUT="$(PYTHONPATH="$OVERRIDE_ROOT" PYTHON_ROOT="$WORK/empty_root" \
  B12X_SESSION_PATH="$OVERRIDE" run_mod)" || fail "override run failed: $OUT"
case "$OUT" in
  *"using B12X_SESSION_PATH override"*) ;;
  *) fail "override was not selected: $OUT" ;;
esac
[ "$(grep -c "b12x-reclaim-gate v4" "$OVERRIDE")" = 2 ] \
  || fail "override target was not patched"
[ "$(checksum "$OVERRIDE_ROOT/flashinfer/experimental/b12x/preparation/session.py")" \
  = "$OVERRIDE_BEFORE" ] || fail "probe target was patched despite override"

# ---------------------------------------------------------------- case 3
# An override that points nowhere fails loudly instead of falling back.
set +e
OUT="$(B12X_SESSION_PATH="$WORK/does-not-exist.py" run_mod)"
RC=$?
set -e
[ "$RC" -ne 0 ] || fail "missing override path should fail"
case "$OUT" in
  *"does-not-exist.py"*) ;;
  *) fail "missing override failure does not name the path: $OUT" ;;
esac

# ---------------------------------------------------------------- case 4
# Idempotent re-run on an already-v4 file: exit 0, no rewrite.
BEFORE="$(checksum "$VENDORED_SESSION")"
OUT="$(PYTHONPATH="$VENDORED" PYTHON_ROOT="$WORK/empty_root" run_mod)" \
  || fail "idempotent re-run failed: $OUT"
case "$OUT" in
  *"already gated (v4)"*) ;;
  *) fail "second run did not report already gated (v4): $OUT" ;;
esac
[ "$(checksum "$VENDORED_SESSION")" = "$BEFORE" ] || fail "idempotent re-run rewrote the file"

# ---------------------------------------------------------------- case 5
# v3 (and older) containers are upgraded in place; v3's warning_once removed.
V3="$WORK/v3_session.py"
cp "$FIX/v3-session.py" "$V3"
grep -q "warning_once" "$V3" || fail "v3 fixture lost its warning_once form"
OUT="$(B12X_SESSION_PATH="$V3" run_mod)" || fail "v3 upgrade failed: $OUT"
grep -q "b12x-reclaim-gate v3" "$V3" && fail "v3 marker survived the upgrade"
[ "$(grep -c "b12x-reclaim-gate v4" "$V3")" = 2 ] || fail "v3 upgrade did not install two v4 markers"
grep -q "warning_once" "$V3" && fail "v3 upgrade left warning_once behind"
[ "$(grep -c "self._program_cache.clear()" "$V3")" = 1 ] \
  || fail "v3 upgrade changed the cache-clear count"
grep -q "keep = frozenset()" "$V3" || fail "v3 upgrade lost the empty-keep binding"
[ "$(grep -c "^        self._program_cache.clear()$" "$V3")" = 1 ] \
  || fail "v3 upgrade did not leave exactly one unconditional cache clear"
! grep -q "            self._program_cache.clear()" "$V3" \
  || fail "v3 upgrade left the cache clear indented inside else"
python3 - "$V3" <<'PY' || fail "cache clear is not unconditional above the gate"
import sys

text = open(sys.argv[1]).read()
gate = '        if os.environ.get("B12X_SKIP_RECLAIM") == "1":\n'
clear = "        self._program_cache.clear()\n"
assert text.count(clear + gate) == 1
start = text.index(clear + gate) + len(clear)
end = text.index("evict_unretained(keep)\n", start)
assert "self._program_cache.clear()" not in text[start:end]
PY
[ "$(grep -c "stack_limit=None if not keep else self._stack_limit" "$V3")" = 1 ] \
  || fail "v3 upgrade did not guard the stack-limit call site"
grep -q "stack_limit=self._stack_limit)" "$V3" \
  && fail "v3 upgrade left the unguarded stack-limit call site"
python3 -m py_compile "$V3" || fail "upgraded v3 fixture does not compile"

# ---------------------------------------------------------------- case 5b
# Legacy layout fallback: probe resolves nothing, $PYTHON_ROOT/b12x has the file.
LEGACY_ROOT="$WORK/legacy_root"
mkdir -p "$LEGACY_ROOT/b12x/preparation"
LEGACY_SESSION="$LEGACY_ROOT/b12x/preparation/session.py"
cp "$FIX/pristine-session.py" "$LEGACY_SESSION"
mkdir -p "$WORK/legacy_bin"
cat > "$WORK/legacy_bin/python3" <<SH
#!/bin/sh
if [ "\$1" = "-c" ]; then
  exit 0
fi
exec "$REAL_PYTHON" "\$@"
SH
chmod +x "$WORK/legacy_bin/python3"
OUT="$(PATH="$WORK/legacy_bin:$PATH" PYTHON_ROOT="$LEGACY_ROOT" run_mod)" \
  || fail "legacy fallback failed: $OUT"
case "$OUT" in
  *"using legacy b12x: $LEGACY_SESSION"*) ;;
  *) fail "legacy fallback did not select the legacy file: $OUT" ;;
esac
[ "$(grep -c "b12x-reclaim-gate v4" "$LEGACY_SESSION")" = 2 ] \
  || fail "legacy fallback did not patch the file"
python3 -m py_compile "$LEGACY_SESSION" || fail "legacy patched file does not compile"

# ---------------------------------------------------------------- case 6
# Neither discovery layout exists: fail with both layouts named plus the probe.
mkdir -p "$WORK/stub_bin" "$WORK/empty_root2"
cat > "$WORK/stub_bin/python3" <<'SH'
#!/bin/sh
exit 0
SH
chmod +x "$WORK/stub_bin/python3"
set +e
OUT="$(PATH="$WORK/stub_bin:$PATH" PYTHON_ROOT="$WORK/empty_root2" run_mod)"
RC=$?
set -e
[ "$RC" -ne 0 ] || fail "missing session should fail"
case "$OUT" in
  *"vendored layout"*) ;;
  *) fail "failure message does not name the vendored layout: $OUT" ;;
esac
case "$OUT" in
  *"legacy layout"*) ;;
  *) fail "failure message does not name the legacy layout: $OUT" ;;
esac
case "$OUT" in
  *"find_spec"*) ;;
  *) fail "failure message does not name the probe: $OUT" ;;
esac
case "$OUT" in
  *"b12x/preparation/session.py"*) ;;
  *) fail "failure message does not name the legacy path: $OUT" ;;
esac

# ---------------------------------------------------------------- case 7
# Warning discipline: both gates use logger.warning, never warning_once, and
# behave correctly at runtime (gated method returns before touching _plans;
# default path still runs the sweep; site 2 binds an empty keep when gated).
python3 - "$VENDORED_SESSION" <<'PY' || fail "gate behavior check failed"
import os
import sys
import textwrap
import types

path = sys.argv[1]
text = open(path).read()

site1 = text.split("    def _reclaim_programs(self):\n", 1)[1].split(
    "\n    @contextmanager\n", 1
)[0]
gate = '        if os.environ.get("B12X_SKIP_RECLAIM") == "1":\n'
clear_line = "        self._program_cache.clear()\n"
idx = text.rindex(clear_line + gate)
site2 = text[idx:].split("        evict_unretained(keep)\n", 1)[0]
site2 += "        evict_unretained(keep)\n"
assert "self._program_cache.clear()" in site2
assert text.count(clear_line + gate) == 1

assert text.count("stack_limit=None if not keep else self._stack_limit") == 1
assert text.count("        self._program_cache.clear()\n" + gate) == 1
assert "stack_limit=self._stack_limit)" not in text

for name, body in (("site1", site1), ("site2", site2)):
    assert "warning_once" not in body, f"{name} calls warning_once"
    assert "logger.warning(" in body, f"{name} has no logger.warning"

assert site1.count("logger.warning(") + site2.count("logger.warning(") == 2


class FakeLogger:
    def __init__(self):
        self.calls = []

    def warning(self, message):
        self.calls.append(message)


logger = FakeLogger()


class FakeTiming:
    def __init__(self, phase, rank=None):
        self.phase = phase

    def span(self, name):
        return _NullCtx()

    def record(self, name, **fields):
        pass


class _NullCtx:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


evicted = []
gc_calls = []


class FakeGC:
    def collect(self):
        gc_calls.append(True)


ns = {
    "os": os,
    "gc": FakeGC(),
    "logger": logger,
    "PreparationTiming": FakeTiming,
    "evict_unretained": lambda keep: evicted.append(keep) or 0,
}
exec(
    "class _Session:\n"
    "    def _reclaim_programs(self):\n"
    + textwrap.indent(textwrap.dedent(site1), "        "),
    ns,
)
Session = ns["_Session"]


class ExplodingSession:
    def __init__(self):
        self._tuning_rank = 1

    @property
    def _plans(self):
        raise AssertionError("gated _reclaim_programs touched _plans")


os.environ["B12X_SKIP_RECLAIM"] = "1"
Session._reclaim_programs(ExplodingSession())
assert evicted == [], "gated site1 still ran evict_unretained"
assert gc_calls == [], "gated site1 still ran gc.collect()"
assert len(logger.calls) == 1, logger.calls


class Plan:
    def __init__(self, program):
        self.prepared = types.SimpleNamespace(programs=(program,))


class LiveSession:
    _tuning_rank = 3

    def __init__(self):
        self._plans = [Plan("p1"), Plan("p2")]


del os.environ["B12X_SKIP_RECLAIM"]
Session._reclaim_programs(LiveSession())
assert evicted == [frozenset({"p1", "p2"})], evicted
assert gc_calls == [True], gc_calls

# Site 2: inject a fake b12x._lib.program_cache for the local import.
cache_mod = types.ModuleType("b12x._lib.program_cache")
cache_evicted = []
cache_mod.evict_unretained = lambda keep: cache_evicted.append(keep) or 0
b12x_pkg = types.ModuleType("b12x")
b12x_pkg.__path__ = []
lib_pkg = types.ModuleType("b12x._lib")
lib_pkg.__path__ = []
sys.modules["b12x"] = b12x_pkg
sys.modules["b12x._lib"] = lib_pkg
sys.modules["b12x._lib.program_cache"] = cache_mod

call_lines = [
    line.strip()
    for line in text.splitlines()
    if line.strip().startswith("reclaim_device_memory(keep,")
]
assert len(call_lines) == 1, call_lines
reclaim_calls = []
ns2 = {
    "os": os,
    "logger": FakeLogger(),
    "reclaim_device_memory": (
        lambda keep, *, stack_limit: reclaim_calls.append((keep, stack_limit))
    ),
}
src2 = (
    "def _tail(self):\n"
    + site2
    + "        result_keep = keep\n"
    + "        " + call_lines[0] + "\n"
    + "        return result_keep\n"
)
exec(src2, ns2)
tail = ns2["_tail"]


class Cache:
    def __init__(self):
        self.cleared = 0

    def clear(self):
        self.cleared += 1


class JobSession:
    def __init__(self):
        self.sync = 0
        self._plans = [Plan("p3")]

    def _synchronize(self):
        self.sync += 1


class Job:
    _stack_limit = 4242

    def __init__(self, session, cache):
        self.session = session
        self._program_cache = cache


class ExplodingJobSession:
    @property
    def _plans(self):
        raise AssertionError("gated site2 touched _plans")

    def _synchronize(self):
        raise AssertionError("gated site2 synchronized")


os.environ["B12X_SKIP_RECLAIM"] = "1"
cache = Cache()
keep = tail(Job(ExplodingJobSession(), cache))
assert keep == frozenset(), keep
assert cache.cleared == 1, "gated site2 skipped the unconditional cache clear"
assert cache_evicted == [], "gated site2 ran evict_unretained"
assert reclaim_calls == [(frozenset(), None)], reclaim_calls

del os.environ["B12X_SKIP_RECLAIM"]
session = JobSession()
cache = Cache()
keep = tail(Job(session, cache))
assert keep == frozenset({"p3"}), keep
assert session.sync == 1, session.sync
assert cache.cleared == 1, cache.cleared
assert cache_evicted == [frozenset({"p3"})], cache_evicted
assert reclaim_calls == [(frozenset(), None), (frozenset({"p3"}), 4242)], reclaim_calls
PY

echo "PASS: b12x reclaim-gate mod"
