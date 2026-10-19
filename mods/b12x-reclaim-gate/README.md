# b12x-reclaim-gate

Gate b12x's warmup reclamation sweep behind `B12X_SKIP_RECLAIM=1`.

## Why

`b12x/preparation/session.py:_reclaim_programs` does three things after a
warmup/profiling batch release:

1. `keep = frozenset(program for plan in self._plans ...)` — a comprehension
   that materializes every prepared plan's programs;
2. `evict_unretained(keep)` — the stale-cache sweep;
3. `gc.collect()` — an eager generational collection.

(1) and (2) both recurse the live retained-owner graph via
`retained_program_keys` -> `program_keys()` (`b12x/_lib/compile_plan.py`).
That recursion has **no cycle guard and no memoization** — DAG-shaped owner
graphs re-materialize `tuple(mapping.values())` per path, so the walk is
exponential. (3) then grinds the same graph.

On the `pd-disagg-mimo-uncensored-exl3-4x` recipe, NIXL-registered KV buffers
(kv_producer/kv_consumer load) join the retained-owner set, and the sweep
spins at 100% CPU (GIL held) indefinitely: py-spy pins every sample in
`retained_program_keys.__iter__` via `evict_unretained`, the worker stops
answering EngineCore RPCs (`determine_available_memory` never returns), and
engine init never completes — no API bind, no KV-cache init, engines idle.

The sweep is a stale-cache eviction **optimization**: `keep` protects live
programs, and a skipped sweep only leaves extra memoized entries in
per-process `WeakValueDictionary`-backed caches (reaped by CPython's automatic
threshold-triggered collection). Skipping is safe.

## What it does

Patches both reclamation sweeps in the installed b12x preparation session:

- `PreparationSession._reclaim_programs`: with `B12X_SKIP_RECLAIM=1` the
  **entire** method early-returns (warning per call), skipping the `keep` walk,
  the `evict_unretained(keep)` sweep, and the eager `gc.collect()`.
- `PreparationJob._run` tail: only the exponential part (`keep` over
  `self.session._plans`, `_synchronize`, `evict_unretained`) is gated. The
  job-local `self._program_cache.clear()` stays unconditional above the gate —
  it is an O(1) dict drop, not the retained-owner walk, and skipping it would
  delay unloading of losing-candidate executables. The skip branch binds
  `keep = frozenset()` because `reclaim_device_memory(keep, ...)` consumes it
  later in `_run`, and the call site is rewritten to
  `reclaim_device_memory(keep, stack_limit=None if not keep else
  self._stack_limit)`. Vendored `reclaim_device_memory` resets and re-sets the
  per-thread CUDA stack limit even for an empty `keep`, so passing the baseline
  would shrink the stack below retained-kernel needs; `stack_limit=None` takes
  its documented host-device path (`gc.collect()` only), preserving the reclaim
  of losing-candidate libraries without touching the CUDA stack.

Default behavior (env unset or `0`) is functionally identical to upstream:
the site-1 body stays byte-identical, while site 2 is re-indented into the new
`else:` branch and its `reclaim_device_memory` call passes `self._stack_limit`
for any non-empty `keep`.

### Target discovery

Current `vllm-node-b12x` images vendor b12x inside flashinfer: `b12x/` is a
shim-only package (a `sys.meta_path` aliaser) and the real module lives at
`flashinfer/experimental/b12x/preparation/session.py`. Discovery order:

1. `B12X_SESSION_PATH` (explicit override; must exist);
2. `importlib.util.find_spec("b12x.preparation.session").origin` — portable
   across both layouts;
3. legacy `$PYTHON_ROOT/b12x/preparation/session.py` (older images);
4. otherwise fail, naming both layouts and the probe.

Version history: **v1** gated only `evict_unretained`; **v2** also gated the
eager `gc.collect()`; both still ran the `keep` frozenset walk above the timing
block, which spins by itself (py-spy pinned the spin at `session.py:606`, the
`keep` comprehension, under v2). **v3** replaces the whole method body and
upgrades v0/v1/v2 containers in place. **v4** adds vendored-layout discovery,
gates the second sweep in `PreparationJob._run`, keeps its O(1)
`self._program_cache.clear()` unconditional above the gate, and rewrites the
later call to `reclaim_device_memory(keep, stack_limit=None if not keep else
self._stack_limit)` so a gated empty `keep` cannot shrink the CUDA stack. It
also replaces v3's `logger.warning_once(...)` with `logger.warning(...)` — the
vendored module's logger is a plain `logging.Logger` (vllm patches
`warning_once` only onto loggers it creates), so v3 raised `AttributeError`
when the gate fired. v4 upgrades v0/v1/v2/v3 containers in place.

## Usage

`mods/b12x-reclaim-gate` runs at container start (launch-cluster.sh
apply-mod). Set the env in the recipe:

```yaml
env:
  B12X_SKIP_RECLAIM: "1"
```

The `pd-disagg-mimo-nvfp4-4x` and `mimo-v2.6-flash-rl-nvfp4-2x`/`-4x` recipes
carry this; other recipes are unaffected (default path unchanged).

Idempotent: re-running the mod on an already-gated wheel is a no-op.