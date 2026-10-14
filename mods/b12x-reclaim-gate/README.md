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

Patches `_reclaim_programs` in the installed b12x wheel: with
`B12X_SKIP_RECLAIM=1` in the environment, the **entire** method early-returns
(a warning logged once per call), skipping the `keep` walk, the
`evict_unretained(keep)` sweep, and the eager `gc.collect()`. Default behavior
(env unset or `0`) is byte-identical to upstream.

Version history: **v1** gated only `evict_unretained`; **v2** also gated the
eager `gc.collect()`; both still ran the `keep` frozenset walk above the timing
block, which spins by itself (py-spy pinned the spin at `session.py:606`, the
`keep` comprehension, under v2). **v3** replaces the whole method body and
upgrades v0/v1/v2 containers in place.

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