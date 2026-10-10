# b12x-reclaim-gate

Gate b12x's warmup reclamation sweep behind `B12X_SKIP_RECLAIM=1`.

## Why

`b12x/preparation/session.py:_reclaim_programs` runs `evict_unretained(keep)`
after a warmup/profiling batch release. `evict_unretained` ->
`retained_program_keys` (`b12x/_lib/compile_plan.py`) walks every live
retained owner via `program_keys()` recursion. That recursion has **no cycle
guard and no memoization** — DAG-shaped owner graphs re-materialize
`tuple(mapping.values())` per path, so the walk is exponential.

On the `pd-disagg-mimo-uncensored-exl3-4x` recipe, NIXL-registered KV buffers
(kv_producer/kv_consumer load) join the retained-owner set, and the sweep
spins at 100% CPU (GIL held) indefinitely: py-spy pins every sample in
`retained_program_keys.__iter__` via `evict_unretained`, the worker stops
answering EngineCore RPCs (`determine_available_memory` never returns), and
engine init never completes — no API bind, no KV-cache init, engines idle.

The sweep is a stale-cache eviction **optimization**: `keep` protects live
programs, and a skipped eviction only leaves extra memoized entries in
per-process `WeakValueDictionary`-backed caches (reaped by the `gc.collect()`
that still runs). Skipping is safe.

## What it does

Patches `_reclaim_programs` in the installed b12x wheel: with
`B12X_SKIP_RECLAIM=1` in the environment, the `evict_unretained(keep)` sweep
is skipped (`removed = -1`, a warning logged once per call), `gc.collect()`
still runs, everything else untouched. Default behavior (env unset or `0`)
is byte-identical to upstream.

## Usage

`mods/b12x-reclaim-gate` runs at container start (launch-cluster.sh
apply-mod). Set the env in the recipe:

```yaml
env:
  B12X_SKIP_RECLAIM: "1"
```

The `pd-disagg-mimo-uncensored-exl3-4x` recipe carries this; other recipes
are unaffected (default path unchanged).

Idempotent: re-running the mod on an already-gated wheel is a no-op.