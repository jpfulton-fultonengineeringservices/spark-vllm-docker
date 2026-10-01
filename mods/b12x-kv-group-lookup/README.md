# b12x KV-cache group-lookup mod

## What it fixes

`allocate_kv_cache` (`vllm/v1/worker/utils.py`) resolves each `KVCacheTensor`
to the cache group that owns its first layer with a bare `next(...)` over
`group.layer_names`. Tensor assembly and group construction disagree on where
a group's layer names live: `get_kv_cache_config_from_groups` builds tensors
from `UniformTypeKVCacheSpecs.kv_cache_specs` keys, while several group
construction sites leave `layer_names` inconsistent with those keys. When the
two diverge the lookup finds nothing and the worker dies with a bare
`StopIteration` during KV cache allocation — observed on the 3x PP=3 DFlash
launch of `mimo-v2.6-flash-pp3` (2026-10-01) inside
`initialize_b12x_tuning_cache -> init_kv_cache -> allocate_kv_cache`.

## How it fixes it

The patched lookup accepts either source of truth (`layer_names` or the
uniform-type `kv_cache_specs` keys) and, when neither owns the layer, raises
a `RuntimeError` carrying the full tensor/group structure so any residual
mismatch is self-explanatory instead of a bare `StopIteration`.

## Behavior changes

- The matching semantics are unchanged for groups whose `layer_names` is
  correct (the first check wins, as before).
- The failure mode changes: an unowned layer now raises `RuntimeError` with
  diagnostics instead of `StopIteration` from `next()`.

## Files

- `vllm/v1/worker/utils.py` — robust group lookup in `allocate_kv_cache`.

## Test

```bash
bash tests/test_b12x_kv_group_lookup_mod.sh
```

Runs under bash 3.2 and 5.x. Covers the stale-`layer_names` case (the bug),
the legacy case, the informative failure, and patch idempotency.
