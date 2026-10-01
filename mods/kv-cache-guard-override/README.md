# KV-cache memory-guard override mod

## What it fixes

`_check_enough_kv_cache_memory` (`vllm/v1/core/kv_cache_utils.py`) refuses to
start the engine when one request at the full `max_model_len` would not fit
in the available KV cache. The assumption is far more conservative than real
workloads — observed on the TP=4 fleet: max 22% KV utilization with 8
concurrent sequences — and on the 3x PP=3 topology it blocks a serviceable
engine (33.84 GiB needed for a single 1M-token request vs 27.17 GiB
available at `gpu_memory_utilization` 0.80).

## How it fixes it

The guard stays the default. With `VLLM_SKIP_KV_CACHE_MEMORY_CHECK=1` (or
`true`) the raise becomes a logged warning and the engine starts. A request
that outgrows the KV pool queues instead of crashing. The
`available_memory <= 0` guard is untouched — that one is always fail-fast.

## Behavior changes

- Default (env unset or any other value): identical fail-fast behavior.
- `VLLM_SKIP_KV_CACHE_MEMORY_CHECK=1|true`: the max-seq-len memory check is
  skipped with a warning carrying the full sizing message.

## Files

- `vllm/v1/core/kv_cache_utils.py` — env-gated guard in
  `_check_enough_kv_cache_memory`.

## Test

```bash
bash tests/test_kv_cache_guard_override_mod.sh
```

Runs under bash 3.2 and 5.x. Covers default fail-fast, non-override values,
both override values, the silent enough-memory path, the untouched
`available_memory <= 0` guard, and patch idempotency.
