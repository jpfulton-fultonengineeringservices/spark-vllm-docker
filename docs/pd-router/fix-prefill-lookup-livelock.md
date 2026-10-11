# Fix: prefill store-lookup livelock (enable_lookup=false on producer)

**Date:** 2026-10-11 ~21:00–21:40Z · **Follows:** `diagnosis-prefill-mooncake-lookup-deadlock.md` · **Fix commit:** fork `7487dce` (branch `fulton/mimo-v2.6-flash`)

## Root cause (precise)

The vendored `MooncakeStoreScheduler.get_num_new_matched_tokens` (fork
`mods/mooncake-worker/src/mooncake_worker/vendor/store/scheduler.py:91-144`)
runs a store-side prefix-cache lookup for **every** new request, and the
deployment never set `lookup_async` (default **False**, `scheduler.py:63`).
Each call therefore **blocks the vLLM scheduler** inside
`LookupKeyClient.lookup → future.result()` (worker.py:2528) until the
single-worker executor round-trips a ZMQ REQ over an **unbounded**
`ipc://…` socket (no SND/RCVTIMEO; `_lookup` at worker.py:2509).

The prefill (kv_producer) role **never needs** this lookup — it only
produces KV. But the producer lookup *hit* stale keys: the mooncake master
retained 15,445 keys persisted across boots (hindsight extraction prompts
repeat verbatim), and `batch_is_exist` returned hits in ~ms
(32,672 `lookup_exists` ops logged, all ≤10 ms, counter still growing
~40/s at 20:52). A hit creates a `LoadSpec` → `update_state_after_alloc`
→ the request is re-queued with `WAITING_FOR_REMOTE_KVS` (deployed
scheduler.py:1754). Loads are only *issued* from the worker-side
`get_finished` (worker.py:2147-2153), which runs only when the worker
receives an `execute_model` for an admitted request — the scheduler never
admits anything because every step re-enters the blocking lookup for the
next waiting request. **Circular wait → prefill EngineCore wedged forever**
(running=0, waiting monotonic 4→5, GPU 0%, scheduler log silent).

## Evidence (all live, 20:37–20:53Z)

- py-spy EngineCore MainThread: blocked in `concurrent/futures/_base.py:451
  result()` ← `worker.py:2528 lookup` ← `scheduler.py:112
  get_num_new_matched_tokens` ← `connector.py:191` ← `scheduler.py:1390
  schedule` — with `num_tokens: 55047` (a real hindsight extraction) and a
  **fresh Future object every sample** → the loop re-runs the lookup each
  step; the *same* request stays wedged.
- Worker (Worker_TP0): LookupKeyServer thread **answering**
  (`process_request` in `recv_multipart`, ~40 lookups/s), send/recv store
  threads idle ⇒ lookups succeed; nothing ever gets admitted to produce/transfer.
- litellm: `RateLimitError: Deployment … max_parallel_requests=4 requests
  in flight` (slots held by the wedged requests), then
  `Available Model Group Fallbacks=None` → OpenRouter 402 (visible symptom).

## Fix shipped (fork 7487dce)

1. **`mods/pd-disagg/dispatch.sh`** — producer extra_config gains
   `"enable_lookup": false`. Honored as the first check in
   `get_num_new_matched_tokens` (scheduler.py:101): the producer skips the
   lookup entirely → recomputes fresh prefill; no deferral, no livelock.
   The **consumer keeps its lookup**: `_capacity_only` is
   `kv_consumer and not enable_lookup` (connector.py:146), so decode stays
   a full transfer participant — the store-only PD path is intact.
2. **`mods/mooncake-worker/.../store/worker.py` (LookupKeyClient)** —
   defense-in-depth so a wedged lookup can never hard-hang a scheduler
   again: socket built with `SNDTIMEO/RCVTIMEO = 10 s`; `zmq.Again` inside
   `_lookup` rebuilds the REQ socket **on the executor thread that owns it**
   (avoids cross-thread close of a blocked pyzmq socket) and returns a
   0-hit; `future.result(timeout=30 s)` bound under it returns 0 as well
   (builtin `TimeoutError`, not `concurrent.futures.TimeoutError`).

## Relaunch & verification plan

`--recreate` relaunch of `pd-disagg-mimo-uncensored-exl3-4x` with the fix
pulled on all 4 nodes (HEAD 7487dce confirmed). Battery:

1. Engines healthy (8100/8200), router `:8000` binds after both healthy.
2. Small + **55k-scale** chat completion through litellm
   (`cluster/mimo-v2.6-flash-rl-uncensored-exl3`) → 200.
3. Router PD counters increment with **litellm-sourced** traffic (not only
   operator probes); no `num_requests_waiting` growth at running=0.
4. 5-minute soak with concurrent hindsight-scale prompts; no
   `max_parallel_requests` leak; spend logs show `success` for the
   cluster group.
