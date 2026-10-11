# Diagnosis: "no litellm traffic hitting the cluster" — prefill EngineCore deadlock in Mooncake store lookup

**Date:** 2026-10-11 ~20:37–20:53Z · **Reporter symptom:** no inference traffic reaching the GX10 cluster after the vllm-router cutover + reconcile.
**Verdict:** traffic *does* reach the cluster; the **prefill rank's EngineCore is deadlocked inside `MooncakeStoreConnector`'s KV-store lookup**, so every request enqueues and never schedules. Not a litellm, router, or network fault.

## Evidence chain (what is NOT the cause)

| Layer | Check | Result |
|---|---|---|
| Route discovery | `cluster_route_manager.health_check` + `list_cluster_routes` | `cluster/mimo-v2.6-flash-rl-uncensored-exl3` **alive**, api_base `http://10.10.10.165:8000`, served `XiaomiMiMo/MiMo-V2.6-Flash-RL-EXL3`, 8 ms |
| litellm routing config | `GET :4000/model/info` | `→ http://10.10.10.165:8000/v1`, `max_parallel_requests=4` — **correct** |
| jev_router decisions | bounded `docker logs litellm` | `cluster/mimo-...-uncensored` chosen ~1/min (source `hindsight-cluster`), latest 20:43:38 — **decision path working** |
| litellm deploy row | `LiteLLM_ProxyModelTable` | re-created **20:41:54** (the user's reconcile) — fresh, non-stale |
| Router container | `docker logs vllm-router`, `:8000/v1/models` | `Router ready | workers: [prefill :8100, decode :8200]`, health checker running |
| Router PD counters | `:29000/metrics` | `pd_requests_total=5` — **only the 5 operator probes** (my 4 battery + 1 test); zero hindsight traffic ever arrived at the router |
| Engine health | `:8100`/`:8200` `/health`, `/v1/models` | **both 200** (HTTP frontends perfectly healthy) |
| Mooncake store | `mooncake-master` admin metrics | `state=serving, service_ready=true, Clients: 4, Keys: 15445` — store itself healthy |

## The actual fault (prefill rank, node1 / gx10-cb11, :8100)

- `vllm:num_requests_running = 0`, `vllm:num_requests_waiting = 4` → grew to **5** after a direct probe; **never drains** (sampled 4→5 over ~3 min, running flat 0).
- `num_requests_waiting_by_reason{reason="capacity"}=2, {reason="deferred"}=2`.
- Engine scheduler log line **stops at 20:41:55** (the last `Engine 000: Running: 0 …`); GPU util 0%; container CPU 113% but **EngineCore thread 1.9%** (idle/blocked, not spinning), `VLLM::Worker_TP` 44%.
- Direct probe: `POST http://10.100.171.3:8100/v1/chat/completions` (1 token) — **hangs**; waiting 4→5, running stays 0.
- Router probe: `POST :8000/v1/chat/completions` {"hi"} → **HTTP 000, 15 s timeout**; pd counters unchanged (request never cleared the prefill scheduler).

**py-spy (`vllm_node`, EngineCore pid) — the smoking gun:**

```
Thread 745 (idle): "MainThread"
    wait (threading.py:355)
    result (concurrent/futures/_base.py:451)
    lookup (vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py:2528)
    get_num_new_matched_tokens (vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/scheduler.py:112)
    get_num_new_matched_tokens (vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/connector.py:191)
    schedule (vllm/v1/core/sched/scheduler.py:1390)
    step_with_batch_queue (vllm/v1/engine/core.py:739)
    run_busy_loop (vllm/v1/engine/core.py:1527)
```

The vLLM Scheduler calls the Mooncake store connector's `get_num_new_matched_tokens` **synchronously on the scheduler path**; that submits a KV-store `lookup()` and calls a **`concurrent.futures` `result()` with no timeout**. The future never resolves → the scheduler blocks mid-`schedule()` → the EngineCore busy loop is wedged → **no request is ever admitted**.

Corroborating: `mooncake-master` shows a steady `ExistKey` storm (~33–40 req/s) with **zero `Get`/`Put`/`PutStart`** — the store is being polled/queried but no KV bytes move, i.e. the lookup path is stuck retrying.

## Why the small e2e battery passed but real traffic fails

The Phase 5 battery (4× tiny prompts at 20:24) completed before the first *real* (hindsight extraction, 18–35k-token, remote-decode) request arrived at ~20:41. The deadlock is triggered once the store lookup path is exercised under real load; tiny probes completed because the connector returned quickly (store empty / no match). The router container, the route registration, and the litellm decision path were all fine throughout.

## Cascade to "no traffic / OpenRouter 402"

1. 4 hindsight requests dispatch → all wedge in the prefill scheduler → never complete → litellm's `max_parallel_requests=4` slots stay full.
2. New cluster request → `litellm.RateLimitError: Deployment has all max_parallel_requests slots in use`.
3. `Available Model Group Fallbacks=None` for `cluster/mimo-...-uncensored-exl3` → falls back to OpenRouter.
4. OpenRouter pool is out of credits → `402 Insufficient credits` → the request surfaces as an OpenRouter error, **masking** the fact that the real problem is the on-cluster prefill deadlock.

## Recommended next step (not yet actioned — read-only diagnosis)

The fix is **engine-side**, not litellm/router: the prefill rank's `MooncakeStoreConnector` synchronous `lookup()` must not block the scheduler (add a timeout / make `get_num_new_matched_tokens` non-blocking, or disable the store-side prefix lookup on the prefill path). Short-term operator recovery is a `--recreate` relaunch of the mooncake twin once the connector is hardened — restarting litellm alone will not help, and the store itself (`mooncake-master`) is healthy so it does not need restarting.

**Open question:** was the earlier NIXL `--kv-connector nixl` router config intended to bypass the vLLM-side `MooncakeStoreConnector` entirely? Per `phase5-verification.md` the router ran `--kv-connector nixl` yet the engine also had its own `MooncakeStoreConnector` active (`kv_role=kv_producer`), so both a router-driven transfer path *and* a store lookup ran — the store lookup is the one that wedged.

*Citations: py-spy dump + `:8100`/`:8200`/`:29000` metrics + `vllm_node`/`vllm-router`/`mooncake-master` docker logs + `litellm` bounded logs + `LiteLLM_ProxyModelTable`/`LiteLLM_SpendLogs`, all sampled live 2026-10-11 20:37–20:53Z.*
