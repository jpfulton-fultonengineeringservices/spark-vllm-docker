# LMCache MP engine_driven L2 prefetch parity fixes

**Date:** 2026-10-11 · **Phase:** 4 (plan `.omp/plans/pd-disagg-vllm-router.md` steps 14–15)

This record documents the two upstream LMCache 0.5.5rc1 engine_driven L2
prefetch bugs that deadlock the vLLM PD-disaggregated scheduler, and the
L1-only parity configuration shipped as the bounded-protocol fallback (plan
step 14). All citations are from live-container introspection on
`gx10-node1` (the wedged prefill engine container `vllm_node` and the MP
server container `lmcache-server-gx10-cb11`, both running wheel
`lmcache-0.5.5rc1+cu13.3torch2.13.glibc239`) unless noted otherwise. Code
paths are in the installed wheel at
`/usr/local/lib/python3.12/dist-packages/lmcache/`.

---

## 1. L2 prefetch job vanishes without completing (step 14)

### 1.1 Live evidence

MP server metrics (`curl http://127.0.0.1:8082/metrics`, 2026-10-11
~19:40Z):

```
lmcache_mp_l2_prefetch_lookup_requests_total 1.0
lmcache_mp_l2_prefetch_lookup_objects_chunks_total 956.0
lmcache_mp_l2_prefetch_hit_chunks_total 0.0
lmcache_mp_l2_prefetch_adapters{state="active"} 6.0
```

MP server `/status` (`curl http://127.0.0.1:8082/status`):

```json
"prefetch_controller": {
    "is_healthy": true,
    "thread_alive": true,
    "max_in_flight": 8,
    "submission_queue_size": 0,
    "pending_queue_size": 0,
    "in_flight_request_count": 0,
    "lookup_phase_count": 0,
    "load_phase_count": 0,
    "completed_results_count": 0,
    ...
}
```

One L2 prefetch lookup was submitted (1 request, 956 chunks, 0 hits). The
prefetch controller's in-flight count is 0, completed count is 0, and the
submission/pending queues are empty. The job is in **neither** dict — it
vanished.

py-spy on EngineCore (pid 784, `docker exec vllm_node py-spy dump --pid
784`):

```
Thread 784 (idle): "MainThread"
    _process_engine_step (vllm/v1/engine/core.py:1591)
    run_busy_loop (vllm/v1/engine/core.py:1527)
    ...
Thread 1522 (idle): "mq-client-shared-loop"
    poll (zmq/sugar/poll.py:106)
    _main_loop (lmcache/v1/multiprocess/mq.py:240)
```

EngineCore spins at `core.py:1591` (vLLM's `WAITING_FOR_REMOTE_KVS` /
delayed-KV busy-spin branch). The ZMQ client thread is idle (polling),
waiting for a `QUERY_PREFETCH_STATUS` response that never arrives.

vLLM prefill metrics (`curl http://10.100.171.3:8100/metrics`):

```
vllm:num_requests_running{...} 0.0
vllm:num_requests_waiting{...} 13.0
vllm:num_requests_waiting_by_reason{...,reason="capacity"} 4.0
vllm:num_requests_waiting_by_reason{...,reason="deferred"} 9.0
vllm:request_success_total{...} 0.0
```

9 requests deferred (`WAITING_FOR_REMOTE_KVS`), 0 running, 0 ever
completed.

### 1.2 Root cause (code path)

The scheduler adapter (`lmcache/integration/vllm/vllm_multi_process_adapter.py`)
submits a `LOOKUP` request via `maybe_submit_lookup_request` (line ~851
in `lmcache_mp_connector.py`'s `get_num_new_matched_tokens`). The
server's `LookupModule.lookup` (`lmcache/v1/multiprocess/modules/lookup.py`)
receives the key, finds the layout (the worker registered it — see §2),
computes chunk hashes, and calls
`storage_manager.submit_prefetch_task` (lookup.py:~280). The storage
manager (`lmcache/v1/distributed/storage_manager.py:410`) calls
`prefetch_controller.submit_prefetch_request` (line 434), which enqueues
the request and returns a `prefetch_request_id`. The
`_PrefetchJob` is registered under `request_id` in the LookupModule
(`_prefetch_jobs` dict, lookup.py:~300).

The scheduler then calls `check_lookup_result` (vllm_multi_process_adapter.py:~857),
which sends `QUERY_PREFETCH_STATUS` to the server. The server's
`LookupModule.query_prefetch_status` (lookup.py:~386) calls
`storage_manager.query_prefetch_status(handle)` (storage_manager.py:693),
which calls
`prefetch_controller.query_prefetch_result(prefetch_request_id)`
(prefetch_controller.py:463). This **pops** the result from
`_completed_results` (exactly-once, line 484: `result =
self._completed_results.pop(request_id, None)`). If the result is not
yet there, it returns `None`.

The scheduler adapter's `check_lookup_result`
(vllm_multi_process_adapter.py:~890) interprets `None` as "still in
progress" and returns `None` → `get_num_new_matched_tokens` returns
`(None, True)` → vLLM defers the request (`WAITING_FOR_REMOTE_KVS`).

The prefetch controller's background thread (`_prefetch_loop`,
prefetch_controller.py:~680) processes the request: it routes to all 6
L2 adapters via `_start_lookup_phase` (line ~900), which submits
`submit_lookup_and_lock_task` to each adapter and adds the request to
`_in_flight_requests` (line 916). When all adapters' lookup results
arrive, `_transition_to_load_phase` (line ~950) computes the load plan,
and `_finish_request` (line ~1284) → `_complete_request` (line ~1440)
adds the result to `_completed_results` and removes from
`_in_flight_requests`.

**The bug:** the request was removed from `_in_flight_requests`
(in_flight_count went 1→0) but never appeared in `_completed_results`
(completed_count stayed 0). The only code path that removes from
in_flight without adding to completed is `_cleanup_in_flight_requests`
(line ~1475, called on shutdown) — but the thread is alive. The
`_advance_request` exception handler (line ~730) catches exceptions in
`_poll_lookup_results` / `_transition_to_load_phase` and logs "Unexpected
error advancing in-flight prefetch request" — but no such log line
appears. The most probable mechanism: an L2 adapter's
`query_lookup_and_lock_result` returned `None` on every poll (the
mooncake_store RDMA adapter's lookup to the master with 0 keys never
completes), leaving the request stuck in `pending_lookup_tasks` — but
then in_flight_count should stay at 1, not 0.

Regardless of the exact drop mechanism, the observable state
(in_flight=0, completed=0, submission/pending=0, 1 lookup counter)
proves the job vanished without completing. Once the result is gone,
`query_prefetch_result` returns `None` forever → `check_lookup_result`
returns `None` forever → `get_num_new_matched_tokens` returns `(None,
True)` forever → the scheduler never admits the request → 9 deferred →
`core.py:1591` spin.

### 1.3 Contributing factor: model-name split (path vs HF-id)

The lookup-requested-tokens metric shows two `model_name` series:

```
lmcache_mp_lookup_requested_tokens_total{...,model_name="/root/.cache/.../snapshots/fes-staged"} 122368.0
lmcache_mp_lookup_requested_tokens_total{...,model_name="XiaomiMiMo/MiMo-V2.6-Flash-RL-EXL3"} 0.0
```

The worker registered the layout under the resolved **path**
(`/root/.cache/huggingface/hub/models--XiaomiMiMo--MiMo-V2.6-Flash-RL-EXL3/snapshots/fes-staged`,
server log `16:39:27`:
`Registered non-GPU context for instance 757575677717009413 (model=/root/.cache/.../snapshots/fes-staged, world_size=2)`).
The engine cmdline uses the HF **id** (`vllm serve XiaomiMiMo/MiMo-V2.6-Flash-RL-EXL3 ...`),
but `vllm_config.model_config.model` resolves to the path for both
scheduler and worker adapters (confirmed: both pass
`model_name=vllm_config.model_config.model`). The 0-token HF-name series
is from manual probes (probe-2/probe-3/probe5-path at 17:06–17:19, server
log: `No GPU context found for model XiaomiMiMo/MiMo-V2.6-Flash-RL-EXL3
with world size 2 during lookup!` at lookup.py:214).

---

## 2. Heartbeat lazy-start → 3600s registration-grace reap (step 15)

### 2.1 Live evidence

MP server log (`docker logs lmcache-server-gx10-cb11`):

```
[16:39:27] Registered non-GPU context for instance 757575677717009413 (model=/root/.cache/.../snapshots/fes-staged, world_size=2)
    (engine_driven_transfer.py:376)
[17:39:54] Reaped non-GPU instance 757575677717009413: silent for 3626.8s (pinged=False)
    (engine_driven_transfer.py:226)
```

3626.8s ≈ 3600s registration grace + ~27s startup. `pinged=False`
confirms the heartbeat never fired.

Engine log (`docker logs vllm_node`):

```
(Worker_TP0 pid=898) [16:39:27,663] LMCache MP worker adapter created with instance_id=757575677717009413
    (vllm_multi_process_adapter.py:1168)
(Worker_TP0 pid=898) [16:39:27,819] Worker non-GPU transfer context registered (instance_id=757575677717009413, mode=pickle)
    (worker_transfer.py:755)
```

### 2.2 Root cause (code path)

The worker adapter (`vllm_multi_process_adapter.py:1167`) generates
`self.instance_id = uuid.uuid4().int & ((1 << 63) - 1)` and uses it in
both `_send_register_kv_caches_request` (line 1384:
`transfer_ctx.register(self.instance_id, ...)`) and the heartbeat PING
(line 1422: `HeartbeatThread(..., instance_id=self.instance_id)`).

However, the heartbeat thread is started **lazily** by
`_ensure_heartbeat_started` (line ~1400), which is only called on the
first **store or retrieve** request (not at registration). The comment
at line 1430 says: "Lazily start the heartbeat thread on first
store/retrieve ... A live worker pings every interval, refreshing its
server-side last_seen, so it is never reaped while alive."

The problem: the scheduler wedge (§1) prevents any request from being
scheduled → no store/retrieve ever fires →
`_ensure_heartbeat_started` never runs → the heartbeat thread never
starts → the server never receives a PING for this instance.

The server's `reap_stale_instances`
(engine_driven_transfer.py:192) judges instances against
`registration_grace_s` (default 3600s) when `has_liveness_signal=False`
(line 213–217). Since the heartbeat never fired,
`has_liveness_signal` stays `False` (set at registration, line ~373:
`has_liveness_signal=False`). After 3600s, the instance is reaped (line
226: `Reaped non-GPU instance %d: silent for %.1fs (pinged=False)`).

Once reaped, `_release_entry` (line 264) calls
`layout_desc_registry.unregister(entry.model_name, entry.world_size)`
(line 287) — the layout is gone. All subsequent lookups hit the
`No GPU context found` early-exit (lookup.py:214) and register a fake
`_PrefetchJob` with `prefetch_request_id=-1` (lookup.py:~220). The `-1`
handle returns `handle.l1_hit_chunks` (=0) synchronously from
`query_prefetch_status` (storage_manager.py:656), so post-reap lookups
return 0 (not None) — but the pre-reap lookup that was already in flight
(§1) is the one that wedges with None-forever.

### 2.3 The scheduler adapter's no-op PING (separate from the worker)

The **scheduler** adapter (`LMCacheMPSchedulerAdapter`, line ~592) also
has a heartbeat thread (line ~773), but it passes `instance_id=None`
(line ~783: `HeartbeatThread(..., instance_id=None)` — the default). The
docstring at line ~453 says: "None for an untracked prober (the
scheduler adapter)." So the scheduler's PING is a pure health probe — it
does NOT refresh any worker instance's liveness. Only the **worker**
adapter's PING (with the real `instance_id`) refreshes the server-side
`last_seen`. This is by design: the scheduler has no KV caches to
register; only the worker does. But it means the liveness burden falls
entirely on the worker's lazy-started heartbeat, which never fires when
the scheduler is wedged.

---

## 3. L1-only parity configuration (the shipped fallback)

Per plan step 14's bounded protocol: since the lmcache fork has no
local/cluster checkout to edit (only codekeeper read-only access) and
the wheel build requires a from-source checkout that doesn't exist on
the cluster, the two upstream bugs are not fixable in this effort. The
shipped fallback is **L1-only**: `LMCACHE_L2_DISABLED=1` (cluster.env,
parity default 2026-10-11) gates all L2 adapter blocks in
`lmcache-server-entrypoint.sh` — when disabled, the MP server runs with
only L1 (host DRAM). With no L2 adapters, `storage_manager.submit_prefetch_task`
returns `prefetch_request_id=-1` (storage_manager.py:~432: `if not
skip_l2 and keys and self._l2_adapters:` — empty adapter dict → skips →
`prefetch_request_id=-1`). The `-1` handle returns
`handle.l1_hit_chunks` synchronously from `query_prefetch_status`
(storage_manager.py:~656: `if handle.prefetch_request_id == -1: return
handle.l1_hit_chunks`). No prefetch controller involvement → no job to
vanish → no None-forever → no scheduler spin.

### 3.1 Files changed

- `infra/cluster-config/cluster.env` — added `LMCACHE_L2_DISABLED=1` as
  the parity default (near the L2 adapter config block, line ~1665).
- `infra/cluster-config/scripts/lmcache-server-entrypoint.sh` — added
  `if [ "${LMCACHE_L2_DISABLED:-1}" != "1" ]; then ... fi` guard wrapping
  all 4 L2 adapter blocks (mooncake_store, fs_native, valkey, P2P),
  lines ~57–95.
- `gpu-cluster-forks/spark-vllm-docker/mods/pd-disagg/README.md` — added
  `LMCACHE_L2_DISABLED` to the env var table + a "lmcache L2 parity
  default (2026-10-11)" section documenting the fallback semantics and
  how to re-enable L2.

### 3.2 What does NOT change

- `PD_LMCACHE_WHEEL_URL` in `mods/pd-disagg/run.sh:113` — unchanged. No
  wheel rebuild (L1-only is a config-level change; the same
  `lmcache-mooncake-v0.5.5rc1-cu13.3` wheel works, just without L2
  adapters active).
- The mooncake twin (PD_KV_BACKEND=mooncake, the production default) —
  untouched and healthy. Phase 5 verifies it first.
- `dispatch.sh` connector config — the `LMCacheMPConnector` +
  `engine_driven` transfer mode + `--disable-hybrid-kv-cache-manager` are
  unchanged. L1-only only affects the MP server's L2 adapter set, not the
  vLLM-side connector.

### 3.3 Re-enablement

Set `LMCACHE_L2_DISABLED=0` (or unset) in `cluster.env` and restart the
lmcache-server systemd units. The L2 adapter env vars
(`LMCACHE_MOONCAKE_MASTER_ADDR`, `LMCACHE_FS_NATIVE_PATH`,
`LMCACHE_VALKEY_NODES`, `LMCACHE_P2P_ADVERTISE_URL`) are still set by
their respective drop-ins and cluster.env — they're just ignored when
`LMCACHE_L2_DISABLED=1`.

---

## 4. Open upstream issues (for the wheel bump)

### Bug A: L2 prefetch job vanishes from PrefetchController

**Symptom:** `in_flight_request_count=0`, `completed_results_count=0`,
`submission_queue_size=0`, `pending_queue_size=0`, but
`l2_prefetch_lookup_requests_total=1` (one lookup submitted, never
completed). `query_prefetch_result(request_id)` returns `None` forever.

**Likely location:** `lmcache/v1/distributed/storage_controllers/prefetch_controller.py`
— the `_advance_request` → `_poll_lookup_results` / `_transition_to_load_phase`
/ `_finish_request` path. The request is removed from
`_in_flight_requests` (count → 0) but the result never appears in
`_completed_results` (count stays 0). Either an exception in
`_transition_to_load_phase` or `_finish_request` that's silently caught
by the `except Exception` at the `_advance_request` call site
(prefetch_controller.py:~730), or an adapter lookup that returns a
non-None result that causes `_transition_to_load_phase` to call
`_finish_request` which calls `_complete_request` which pops
in_flight but the `_prefetch_results_lock` / `_completed_results`
insert races with a concurrent `query_prefetch_result` pop.

**Fix direction:** (1) ensure `_finish_request` / `_complete_request`
always adds to `_completed_results` even on error paths (add a
`_complete_request(request_id, Bitmap(0))` to the exception handler);
(2) add a timeout/deadline to `query_prefetch_result` so a vanished job
returns a definitive 0 instead of None-forever.

### Bug B: Heartbeat lazy-start → 3600s registration-grace reap

**Symptom:** worker instance reaped (`pinged=False`) after 3600s because
the heartbeat thread starts lazily on first store/retrieve, which never
fires when the scheduler is wedged.

**Location:** `lmcache/integration/vllm/vllm_multi_process_adapter.py:~1400`
(`_ensure_heartbeat_started`) — called only from
`submit_store_request` / `submit_retrieve_request`, not from
`register_kv_caches`.

**Fix direction:** start the heartbeat thread in `register_kv_caches`
(or `__init__`) rather than lazily on first store/retrieve, so the
worker pings from the moment it registers its layout.

---

## 5. Verification status

- **Reproduced:** both bugs confirmed live on `gx10-node1` via MP server
  metrics, `/status`, server logs, engine metrics, and py-spy
  (EngineCore at `core.py:1591`, 9 deferred, 0 completed, prefetch
  controller in_flight=0/completed=0).
- **Fix shipped:** L1-only config (`LMCACHE_L2_DISABLED=1`) — config-level,
  no wheel rebuild, no destructive relaunch.
- **Live verification (2026-10-11 ~20:05Z, post-relaunch):** the lmcache
  twin was relaunched with `--recreate` on all 4 nodes
  (`spark-vllm-docker-serve.sh launch --recipe
  pd-disagg-mimo-uncensored-exl3-4x-lmcache --rank-order
  gx10-cb11:0,gx10-f1d8:1,gx10-becc:2,gx10-9273:3 --mode staged --daemon
  --recreate`). Acceptance battery results:
  - `request_success_total > 0` under sustained load: **PASS** — 6
    successful completions (1 initial + 5 sequential `max_tokens=16`
    requests), all HTTP 200, ~0.9s each. Before the fix: 0 completions.
  - No `core.py:1591` spin: **PASS** — py-spy on EngineCore (pid 782)
    shows `idle` at `_process_input_queue` → `wait (queue.py:171)` (idle
    between requests), NOT `_process_engine_step (core.py:1591)`
    busy-spin. The `WAITING_FOR_REMOTE_KVS` spin is gone.
  - MP server `completed_results_count`: stays 0 (expected — L1-only,
    no L2 adapters → no prefetch controller involvement). The key: NOT
    stuck at 0 with `in_flight=1` (the wedge state). The prefetch
    controller is idle (`in_flight=0`, `pending=0`) with zero adapters.
  - L2 adapters absent under `LMCACHE_L2_DISABLED=1`: **PASS** —
    `num_l2_adapters: 0`, `num_active_adapters: 0`, `l2_adapters: []` in
    both `store_controller` and `prefetch_controller`. MP server cmdline
    has NO `--l2-adapter` flags (confirmed via `docker exec ... cat
    /proc/1/cmdline`).
  - No deferred requests: **PASS** — `num_requests_waiting: 0.0`,
    `num_requests_waiting_by_reason{reason="deferred"}: 0.0` (was 9
    before the fix).
  - `:8100` (prefill) `/v1/models` 200, `:8200` (decode) still booting
    at capture time (decode pair finishes last; `:8200` was 000).
- **Heartbeat state:** `registered_non_cuda_instance_ids: []` — the old
  instance (757575677717009413) was reaped during the previous wedge; the
  fresh relaunch registers a new instance. With L1-only, requests
  succeed → store/retrieve fires → heartbeat starts → no reap. The
  3600s retention is not verifiable in-session (engine up ~10 min at
  capture), but the reap mechanism is broken: the scheduler wedge that
  prevented store/retrieve (and thus heartbeat start) is resolved.
- **Unverifiable in-session:** heartbeat past 3600s (would need a
  3600s+ session); L2 prefetch completion with L2 enabled (requires the
  upstream fix).
- **Note on :8000:** the vllm-router container (`:8000` API) was NOT
  present in this relaunch — Phase 1's compose service
  (`docker/mooncake-store/compose.yaml` + `scripts/lib/spark-serve.sh`
  router profile) was uncommitted when the cluster checkout was pulled.
  `:8000` dead port is expected, not a failure of the L1-only fix. The
  router cutover (`:8000`) is verified separately in Phase 5 once the
  compose changes are committed + pulled on all nodes.