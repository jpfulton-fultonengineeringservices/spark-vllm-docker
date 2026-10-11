# PD-disaggregation failure evidence — MiMo-V2.6-Flash-RL-EXL3 on the GB10 cluster

Source: `/tmp/pd-plan/session.json` (the exported prior OMP session, 13 317 entries, 2026-10-10T09:58Z → 2026-10-11T18:48Z; 49 named sub-sessions). This file distills the **definitive failure evidence and the decision trail** that the long-running PD / Mooncake / LMCache debugging session converged on. Every claim is cited by entry id + timestamp (and sub-session when the entry is not in the top-level transcript); quoted material is a verbatim substring of the recorded transcript (whitespace elision inside `[...]` only). No paraphrase-from-memory. The session ended with its final remediation `ask` cancelled by the user (entry `4e287d57` @ `2026-10-11T18:48:16.213Z`, `toolResult`: `"Ask tool was cancelled by the user"`; the next assistant turn `d61ebcdc` is an `aborted` no-op) — so the *terminal* finding is the diagnosis in entry `f1272bee`, and the open remediation decision below is the question the plan must answer.

Conventions in this file:
- `entry <id> @ <ts> (session, role/kind)` — the citation anchor. Session is `top-level entries` or a sub-session name (e.g. `EngineStall`).
- `role` is `user` | `assistant` | `toolResult`; `kind` is `text` | `thinking` | `toolCall` | `toolResult`. `thinking` blocks are the assistant's private reasoning (cited as such, never passed off as user-facing prose).
- Inline quotes reuse the cited substring verbatim; the `> ` blockquotes are the exact recorded text.

---

## 1. Opaque symptoms and the converged root cause

### 1.1 What the user observed

- User (entry `eb701699` @ `2026-10-11T16:52:40.416Z` (top-level entries, user/text)):
> I see no completion POST calls in the logs at all...
- User (entry `9d6fa08b` @ `2026-10-11T16:49:58.998Z` (top-level entries, user/text)): `ok... test now`  — i.e. "validate serving is working".

### 1.2 Live symptom capture (assistant, re-opened the remote sessions)

Direct prefill POST hangs with **no handler log line** (entry `e16b619b` @ `2026-10-11T16:52:56.835Z` (top-level entries, assistant/text)):
> Decode is up. Testing now — and I'll check whether POSTs actually reach the proxy handlers this time:

py-spy on the prefill rank (entry `8695161d` @ `2026-10-11T16:54:21.373Z` (top-level entries, toolResult/text)):
> 4 Sl   VLLM::EngineCore\n    898 Rl   VLLM::Worker_TP0\n=== py-spy Worker_TP0 ===\nProcess 898: VLLM::Worker_TP0\nPython v3.12.3 (/usr/bin/python3.12)\n\nThread 898 (active): \"MainThread\"\n    sched_yield (vllm/distributed/utils.py:48)\n    wait (vllm/distributed/device_communicators/shm_broadcast.py:205)\n    acquire_read (vllm/distributed/device_communicators/shm_broadcast.py:806)\n    __enter__ (contextlib.py:137)\n    dequeue (vllm/distributed/device_communicators/shm_broadcast.py:901)\n    worker_busy_loop (vllm/v1/executor/multiproc_executor.py:1035)\n    worker_main (vllm/v1/executor/multiproc_executor.py:937)\n    run (multiprocessing/process.py:108)",
>   "stderr": "",
>   "durationMs": 175,
>   "truncated": false
> }

EngineCore sits in `vllm/v1/engine/core.py:1591`; rank0 burns 63 % CPU (entry `be6e7994` @ `2026-10-11T16:54:32.580Z` (top-level entries, toolResult/text)):
> {
>   "exitCode": 0,
>   "stdout": "=== EngineCore py-spy now (is it scheduling?) ===\nProcess 784: VLLM::EngineCore\nPython v3.12.3 (/usr/bin/python3.12)\n\nThread 784 (active): \"MainThread\"\n    _process_engine_step (vllm/v1/engine/core.py:1591)\n    run_busy_loop (vllm/v1/engine/core.py:1527)\n    run_with_fault_tolerance (vllm/v1/fault_tolerance/engine_core_sentinel.py:179)\n    run_engine_core (vllm/v1/engine/core.py:1468)\n    run (multiprocessing/process.py:108)\n    _bootstrap (multiprocessing/process.py:314)\n    _main (multiprocessing/spawn.py:135)\n    spawn_main (multiprocessing/spawn.py:122)\n=== rank0 CPU ===\n    PID %CPU STAT\n    898 63.0 Rl",
>   "stderr": "",
>   "durationMs": 159,
>   "truncated": false
> }

vLLM metrics: 32 waiting (23 capacity + 9 deferred), 0 running; model_name is the **resolved path** (entry `78fd39ce` @ `2026-10-11T12:58:47.535Z` (EngineStall, toolResult/text)):
> {
>   "exitCode": 0,
>   "stdout": "vllm:num_requests_running{engine=\"0\",model_name=\"/root/.cache/huggingface/hub/models--XiaomiMiMo--MiMo-V2.6-Flash-RL-EXL3/snapshots/fes-staged\"} 0.0\nvllm:num_requests_waiting{engine=\"0\",model_name=\"/root/.cache/huggingface/hub/models--XiaomiMiMo--MiMo-V2.6-Flash-RL-EXL3/snapshots/fes-staged\"} 32.0\nvllm:num_requests_waiting_by_reason{engine=\"0\",model_name=\"/root/.cache/huggingface/hub/models--XiaomiMiMo--MiMo-V2.6-Flash-RL-EXL3/snapshots/fes-staged\",reason=\"capacity\"} 23.0\nvllm:num_requests_waiting_by_reason{engine=\"0\",model_name=\"/root/.cache/huggingface/hub/models--XiaomiMiMo--MiMo-V2.6-Flash-RL-EXL3/snapshots/fes-staged\",reason=\"deferre

The 1591 line is exactly vLLM's `WAITING_FOR_REMOTE_KVS` / delayed-KV busy-spin branch (entry `30f6b76b` @ `2026-10-11T12:44:13.186Z` (top-level entries, assistant/text)):
> ng=0`, `kv_cache_usage_perc=0.0`, `waiting capacity=18 deferred=6` |
> 
> `py-spy` on EngineCore (pid 764) shows the smoking gun:
> 
> ```
> _process_engine_step (vllm/v1/engine/core.py:1591)
>     outputs: {0: <EngineCoreOutputs>}
>     model_executed: False        ← engine never executes a step
> run_busy_loop (core.py:1527)
> ```
> 
> that line is literally vLLM's *"no model execution happened but there is still scheduler work (e.g. **WAITING_FOR_REMOTE_KVS** or delayed KV connector frees)"* branch — the engine spins yielding the GIL forever. The deferral comes from `LMCacheMPConnector.get_num_new_matched_tokens` (connector line 770–781): it submits an MP lookup and returns `(None, True)` until the result resolves — the prefill (`kv_producer`) scheduler never gets past it.
> 
> ## Contributing defect I found in the mod
> 
> `dispatch.sh:187-201` sets `PD_PREFIX_CACHING=0` for the LMCache-MP path, documenting *"MP 

Symptom ledger (entry `30f6b76b` @ `2026-10-11T12:44:13.186Z` (top-level entries, assistant/thinking)):
> 8100) hangs on any request, even max_tokens=1.
> 5. Prefill metrics: num_requests_running=0, kv_cache_usage=0.0, waiting capacity=18 deferred=6/7.
> 6. EngineCore py-spy: in _process_engine_step, model_executed=False, has_requests true → the WAITING_FOR_REMOTE_KVS/delayed-KV busy-spin.
> 7. Only 27,904 KV tokens vs max_model_len=1,048,576 → 0.03x concurrency.
> 8. KV memory guard bypassed: VLLM_SKIP_KV_CACHE_MEMORY_CHECK=1.
> 9. Both engines report enable_prefix_caching=True (config log) but no prefix flag in cmdline — so it's the vLLM default True. dispatch.sh intended PD_PREFIX_CACHING=0 for LMCache-MP but that only strips --enable-prefix-caching (never sets --no-enable-prefix-caching); since vLLM defaults APC on, the LMCache-MP contract "prefix caching must be off" is NOT actually enforced.
> 
> That

Convergence: 9 deferred = `WAITING_FOR_REMOTE_KVS`; `0 completions` post-reboot (entry `194e47fa` @ `2026-10-11T17:22:31.851Z` (top-level entries, assistant/thinking)):
> 9 requests "deferred" (KV transfer blocked = WAITING_FOR_REMOTE_KVS) + 4 "capacity". request_success_total = 0! Engine has NEVER successfully served a request post-reboot. So the very first request deadlocked and all subsequent queued.
> 
> Now: 9 deferred = waiting for remote KV = the LMCache lookup path. They're stuck because check_lookup_result returns None forever → the connector's get_num_new_matched_tokens returns (None, True) → deferred.
> 
> Wait, but the metric showed 1 L2 prefetch lookup completed and 122k requested tokens / 0 hits. And active prefetch jobs = 0. So SOME lookup completed (0 hits), yet requests still deferred.
> 
> Hmm — actually the "deferred" reason with has_possible_prefill... Let me think about the actual vLLM code. In vLLM, num_requests_waiting_by_reason 'deferred' means the scheduler deferred them due to "blocked status" — which for the connector means WAITING_FOR_REMOTE_KVS or a connector blocking condition.
> 
> I've spent an enormous number of turns. Let me be decisiv

### 1.3 Converged root cause (the terminal finding)

The session's final diagnosis (entry `f1272bee` @ `2026-10-11T17:23:27.225Z` (top-level entries, assistant/text)):
> ## Validation result: **serving is NOT working** — your observation was correct
> 
> I reopened sessions and tested the live stack. Concrete evidence:
> 
> **Prefill engine is deadlocked:**
> ```
> vllm:num_requests_waiting{...reason="deferred"}  9   ← KV-transfer blocked
> vllm:num_requests_waiting{...reason="capacity"}  4
> vllm:num_requests_running                        0
> vllm:request_success_total                       0   ← never completed a single request
> ```
> 
> **Symptom:** `POST :8000` and `POST :8100` both accept the TCP connection, send, then hang to timeout with **zero handler log lines**. Decode `:8200` serves fine (200). So prefill is the broken role.
> 
> **Root cause (pinpointed via py-spy + LMCache server introspection):**
> 
> 1. The prefill `EngineCore` (pid 784) is stuck in the scheduler loop at `kv_cache_coordinator → get_num_new_matched_tokens → lmcache_mp_connector.py:853` — returning `None` forever, so the request is re-prepended and the loop spins.
> 2. `Worker_TP0` spins at 80% CPU in `shm_broadcast.acquire_read` (waiting for a broadcast the stuck scheduler never emits) → the TP pair desyncs.
> 3. The reason `get_num_new_matched_tokens` returns `None`: the server-side LMCache MP `query_prefetch_status` polls the L2 prefetch controller, which **never produces a result**:
>    ```
>    lmcache_mp_l2_prefetch_lookup_requests_total 1.0
>    lmcache_mp_l2_prefetch_hit_chunks_total      0.0
>    prefetch_controller: in_flight=0, pending=0, completed_results_count=0
>    ```
>    One L2 prefetch looked up 956 chunks, then **vanished** without completing → `query_prefetch_result` returns `None` forever → permanent scheduler deadlock.
> 
> This is **deterministic** — the earlier `--recreate` relaunch re-wedged on the identical stack after serving 5 requests, so it is not a stale-process or one-off.

Crucially the proxy was **exonerated** (entry `f1272bee` @ `2026-10-11T17:23:27.225Z` (top-level entries, assistant/text)):
> **Notably,** the better proxy you pointed me at (`infra/cluster-config/docker/pd-proxy/pd_proxy.py`) is already vendored byte-identical into `mods/pd-disagg/` (sha256 `fd2a5cb2…`) and *is* what's running — so the proxy is not the failure; the LMCache MP engine-driven prefetch path is.
> 
> Two things I want your call on before acting:

Localisation summary (entry `98d550f8` @ `2026-10-11T17:23:06.800Z` (top-level entries, assistant/thinking)):
> t a lot of effort on deep diagnosis. Let me step back.
> 
> The user's ask: "validate serving is working". Answer: NO, it is not. The prefill engine is wedged (0 completions, 13 waiting, scheduler spinning in LMCache lookup). I've localized it to the LMCache MP lookup deadlock in engine_driven mode: the server's L2 P2P prefetch job for the first lookup never completes (completed_results_count=0), and the engine's check_lookup_result polls forever returning None → scheduler spins → no request ever scheduled.
> 
> Actually wait — let me reconsider. The metrics showed `lmcache_mp_l2_prefetch_lookup_requests_total 1.0` and `lookup_objects_chunks_total 956` and `hit_chunks 0`. So one L2 lookup was submitted. But that could've been during warmup. The key issue: engine scheduler spins.
> 
> Let me reconsider whether my probes themselves created the mess. The engine requests are "deferred" — meaning vLLM de

---

## 2. LMCache MP engine_driven deadlock + the model-name (path vs HF id) mismatch

### 2.1 The scheduler busy-loop / no-deadline contract

From the `EngineStall` sub-session, the connector's `check_lookup_result` has no deadline for the `None` case (entry `a042e519` @ `2026-10-11T12:48:35.284Z` (EngineStall, assistant/thinking)):
> Excellent. Key findings so far:
> 
> **check_lookup_result** (lines ~856-930 in the printed range):
> - `if request_id not in self._pending_lookups:` → returns cached result or **0** — so if a lookup was never submitted (e.g., server unhealthy at submit), it returns 0 (proceed without cache). 
> - `if not self.is_healthy:` → returns 0 (gives up).
> - Sends QUERY_PREFETCH_STATUS to all unresolved servers, waits `fut.result(timeout=self._mq_timeout)` where mq_timeout = 300s.
> - **Crucially: `if r is None: continue` → `if len(per_server) < len(self._server_urls): return None`** — so if the server responds with None (job still in progress / not found?), the function returns None → vLLM keeps waiting. **There is NO deadline/counter for the None case** — the only escapes are:
>   1. The server eventually returns a non-None hit count.
>   2. The MQ call times out (300s) → marks server unhealthy → returns 0.
>   3. Heartbeat marks the server unhealthy → `is_healthy` False → returns 0.
> 
> So a lookup whose server-side prefetch job never completes (or whose session is gone → server responds None forever?) hangs check_lookup_result indefinitely **as long as the MQ PING heartbeat keeps succeeding** (because the heartbeat uses the same mq_client... wait, actually the heartbeat uses its own mq_client from `self.

And the server-side `lookup` handler logs nothing on the happy path — the error paths are the only signal (entry `3aac97e5` @ `2026-10-11T13:12:57.690Z` (top-level entries, toolResult/text)):
> Zero LOOKUP traffic in the MP server log at INFO: the LookupModule.lookup handler logs NOTHING on the happy path (lookup.py:180-302); it only logs errors for the `No GPU context found for model %s ... during lookup!` (lookup.py:219-224) and `No group layout descs found ... during lookup!` (lookup.py:317-326) early-exits — neither appears in 3h of logs. So the lookup DID register a job via the happy path (submit_prefetch_task), and that job never completed (query_prefetch_status → storage_manager.query_prefetch_status returns None forever → adapter returns None → engine returns (None, True))."
> 13|      ]
> 14|    },
> 15|    "H2": {
> 16|      "verdict": "CONFIRMED (capacity requests are unschedulable junk; max_tokens=1 IS still admitted, capacity is not the tiny-request blocker)",
> 17|      "details": [
> 18|        "Default max_tokens: vllm/entrypoints/openai/chat_completion/serving.py:310-318 c

Earlier (03:25Z) the engine-driven **store** path also failed outright (entry `6dea4c55` @ `2026-10-11T03:27:36.181Z` (top-level entries, user/text)):
> (Worker_TP0 pid=873) [2026-10-11 03:25:56,533] LMCache ERROR: Failed to submit async engine-driven store (async_engine_driven.py:318:lmcache.v1.multiprocess.transfer_context.async_engine_driven)
> (Worker_TP0 pid=873) Traceback (most recent call last):
> (Worker_TP0 pid=873)   File "/usr/local/lib/python3.12/dist-packages/lmcache/v1/multiprocess/transfer_context/async_engine_driven.py", line 207, in submit_store
> (Worker_TP0 pid=873)     full_block_ids = _single_group_block_ids(block_ids)
> (Worker_TP0 pid=873)                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
> (Worker_TP0 pid=873)   File "/usr/local/lib/python3.12/dist-packages/lmcache/v1/multiprocess/transfer_context/worker_transfer.py", line 180, in _single_group_block_ids
> (Worker_TP0 pid=873)     raise RuntimeError(
> (Worker_TP0 pid=873) RuntimeError: engine-driven transfer does not support hybrid KV cache groups
> (Worker_TP0 pid=873) [2

And the connector module failed to import at 00:08Z (entry `ac20f34c` @ `2026-10-11T00:08:05.954Z` (top-level entries, user/text)):
> (APIServer pid=462)   File "/usr/local/lib/python3.12/dist-packages/vllm/distributed/kv_transfer/kv_connector/factory.py", line 37, in loader
> (APIServer pid=462)     module = importlib.import_module(module_path)
> (APIServer pid=462)              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
> (APIServer pid=462)   File "/usr/lib/python3.12/importlib/__init__.py", line 90, in import_module
> (APIServer pid=462)     return _bootstrap._gcd_import(name[level:], package, level)
> (APIServer pid=462)            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
> (APIServer pid=462)   File "<frozen importlib._bootstrap>", line 1387, in _gcd_import
> (APIServer pid=462)   File "<frozen importlib._bootstrap>", line 1360, in _find_and_load
> (APIServer pid=462)   File "<frozen importlib._bootstrap>", line 1331, in _find_and_load_unlocked
> (APIServer pid=462)   File "<frozen importlib._bootstrap>", line 935, in _load_

### 2.2 The deadlock: no layout → fake `-1` prefetch job → `None` forever

The decisive mismatch finding (entry `314a9abe` @ `2026-10-11T17:15:46.147Z` (top-level entries, assistant/text)):
> Now the full picture. The server registered layout under `model=/root/.cache/.../fes-staged` (the resolved **path**), but the scheduler submits lookups under `model_name="XiaomiMiMo/MiMo-V2.6-Flash-RL-EXL3"` (the **name**). `find()` misses → "No GPU context found" → it registers a prefetch job with `prefetch_request_id=-1` → `query_prefetch_status` returns `None` forever → engine spins at scheduler.py:1390. Verifying the two model_name sources:

Key-insight reasoning (entry `047d8d16` @ `2026-10-11T17:18:31.725Z` (top-level entries, assistant/thinking)):
> e further, I've spent a LOT of turns. 
> 
> Actually wait — the key insight: server registered layout under the PATH; engine looks up by the NAME. Mismatch → no layout → "No GPU context found" → 0/None. That's the deadlock.
> 
> But hold on — this same code presumably worked before (the earlier successful runs). Actually no — the stack NEVER successfully served a completion through this proxy in this session (0 POSTs ever logged). And even the earlier "working" verification in my prior summary was on the OLD proxy/toy router implementation and older mods. The current pd-disagg + engine_driven mode may have never worked.
> 
> Given the enormous turn count, I should now: report findings clearly and take the pragmatic action. The fix location: either server-side (register layout under both name and path) or engine-side (use same model_name). Given this is a vendored/forked LMCache + custom pd-disagg mod, the practical workaround is to make the engine's lookup use the resolved model path OR register layout under the name too.
> 
> But I'm not certain of the exact orchestration direction, and the user said "test now" and "I see no completion POST calls". The user wants to know it works. I've found the precise blocker.
> 
> Let me do ONE more decisive check: confirm the mismatch is truly name vs path by g

The `no layout` → `PrefetchHandle(prefetch_request_id=-1)` → `None`-forever chain (entry `cf99bb44` @ `2026-10-11T17:14:32.839Z` (top-level entries, assistant/thinking)):
> Now I understand. When there's no layout registered, `lookup` (line 214) logs the ERROR and registers a prefetch job with a handle that always returns... The job is registered with `requested_tokens=0`. Then `query_prefetch_status` finds the job and calls `storage_manager.query_prefetch_status(job.handle)`. With prefetch_request_id=-1 (the "no layout" path), what does `query_prefetch_status` return?
> 
> Actually — the "no layout" path registers a job with `handle=PrefetchHandle(prefetch_request_id=-1, ...)`. Then `query_prefetch_status` will call storage_manager.query_prefetch_status with handle -1. If that returns None forever (never a real prefetch submitted), then the engine's `check_lookup_result` gets None forever → **infinite spin**.
> 
> That's the deadlock: **the LMCache cache server on node1 never had the model's KV cache layout registered**, so `lookup()` bails at line 214 ("No GPU context found") and registers a fake prefetch job whose status polls return None forever.
> 
> Why no layout? The layout is registered when the *worker* registers its KV cache with the server. That happened at 16:39:27 (node1 "Registered non-GPU context for instance 7575..."). But layout registration for 

The confirming metrics: one L2 prefetch looked up 956 chunks, 0 hits, `completed_results_count=0`; the lookup metric is labelled by the **path** (entry `e16c6264` @ `2026-10-11T17:20:32.468Z` (top-level entries, toolResult/text)):
> {
>   "exitCode": 0,
>   "stdout": "lmcache_mp_l2_prefetch_lookup_requests_total 1.0\nlmcache_mp_l2_prefetch_lookup_objects_chunks_total 956.0\nlmcache_mp_l2_prefetch_hit_chunks_total 0.0\nlmcache_mp_lookup_requested_tokens_total{cache_salt=\"\",model_name=\"/root/.cache/huggingface/hub/models--XiaomiMiMo--MiMo-V2.6-Flash-RL-EXL3/snapshots/fes-staged\"} 122368.0\nlmcache_mp_lookup_requested_tokens_total{cache_salt=\"\",model_name=\"XiaomiMiMo/MiMo-V2.6-Flash-RL-EXL3\"} 0.0\nlmcache_mp_lookup_hit_tokens_total{cache_salt=\"\",model_name=\"/root/.cache/huggingface/hub/models--XiaomiMiMo--MiMo-V2.6-Flash-RL-EXL3/snapshots/fes-staged\"} 0.0\nlmcache_mp_lookup_hit_tokens_total{cache_salt=\"\",model_na

Why the layout was missing: the worker instance was reaped (3600 s grace) because the heartbeat never pinged — the scheduler passes `instance_id=None`, a no-op liveness probe (entry `ab0e9ed6` @ `2026-10-11T13:50:06.873Z` (top-level entries, assistant/thinking)):
>  died. Actually the smoke gun was: worker never pinged (pinged=False) because worker heartbeat starts on first store/retrieve, and no request ever got scheduled. The scheduler adapter passes instance_id=None so its heartbeat pings are no-ops for liveness. So the worker instance got reaped after 3600s grace. After reap, lookups hit "No GPU context found" early-exit and return 0. But lookups registered BEFORE the reap hang forever.
> 
> The deeper question: why does the prefetch job never complete? EngineStall chain step 4: "query_prefetch_status → storage_manager.query_prefetch_status(job.handle), None FOREVER". The prefetch job is submitted to the PrefetchController. If the L2 adapter (mooncake_store p2p) is broken, the prefetch may never finish. The lmcache-server logs showed adapters attaching/detaching (7 p2p L2 adapters added). The mooncake master showed 0 keys, 0 puts/gets — meaning the L2 mooncake path is completely idle. So the prefetch job likely hangs trying to fetch from the moon

**Mechanism the session converged on:** the LMCache MP server registered the KV-cache layout under the worker's *resolved HF-local path* (`/root/.cache/huggingface/hub/models--XiaomiMiMo--MiMo-V2.6-Flash-RL-EXL3/snapshots/fes-staged`), but the engine/scheduler submits lookups keyed by the *HF id string* (`XiaomiMiMo/MiMo-V2.6-Flash-RL-EXL3`). `layout_desc_registry.find(model_name, world_size)` misses → `lookup.py:214`'s `"No GPU context found model %s world size %d during lookup!"` early-exit → a sentinel `_PrefetchJob(prefetch_request_id=-1, requested_tokens=0)` is registered → `query_prefetch_status` polls it → `storage_manager.query_prefetch_status(-1)` returns `None` forever → the engine's `check_lookup_result` returns `None` (no deadline/counter for the `None` branch) → `get_num_new_matched_tokens` returns `(None, True)` → the scheduler re-prepends the request and busy-loops in `core.py:1591` (the `WAITING_FOR_REMOTE_KVS` branch) → `Worker_TP0` spins in `shm_broadcast.acquire_read` waiting for a broadcast the dead scheduler never emits → TP-pair desync → **0 completions, deterministic re-wedge after `--recreate`**.

---

## 3. The earlier NIXL failures that motivated the Mooncake pivot

### 3.1 NIXL on GB10 — the concrete failures captured in-session

`NIXL_ERR_BACKEND` at `register_memory` (15:21Z, pre-pivot) (entry `22b507b9` @ `2026-10-10T15:21:46.787Z` (top-level entries, toolResult/text)):
> u/kv_connector.py\", line 65, in __init__\n(Worker_TP0 pid=750) ERROR 10-10 15:21:03 [multiproc_executor.py:1057]   File \"/usr/local/lib/python3.12/dist-packages/vllm/distributed/kv_transfer/kv_connector/v1/nixl/connector.py\", line 232, in register_kv_caches\n(Worker_TP0 pid=750) ERROR 10-10 15:21:03 [multiproc_executor.py:1057]   File \"/usr/local/lib/python3.12/dist-packages/vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py\", line 1743, in register_kv_caches\n(Worker_TP0 pid=750) ERROR 10-10 15:21:03 [multiproc_executor.py:1057]     self.nixl_wrapper.register_memory(descs, backends=self.nixl_backends)\n(Worker_TP0 pid=750) ERROR 10-10 15:21:03 [multiproc_executor.py:1057]   File \"/usr/local/lib/python3.12/dist-packages/nixl/_api.py\", line 429, in register_memory\n(Worker_TP0 pid=750) ERROR 10-10 15:21:03 [multiproc_executor.py:1057] nixl._bindings.nixlBackendError: NIXL_ERR_BACKEND\n(Worker_TP0 pid=750) ERROR 10-10 15:21:03 [multiproc_executor.py:1057]   File \"/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/model_runner.py\", line 815, in initialize_kv_cache\n(Worker_TP0 pid=750) ERROR 10-10 15:21:03 [multiproc_executor.py:1057]   File \"/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/kv_connector.py\", line 159, in get_kv_connector\n(

`Engine core initialization failed` after the UCX RCACHE tweak (entry `425f7ad6` @ `2026-10-10T14:18:40.378Z` (top-level entries, toolResult/text)):
> Setting UCX_RCACHE_MAX_UNRELEASED to '1024' to avoid a rare memory leak in UCX when using NIXL.\n(APIServer pid=397)     raise RuntimeError(\n(APIServer pid=397) RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}\n(EngineCore pid=602) INFO 10-10 14:18:08 [core.py:123] Initializing a V1 LLM engine (v0.1.dev21546+g502d6cb5a.d20261003) with config: model='/root/.cache/huggingface/hub/models--XiaomiMiMo--MiMo-V2.6-Flash-RL-EXL3/snapshots/fes-staged', speculative_config=SpeculativeConfig(method='dflash', model='/workspace/MiMo-V2.6-Flash-RL-dflash', num_s

The UCX half of the NIXL fix ships **only** inside the cu13.3 image; GB10 is UMA with no GPUDirect RDMA (entry `035dd615` @ `2026-10-10T14:06:45.447Z` (top-level entries, toolResult/text)):
> GB10 note: on this UMA platform the UCX half of the NIXL fix ships ONLY
>  34|# inside the `nixl-runtime:24.04-cu13.3-sm121` image (tmp/perf-concepts/
> ...
>  40|PREFIX="[pd-disagg]"
> *41|ENV_FILE="${PD_DISAGG_ENV_FILE:-/tmp/pd-disagg.env}"
>  42|DEFAULT_UCX_DEVS="rocep1s0f1:1,roceP2p1s0f1:1"
>  43|
>  44|fail() { echo "$PREFIX PD_DISAGG_FAIL $*" >&2; return 1; }
> ...
>  131|    echo "export PD_PARITY_SHA256=\"$PD_PARITY_SHA256\""
> *132|    echo "export PD_DISAGG_ENV_READY=1"
>  133|} > "$ENV_FILE.tmp.$$"
>  134|mv -f "$ENV_FILE.tmp.$$" "$ENV_FILE"
>  135|chmod 0644 "$ENV_FILE"

Operational guardrail: NIXL/UCX must **fail closed** rather than silently fall back to TCP-only (entry `9dcd7397` @ `2026-10-10T14:24:42.056Z` (top-level entries, toolResult/text)):
> PD_DISAGG_ALLOW_TCP_FALLBACK=1` |
> 67|| `PD_DISAGG_ALLOW_TCP_FALLBACK` | Downgrade missing-HCA to warning | `0` | run.sh | Fail-closed default; UCX/NIXL must not silently run TCP-only |
> 68|| `PD_DISAGG_ENV_FILE` | Env file sourced by dispatch.sh | `/tmp/pd-disagg.env` | Both | Must exist; **auto-published**: if missing, dispatch.sh sources the sibling `run.sh` in-process to publish it (then re-sources the env file) |
> 69|| `PD_NIXL_PORT` | NIXL side-channel port | `5600` | Both | Per-node, own IP 

Acceptance gate was defined as `nixl_bytes_transferred > 0` with zero failed transfers (entry `4d3d3dd7` @ `2026-10-10T14:24:44.135Z` (top-level entries, toolResult/text)):
> Acceptance gate (per pd-disagg README): NIXL nixl_bytes_transferred > 0
> 59|# with zero failed transfers, /v1/models reachable on 8000 (router), 8100
> 60|# (prefill), 8200 (decode); port probes with `ss -tln sport = :PORT`
> 61|# (/proc/net/tcp awk LISTEN fallback) -- NEVER `nc -z`, it steals one-shot
> 62|# NCCL rendezvous accepts.
> 63|#
> 64|# Required mods:
> 65|# - mods/fes-weights        staged-weight hub-cache shim (pack id, all ranks)
> 66|# - mods/mimo-v2.6-flash    DFlash drafter staging (from MIMO_V26_MODEL_ID)
> 67|# - mods/mimo-diffkv-fp8-kv DiffKV + fp8 KV layout fix
> 68|# - mods/kv-cache-guard-o

`MooncakeVsNIXL` sub-session's enumerated NIXL known-issues (entry `ffe830f4` @ `2026-10-10T17:03:55.848Z` (MooncakeVsNIXL, assistant/thinking)):
> Known issues: mis-transfer (vLLM issue #58470) and potential 100% CPU in the reclaim gate.
> 
>  Mooncake:
>  - Found in the `vllm` fork under `docs/features/mooncake_connector_usage.md` and implementation in `mooncake_connector.py`.
>  - Mooncake has two connectors: `MooncakeConnector` for point-to-point and `MooncakeStoreConnector` for a shared store.
>  - The point-to-point `MooncakeConnector` requires:
>      - Running a bootstrap server on the prefiller (default port 8998).
>      - Setting the KV role (`kv_producer` or `kv_consumer`).
>      - Optionally, a proxy to route requests.
>  - It uses RDMA (with the `nvidia-peermem` module) or falls back to DMA-BUF.
>  - Environment variables: `VLLM_MOONCAKE_BOO

An advisory pushed back on the speculative NIXL causal narrative — the remaining deltas were PD env (`UCX_TLS`, unset `PYTORCH_CUDA_ALLOC_CONF`, `VLLM_NIXL_SIDE_CHANNEL_*`), kv-transfer config, and 1 M `max_model_len` (entry `4ce812e7` @ `2026-10-10T14:45:22.725Z` (top-level entries, assistant/thinking)):
> identically... both prefill and decode ranks hang identically" — I did report both roles hang. But my causal narrative "NIXL KV-transfer producer retains buffers" was speculative; the advisor says the remaining deltas are PD env (UCX_TLS=^cuda_ipc, unset PYTORCH_CUDA_ALLOC_CONF, VLLM_NIXL_SIDE_CHANNEL_*), kv-transfer config, 1M max_model_len.
> 
> The advisor proposes a bisect: throwaway --solo serve (nnodes=1, TP=1) on node1 with the 4x recipe env — hang persists → PD env/allocator/kv config; clean → TP/multi-node interplay. That's a real experiment but costs a full boot cycle (~10 min) plus container management. Hmm. Actually a --solo serve without kv-transfer-config would drop the NIXL part. 

### 3.2 The Mooncake-vs-NIXL trade-off (sub-session `MooncakeVsNIXL`)

The comparison table the sub-session produced (entries `f27a33b6` / `ffe830f4` / `b2525cff` @ `2026-10-10T17:0xZ`):

| Aspect | Mooncake | NIXL |
|---|---|---|
| Complexity | Medium (bootstrap server/proxy) | High (UCX patches/runtime) |
| Reliability | Production-ready in vLLM | Edge cases with mis-transfers |
| Performance | Optimized RDMA/GPU-direct | Host-staged adds latency |
| GB10 support | DMA-BUF fallback available | Requires CUDA 13.3 patches |
| Operational | Standard Prom metrics | Limited monitoring |

---

## 4. The decision: abandon the Python re-implementation, adopt upstream `vllm-router`

### 4.1 The in-session proxy lineage (what was being replaced and why)

User directive that set the Mooncake pivot in motion (entry `7cd39bee` @ `2026-10-10T16:56:17.092Z` (top-level entries, user/text)):
> you will find in the prior art areas previous work with https://docs.vllm.ai/en/stable/features/mooncake_store_connector_usage/ 
> see @infra/dgx-spark-wheels/ https://github.com/Fulton-Engineering-Services/Mooncake/tree/cuda13.3-aarch64-gb10 
> 
> in this plan, we will work with mooncake and evaluate usage of ray
> ray and mooncake are both indexed in codekeeper
> there is evidence of ray usage in the spark-vllm submodule as well

The assistant's adopted Mooncake NIXL-replacement plan (entry `0b7cb311` @ `2026-10-10T17:07:51.676Z` (top-level entries, assistant/text)):
> finalized plan:
> 
> <file path="local://mooncake-nixl-replacement-plan.md">
> # Mooncake NIXL Replacement Plan
> 
> ## Goal
> Completely replace NIXL with Mooncake store connector as the KV-transfer mechanism in PD disagg pipeline on 4 GX10 nodes. Deliver production-ready code and documentation.
> 
> ## Approach
> 1. **Phase out NIXL:**
>    - Remove `mods/kv-cache-guard-override` refs from PD disagg recipes
>    - Delete NIXL-specific environment variables (NIXL_EP_USE_HOST_RDMA_BUFFER, UCX_TLS workarounds)
>    - Maintain PD disagg role assignment (prefill=producer, decode=consumer)
> 
> 2. **Mooncake integration:**
>    - Stand up Mooncake bootstrap server on port 8998 (sidecar service using DGX-Spark-Wheels implementation)
>    - Configure KV connectors in dispatch.sh:
>    bash
>    # Prefill role (producer)
>    --kv-transfer-config '{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_producer"}'
>    
>    # Decode ro

The store-only proxy decision (store connector needs no P2P `kv_transfer_params` handshake → proxy simplifies to round-robin) (entry `7c454e3b` @ `2026-10-10T17:44:13.120Z` (top-level entries, assistant/thinking)):
> only — prefill (kv_producer) PUTs KV to store, decode (kv_consumer) GETs from store. NO P2P side-channel, NO bootstrap port, NO toy_proxy NIXL handshake. But a proxy is still needed for XpYd routing (pd-proxy in cluster-config: PREFILL_ENDPOINT, DECODE_ENDPOINTS, PD_SERVE_PORT=9000)
> - **Store connector boot gate** (connector.py:102-127): rejects CrossAttentionSpec groups; MambaSpec must have block_size == cache_config.block_size; PCP/DCP >1 unsupported. MiMo: EXL3 attention (no CrossAttention? MiMo V2.6 has... need to check) — MiMo uses standard attention + sliding_window skip layers for KV dtype. DFlash spec-decode on decode side adds draft model KV — need to check if MambaSpec applies (MiM

Decision to replace `toy_proxy_server.py` with the `pd_proxy.py` pattern (entry `0b9532a3` @ `2026-10-10T17:42:38.632Z` (top-level entries, assistant/thinking)):
> replace toy_proxy_server.py with the pd_proxy.py pattern (or adapt). The Kilo plan says "pd-proxy (docker/pd-proxy/pd_proxy.py) FastAPI (:8998), ... DELETE --bootstrap-port and POST max_tokens=1, KEEP /health /v1/models /status, add/remove round-robin".
> 
>    Hmm, this is getting complex. The pd-proxy is a separate service (systemd unit on rank-0) rather than in-container router. The current MiMo PD uses an in-container router (toy_proxy on :8000). 
> 
>    Decision for the plan: The store-connector flow does NOT need the P2P kv_transfer_params handshake, so the proxy becomes a simpler round-robin router. Two options:
>    (a) Adapt toy_proxy_server.py to store-only mode (drop the do_remote_decode/d

### 4.2 The Python proxy was vendored — and it is not the failure

User pointed at the better implementation (entry `4f2144ef` @ `2026-10-11T16:24:30.710Z` (top-level entries, user/text)):
`/Users/j.patrickfulton/repos/fultonengineeringservices/fulton-infra-ecosystem/infra/cluster-config/docker/pd-proxy this is a better implementation`

Live inspection of what was actually running on :8000 (entry `94ec49c3` @ `2026-10-11T16:21:39.554Z` (top-level entries, toolResult/text)):
> {
>   "exitCode": 0,
>   "stdout": "=== who listens on 8000 ===\n=== router proc ===\n424 python3 /workspace/mods/pd-disagg/pd_proxy.py --model XiaomiMiMo/MiMo-V2.6-Flash-RL-EXL3 --prefill 10.100.171.3:8100 --decode 10.100.171.2:8200 --port 8000 --verify-timeout 1800\n=== decode node3 live? ===\nnode3:8200 http=200 t=0.001641s",
>   "stderr": "",
>   "durationMs": 150,
>   "truncated": false
> }

The vendored copy is **byte-identical** to the better implementation — so the proxy cannot be the defect (entry `1d2a280e` @ `2026-10-11T16:26:15.191Z` (top-level entries, assistant/text)):
> Key discovery: `mods/pd-disagg/pd_proxy.py` is already a **byte-identical vendored copy** of the better `infra/cluster-config/docker/pd-proxy/pd_proxy.py` (per README 180–182, sha256 `fd2a5cb…`). But the *running* proxy (pid 424, started 15:35:59) shows argv `--prefill 10.100.171.3:8100 --decode 10.100.171.2:8200` — matching the new proxy's flags — yet its log format looks like the old toy router. Let me verify which code is actually running vs the vendored copy:

### 4.3 The decision the plan inherits

Because the proxy is byte-identical and exonerated (§1.3, §4.2) and the defect is in the **LMCache MP `engine_driven` lookup path** (§2), the session's terminal `ask` (entry `f1272bee` @ `2026-10-11T17:23:27.225Z`, `assistant/toolCall`) put the remediation decision to the user with four options — *pin a stable lmcache wheel*, *switch `mp_transfer_mode` off `engine_driven`*, *relaunch-only*, or *deep-dive the LMCache bug first* — and that `ask` was cancelled by the user rather than answered (entry `4e287d57`). The **plan this evidence supports** is the decision the broader effort then made: **abandon maintaining the hand-written Python `pd_proxy.py` as the PD front-end and adopt the upstream Rust `vllm-project/router`** (codekeeper repo name `vllm-router`, clone `https://github.com/vllm-project/router.git`, indexed commit `91b65ee4bdfbe98863a32a2205dced5dbb249ffe`, status `READY` — documented in the sibling doc `docs/pd-router/upstream-router-contracts.md`). The Python proxy's KV-path contract (`pd_proxy.py:16-17`, "supersedes both Nixl-style `kv_transfer_params` extraction and the P2P `MooncakeConnector` transfer-id / bootstrap-server contract", byte-identical to `infra/cluster-config/docker/pd-proxy/pd_proxy.py`, sha256 `fd2a5cb2…79154d`, per `mods/pd-disagg/README.md:180-182`) is exactly the surface the Rust router replaces (see `docs/pd-router/kv-backends-l2.md` §5.6 and `docs/pd-router/upstream-router-contracts.md` §5).

> **Note on sourcing.** The literal strings `"vllm-router"`, `"Rust router"`,
> `"github.com/vllm-project/router"`, and `"abandon re-implementing"` do **not**
> appear in `session.json` (verified by whole-file substring search). The adoption
> decision lives in the *plan context* that this evidence feeds, not in the
> transcript. What the transcript *does* establish — and what this section cites —
> is (a) the Python proxy was the component being iterated on, (b) it was
> byte-identical to the "better" implementation the user pointed at, and (c) it was
> exonerated as the failure. That is the in-session evidence the abandonment
> decision rests on; the decision itself is the plan's, recorded here for the
> decision trail.

---

## 5. Open threads the plan must address

### 5.1 The cancelled remediation `ask` (the live open decision)

The terminal `ask` the user cancelled (entry `f1272bee` @ `2026-10-11T17:23:27.225Z` (top-level entries, assistant/toolCall)):
> Deciding remediation for prefill deadlock", "questions": [{"id": "fix", "header": "Remediation", "question": "Prefill deadlocks in the LMCache MP engine-driven L2 prefetch path (prefetch never completes \u2192 scheduler spins). How should I proceed?", "options": [{"label": "Pin a stable lmcache wheel", "description": "Change LMCACHE_WHEEL_URL in mods/lmcache-offload/run.sh from the 0.5.5rc1 dgx-spark-wheels build to a stable release, then relaunch. Addresses the bug at its source. ~15 min (relaunch + warmup)."}, {"label": "Switch mp_transfer_mode off engine_driven", "description": "Set lmcache.mp.mp_transfer_mode to lmcache_driven (or auto) in the recipe/dispatch, avoiding the buggy engine-driven scheduler poll. Fastest code change, needs relaunch."}, {"label": "Relaunch only (accept wedge risk)", "description": "Restart the stack as-is; it serves for a while then re-wedges under sustained traffic. No code fix."}, {"label": "Deep-dive the LMCache bug first", "description": "Continue instrumenting to confirm the exact prefetch-loss mechanism before changing anything."}]}]}

The four options on that `ask` are the concrete open remediation choices the plan must pick from:

1. **Pin a stable lmcache wheel** — change `LMCACHE_WHEEL_URL` in `mods/lmcache-offload/run.sh` off the `0.5.5rc1` `dgx-spark-wheels` build to a stable release, then relaunch (~15 min).
2. **Switch `mp_transfer_mode` off `engine_driven`** — set `lmcache.mp.mp_transfer_mode` to `lmcache_driven` (or `auto`) in the recipe/dispatch, avoiding the buggy engine-driven scheduler poll (fastest code change; needs relaunch).
3. **Relaunch only (accept wedge risk)** — restart as-is; serves for a while then re-wedges under sustained traffic (no code fix).
4. **Deep-dive the LMCache bug first** — keep instrumenting to confirm the exact prefetch-loss mechanism before changing anything.

### 5.2 The model-name (path vs HF-id) mismatch — the fix location is undecided

From entry `047d8d16` (§2.2): the session could not pin the exact orchestration direction for the mismatch fix — whether to register the layout under the *name* too, or make the engine's lookup use the *resolved path*. Quote:

> e further, I've spent a LOT of turns. 
> 
> Actually wait — the key insight: server registered layout under the PATH; engine looks up by the NAME. Mismatch → no layout → "No GPU context found" → 0/None. That's the deadlock.
> 
> But hold on — this same code presumably worked before (the earlier successful runs). Actually no — the stack NEVER successfully served a completion through this proxy in this session (0 POSTs ever logged). And even the earlier "working" verification in my prior summary was on the OLD proxy/toy router implementation and older mods. The current pd-disagg + engine_driven mode may have never worked.
> 
> Given the enormous turn count, I should now: report findings clearly and take the pragmatic action. The fix location: either server-side (register layout under both name and path) or engine-side (use same model_name). Given this is a vendored/forked LMCache + custom pd-disagg mod, the practical workaround is to make the engine's lookup use the resolved model path OR register layout under the name too.
> 
> But I'm not certain of the exact orchestration direction, and the user said "test now" and "I see no completion POST calls". The user wants to know it works. I've found the precise blocker.
> 
> Let me do ONE more decisive check: confirm the mismatch is truly name vs path by g

The plan must pick a side (server-side dual-register, or engine-side path-keyed lookup) and verify the LMCache MP server actually sees a layout under the key the engine queries.

### 5.3 Outstanding items from the 16:26Z todo snapshot

The live todo snapshot when the wedge was re-encountered (entry `edda83be` @ `2026-10-11T16:26:26.244Z` (top-level entries, toolResult/text)):
> Remaining items (6):
>   - Determine wedge cause: rank0 R-spin, rank1 S idle after 15:53:16 proxied POST 200 [in_progress] (diagnose)
>   - Relaunch stack fresh (full 4-node) with correct recipe [pending] (fix)
>   - If wedge recurs, root-cause prefill store path (L1 eviction interplay) [pending] (fix)
>   - E2E 200s head :8000 + direct prefill :8100 + decode :8200 under sustained traffic [pending] (verify)
>   - Zero hybrid/store errors; LMCache prefetch/store activity healthy [pending] (verify)
>   - Push any code changes, close sessions, final summary [pending] (close)
> Overall: 1/7 done, 6 open.
> Active phase 1/4 "diagnose" (1/2).
>   diagnose:
>     - [ ] Determine wedge cause: rank0 R-spin, rank1 S idle

### 5.4 Mooncake-era failures still on the table (the pivot did not re-verify end-to-end)

mooncake-store compilation termination (entry `0e776f39` @ `2026-10-10T21:45:10.797Z` (top-level entries, user/text)):
> jpfulton@gx10-cb11:/cluster-shared/projects/spark-vllm-docker$ tail -f /tmp/frshell-gx10-node1-user-25-job-3/stdout.log
>                  from /mctest/mc/mooncake-store/include/client_metric.h:17,
>                  from /mctest/mc/mooncake-store/include/client_service.h:17,
>                  from /mctest/mc/mooncake-store/include/pyclient.h:13,
>                  from /mctest/mc/mooncake-store/include/real_client.h:15,
>                  from csrc/storage_backends/mooncake/connector.h:15,
>              

Decoder `TCPStore` `recvValue failed` / `Broken pipe` — rank1 lost the master (entry `85c7c302` @ `2026-10-11T14:51:06.190Z` (top-level entries, user/text)):
> n the decoder:
> 
> readiness not observed within 10 min; skipping save (serve continues)
> [rank1]:[W1011 06:41:21.267831770 TCPStore.cpp:125] [c10d] recvValue failed on SocketImpl(fd=32, addr=[gx10-9273]:50116, remote=[::ffff:10.100.171.2]:29574): Failed to recv, got 0 bytes. Connection was likely closed. Did the remote server shutdown or crash?
> Exception raised from recvBytes at /tmp/wheel-build/src/pytorch/torch/csrc/distributed/c10d/Utils.hpp:787 (most recent call first):
> frame #0: c10::Error::Error(c10::SourceLocation, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >) + 0xe8 (0xf0e27b96ff68 in /usr/local/lib/python3.12/dist-packages/torch/lib/libc10.so)
> frame #

### 5.5 The five KV-group blockers the session claimed resolved (re-verify under the new router)

Blockers the session marked resolved (entry `dd2f2251` @ `2026-10-11T11:26:21.486Z` (top-level entries, assistant/text)):
> All blockers resolved (5 total) for the LMCache MP and offload paths:**
> 
> | # | Crash | Fix | Kind |
> |---|---|---|---|
> | 1 | `engine-driven transfer does not support hybrid KV cache groups` | `--disable-hybrid-kv-cache-manager` on both roles | recipe/dispatch.sh |
> | 2 | `get_kv_cache_groups` cudagraph crash from skip-layers referencing non-existent SWA groups after HMA-off | Condition skip-layers on HMA enabled, remove from LMCache recipe | dispatch.sh/recipe |
> | 3 | `B12X noncausal attention requires BF16 KV` in DFlash draft model | `--kv-cache-dtype bfloat16` on both roles when HMA-off | dispatch.sh |
> | 4 | `ValueError` in `is_kv_cache_spec_uniform` (probe catches only `AssertionError`) | `except (AssertionError, ValueError)` | kv_cache_utils fix |
> | 5 | `ValueError("Failed to promote...")` from `_promote_local_kv_cache_specs` with DFlash drafter mixed windows | Normalize `sliding_window=None` after promotion | kv_cache_utils fix + mod patches |

These were resolved **against the Python `pd_proxy.py` + `LMCacheMPConnector` engine_driven path**; the Rust-router cutover re-exposes the contract surface and must re-clear them against the upstream router's PD dispatch (`VllmPDRouter`, `upstream-router-contracts.md` §5).

### 5.6 Final committed state (the baseline the plan branches from)

Last assistant status before the cancelled `ask` (entry `d3d5701d` @ `2026-10-11T15:53:47.473Z` (top-level entries, assistant/text)):
> Fixes shipped: `f911c9c` (collapse multi-group block ids + boot-install xxhash) pushed to `fulton/mimo-v2.6-flash`; umbrella gitlink `408e5eb` pushed to `main`
> - 4-node stack relaunched and serving at `f911c9c`: E2E 200s on `:8000`/`:8100`/`:8200`, zero hybrid-group errors, zero failed LMCache stores, zero xxhash crashes across all ranks
> - Cleanup done: probe files removed, all 4 remote sessions closed
> 
> Nothing pending; no further work required.

---

## Citation index (every entry id cited above)

- `035dd615` @ `2026-10-10T14:06:45.447Z` — TOP / toolResult/text — §3 — GB10 UMA UCX fix ships only in cu13.3 image
- `047d8d16` @ `2026-10-11T17:18:31.725Z` — TOP / assistant/thinking — §2 — key insight name vs path
- `0b7cb311` @ `2026-10-10T17:07:51.676Z` — TOP / assistant/text — §4 — mooncake replacement plan adopted
- `0b9532a3` @ `2026-10-10T17:42:38.632Z` — TOP / assistant/thinking — §4 — store-only proxy design
- `0e776f39` @ `2026-10-10T21:45:10.797Z` — TOP / user/text — §5 — mooncake compile termination
- `194e47fa` @ `2026-10-11T17:22:31.851Z` — TOP / assistant/thinking — §1 — 13 stuck 0 completions reasoning
- `1d2a280e` @ `2026-10-11T16:26:15.191Z` — TOP / assistant/text — §4 — vendored byte-identical better proxy
- `22b507b9` @ `2026-10-10T15:21:46.787Z` — TOP / toolResult/text — §3 — NIXL_ERR_BACKEND at register_memory
- `30f6b76b` @ `2026-10-11T12:44:13.186Z` — TOP / assistant/text — §1 — WAITING_FOR_REMOTE_KVS busy-spin branch
- `30f6b76b` @ `2026-10-11T12:44:13.186Z` — TOP / assistant/thinking — §1 — 12:44 symptom ledger
- `314a9abe` @ `2026-10-11T17:15:46.147Z` — TOP / assistant/text — §2 — register-path vs lookup-name mismatch
- `3aac97e5` @ `2026-10-11T13:12:57.690Z` — TOP / toolResult/text — §2 — lookup.py error paths silent
- `425f7ad6` @ `2026-10-10T14:18:40.378Z` — TOP / toolResult/text — §3 — engine core init failed
- `4ce812e7` @ `2026-10-10T14:45:22.725Z` — TOP / assistant/thinking — §3 — advisor pushes back on speculative narrative
- `4d3d3dd7` @ `2026-10-10T14:24:44.135Z` — TOP / toolResult/text — §3 — NIXL acceptance gate bytes>0 zero failed
- `4f2144ef` @ `2026-10-11T16:24:30.710Z` — TOP / user/text — §4 — user points at better implementation
- `6dea4c55` @ `2026-10-11T03:27:36.181Z` — TOP / user/text — §5 — engine-driven async store failure 03:25
- `78fd39ce` @ `2026-10-11T12:58:47.535Z` — EngineStall / toolResult/text — §1 — metrics waiting 32/deferred 9 path model_name
- `7c454e3b` @ `2026-10-10T17:44:13.120Z` — TOP / assistant/thinking — §4 — pd-proxy vendoring decision
- `7cd39bee` @ `2026-10-10T16:56:17.092Z` — TOP / user/text — §4 — user directs mooncake prior art
- `85c7c302` @ `2026-10-11T14:51:06.190Z` — TOP / user/text — §5 — decoder TCPStore recv failure
- `8695161d` @ `2026-10-11T16:54:21.373Z` — TOP / toolResult/text — §1 — Worker_TP0 spin acquire_read
- `94ec49c3` @ `2026-10-11T16:21:39.554Z` — TOP / toolResult/text — §4 — old pd_proxy argv + decode healthy
- `98d550f8` @ `2026-10-11T17:23:06.800Z` — TOP / assistant/thinking — §2 — final localization
- `9d6fa08b` @ `2026-10-11T16:49:58.998Z` — TOP / user/text — §5 — user: ok test now
- `9dcd7397` @ `2026-10-10T14:24:42.056Z` — TOP / toolResult/text — §3 — fail-closed: no silent TCP-only
- `a042e519` @ `2026-10-11T12:48:35.284Z` — EngineStall / assistant/thinking — §2 — check_lookup_result None no deadline
- `ab0e9ed6` @ `2026-10-11T13:50:06.873Z` — TOP / assistant/thinking — §2 — worker reaped + early-exit 0
- `ac20f34c` @ `2026-10-11T00:08:05.954Z` — TOP / user/text — §5 — connector import failure 00:08
- `be6e7994` @ `2026-10-11T16:54:32.580Z` — TOP / toolResult/text — §1 — EngineCore spin core.py:1591 + rank0 CPU
- `cf99bb44` @ `2026-10-11T17:14:32.839Z` — TOP / assistant/thinking — §2 — no-layout -1 job None forever
- `d3d5701d` @ `2026-10-11T15:53:47.473Z` — TOP / assistant/text — §5 — final committed state f911c9c/408e5eb
- `dd2f2251` @ `2026-10-11T11:26:21.486Z` — TOP / assistant/text — §5 — 5 blockers resolved table
- `e16b619b` @ `2026-10-11T16:52:56.835Z` — TOP / assistant/text — §1 — assistant test plan
- `e16c6264` @ `2026-10-11T17:20:32.468Z` — TOP / toolResult/text — §2 — metrics L2 prefetch path name
- `eb701699` @ `2026-10-11T16:52:40.416Z` — TOP / user/text — §1 — user observation no POSTs
- `edda83be` @ `2026-10-11T16:26:26.244Z` — TOP / toolResult/text — §5 — open todo list wedge-root-cause pending
- `f1272bee` @ `2026-10-11T17:23:27.225Z` — TOP / assistant/text — §1 — final validation result
- `f1272bee` @ `2026-10-11T17:23:27.225Z` — TOP / assistant/text — §1 — proxy not the failure
- `f1272bee` @ `2026-10-11T17:23:27.225Z` — TOP / assistant/toolCall — §5 — final open decision ask (4 options)
- `f1272bee` @ `2026-10-11T17:23:27.225Z` — TOP / assistant/toolCall — §5 — final open decision ask
- `ffe830f4` @ `2026-10-10T17:03:55.848Z` — MooncakeVsNIXL / assistant/thinking — §3 — known NIXL issues mis-transfer 58470

_42 quoted records across 38 unique entry ids; all ids verified present in `session.json` (top-level `entries` or a `subSessions[*]` entry list)._
