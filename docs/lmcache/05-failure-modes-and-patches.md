# 05 - Failure modes, patches, and false positives

Every defect below was hit during the real bring-up of the working corridor. Each
costs at minimum one engine boot (5-15 minutes) because the failures surface at
KV-cache registration or at first request, so recognising the signature early is
the whole point of this document. Read it before debugging anything that "worked
yesterday", and read the false-positives section **before** chasing a number that
looks wrong.

## 1. Chunk size mismatch

**Symptom**

```
ValueError: LMCache chunk size 256 must be a multiple of 1648
(the vLLM block size scaled by decode_context_parallel_size).
Set --chunk-size to a multiple of 1648.
```

**Where it fires.** `lmcache/v1/dstributed/l2_adapters/...` - more usefully, at
KV-cache registration in
`lmcache/integration/vllm/vllm_multi_process_adapter.py`, validating against the
server's advertised chunk size.

**Why.** The chunk size must be a multiple of the engine's *resolved* attention
block size, which for a mamba-align hybrid is padded above the raw `--block-size`.
Qwen3_5 pads the attention block so the attention page covers the padded mamba
page, resolving to **1648 tokens** (103 × 16); the engine logs
`Setting attention block size to 1648 tokens to ensure that attention page size
is >= mamba page size`. The wheel default is 256, which does not divide 1648.

**Fix - three channels, all required.**

| Channel | Where | Why it is needed |
|---|---|---|
| Recipe env | `recipes/<recipe>.yaml` → `LMCACHE_CHUNK_SIZE: "1648"` | The engine's own lmcache config loader reads env |
| Container env | launcher `-e LMCACHE_CHUNK_SIZE=${LMCACHE_CHUNK_SIZE:-256}` | Recipe `env:` **does not reach `docker run`** - mods and serve run in a separate process |
| Server CLI | `cluster-config/scripts/lmcache-server-entrypoint.sh` → `--chunk-size "${LMCACHE_CHUNK_SIZE:-256}"` | **The MP server reads it from argparse only**; `MPServerConfig` takes `chunk_size=args.chunk_size`, so an env var on the server container is inert |

This is the defect people get wrong most often, because fixing one or two of the
three looks like it should be sufficient - and each partial fix produces the same
error text.

**Confirm.**

```bash
docker exec vllm_node sh -c 'echo $LMCACHE_CHUNK_SIZE'      # 1648
curl -s http://<node>:8082/status | python3 -c 'import json,sys;print(json.load(sys.stdin)["chunk_size"])'
# 1648 - the server's view, which is what validation uses
```

## 2. `engine-driven transfer does not support hybrid KV cache groups`

**Symptom.** At first store, not at boot:

```
RuntimeError: engine-driven transfer does not support hybrid KV cache groups
```

**Where.** `lmcache/v1/multiprocess/transfer_context/worker_transfer.py` →
`_single_group_block_ids()`, reached from `async_engine_driven.submit_store()`.

**Why.** That helper does `if len(block_ids) != 1: raise ...`. A mamba-align hybrid
always has more than one KV cache group (full-attention plus Mamba/GDN), so the
engine-driven store path structurally cannot represent it. Note the asymmetry that
makes this confusing: **lookups work** (they take a different code path), so L1
hits appeared while stores silently failed.

**Fix.** Switch the transfer mode to the multi-group-capable context:

```json
"lmcache.mp.mp_transfer_mode": "lmcache_driven"
```

plus `--supported-transfer-mode` on the server that includes it (`auto` is
safest). See defect 3 - this switch has a prerequisite.

## 3. Cross-container CUDA IPC failure

**Symptom.** Either a hard failure at registration or the same
`register_kv_caches` timeout as a mode mismatch:

```
cudaErrorMapBufferObjectFailed
```

**Where.** The default IPC wrapper in
`lmcache/v1/platform/cuda/ipc_wrapper.py`, which rides PyTorch's
`UntypedStorage._share_cuda_`.

**Why.** `_share_cuda_` needs a `/dev/shm` **shared between the two containers**.
The engine container and the MP-server container are separate containers with
separate `/dev/shm`, so the PyTorch storage-IPC path cannot rendezvous. This is the
failure documented in spark-vllm-docker commit `3a587079cc`, which is why the PD
corridor was pinned to `engine_driven` in the first place.

**Fix.** Use the driver-level wrapper on both sides:

- engine: `"lmcache.mp.isolated_ipc": true` in the recipe's
  `--kv-transfer-config` extra_config
- server: `--isolated-ipc`, emitted from `LMCACHE_ISOLATED_IPC` via the drop-in

Driver-level `cudaIpcGetMemHandle`/`cudaIpcOpenMemHandle` rendezvous in the kernel
driver, not in a filesystem, so they work across fully isolated containers.

**Confirm.** Both sides report a mode other than the default, and registration
completes:

```bash
docker logs vllm_node 2>&1 | grep -i 'Registering kv caches'
# present, and no timeout
curl -s http://<node>:8082/status | grep -i transfer
```

## 4. `'cudaIpcMemHandle_t' object has no attribute 'reserved'`

**Symptom.** At KV-cache registration, immediately after enabling `isolated_ipc`:

```
RuntimeError: Device backend 'cuda' does not support timeline-semaphore event IPC
on device 0: 'cuda.bindings.runtime.cudaIpcMemHandle_t' object has no attribute 'reserved'
```

**Where.** Four sites across **two** files - this is why fixing one file is not
enough:

- `lmcache/v1/platform/cuda/timeline_semaphore_event_ipc.py` - export
  (`bytes(handle.reserved)`) and import (`ipc_handle.reserved = handle_bytes`)
- `lmcache/v1/platform/cuda/ipc_wrapper.py` - `RawCudaIPCWrapper` stash
  (`self._ipc_handle_reserved = bytes(ipc_handle.reserved)`) and restore
  (`handle.reserved = self._ipc_handle_reserved`)

**Why.** The pinned cuda-python bindings expose `cudaIpcMemHandle_t` as an opaque
object whose only accessor is `getPtr()` - the address of the internal 64-byte
buffer. There is no `.reserved` byte-array field. The object *is* ctypes-backed, so
`getPtr()` gives a live, writable buffer.

**Fix.** Shim all four sites via ctypes:

```python
ipc_handle.reserved = handle_bytes            -> ctypes.memmove(ipc_handle.getPtr(), bytes(handle_bytes), 64)
bytes(handle.reserved)                        -> ctypes.string_at(handle.getPtr(), 64)
self._ipc_handle_reserved = bytes(ipc_handle.reserved)
                                              -> ctypes.string_at(ipc_handle.getPtr(), 64)
handle.reserved = self._ipc_handle_reserved   -> ctypes.memmove(handle.getPtr(), bytes(self._ipc_handle_reserved), 64)
```

Apply **both** sides: engine via `mods/lmcache-mp-engine`, server baked into the
image `Dockerfile`. The server must be baked, not `docker exec`'d from the
entrypoint - see `03-server-side.md`.

**The anchoring trap.** The patch must also ensure `import ctypes`. Anchor that
insertion on the file's first plain import:

```python
m = re.search(r"^(import \w+\n)", src, re.M)
if m is None:
    m = re.search(r"^(from __future__ .*\n)", src, re.M)
src = src[:m.end()] + "import ctypes\n" + src[m.end():]
```

A `struct`-anchored insert silently no-ops on `ipc_wrapper.py`, which imports
neither `struct` nor `ctypes` - and the "patched" file then NameErrors at runtime.

### Sub-failures where the shim looks applied but the engine still crashes

- **The mod never ran** because the shared checkout had not been pulled. Check the
  mod's own log line, not just the presence of the directory:
  `docker logs vllm_node 2>&1 | grep 'lmcache-mp-engine'`.
- **The image build served from cache**, so the new RUN never executed and the
  registry digest was unchanged. Compare digest before/after - an unchanged digest
  is proof the layer did not rebuild.
- **The patch targeted the wrong file.** Dump the path list and `grep -c` the
  anchor inside the container before trusting either side.

## 5. Multi-group block-id rejection, and the dead end that follows it

**Symptom.** A partial L2 hit (for example after a restart) kills EngineCore:

```
RuntimeError: A KV connector reported block-level load failures (invalid_block_ids)
which is not supported for models with multiple KV cache groups.
Connectors must report failed requests via KVConnectorTransferResults.failed_recving instead.
```

**Where.** `vllm/v1/core/sched/scheduler.py` → `_handle_invalid_blocks()`, which
raises whenever `len(kv_cache_config.kv_cache_groups) > 1`.

**Why.** Flat block ids are only unique within a KV cache group, so vLLM cannot map
them back to requests on a multi-group model. LMCache's MP adapter routes any failed
retrieve into `error_block_ids`, which vLLM surfaces as `invalid_block_ids`.

**Fix - and the dead end.** Make the adapter stop emitting block-level failures:

```python
# get_block_ids_with_load_errors()
self.error_block_ids.clear()
return set()
```

Under vLLM's default `kv_load_failure_policy=recompute`, a request whose KV never
arrives simply recomputes its prefix - the desired degraded behaviour - and it is
still released through `finished_recving`, because upstream's own contract states
"Failed receives also appear in finished_recving so the scheduler can release the
request from its transfer wait state".

**Do not take the message's literal suggestion.** Implementing per-request
`failed_recving` was tried and rejected: it trips

```
AssertionError   # scheduler.py:_update_from_kv_xfer_finished -> assert req_id in self.requests
```

because a failed id can be reported in a later snapshot than the one that finished
the request, so the scheduler has already dropped it. See the residual-risk note at
the end.

## 6. Device-ordinal clamp (inert here, live in-process)

**Symptom.** `CUDA error: invalid device ordinal`.

**Why.** Cross-host TP=2 gives rank 1 `local_worker_id=1` while the node has one GPU
(`device_count=1`), so a direct `set_device(1)` is out of range.

**Status in this corridor: inert.** The MP connector derives its device from the
registered tensors; there is no direct `set_device(local_worker_id)` call in the
pinned wheel's MP surface. The mod greps for the pattern and logs
`device-ordinal clamp not applicable` rather than shipping a no-op edit. It *is*
live for the in-process corridor, which patches
`lmcache/v1/gpu_connector/__init__.py`.

## 7. Stale MP sidecar: port and container-name collision

**Symptom.** After a preset swap or a rollback, the new placement's engine hangs at
registration, or `docker compose up` reports a name conflict. Look for an
`lmcache-server-<host>` container that survived the previous stack.

**Why.** The unit binds 5555/8082 and uses a fixed container name. A leftover unit
also keeps a `20-l2.conf` pointing at a master that no longer exists.

**Fix.** `cmd_stop` in `cluster-config/scripts/lib/spark-serve.sh` stops and
disables **this host's** unit, removes its drop-in, and emits
`LMCACHE_MP_STOPPED host=<h>`. That line in a teardown log is the proof the reap
happened. L2 data survives because fs_native and valkey are persistent - the
sidecar restart is cheap, the data is not reloaded.

## 8. Wrong staged-weights slug

**Symptom.** Launcher warning, then an engine exit before any model load:

```
[!] recipe serves <model> but staged-weights slug falls back to '<other-model>'
...
weights dir not found: /opt/llm/models/<other-model>
error: staged weights failed verification at /opt/llm/models/<other-model>
```

**Why.** `SPARK_VLLM_DOCKER_SLUGS` in `cluster.env` maps recipe name → staged slug.
A recipe with no mapping silently inherits whatever fallback the launcher picks,
which for a new recipe is some other model's directory.

**Fix.** Add the mapping alongside the recipe's siblings, plus the matching entries
in `SPARK_VLLM_DOCKER_MAX_PARALLEL` and `SPARK_VLLM_DOCKER_LITELLM_ROUTES` so the
route registration and concurrency cap match too.

## 9. CCI silent no-op applies

**Symptom.** `Apply: 0 to tear down, 0 to start, 2 unchanged`, or
`1 already serving, reused` - and nothing changed.

**Why.** The orchestrator diffs against the stored `current` profile. Once a profile
is committed as current, re-applying the same profile has no diff. `?recreate=1`
only forces **arriving** instances, so it does nothing for an instance already in
`current`.

**Fix.** Apply the builtin `none` profile to clear current, then apply the target
profile. **Caveat that bites:** `none` tears down *every* instance in current,
including ones on other nodes (for example an embed server on node 4). If you only
want to restart one instance, remove `vllm_node` on that instance's own ranks first
and then re-apply - the arrival probe fails and setup runs fresh.

---

## Patch-verification discipline

1. **Grep the anchor inside the container** before believing a patch. A patch
   written against the wrong file, or against a wheel version you are not running,
   looks identical to success in the diff.
2. **Prove the artifact changed.** For an image patch, compare the registry digest
   before and after; an unchanged digest means the layer came from cache. For a mod,
   grep the patched file inside the container for your replacement string.
3. **Beware the single-file `:ro` bind-mount inode trap.** A `git pull` replaces the
   file with a new inode while the running container keeps the old one, so the file
   on the host looks correct and the container still reads stale bytes. Force-recreate
   the consuming container (victoria-metrics, grafana, ...) after such a change.
4. **Re-read the whole boot from the top.** A crash at 05:01 can leave a
   `register_buffer ... -600` line from 04:24 as the most visible error, and chasing
   the older line wastes a cycle.

## False positives - do not chase these

- **Identical-prompt repeat with fast TTFT and flat `lookup_hit_tokens`.** vLLM's
  own APC answers the second call before LMCache is consulted. This is the single
  most convincing false positive in the system; vary the tail or restart the engine.
- **`prefetch_failure_chunks_total{reason="l1_oom"}` in the hundreds.** Expected at
  `LMCACHE_L1_SIZE_GB=1` with 1648-token chunks: the chunk was fetched from L2 and
  could not be retained. Not an L2 fault.
- **`l2_usage_bytes{l2_name="p2p"} == 0` while p2p's store counter advances.** The
  p2p adapter is attached but unused in this deployment.
- **Identical `l2_usage_bytes` for `fs_native` and `valkey`.** Write fan-out, not
  double counting. (Summing **across ranks**, however, does double count - both
  tiers are shared stores.)
- **`rdma_context.cpp ... register memory ... Bad address [14]`.** Documented harmless
  under host staging; registration still succeeds via the staging path.
- **`l1_chunk_lifetime_seconds` panel reading "No data".** Evictions are sparse; a
  `rate()` window is usually empty. Query it cumulatively.

## Residual risks (unproven - state them as such)

- **Partial-KV correctness.** Defect 5's fix lets a failed retrieve proceed with
  partially-loaded KV rather than recomputing it. No incoherent output has been
  observed across the whole coherence series, but it has not been proven safe. If
  wrong output ever appears after a partial hit, the correct fix is to **retract the
  matched tokens** so the scheduler recomputes, not to suppress the report harder.
- **Engine-wheel versus server-image skew.** Both sides install the pinned wheel and
  must move together. A skew is silent until a cell-format difference appears, at
  which point transfers look corrupt rather than failing loudly.
- **Sidecar reap on plain restarts.** Because `cmd_stop` reaps the MP units, a
  non-CCI restart always cold-starts L1. Correct for profile swaps, slightly wasteful
  for a bare engine restart, and worth knowing before reading a hit-ratio dip.
- **Two upstream L2-prefetch bugs remain open in this wheel version.** See
  `docs/pd-router/lmcache-parity-fixes.md`: a vanished prefetch job wedges the
  scheduler in `WAITING_FOR_REMOTE_KVS`, and a lazily-started heartbeat can reap an
  idle instance after the 3600 s registration grace. They present as a scheduler
  spin with `num_requests_running=0` while `waiting` grows - check the MP server's
  prefetch-controller counters before blaming the adapters.
