# P2P `MooncakeConnector` — architecture, handshake, and GB10 feasibility

**Scope:** The P2P `MooncakeConnector` (vLLM `--kv-connector MooncakeConnector`,
**not** `MooncakeStoreConnector`) as implemented in the Fulton-Engineering-Services
vLLM fork. This is the connector mode the upstream `vllm-project/router`
`--kv-connector mooncake` deployment targets.

**Fork:** `Fulton-Engineering-Services/vllm` @ branch
`cuda13.3-aarch64-gb10-glm5next-modular` (tip `4e79d6be6f`, per
`mods/mooncake-worker/README.md:3-5`). Source fetched via `raw.githubusercontent.com`
and cross-checked against `vllm-project/vllm:main` for upstream diff.

**Upstream:** `vllm-project/vllm` @ `main` (fetched via `raw.githubusercontent.com`).

**Citation convention:** `fork:<path>` = file in the fork branch above;
`upstream:<path>` = same path in `vllm-project/vllm:main`; `eco:<path>` =
file in `fulton-infra-ecosystem` repo root.

---

## 1. Architecture: bootstrap server, HTTP endpoints, env vars

### 1.1 Module structure (fork refactors upstream's monolith)

Upstream `main` ships a single monolithic `mooncake_connector.py` (~2882 lines)
containing the entire `MooncakeConnector` class, scheduler, worker, bootstrap
server, and all transfer logic (`upstream:vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py`).

The fork refactored this into sibling modules and added a fork-specific
host-staging layer:

| Module (fork) | Role | Upstream equivalent |
|---|---|---|
| `mooncake_connector.py` | Legacy re-export shim: `from .base import MooncakeConnector` (`fork:mooncake_connector.py:8`) | Monolith |
| `base.py` | `MooncakeConnector(KVConnectorBase_V1, SupportsHMA)` class — delegates to scheduler/worker (`fork:base.py:31`) | Inline in monolith |
| `scheduler.py` | `MooncakeConnectorScheduler` — scheduler-side KV transfer planning (`fork:scheduler.py:18`) | Inline |
| `worker.py` | `MooncakeConnectorWorker(_WorkerSendMixin, _WorkerReceiveMixin, _WorkerSetupMixin, _StagingMixin)` (`fork:worker.py:23`) | Inline |
| `_worker_setup.py` | `_WorkerSetupMixin` — `register_kv_caches()`, GPU-VA vs host-staging decision (`fork:_worker_setup.py`) | Inline |
| `_worker_send.py` | `_WorkerSendMixin` — producer send path, ZMQ side-channel, bootstrap registration (`fork:_worker_send.py:26`) | Inline |
| `_worker_receive.py` | `_WorkerReceiveMixin` — consumer receive path, bootstrap query, H2D copy (`fork:_worker_receive.py:18`) | Inline |
| `staging.py` | `_PinnedStagingArena` + `_StagingMixin` — **fork-specific** host-staging for GB10 (`fork:staging.py`) | **Does not exist upstream** |
| `_protocol.py` | `MooncakeConnectorMetadata`, `PullReqMeta`, `SendBlockMeta`, `should_launch_bootstrap_server`, `get_mooncake_bootstrap_addr` (`fork:_protocol.py`) | Inline in monolith |
| `mooncake_utils.py` | `MooncakeBootstrapServer`, `RegisterWorkerPayload`, `EngineEntry` (`fork:mooncake_utils.py`) | Same file upstream |
| `__init__.py` | Re-exports `MooncakeConnector`, `MooncakeConnectorScheduler`, `MooncakeConnectorWorker` (`fork:__init__.py`) | Inline |

The `staging.py` module and the mixin decomposition (`_WorkerSendMixin`,
`_WorkerReceiveMixin`, `_WorkerSetupMixin`, `_StagingMixin`) are **fork-specific
additions** — upstream `main` has neither. The `mooncake_utils.py` bootstrap
server is present in both fork and upstream with identical route registration.

### 1.2 Bootstrap server

**Class:** `MooncakeBootstrapServer` (`fork:mooncake_utils.py`, class definition;
identical to `upstream:mooncake_utils.py`).

**Purpose:** "A centralized registry for prefiller connection info (IP, port,
ranks)" (`upstream:mooncake_utils.py`, class docstring).

**Process:** The bootstrap server runs **in-process inside the prefiller
(producer) vLLM worker** — not as a separate container. It is launched by
`MooncakeConnectorWorker.__init__` when `should_launch_bootstrap_server()`
returns True (`fork:worker.py`, `__init__` method):

```python
if should_launch_bootstrap_server(vllm_config):
    _, port = get_mooncake_bootstrap_addr(vllm_config)
    self.bootstrap_server = MooncakeBootstrapServer("0.0.0.0", port)
    self.bootstrap_server.start()
```

`should_launch_bootstrap_server()` returns True only for the TP-rank-0,
PP-rank-0, DP-rank-0 worker — i.e., one bootstrap server per prefiller engine
(`fork:_protocol.py`, `should_launch_bootstrap_server` function).

**Default port:** `VLLM_MOONCAKE_BOOTSTRAP_PORT = 8998`
(`fork:vllm/envs.py:241`; env resolver at `fork:vllm/envs.py:1759-1760`).

### 1.3 HTTP endpoints

The bootstrap server is a FastAPI/uvicorn app with two routes
(`fork:mooncake_utils.py:52-53`, identical to `upstream:mooncake_utils.py`):

| Method | Path | Handler | Payload / Response |
|---|---|---|---|
| `POST` | `/register` | `register_worker(payload: RegisterWorkerPayload)` | Request: `{"engine_id": str, "dp_rank": int, "tp_rank": int, "pp_rank": int, "addr": str}` (`RegisterWorkerPayload`, `upstream:mooncake_utils.py`, `class RegisterWorkerPayload(BaseModel)`). Response: `{"status": "ok"}` or HTTP 400 on engine-id/rank mismatch. |
| `GET` | `/query` | `query()` | Response: `dict[int, EngineEntry]` — `{dp_rank: {"engine_id": str, "worker_addr": {tp_rank: {pp_rank: addr}}}}` (`upstream:mooncake_utils.py`, `class EngineEntry` and `query` method). |

**`/v1/query` was not found.** Only `/register` (POST) and `/query` (GET) are
registered in the bootstrap server's `_register_routes` method
(`fork:mooncake_utils.py:52-53`). If `/v1/query` exists it would be in the
`vllm-project/router` (codekeeper repo `vllm-router`, pin `eb57b66b`), not in
the vLLM mooncake bootstrap server. The `/query` endpoint returns the full
worker registry keyed by `dp_rank → tp_rank → pp_rank → addr`.

### 1.4 Environment variables

All `VLLM_MOONCAKE_*` and `MOONCAKE_*` env vars defined in the fork
(`fork:vllm/envs.py`, grep-confirmed):

| Env var | Default | Scope | Source |
|---|---|---|---|
| `VLLM_MOONCAKE_BOOTSTRAP_PORT` | `8998` | P2P — bootstrap server port | `fork:vllm/envs.py:241`; resolver `:1759-1760` |
| `VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT` | `480` | P2P — KV-send timeout (s); producer releases blocks after timeout | `fork:vllm/envs.py:267`; resolver `:1889-1890`; usage doc `fork:docs/features/mooncake_connector_usage.md` |
| `VLLM_MOONCAKE_STORE_TIER_LOG` | `False` | Store only — per-batch memory/disk tier breakdown | `fork:vllm/envs.py:242` |
| `VLLM_MOONCAKE_LOAD_RECV_THREADS` | `1` | Store only — KV-load receive threads | `fork:vllm/envs.py:243` |
| `VLLM_MOONCAKE_DISK_STAGING_USABLE_RATIO` | `0.9` | Store only — DirectIO GET usable ratio | `fork:vllm/envs.py:244` |
| `MOONCAKE_PREFERRED_SEGMENT` | `None` | Store only — preferred RDMA segment address | `fork:vllm/envs.py:245` |
| `MOONCAKE_REQUESTER_LOCAL_HOSTNAME` | `None` | Store only — requester local hostname | `fork:vllm/envs.py:246` |

**Upstream-only env var not in fork:** `VLLM_MOONCAKE_CONNECTOR_TIMEOUT: float = 30.0`
(`upstream:vllm/envs.py:268`). The fork removed or never picked up this timeout.

**Mooncake transfer-engine env var (not vLLM):** `WITH_NVIDIA_PEERMEM` —
documented in the fork usage doc (`fork:docs/features/mooncake_connector_usage.md`,
"Environment Variables" section) as "Read mooncake, not vLLM." This controls
whether the Mooncake `TransferEngine` registers GPU memory via `nvidia-peermem`
kernel module (`WITH_NVIDIA_PEERMEM=1`, default) or uses the DMA-BUF path
(`WITH_NVIDIA_PEERMEM=0`). Symptom when unset on a host without `nvidia-peermem`:
`Failed register memory <addr>: Bad address [14]` in `rdma_context.cpp`
(`fork:docs/features/mooncake_connector_usage.md`, `WITH_NVIDIA_PEERMEM` section).

**`kv_connector_extra_config` keys** (not env vars — set in `--kv-transfer-config`
JSON; read by `MooncakeConnectorWorker.__init__`, `fork:worker.py`):

| Key | Default | Purpose |
|---|---|---|
| `host_staging` | `False` | Enable `_PinnedStagingArena` host-staged transfer (GB10) |
| `staging_send_mib` | `512` | Send arena size (MiB) |
| `staging_recv_mib` | `6144` | Recv window size (MiB) |
| `num_workers` | `10` | Sender thread pool size |
| `mooncake_protocol` | `"rdma"` | Transfer protocol (`fork:docs/features/mooncake_connector_usage.md`) |

---

## 2. Two-leg handshake

### 2.1 Roles and scheduler-side entry points

The P2P `MooncakeConnector` uses vLLM's `kv_role` to split prefiller (producer)
and decoder (consumer) (`fork:docs/features/mooncake_connector_usage.md`,
"KV Role Options"):

- **`kv_producer`** — prefiller generates KV, pushes to decoder via
  `TransferEngine.batch_transfer_sync_write()`
- **`kv_consumer`** — decoder pulls KV from prefiller via ZMQ request → RDMA write

The scheduler-side entry points (`fork:scheduler.py`,
`MooncakeConnectorScheduler.get_num_new_matched_tokens`) branch on
`kv_transfer_params` flags set by the `vllm-project/router`:

- **`do_remote_prefill`** (consumer side): `assert not self.is_kv_producer` —
  pulls all prompt blocks from the remote prefiller
  (`fork:scheduler.py`, `get_num_new_matched_tokens`, `do_remote_prefill` branch).
  Sets `self._reqs_need_recv[request_id]`.

- **`do_remote_decode`** (producer side): `assert not self.is_kv_consumer` —
  queues KV blocks for send to the decoder
  (`fork:scheduler.py`, `get_num_new_matched_tokens`, `do_remote_decode`
  branch). Sets `self._reqs_need_send[request.request_id]`.

### 2.2 Producer side (`do_remote_decode`)

1. **Bootstrap server launch:** The TP=0/PP=0/DP=0 producer worker launches
   `MooncakeBootstrapServer("0.0.0.0", 8998)` in-process
   (`fork:worker.py`, `__init__`; `fork:_protocol.py`,
   `should_launch_bootstrap_server`).

2. **Bootstrap registration:** The producer worker calls
   `register_worker_with_bootstrap()` which sends `POST /register` with
   `RegisterWorkerPayload(engine_id, dp_rank, tp_rank, pp_rank, addr=<zmq_side_channel_addr>)`
   (`fork:_worker_send.py`, `register_worker_with_bootstrap`).

3. **ZMQ side-channel bind:** The producer binds a ZMQ `ROUTER` socket on a
   random TCP port (`sock.bind_to_random_port(f"tcp://{self.hostname}")`,
   `fork:_worker_send.py`, `_mooncake_sender_listener`). This port is the
   `addr` advertised to the bootstrap server.

4. **KV registration:** `register_kv_caches()` registers the producer's KV
   cache memory with `TransferEngine` for RDMA read
   (`fork:_worker_setup.py`, `register_kv_caches`). When `host_staging=True`,
   the host-pinned send arena is registered instead (see §3).

5. **KV transfer:** When a consumer sends a `MooncakeXferMetadata` message via
   ZMQ, the producer's `send_kv_to_decode()` computes transfer regions and
   calls `engine.batch_transfer_sync_write()` to RDMA-write KV blocks to the
   consumer's registered memory regions (`fork:_worker_send.py`,
   `send_kv_to_decode`).

### 2.3 Consumer side (`do_remote_prefill`)

1. **Bootstrap query:** The consumer reads `remote_bootstrap_addr` from
   `PullReqMeta.remote_bootstrap_addr` (set by the router in
   `kv_transfer_params`) and calls
   `_connect_to_prefiller_bootstrap(remote_bootstrap_addr)` which does
   `GET <remote_bootstrap_addr>/query` to discover the producer's worker
   addresses (`fork:_worker_receive.py`, `handle_new_engine_id` →
   `_connect_to_prefiller_bootstrap`).

   ```python
   remote_bootstrap_addr = next(iter(pull_metas.values())).remote_bootstrap_addr
   ...
   await self._connect_to_prefiller_bootstrap(remote_bootstrap_addr)
   ```

2. **Engine ID lookup:** The consumer uses `remote_engine_id` (also from
   `kv_transfer_params`) to find the producer's worker addresses in the
   bootstrap response:
   `self._remote_agents[remote_engine_id][remote_tp_rank][pp_rank]`
   (`fork:_worker_receive.py`, `receive_kv` → `handle_new_engine_id`).

3. **KV transfer request:** The consumer sends `MooncakeXferMetadata` (containing
   its own `remote_hostname`, `remote_port`, `kv_caches_base_addr`,
   `block_lens`, `registered_layer_names`, etc.) via ZMQ `DEALER` socket to the
   producer's side-channel port (`fork:_worker_receive.py`,
   `receive_kv_from_single_worker`).

4. **RDMA write:** The producer receives the metadata, computes transfer
   regions, and calls `engine.batch_transfer_sync_write()` to write KV blocks
   into the consumer's registered memory (`fork:_worker_send.py`,
   `send_kv_to_decode` → `_send_blocks`).

5. **Completion + H2D:** When `host_staging=True`, after the RDMA write lands in
   the consumer's pinned recv window, the consumer H2D-copies from the window
   to real GPU KV blocks before marking the request received
   (`fork:_worker_receive.py`, `process_pulling_result` →
   `_staging_h2d_copy_request`; `fork:staging.py`, `_StagingMixin`).

### 2.4 KV block transport mechanism

- **Protocol:** RDMA (`protocol="rdma"`, `device_name` like `roceP2p1s0f1`)
  via `TransferEngine.initialize(hostname, "P2PHANDSHAKE", protocol, device_name)`
  (`fork:worker.py`, `__init__`).

- **Transfer API:** `engine.batch_transfer_sync_write()` — synchronous RDMA
  write from producer-registered memory to consumer-registered memory
  (`fork:_worker_send.py`, `_send_blocks`).

- **Without `host_staging`:** Direct GPU-VA to GPU-VA RDMA (requires
  `nvidia-peermem` + GPUDirect RDMA). GPU memory registered via
  `engine.batch_register_memory(kv_data_ptrs)`.
  → **Fails on GB10** (see §3).

- **With `host_staging=True`:** D2H → pinned host arena → RDMA → consumer's
  pinned recv window → H2D. The GPU-to-host and host-to-GPU copies use
  `cupy-cuda13x` `MemoryPointer.copy_from_device_async` / `copy_from_host_async`
  on a non-blocking `cupy.cuda.Stream` (`fork:staging.py`,
  `_PinnedStagingArena.d2h` / `h2d`). The RDMA write goes from the registered
  host-pinned arena, not GPU VAs.
  → **Works on GB10** (see §3).

---

## 3. GB10 feasibility (critical)

### 3.1 The GPU-VA wall

The fork's `staging.py` contains the definitive empirical evidence in the
`_PinnedStagingArena` class docstring (`fork:staging.py`, class docstring):

> Why exists (GB10 / DGX Spark): on unified-memory platforms GPUDirect
> architecturally unavailable — nvidia-peermem does not load,
> `CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED == 0`, and `ibv_reg_mr()` on CUDA VAs
> fails EFAULT (all verified empirically on driver 580.173.02).

This is the same wall documented in our store-only decision doc
(`eco:gpu-cluster-forks/spark-vllm-docker/docs/pd-router/kv-backends-l2.md:191-194`):

> GB10 no GPUDirect RDMA NIXL-style GPU-VA registration: store connector's
> direct `store.register_buffer(gpu_addr)` fails `ibv_reg_mr` EFAULT ("Bad
> address [14]", `num_segments=0`). `_StagingSlotPool` D2H-copies GPU KV and
> RDMA-writes.

And the same symptom documented in the usage doc
(`fork:docs/features/mooncake_connector_usage.md`, `WITH_NVIDIA_PEERMEM`):

> Symptom left unset on host: `Failed register memory <addr>: Bad address [14]`
> `rdma_context.cpp`, and KV transfers fail

### 3.2 The three registration paths and their GB10 viability

| Path | Env / config | Mechanism | GB10 viable? | Evidence |
|---|---|---|---|---|
| GPU-VA `nvidia-peermem` | `WITH_NVIDIA_PEERMEM=1` (default) | `ibv_reg_mr()` on CUDA VAs | **No** — EFAULT | `fork:staging.py` docstring; `eco:kv-backends-l2.md:191-194`; `fork:docs/features/mooncake_connector_usage.md` |
| DMA-BUF | `WITH_NVIDIA_PEERMEM=0` | DMA-BUF fd export → `ibv_reg_mr` on fd | **No** — `CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED == 0` | `fork:staging.py` docstring |
| Host-staging | `host_staging=True` in `kv_connector_extra_config` | D2H → pinned host buffer → RDMA → H2D | **Yes** — host DRAM registers cleanly | `fork:staging.py` docstring: "NVIDIA-sanctioned fallback (DGX Spark Porting Guide) transfer through cudaHostAlloc-style pinned host buffers, register fine" |

**Critical finding:** The `WITH_NVIDIA_PEERMEM=0` DMA-BUF fallback that the
usage doc recommends for "hosts where nvidia-peermem not loaded" does NOT work
on GB10 — the fork's `staging.py` explicitly verifies that
`CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED == 0` on GB10 (driver 580.173.02).
Both GPU-VA registration paths (peermem and DMA-BUF) are architecturally
unavailable on GB10's unified-memory platform. The ONLY viable path is
`host_staging=True`.

### 3.3 The code path that decides

The decision point is in `_WorkerSetupMixin.register_kv_caches()`
(`fork:_worker_setup.py`):

```python
self.host_staging = kv_transfer_config.kv_connector_extra_config.get(
    "host_staging", False)
...
if self.host_staging:
    # NOT GPU KV — unified-memory (GB10) CUDA VAs
    # EFAULT nvidia-peermem, dma-buf
    # KV GPU<->pinned
    self._setup_host_staging()
# engine.batch_register_memory(...)  -- registers local memory for RDMA
```

When `host_staging=True`, `_setup_host_staging()` (`fork:staging.py`,
`_StagingMixin._setup_host_staging`) creates `_PinnedStagingArena` instances
for send (producer) and/or recv (consumer), each registering a host-pinned
buffer with `engine.register_memory(self.base_ptr, self.num_bytes, ...)`
(`fork:staging.py`, `_PinnedStagingArena.__init__`).

The send path branches in `_WorkerSendMixin._send_blocks()`
(`fork:_worker_send.py`):

```python
if self.host_staging:
    self._send_blocks_staged(remote_session, src_ptrs, dst_ptrs, lengths)
else:
    self.engine.batch_transfer_sync_write(...)
```

`_send_blocks_staged()` (`fork:staging.py`, `_StagingMixin._send_blocks_staged`):
1. D2H-copies GPU KV into the pinned send arena (`arena.d2h(...)`)
2. Synchronizes the cupy stream (`arena.synchronize()`)
3. RDMA-writes from the arena to the consumer's advertised addresses via
   `engine.batch_transfer_sync_write()` from host-pinned memory

The receive path in `_WorkerReceiveMixin.receive_kv_from_single_worker()`
(`fork:_worker_receive.py`) advertises the staging window's host addresses
instead of GPU addresses when `host_staging=True`:

```python
if self.host_staging:
    advertised_bases = self._staging_window_bases  # host-pinned addresses
else:
    advertised_bases = self.kv_caches_base_addr     # GPU addresses
```

After RDMA completes, the consumer H2D-copies from the pinned recv window to
real GPU KV blocks (`_staging_h2d_copy_request`, `fork:_worker_receive.py`,
`process_pulling_result`).

### 3.4 The `host_staging` path is fork-specific

The `staging.py` module, `_PinnedStagingArena` class, `_StagingMixin`, and the
`host_staging` / `staging_send_mib` / `staging_recv_mib` config keys do NOT
exist in upstream `vllm-project/vllm:main` — they are fork-specific additions
(verified: upstream `mooncake_connector.py` is a monolith with no `staging.py`
sibling; `upstream:vllm/envs.py` has no `host_staging`-related env var).

The store connector has its own independent host-staging implementation
(`_StagingSlotPool` in `fork:store/worker.py`) that was vendored by
`mods/mooncake-worker` (`eco:gpu-cluster-forks/spark-vllm-docker/mods/mooncake-worker/README.md:27-31`).
The P2P connector's `_PinnedStagingArena` is a separate implementation of the
same concept, added to the P2P path by the fork.

---

## 4. Contrast with `MooncakeStoreConnector`

### 4.1 Architecture difference (from store connector docstring)

`fork:store/connector.py` docstring (lines 5-9):

> `MooncakeStoreConnector` - KV cache connector using MooncakeDistributedStore.
> Unlike [MooncakeConnector which] does direct P2P transfer, [this] connector
> uses MooncakeDistributedStore shared KV cache pool. Both producer and
> consumer instances read/write KV to/from store independently, enabling prefix
> caching [and] hash-based deduplication.

| Aspect | P2P `MooncakeConnector` | `MooncakeStoreConnector` |
|---|---|---|
| Transfer mode | Direct P2P: `engine.batch_transfer_sync_write()` producer→consumer | Store-mediated: `store.put()` / `store.get()` via shared KV pool |
| Sidecar cohort | **None** — bootstrap server runs in-process in the prefiller vLLM worker | `mooncake-master` (rank-0, :50051 TCP) + `mooncake-client` (every node, :50053 RoCE) (`eco:kv-backends-l2.md:287-296`) |
| Discovery | Bootstrap server (`POST /register` + `GET /query`, :8998) | Mooncake master server (`MOONCAKE_MASTER_SERVER_ADDRESS:50051`, `metadata_server: P2PHANDSHAKE`) (`eco:kv-backends-l2.md:112-113`) |
| Prefix dedup | **No** — each request transfers KV directly | **Yes** — hash-based dedup with prefix caching (`fork:store/connector.py:8-9`) |
| Host-staging | `_PinnedStagingArena` (`fork:staging.py`) — fork-specific to P2P path | `_StagingSlotPool` (`fork:store/worker.py`; `eco:mods/mooncake-worker/README.md:27-31`) — separately implemented |
| Transfer-engine wheel | **Yes** — imports `mooncake.engine.TransferEngine` (`fork:worker.py`, `from mooncake.engine import TransferEngine`) | **Yes** — `MooncakeDistributedStore` wraps `TransferEngine`; same wheel `mooncake-transfer-engine-cuda13==0.3.12.post1` (`eco:kv-backends-l2.md:575`) |
| `cupy-cuda13x` dep | **Yes** — `_PinnedStagingArena` uses cupy for D2H/H2D (`fork:staging.py`) | **Yes** — `_StagingSlotPool` uses cupy for D2H/H2D (`eco:kv-backends-l2.md:575`; `eco:mods/mooncake-worker/README.md:27-31`) |
| Config keys | `host_staging`, `staging_send_mib`, `staging_recv_mib`, `num_workers` | `host_staging=true, staging_num_slots=2, staging_slot_size_mb=2048` (`eco:kv-backends-l2.md:573`) |
| Ports | Bootstrap :8998 (`VLLM_MOONCAKE_BOOTSTRAP_PORT`) + ephemeral ZMQ side-channel | Master :50051 + client :50053 (RoCE) (`eco:kv-backends-l2.md:574`) |
| Standalone store image | **Not needed** | `mooncake-store:v0.3.12.post1-1` (`eco:kv-backends-l2.md:284-285`) |

### 4.2 Does P2P avoid the store cohort entirely?

**Yes.** The P2P `MooncakeConnector` does not use `MooncakeDistributedStore`,
`mooncake-master`, or `mooncake-client` at all. The bootstrap server is a
lightweight FastAPI registry that runs inside the producer vLLM worker process.
No standalone store containers are required.

### 4.3 Does P2P still need the Mooncake transfer-engine wheel?

**Yes.** Both paths import from `mooncake.engine`:
- P2P: `from mooncake.engine import TransferEngine` (`fork:worker.py`)
- Store: `MooncakeDistributedStore` (which wraps `TransferEngine` internally)

Both need `mooncake-transfer-engine-cuda13` installed in the vLLM container.
The P2P path additionally needs `cupy-cuda13x` when `host_staging=True` (for
the D2H/H2D copies in `_PinnedStagingArena`).

---

## 5. GB10 feasibility verdict

### 5.1 Without `host_staging`

**Does not work on GB10.** The default P2P path registers GPU VAs with
`TransferEngine` via `ibv_reg_mr()`, which requires either `nvidia-peermem`
(`WITH_NVIDIA_PEERMEM=1`, default) or DMA-BUF (`WITH_NVIDIA_PEERMEM=0`). Both
are architecturally unavailable on GB10's unified-memory platform:

- `nvidia-peermem` does not load (no GPUDirect RDMA kernel module)
- `CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED == 0` (DMA-BUF export not supported)
- `ibv_reg_mr()` on CUDA VAs fails EFAULT ("Bad address [14]")

(`fork:staging.py`, `_PinnedStagingArena` docstring; `eco:kv-backends-l2.md:191-194`;
`fork:docs/features/mooncake_connector_usage.md`, `WITH_NVIDIA_PEERMEM`)

This is the **same GPU-VA wall** that forced the store-only pivot from NIXL
(`eco:gpu-cluster-forks/spark-vllm-docker/docs/pd-router/pd-failure-evidence.md`,
§3.1, `NIXL_ERR_BACKEND` at `register_memory`).

### 5.2 With `host_staging=True`

**Architecturally viable on GB10.** The fork's `_PinnedStagingArena`
(`fork:staging.py`) provides a D2H → pinned-host → RDMA → H2D path that
bypasses GPU-VA registration entirely:

- Host-pinned buffers register cleanly with `engine.register_memory()`
  (`fork:staging.py`, `_PinnedStagingArena.__init__`: "NVIDIA-sanctioned
  fallback (DGX Spark Porting Guide) transfer through cudaHostAlloc-style
  pinned host buffers, register fine")
- D2H/H2D copies via `cupy-cuda13x` `MemoryPointer` on non-blocking streams
  (`fork:staging.py`, `d2h` / `h2d` methods)
- RDMA writes from the host-pinned arena, not GPU VAs
  (`fork:staging.py`, `_send_blocks_staged`; `fork:_worker_receive.py`,
  `receive_kv_from_single_worker` — advertises `_staging_window_bases`)

This is the **same concept** as the store connector's `_StagingSlotPool`
(`eco:kv-backends-l2.md:191-194`; `eco:mods/mooncake-worker/README.md:27-31`),
but independently implemented for the P2P path. The store path's host-staging
IS production-deployed on GB10.

### 5.3 Caveats

1. **`staging.py` is fork-specific and may be untested on GB10 in the P2P
   configuration.** The production deployment uses `MooncakeStoreConnector`
   with `_StagingSlotPool` (`eco:kv-backends-l2.md:16-24`), not
   `MooncakeConnector` with `_PinnedStagingArena`. The P2P `host_staging` path
   was added by the fork but has not been the production-exercised path.

2. **The `vllm-project/router` `--kv-connector mooncake` mode** targets the
   P2P `MooncakeConnector` (`eco:pd-failure-evidence.md`, §3.1
   `MooncakeVsNIXL` sub-session: "The point-to-point `MooncakeConnector`
   requires: Running a bootstrap server on the prefiller (default port 8998)").
   The router sets `kv_transfer_params` with `do_remote_prefill` /
   `do_remote_decode` / `remote_bootstrap_addr` / `remote_engine_id` that the
   P2P connector reads.

3. **`WITH_NVIDIA_PEERMEM=0` (DMA-BUF) is NOT a GB10 fallback.** The usage doc
   recommends it for "hosts where nvidia-peermem not loaded" but the fork's
   `staging.py` empirically confirms `CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED
   == 0` on GB10. Only `host_staging=True` works.

### 5.4 Bottom line

**The P2P `MooncakeConnector` path is expected to work on GB10 if and only if
`host_staging=True` is set in `kv_connector_extra_config`**, using the fork's
`_PinnedStagingArena` host-staged D2H→RDMA→H2D path. Without `host_staging`, it
hits the same `ibv_reg_mr` EFAULT GPU-VA wall that forced the store-only
pivot from NIXL. The DMA-BUF fallback (`WITH_NVIDIA_PEERMEM=0`) is also
unavailable on GB10. The P2P path avoids the `mooncake-master`/`mooncake-client`
store cohort entirely but still requires the `mooncake-transfer-engine` wheel
and `cupy-cuda13x` for host-staging. The `host_staging` P2P path is
architecturally sound but has not been production-validated on GB10 — the
store-only path with `_StagingSlotPool` remains the only GB10-proven
deployment.

---

## Sources

- Fork source: `https://raw.githubusercontent.com/Fulton-Engineering-Services/vllm/cuda13.3-aarch64-gb10-glm5next-modular/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/{base,worker,scheduler,staging,_worker_setup,_worker_send,_worker_receive,_protocol,mooncake_utils,__init__}.py`
- Fork env vars: `https://raw.githubusercontent.com/Fulton-Engineering-Services/vllm/cuda13.3-aarch64-gb10-glm5next-modular/vllm/envs.py`
- Fork usage doc: `https://raw.githubusercontent.com/Fulton-Engineering-Services/vllm/cuda13.3-aarch64-gb10-glm5next-modular/docs/features/mooncake_connector_usage.md`
- Fork store connector: `https://raw.githubusercontent.com/Fulton-Engineering-Services/vllm/cuda13.3-aarch64-gb10-glm5next-modular/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/{connector,worker}.py`
- Upstream mooncake: `https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/{mooncake_connector,mooncake_utils}.py`
- Upstream envs: `https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/envs.py`
- Store-only decision doc: `gpu-cluster-forks/spark-vllm-docker/docs/pd-router/kv-backends-l2.md`
- PD failure evidence: `gpu-cluster-forks/spark-vllm-docker/docs/pd-router/pd-failure-evidence.md`
- Mooncake-worker mod: `gpu-cluster-forks/spark-vllm-docker/mods/mooncake-worker/README.md`
- Codekeeper repos: `vllm-router` (READY), `vllm-cuda13.3-aarch64-gb10-glm53-flash` (fork alias, READY)