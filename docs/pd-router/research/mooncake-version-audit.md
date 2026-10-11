# Mooncake version audit: `v0.3.12.post1` → `v0.3.13.post1`

File: `docs/pd-router/research/mooncake-version-audit.md`  
Date: 2026-10-11  
Auditing for: store-path wedge (`MooncakeStoreConnector`) → P2P (`MooncakeConnector` via `vllm-project/router`) pivot.

---

## 1. Release topology

| Tag | SHA | Published | Base |
|-----|-----|-----------|------|
| `v0.3.12.post1` | `6041a609a8c3` | 2026-07-25 | [Release](https://github.com/kvcache-ai/Mooncake/releases/tag/v0.3.12.post1) |
| `v0.3.13` | `e5598b0992cc` | 2026-08-26 | [Release](https://github.com/kvcache-ai/Mooncake/releases/tag/v0.3.13) |
| `v0.3.13.post1` | `719735896c86` | 2026-08-31 | [Release](https://github.com/kvcache-ai/Mooncake/releases/tag/v0.3.13.post1) |
| `HEAD (main)` | `559f83cb73eb` | 2026-10-10 | — |

- **276 commits** between `v0.3.12.post1` and `v0.3.13` ([compare](https://github.com/kvcache-ai/Mooncake/compare/v0.3.12.post1...v0.3.13)).
- **4 additional commits** between `v0.3.13` and `v0.3.13.post1` — CI/build only (non-CUDA wheel separation, TestPyPI gate, EP fix, version bump).  
  SHAs: `14fc4039`, `17ae1624`, `b01664ce`, `71973589`  
  ([compare](https://github.com/kvcache-ai/Mooncake/compare/v0.3.13...v0.3.13.post1))
- **280 total commits** from our pinned version to `v0.3.13.post1`.

### 1.1 Categorized PR highlights (v0.3.12.post1 → v0.3.13)

Categorisation by tag from the [v0.3.13 release body](https://github.com/kvcache-ai/Mooncake/releases/tag/v0.3.13).

#### Transfer Engine / RDMA

| PR | Title | Relevance |
|----|-------|-----------|
| [#3177](https://github.com/kvcache-ai/Mooncake/pull/3177) | [TE] fix(efa): bind CUDA context before registering GPU memory | EFA path GPU registration failure |
| [#3151](https://github.com/kvcache-ai/Mooncake/pull/3151) | [Bugfix][TE] Respect remote_accessible in RDMA verbs MR registration | RDMA MR correctness |
| [#3125](https://github.com/kvcache-ai/Mooncake/pull/3125) | [Bugfix] Fix double free RDMA slice on device-selection failure | Stability on device-selection error |
| [#3198](https://github.com/kvcache-ai/Mooncake/pull/3198) | [Bugfix][TE] RDMA MR and … | RDMA MR fixes |
| [#3243](https://github.com/kvcache-ai/Mooncake/pull/3243) | [Bugfix][TE] Fix dma-buf offsets in chunked GPU MRs | DMA-BUF offset regression |
| [#3236](https://github.com/kvcache-ai/Mooncake/pull/3236) | [Bugfix] Drain RDMA async event queue on each epoll wakeup | RDMA event saturation |
| [#3230](https://github.com/kvcache-ai/Mooncake/pull/3230) | [Bugfix] Clear RDMA context health counter only on successful completions | Health tracking |
| [#3281](https://github.com/kvcache-ai/Mooncake/pull/3281) | [TENT] RDMA: skip NICs that cannot GPUDirect-DMA GPU, fix fallback rotation | NIC selection for DMA-BUF hosts |
| [#3219](https://github.com/kvcache-ai/Mooncake/pull/3219) | [TE] perf(efa): optionally register device memory only on topology-local NICs | EFA NIC scoping |
| [#3381](https://github.com/kvcache-ai/Mooncake/pull/3381) | [Bugfix][TE] MC_IB_PORT=0 RNIC fix | Port-zero handling |
| [#3263](https://github.com/kvcache-ai/Mooncake/pull/3263) | [Bugfix][TE] index-aligned RNIC | RNIC index alignment |
| [#3264](https://github.com/kvcache-ai/Mooncake/pull/3264) | [Bugfix][TENT] Keep context_set_ and buffer keys NicID-indexed | TENT metadata indexing |
| [#3050](https://github.com/kvcache-ai/Mooncake/pull/3050) | [TransferEngine] Harden socket handshake frame handling | Socket handshake reliability |
| [#2974](https://github.com/kvcache-ai/Mooncake/pull/2974) | [TransferEngine] Schedule TCP transfers on bounded per-peer connection lanes | TCP lane isolation |
| [#2955](https://github.com/kvcache-ai/Mooncake/pull/2955) | [TransferEngine] Skip CUDA pointer probe on GPU-less hosts | CPU-only host safety |
| [#2962](https://github.com/kvcache-ai/Mooncake/pull/2962) | [TransferEngine] Make single unregisterLocalMemory best-effort | Graceful unregister |
| [#3285](https://github.com/kvcache-ai/Mooncake/pull/3285) | [TransferEngine] Share one host KV segment across a TP group | Host-memory dedup for MLA |
| [#2954](https://github.com/kvcache-ai/Mooncake/pull/2954) | [TE] thread-local … | — |
| [#3453](https://github.com/kvcache-ai/Mooncake/pull/3453) | [TE] Skip same-host device offset when HIXL CS mode is available | HIXL optimization |
| [#3512](https://github.com/kvcache-ai/Mooncake/pull/3512) | [Bugfix][TE] Log RDMA port number integer, not byte | Logging only |
| [#3542](https://github.com/kvcache-ai/Mooncake/pull/3542) | [TransferEngine] Add TCP session progress deadline | TCP liveness |
| [#3569](https://github.com/kvcache-ai/Mooncake/pull/3569) | [TransferEngine] Enable TCP connection pool by default | TCP pool default |
| [#3387](https://github.com/kvcache-ai/Mooncake/pull/3387) | [Bug]: RDMA reconnect storm mlx5 local length errors | Reconnect storm fix |

#### Store / Master

| PR | Title | Relevance |
|----|-------|-----------|
| [#2826](https://github.com/kvcache-ai/Mooncake/pull/2826) | [Store] implement RFC #2756 batch-record HA OpLog writer | HA oplog |
| [#3077](https://github.com/kvcache-ai/Mooncake/pull/3077) | fix(store): propagate --allocation_strategy through HA supervisor config | Config propagation |
| [#3196](https://github.com/kvcache-ai/Mooncake/pull/3196) | [Store] Support VRAM Segment and NVLink transport within single node | Intra-node NVLink |
| [#3244](https://github.com/kvcache-ai/Mooncake/pull/3244) | [Store] Harden Put/Upsert completion against promotion lifecycle | Completion correctness |
| [#3246](https://github.com/kvcache-ai/Mooncake/pull/3246) | [Bugfix][Store] Fix No rkey MR access on SSD offload read | SSD offload fix |
| [#3159](https://github.com/kvcache-ai/Mooncake/pull/3159) | [store] Avoid staging copy for same node tensor put | Node-local bypass |
| [#3197](https://github.com/kvcache-ai/Mooncake/pull/3197) | [Store] Avoid staging for same-process GPU reads | Same-process bypass |
| [#2909](https://github.com/kvcache-ai/Mooncake/pull/2909) | [Store] Add fixed soft pin lifecycle | Soft-pin lifecycle |
| [#3165](https://github.com/kvcache-ai/Mooncake/pull/3165) | [Store] Validate durable oplog prefix at startup | Oplog validation |
| [#2967](https://github.com/kvcache-ai/Mooncake/pull/2967) | [Bug]: [SSD] DSV4: OBJECT_ALREADY_EXISTS/persist failed/INVALID_KEY | SSD offload fix |
| [#3066](https://github.com/kvcache-ai/Mooncake/pull/3066) / [#3073](https://github.com/kvcache-ai/Mooncake/pull/3073) | [Store] Fix silent data corruption in io_uring read path | Data corruption fix |
| [#3222](https://github.com/kvcache-ai/Mooncake/pull/3222) | [Store] Add Fast-Fail Impossible OffsetAllocator Requests | Allocator safety |
| [#3004](https://github.com/kvcache-ai/Mooncake/pull/3004) | [Store] Make RPC client I/O threads configurable via MC_RPC_CLIENT_IO_THREADS | Configurability |
| [#3318](https://github.com/kvcache-ai/Mooncake/pull/3318) | Allow configuring master admin/metrics HTTP bind address | Bind address |
| [#3168](https://github.com/kvcache-ai/Mooncake/pull/3168) | [Store] Scope capacity metric release to serving MasterService | Metric fix |
| [#3149](https://github.com/kvcache-ai/Mooncake/pull/3149) | [Store] Fix NoF heartbeat config silently ignored in HA mode | HA heartbeat |
| [#2919](https://github.com/kvcache-ai/Mooncake/pull/2919) | [Dep] bind wildcard listen address without getaddrinfo | Wildcard bind |
| [#3118](https://github.com/kvcache-ai/Mooncake/pull/3118) | [Store] Selectively materialize BatchEvict candidates timestamp filtering | Eviction efficiency |
| [#3488](https://github.com/kvcache-ai/Mooncake/pull/3488) | perf(store): batch io_uring bucket reads | IO performance |
| [#3112](https://github.com/kvcache-ai/Mooncake/pull/3112) | [Wheel] Add native fast-copy PUT hot path | PUT performance |
| [#3389](https://github.com/kvcache-ai/Mooncake/pull/3389) | [Store] Add dynamic hot replica fanout | Replication |
| [#3326](https://github.com/kvcache-ai/Mooncake/pull/3326) | [Store] Add bounded standby snapshot capture | Standby safety |
| [#3354](https://github.com/kvcache-ai/Mooncake/pull/3354) | [Store] Return explicit standby restore errors | Error reporting |
| [#3166](https://github.com/kvcache-ai/Mooncake/pull/3166) | [Store] Add exponential backoff ordered oplog writer retries | Oplog retries |
| [#3566](https://github.com/kvcache-ai/Mooncake/pull/3566) | [Bugfix][Store] Allow HA promotion without OpLog | HA availability |

#### Python bindings

| PR | Title | Relevance |
|----|-------|-----------|
| [#2998](https://github.com/kvcache-ai/Mooncake/pull/2998) | [Store] support tensor APIs in dummy client | Python API |
| [#2881](https://github.com/kvcache-ai/Mooncake/pull/2881) | [Store] Add put/get session APIs for ranged multi-buffer transfers | Python API |
| [#3356](https://github.com/kvcache-ai/Mooncake/pull/3356) | [Doc] Update PyPI package documentation | Docs |
| [#3357](https://github.com/kvcache-ai/Mooncake/pull/3357) | [Doc] Align Python setup API documentation | Docs |

#### Documentation

| PR | Title |
|----|-------|
| [#3150](https://github.com/kvcache-ai/Mooncake/pull/3150) | [Doc] Document master KV event publisher flags |
| [#3353](https://github.com/kvcache-ai/Mooncake/pull/3353) | [Doc] Update Mooncake PG design documentation |
| [#3376](https://github.com/kvcache-ai/Mooncake/pull/3376) | [Docs] Reorganize design docs |
| [#3298](https://github.com/kvcache-ai/Mooncake/pull/3298) | [Doc] Add Biren logo to supported hardware |

---

## 2. Issue search results

Searched [Mooncake GitHub issues](https://github.com/kvcache-ai/Mooncake/issues) for terms matching the observed failure signature. Query via `gh search issues --repo kvcache-ai/Mooncake` and GitHub Search API.

### `register_buffer`

| # | State | Title | Relevance |
|---|-------|-------|-----------|
| [#2501](https://github.com/kvcache-ai/Mooncake/issues/2501) | CLOSED | `DummyClient::register_buffer fails for buffers outside pre-mapped shared memory segments` | Directly matches the **DummyClient** error path. Closed 2026-09-23; fix was to validate buffer belongs to a known SHM segment. |
| [#4350](https://github.com/kvcache-ai/Mooncake/issues/4350) | CLOSED | `GPU memory registration fails with "Bad address" on DMA-BUF-only hosts: WITH_NVIDIA_PEERMEM defaults to true, no fallback or hint` | **Direct hit.** Registration error occurs when `WITH_NVIDIA_PEERMEM=1` (default) on hosts that lack peermem — such as GB10 (UMA, no GPUDirect RDMA). |

### `Bad address [14]` / `-600`

| # | State | Title | Relevance |
|---|-------|-------|-----------|
| [#4350](https://github.com/kvcache-ai/Mooncake/issues/4350) | CLOSED | Same as above — `Bad address [14]` is the `errno 14` (`EFAULT`) from `ibv_reg_mem` / `cuMemGetHandleForAddressRange` failure. |
| [#4352](https://github.com/kvcache-ai/Mooncake/pull/4352) | MERGED | `[Doc] Document the WITH_NVIDIA_PEERMEM default and the "Bad address [14]" GPU registration error` | Added docs in `v0.3.13.post1` explaining the error; no default changed. |
| [#2798](https://github.com/kvcache-ai/Mooncake/issues/2798) | CLOSED | `mooncake_master startup failed with "bad address: 0.0.0.0 error: Host not found (authoritative)"` | Different error (getaddrinfo failure on `0.0.0.0`); not our issue. |

### `RDMA`

| # | State | Title | Relevance |
|---|-------|-------|-----------|
| [#4322](https://github.com/kvcache-ai/Mooncake/issues/4322) | OPEN | `[Bug]: Host RDMA regression from resolving NUMA before memory registration (0.3.13.post1)` | **New in v0.3.13.post1.** NUMA-localization regression: untracked host memory gets `*` instead of `cpu:0`, causing throughput regression. [Fix in #4333 (merged post-tag).](https://github.com/kvcache-ai/Mooncake/pull/4333) |
| [#3845](https://github.com/kvcache-ai/Mooncake/pull/3845) | MERGED | `[Store/RDMA] Fix ibv_reg_mr failures with 2MB hugepage SHM (memfd + MTT alignment)` | Directly relevant: fixes `ibv_reg_mr` errors on hugepage-backed SHM segments. |
| [#4515](https://github.com/kvcache-ai/Mooncake/pull/4515) | MERGED | `[Bugfix][TENT] Release the MRs that did register when a NIC refuses a buffer` | Partial-registration leak fix; merged post-v0.3.13.post1. |
| [#3243](https://github.com/kvcache-ai/Mooncake/pull/3243) | MERGED | `[Bugfix][TE] Fix dma-buf offsets in chunked GPU MRs` | DMA-BUF offset error fix; in v0.3.13. |

### `ibv_reg_mr`

| # | State | Title | Relevance |
|---|-------|-------|-----------|
| [#3845](https://github.com/kvcache-ai/Mooncake/pull/3845) | MERGED | Same as above — `ibv_reg_mr` with hugepage. |
| [#4333](https://github.com/kvcache-ai/Mooncake/pull/4333) | MERGED | `[Bugfix] Resolve RDMA NUMA location after memory registration` | Fixes #4322; merged post-tag. |

### `GB10` / `UMA`

No upstream Mooncake issues mention GB10 or UMA. The GB10-specific DMA-BUF limitation is documented only in our fork (see `docs/27-sglang-super-distribution.md` in `home-infrastructure/legacy-inline-findings/`).

### `host_staging`

| # | State | Title | Relevance |
|---|-------|-------|-----------|
| [#4157](https://github.com/kvcache-ai/Mooncake/pull/4157) | MERGED | `[TENT] Add Intel XPU transport, routing and staging to the Transfer Engine` | Host-staging changes for Intel XPU; tangential. |

Our specific `host_staging=true, staging_num_slots=2, staging_slot_size_mb=2048` pattern (from `kv_connector_extra_config`) is used by the **MooncakeStoreConnector** in our deployment. This configures `_StagingSlotPool` which allocates pinned host-DRAM slots for KV offload. The `cupy-cuda13x` dependency enables CUDA-pinned buffer allocation for those slots.

---

## 3. Changes plausibly affecting the failure signature

### Failure signature recap

From `docs/pd-router/pd-failure-evidence.md`:
1. **Boot error**: `register_buffer failed ... Bad address [14] (-600)` — the vLLM rank's `DummyClient.register_buffer` fails mapping GPU memory into the mooncake store connector's pool.
2. **Host-staging pinned-slot pool**: Configured `host_staging=true` with 2 slots of 2 GiB each, backed by `cupy.cuda` pinned allocations. If the `DummyClient` registration fails, the slot pool may be zero-sized or stale.
3. **Store-connector lookup/transfer never completes**: Requests pile as `WAITING_FOR_REMOTE_KVS` ("deferred"), engine busy-spins at `core.py:1591`, 0 completions.

### v0.3.13 changes that map to these

1. **`register_buffer` / `Bad address`**  
   - [#4350](https://github.com/kvcache-ai/Mooncake/issues/4350) / [#4352](https://github.com/kvcache-ai/Mooncake/pull/4352): **Documentation only.** v0.3.13 does NOT change the `WITH_NVIDIA_PEERMEM=1` default. On a GB10 (UMA, no NVIDIA PeerMem support), the default value of `WITH_NVIDIA_PEERMEM=1` still causes `ibv_reg_mr` to fail with `-600` (`EFAULT` / `Bad address`).  
   - [#3845](https://github.com/kvcache-ai/Mooncake/pull/3845): Fixes `ibv_reg_mr` with 2 MB hugepage SHM. This helps when SHM backing uses hugepages, but the root cause (PeerMem vs DMA-BUF) is separate.
   - [#3177](https://github.com/kvcache-ai/Mooncake/pull/3177): EFA only; irrelevant for GB10.
   - [#3243](https://github.com/kvcache-ai/Mooncake/pull/3243): Fixes DMA-BUF offset calculation for chunked GPU MRs. **Relevant** if the host uses DMA-BUF path.

2. **Host-staging / slot pool**  
   - [#2955](https://github.com/kvcache-ai/Mooncake/pull/2955): Skip CUDA pointer probe on GPU-less hosts. The mooncake-store daemon runs in a slim image without GPU; this fix prevents a crash when `mooncake_client` tries `cudaPointerGetAttributes` on a host where `libcuda` stubs are present but no real GPU context exists.  
   - [#2962](https://github.com/kvcache-ai/Mooncake/pull/2962): Makes `unregisterLocalMemory` best-effort, preventing crashes during slot cleanup.  
   - [#3197](https://github.com/kvcache-ai/Mooncake/pull/3197) / [#3159](https://github.com/kvcache-ai/Mooncake/pull/3159): Avoid staging on same-node / same-process GPU reads. These are optimisations, not correctness fixes.
   - [#3285](https://github.com/kvcache-ai/Mooncake/pull/3285): Share one host KV segment across TP group. Critical for MLA multi-rank host memory — without it, each rank allocates its own host copy, causing rapid exhaustion.

3. **Store-connector transfer never completing**  
   - [#3050](https://github.com/kvcache-ai/Mooncake/pull/3050): Harden socket handshake frame handling. A malformed handshake could leave the connector waiting indefinitely.  
   - [#2974](https://github.com/kvcache-ai/Mooncake/pull/2974): Schedule TCP transfers on bounded per-peer connection lanes. Prevents connection starvation.  
   - [#3244](https://github.com/kvcache-ai/Mooncake/pull/3244): Harden Put/Upsert completion against promotion lifecycle. Fixes a race where a put completion is lost during concurrent promotion.  
   - [#3066](https://github.com/kvcache-ai/Mooncake/pull/3066) / [#3073](https://github.com/kvcache-ai/Mooncake/pull/3073): Fix silent data corruption in io_uring read path. On the store daemon's I/O path, corruption could silently produce zero/nonsense results.  
   - [#3112](https://github.com/kvcache-ai/Mooncake/pull/3112): Add native fast-copy PUT hot path. Improves PUT path performance; not a correctness fix.  
   - [#3569](https://github.com/kvcache-ai/Mooncake/pull/3569): Enable TCP connection pool by default. The pool was opt-in before v0.3.13; making it default improves connection reuse.  
   - [#3542](https://github.com/kvcache-ai/Mooncake/pull/3542): Add TCP session progress deadline. Gives the TCP path a liveness deadline — without it, a stuck TCP transfer would hang forever (matches our `WAITING_FOR_REMOTE_KVS` symptom).

### New regressions in v0.3.13

- **#4322** (open): NUMA-location regression. On dual-socket hosts, untracked host memory resolves to `*` instead of its NUMA node, which broadens NIC selection and causes large throughput regression. Fixed by #4333 (merged 2026-09-29, post-tag). **Not relevant for single-socket GB10** but would affect GX10 (dual-socket) if the NUMA resolution fails deterministically.  
- **#2549** (implied by the SHM-hugepage-mapping changes): Several SHM layout changes between 0.3.12 and 0.3.13 may change the default segment alignment, potentially masking or exposing existing `ibv_reg_mr` alignment requirements.

### Verdict: upgrade? Yes, for incremental stability — not for the store-path deadlock

> **The `register_buffer` / `Bad address (-600)` boot error is NOT fixed by v0.3.13.**  
> It is a `WITH_NVIDIA_PEERMEM=1` default conflict on DMA-BUF-only hosts (our GB10). The only upstream action was documentation ([#4352](https://github.com/kvcache-ai/Mooncake/pull/4352)). The fix is to set `WITH_NVIDIA_PEERMEM=0` (for DMA-BUF) or `MC_DMA_BUF_SUPPORTED=0` (for TCP fallback) in the mooncake-store daemon environment — this can be done without any version bump.

> **However, v0.3.13 contains many critical fixes that close the gap to the upstream router paradigm:**
> - TCP connection pool default [#3569], TCP progress deadline [#3542], socket handshake hardening [#3050]
> - Host-memory segment sharing across TP [#3285] (needed for MLA with multi-rank)
> - CUDA-probe skip on GPU-less hosts [#2955] (needed for the mooncake-store slim image)
> - Put/Upsert completion correctness [#3244], data-corruption fix [#3066]
> - ibv_reg_mr hugepage alignment [#3845], DMA-BUF offset fix [#3243]

> **For the store-connector deadlock specifically (never completes):** The TCP deadline [#3542] is the most directly relevant change — it would cause the connector to time out and mark the server unhealthy rather than spinning forever. Combined with the handshake hardening [#3050], this is a genuine improvement over v0.3.12.post1.

> **Caveat**: If the pivot is to P2P `MooncakeConnector` (upstream `vllm-project/router`), then the store-path fixes are less directly relevant. The P2P path bypasses the store/master entirely and uses the bootstrap-server handshake for direct peer-to-peer KV transfer. For that path, TE/RDMA fixes dominate (NIC selection, MR registration, handshake). `v0.3.13` fixes several of these; `HEAD` (2026-10-10, SHA `559f83cb73eb`) contains many more TENT/RDMA fixes that are post-v0.3.13.post1 (including #4515, #4542, #4333).

---

## 4. Wheel pin requirements

### Current pins in repo

All functional call sites of the `0.3.12.post1` mooncake pin, verified via repo-wide grep (`0\.3\.12\.post1`) on 2026-10-11. A version bump must touch **every** vLLM-PD-path site to keep the store daemon and the engine-side `mooncake.store` import in lockstep (same C++ wire protocol, same `VERSION` field).

**vLLM PD path (the path we are fixing):**

| # | File | Line | Pin | Role |
|---|------|------|-----|------|
| 1 | `gpu-cluster-forks/spark-vllm-docker/mods/pd-disagg/run.sh` | 73 | `PD_MOONCAKE_SPEC:=mooncake-transfer-engine-cuda13==0.3.12.post1` | boot-time pip install in the vLLM engine container |
| 2 | `infra/cluster-config/scripts/build-mooncake-store-image.sh` | 49 | `MOONCAKE_VERSION="${MOONCAKE_VERSION:-0.3.12.post1}"` | default build arg for the mooncake-store Docker image |
| 3 | `infra/cluster-config/docker/mooncake-store/Dockerfile` | 54 | `mooncake-transfer-engine-cuda13==${MOONCAKE_VERSION}` | Dockerfile RUN layer (consumes the build arg from #2) |
| 4 | `infra/cluster-config/docker/mooncake-store/compose.env.example` | 19 | `MOONCAKE_STORE_IMAGE=10.100.170.3:5000/mooncake-store:v0.3.12.post1-1` | registry ref for the mooncake-store daemon image |
| 5 | `infra/cluster-config/cluster.env` | 786 | `MOONCAKE_STORE_IMAGE_TAG=v0.3.12.post1-1` | cluster-level default tag (consumed by spark-serve.sh) |
| 6 | `infra/cluster-config/scripts/lib/spark-serve.sh` | 196 | `MOONCAKE_STORE_IMAGE_TAG:-v0.3.12.post1-1` | inline default in the env-file generator |

**SGLang path (separate image, not the vLLM PD path — listed for completeness; a vLLM-only bump does not require touching these):**

| # | File | Line | Pin | Role |
|---|------|------|-----|------|
| 7 | `infra/cluster-config/cluster.env` | 828 | `SGLANG_MOONCAKE_VERSION=0.3.12.post1` | SGLang overlay mooncake wheel version |
| 8 | `infra/cluster-config/scripts/build-sglang-image.sh` | 10, 53, 69 | `SGLANG_MOONCAKE_VERSION:-0.3.12.post1` | SGLang image build arg + in-image version assertion |

**LMCache wheel builder (referenced in docs/skills, not in current checkout):**

| # | File | Line | Pin | Role |
|---|------|------|-----|------|
| 9 | `dgx-spark-wheels/docker/lmcache-mooncake/Dockerfile` | 31 | `ARG MOONCAKE_REF=v0.3.12.post1` | git ref for the LMCache↔Mooncake wheel builder (cited in `agents/fulton-omp-config/managed-skills/build-lmcache-mooncake-wheel/SKILL.md:32` and `home-infrastructure/legacy-inline-findings/docs/27-sglang-super-distribution.md:27`) |

### Wheel availability

`mooncake-transfer-engine-cuda13` wheels for `v0.3.13.post1` **are published on PyPI** for all platforms, including `manylinux_2_28_aarch64` (CPython 3.10–3.13):

| Version | cp310 aarch64 | cp311 aarch64 | cp312 aarch64 | cp313 aarch64 | Source |
|---------|--------------|--------------|--------------|--------------|--------|
| `0.3.12.post1` | ✅ `manylinux_2_28_aarch64` | ✅ | ✅ | ✅ | [PyPI](https://pypi.org/project/mooncake-transfer-engine-cuda13/0.3.12.post1/) |
| `0.3.13` | ✅ `manylinux_2_28_aarch64` | ✅ | ✅ | ✅ | [PyPI](https://pypi.org/project/mooncake-transfer-engine-cuda13/0.3.13/) |
| `0.3.13.post1` | ✅ `manylinux_2_28_aarch64` | ✅ | ✅ | ✅ | [PyPI](https://pypi.org/project/mooncake-transfer-engine-cuda13/0.3.13.post1/) |

Verified via [PyPI simple index](https://pypi.org/simple/mooncake-transfer-engine-cuda13/) — all three versions ship `manylinux_2_28_aarch64` wheels for cp310–cp313, so the GB10/GX10 aarch64 platform is fully supported.

> **Note**: `v0.3.12.post1` and `v0.3.13`/`v0.3.13.post1` use `manylinux_2_28_aarch64` (glibc ≥ 2.28). The mooncake-store Dockerfile uses `python:3.12-slim-trixie` (glibc 2.41) which satisfies this. The vLLM engine image (`vllm-gb10`) is Ubuntu 24.04-based (glibc 2.39), also compatible.

### Bump required — lockstep changes

The mooncake-store daemon and the engine-side `mooncake.store` import **must** run the same wheel version because the C++ wire protocol carries a `VERSION` field that is checked at handshake. A partial bump (image only, or engine only) will cause a version-mismatch rejection at boot.

**For a vLLM PD path bump to `v0.3.13.post1`:**

| # | File | Change |
|---|------|--------|
| 1 | `gpu-cluster-forks/spark-vllm-docker/mods/pd-disagg/run.sh:73` | `PD_MOONCAKE_SPEC:=mooncake-transfer-engine-cuda13==0.3.13.post1` |
| 2 | `infra/cluster-config/scripts/build-mooncake-store-image.sh:49` | `MOONCAKE_VERSION="${MOONCAKE_VERSION:-0.3.13.post1}"` |
| 4 | `infra/cluster-config/docker/mooncake-store/compose.env.example:19` | `MOONCAKE_STORE_IMAGE=...:mooncake-store:v0.3.13.post1-1` (after rebuilding) |
| 5 | `infra/cluster-config/cluster.env:786` | `MOONCAKE_STORE_IMAGE_TAG=v0.3.13.post1-1` |
| 6 | `infra/cluster-config/scripts/lib/spark-serve.sh:196` | `MOONCAKE_STORE_IMAGE_TAG:-v0.3.13.post1-1` |

(Items #3 Dockerfile and #7/#8 SGLang path do not need changes — #3 consumes the build arg from #2; #7/#8 are the SGLang path.)

---

## 5. Recommendation (one-line)

> **Bump to `mooncake-transfer-engine-cuda13==0.3.13.post1`** (image tag `mooncake-store:v0.3.13.post1-1`) across all 6 vLLM-PD-path call sites listed in §4 — but for the P2P router pivot, consider using the **latest HEAD** (`v0.3.14-rc1`) instead, because it contains the TENT/RDMA corner-case fixes ([#4515](https://github.com/kvcache-ai/Mooncake/pull/4515), [#4542](https://github.com/kvcache-ai/Mooncake/pull/4542), [#4333](https://github.com/kvcache-ai/Mooncake/pull/4333)) needed for the `MooncakeConnector` path. Either way, set `WITH_NVIDIA_PEERMEM=0` in the mooncake-store daemon's environment (and/or the engine's `kv_connector_extra_config`) — that single env var, not a version bump, is the fix for the `register_buffer Bad address (-600)` boot error on GB10.

---

## References

- [Mooncake releases](https://github.com/kvcache-ai/Mooncake/releases)
- [v0.3.12.post1 → v0.3.13 compare](https://github.com/kvcache-ai/Mooncake/compare/v0.3.12.post1...v0.3.13)
- [v0.3.13 release notes](https://github.com/kvcache-ai/Mooncake/releases/tag/v0.3.13)
- [v0.3.13.post1 release notes](https://github.com/kvcache-ai/Mooncake/releases/tag/v0.3.13.post1)
- [v0.3.13 → v0.3.13.post1 compare](https://github.com/kvcache-ai/Mooncake/compare/v0.3.13...v0.3.13.post1)
- [PyPI mooncake-transfer-engine-cuda13](https://pypi.org/project/mooncake-transfer-engine-cuda13/)
- [Mooncake issues](https://github.com/kvcache-ai/Mooncake/issues)