# KV backends & the L2 cache stack (GB10, spark-vllm-docker / cluster-config)

Audience: the Rust `vllm-project/router` (codekeeper `vllm-router`) cutover that replaces
`mods/pd-disagg/pd_proxy.py`. This document states, with citations, which KV-transfer path is
production-intended, what each path's container/port/env contract is, and which L2 adapters the
stack can front together with their known GB10 constraints.

Legend: `path:line` citations are relative to the repo root
(`fulton-infra-ecosystem`). Everything below is quoted/derived from the cited file; nothing is
speculative except the two `[INFERENCE]` items explicitly marked.

---

## 1. Which KV path is production-intended

**Production-intended: the Mooncake store-only path (`PD_KV_BACKEND=mooncake`, the default).**
The LMCache MP path is the exercised alternative, selected per-recipe; both are wired end-to-end,
but the store-only path is the default everywhere and is the one the legacy `pd_proxy.py` sits
in front of.

Evidence:

- Default value: `PD_KV_BACKEND="${PD_KV_BACKEND:-mooncake}"`
  (`mods/pd-disagg/mooncake-env.sh:55`); same default re-read in the backend gate
  (`mods/pd-disagg/run.sh:66`, `:112`) and in connector selection
  (`mods/pd-disagg/dispatch.sh:188`).
- Selector documentation: `PD_KV_BACKEND | KV backend selector (mooncake) | mooncake | recipe env`
  (`mods/pd-disagg/README.md:64`) — "mooncake (default)" and two interchangeable paths with
  identical topology: "`mooncake` (default) — MooncakeStoreConnector … lowest indirection; one
  store hop" vs "`lmcache` — LMCacheMPConnector … one extra hop"
  (`mods/pd-disagg/README.md:89-98`).
- Recipe default: the MiMo 4x PD recipe sets the topology on the store-only path and the
  `-lmcache` sibling recipe differs "only in `pd_kv_backend: lmcache`"
  (`infra/cluster-config/scripts/lib/spark-serve.sh:432-434`); backend resolution defaults to
  `mooncake` when the recipe does not override (`scripts/lib/spark-serve.sh:146`, via
  `SPARK_VLLM_DOCKER_KV_BACKEND="${v:-mooncake}"`).
- The retired NIXL path is explicitly replaced: "**No Ray.** Replaces retired NIXL path"
  (`mods/pd-disagg/README.md:5`) and "No NIXL 5600" port (`mods/pd-disagg/README.md:57`).
- The pd-disagg README title itself states the production shape: "PD-disaggregated
  MiMo-V2.6-Flash-RL-UNCENSORED EXL3 (4-node GB10, **Mooncake store-only**)"
  (`mods/pd-disagg/README.md:1`).

Topology of the production path (4-node, `NNODES=4` fail-closed,
`mods/pd-disagg/README.md:7`):

- Ranks 0–1 (prefill, EXL3 TP=2): API 8100, `kv_role=kv_producer`; rank 0 also runs pd-proxy on
  8000 (`mods/pd-disagg/README.md:9-11`; ports at `:56-57`).
- Ranks 2–3 (decode, EXL3 TP=2 + DFlash): API 8200, `kv_role=kv_consumer`
  (`mods/pd-disagg/README.md:12-13`).
- KV data path: "vLLM ranks host-stage GPU KV → pinned DRAM → RDMA-write into client's
  registered segment (hash-based dedup; GPU-VA registration GB10 dead end)"
  (`mods/pd-disagg/README.md:21-23`).

---

## 2. Backend selection mechanics (`mods/pd-disagg/run.sh` + `mooncake-env.sh`)

### 2.1 Selection point

`mods/pd-disagg/run.sh` is the mod that runs in-container on head and every worker before the
serve command (`mods/pd-disagg/run.sh:1-7`), and is inert unless `PD_DISAGG_ENABLED=1`
(`run.sh:21-24`, gate at `run.sh:49-52`).

The backend branch:

- mooncake-only boot installs are gated by
  `if [ "${PD_KV_BACKEND:-mooncake}" = "mooncake" ]; then` (`mods/pd-disagg/run.sh:66`) and end
  at `fi # PD_KV_BACKEND=mooncake` (`run.sh:103`).
- lmcache boot installs are gated by
  `if [ "${PD_KV_BACKEND:-mooncake}" = "lmcache" ];` (`run.sh:112`) and end at
  `fi PD_KV_BACKEND=lmcache` (`run.sh:142`).

### 2.2 What the mooncake branch sets up

- Installs `mooncake-transfer-engine-cuda13==0.3.12.post1` in-process (the connector imports
  `mooncake.store` in vLLM; `run.sh:68-73`), version pinned to the same version the
  `mooncake-store` sibling image runs (`run.sh:72-73`).
- Installs `cupy-cuda13x<14.1.0`, "required [by the] host-staged `_StagingSlotPool`"
  (`run.sh:87-93`).
- Installs `xxhash` unconditionally (APC block-hash backend for `--prefix-caching-hash-algo
  xxhash`; a missing module kills EngineCore mid-serve) (`run.sh:144-155`).
- Applies two vLLM fork patches (mixed-window uniformity probe `ValueError` tolerance
  `run.sh:157-187`; per-layer promotion normalisation `run.sh:189-221`).
- Sources `mooncake-env.sh` (`run.sh:223`).
- Publishes `/tmp/pd-disagg.env` (`PD_DISAGG_ENV_FILE`, `run.sh:31`, `:41`) with the parity
  payload and the mooncake export set: `MOONCAKE_CONFIG_PATH`,
  `MOONCAKE_PREFERRED_SEGMENT`, `MOONCAKE_REQUESTER_LOCAL_HOSTNAME`,
  `MOONCAKE_MASTER_SERVER_ADDRESS`, `MOONCAKE_MASTER_PORT`, `MOONCAKE_CLIENT_PORT`
  (`run.sh:333-340`).

### 2.3 What the lmcache branch sets up

- Installs the GB10/sm_121 `lmcache` wheel (`lmcache-0.5.5rc1+cu13.3torch2.13.glibc239-cp312…`
  from dgx-spark-wheels) with `--no-deps`, then the filtered canonical requirements list
  (`run.sh:105-133`); the filter excludes `torch|numpy|cufile-python|pytest|setuptools|
  setuptools_scm|nixl` (`run.sh:129`).
- Installs `xxhash` (same unconditional block, `run.sh:149-155`).
- Applies the engine-driven-transfer fork patch: LMCache's
  `lmcache/v1/multiprocess/transfer_context/worker_transfer.py`
  `_single_group_block_ids()` is patched so identical multi-group block ids collapse to one
  list instead of hard-rejecting `len != 1` with "engine-driven transfer does not support
  hybrid KV cache groups" (`run.sh:225-274`; failure text at `:231`, `:246`, `:258`).
- Publishes the lmcache export set: `LMCACHE_MP_HOST`, `LMCACHE_MP_PORT`,
  `MOONCAKE_MASTER_SERVER_ADDRESS`, `MOONCAKE_MASTER_PORT` (`run.sh:326-332`) — "LMCache MP
  mode: MP server owns store client; vLLM rank only needs MP endpoint" (`run.sh:327-328`).

### 2.4 `mooncake-env.sh`: what each branch computes

Sourced by `run.sh` (`run.sh:223`); renders the store JSON and exports node-local steering env
(`mooncake-env.sh:1-12`).

- Required inputs, both backends: `MOONCAKE_MASTER_SERVER_ADDRESS` (rank-0 Rail B IP) and
  `MOONCAKE_MASTER_PORT` (50051) (`mooncake-env.sh:32-34`).
- `PD_KV_BACKEND=lmcache` branch (`mooncake-env.sh:57-74`): NO Mooncake connector config — "MP
  server owns store client" (`:58-60`). Exports `LMCACHE_MP_HOST` (default
  `tcp://127.0.0.1`), `LMCACHE_MP_PORT` (default 5555), `PYTHONHASHSEED=0`
  (`:62-68`). Defaults at `:62-63`; required-value guards at `:64-65`.
- mooncake branch (`mooncake-env.sh:76-101`): renders
  `/tmp/pd-disagg/mooncake-store.json` (`:77`):

  ```json
  { "mode": "standalone-store",
    "metadata_server": "P2PHANDSHAKE",
    "master_server_address": "<rank0-railb>:50051",
    "global_segment_size": 0,
    "local_buffer_size": "<MOONCAKE_LOCAL_BUFFER_GIB>GB",
    "protocol": "rdma",
    "device_name": "roceP2p1s0f1",
    "enable_offload": false }
  ```

  (`mooncake-env.sh:80-91` — fields at `:82-89`.) The two MiMo-specific values:
  `global_segment_size=0` — "vLLM ranks pure requesters; DRAM+SSD pool owned by sibling
  mooncake-client container, NOT in-process" (`mooncake-env.sh:9-11`); `enable_offload=false`
  — "SSD tier client-owned; vLLM rank offload stays off" (`:11-12`).
  Exports: `MOONCAKE_CONFIG_PATH`, `MOONCAKE_PREFERRED_SEGMENT` =
  `<railb-ip>:50053`, `MOONCAKE_REQUESTER_LOCAL_HOSTNAME` = `<railb-ip>`,
  `PYTHONHASHSEED=0` (`mooncake-env.sh:94-97`).
- No UCX exports here in either branch: "Mooncake's engine.so links libmlx5 directly
  (mooncake-store Dockerfile) — UCX not on store data path, unlike NIXL"
  (`mooncake-env.sh:14-16`).
- `PYTHONHASHSEED=0` is a silent-correctness requirement: "hash-based KV dedup across
  ranks/instances must not vary processes" (`mooncake-env.sh:18-20`), restated as
  "**`PYTHONHASHSEED=0` everywhere** — vLLM connectors + client must agree"
  (`mods/pd-disagg/README.md:172`).

### 2.5 Connector config actually handed to vLLM (`dispatch.sh`)

- mooncake: `kv-transfer-config` JSON
  `{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_producer|kv_consumer",
  "kv_connector_extra_config":{"host_staging":true,"staging_num_slots":2,
  "staging_slot_size_mb":2048}}` (`mods/pd-disagg/dispatch.sh:207-208`).
- lmcache: `{"kv_connector":"LMCacheMPConnector", …, "kv_connector_extra_config":
  {"lmcache.mp.host":"tcp://127.0.0.1","lmcache.mp.port":5555,
  "lmcache.mp.mp_transfer_mode":"engine_driven"}}` (`dispatch.sh:188-199`), with
  `--disable-hybrid-kv-cache-manager` forced (`PD_DISABLE_HYBrid=1`,
  `dispatch.sh:200-204`) because `LMCacheMPConnector` declares SupportsHMA and MiMo's
  DiffKV/SWA layers produce multiple KVCacheGroups the connector rejects
  (`dispatch.sh:200-203`).
- Both roles run vLLM APC ON with `--prefix-caching-hash-algo xxhash`
  (`dispatch.sh:187`, `:296-308` prefill, `:376-382` decode); rationale at
  `dispatch.sh:296-302`.

---

## 3. `mods/mooncake-worker/` — host-staged connector installer

**Not a daemon**: it is an installer mod, packaged like `mods/exl3-pack` — uv-managed
`pyproject.toml`, `src/mooncake_worker/` library, CLI, ruff/mypy/pytest, `run.sh` thin launcher
(`mods/mooncake-worker/README.md:7-9`). "Installs the host-staged Mooncake store connector
package into the `vllm-node-b12x` image's vLLM tree, from the Fulton-Engineering-Services vLLM
fork branch `cuda13.3-aarch64-gb10-glm5next-modular` (tip `4e79d6be6f`)"
(`README.md:3-5`).

- `run.sh` (thin launcher): exports `PYTHONPATH=<mod>/src`, delegates to
  `python3 -m mooncake_worker.cli <install|verify|selftest|manifest>` (default `install`),
  resolving the vLLM site root via `importlib.util.find_spec("vllm")` when
  `VLLM_PACKAGE_ROOT` is unset, with `VLLM_SITE_PACKAGES` as override
  (`mods/mooncake-worker/run.sh:10-40`; command switch `:21-23`, site resolution `:25-38`).
- CLI verbs map to `cli.py` (`install --site`, `verify --site`, `selftest`, `manifest`;
  `src/mooncake_worker/cli.py:67-80`); default install site
  `/usr/local/lib/python3.12/dist-packages` (`src/mooncake_worker/installer.py:18`); the
  connector package target is
  `vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store`
  (`installer.py:20`).
- What it installs (the matched 8-file set): `worker.py` (md5 `09388dea…`), `data.py`,
  `connector.py`, `scheduler.py`, `coordinator.py`, `protocol.py`, `metrics.py`,
  `__init__.py` (`README.md:38-43`); shipped because the FES `worker.py` reads
  `ReqMeta.partial_tail_offloads` while the stock image `data.py` defines
  `boundary_state_offloads` — a lone-worker swap would `AttributeError` (`README.md:38-41`).
- Why host staging: "GB10 has no GPUDirect RDMA for NIXL-style GPU-VA registration: the store
  connector's direct `store.register_buffer(gpu_addr)` fails with `ibv_reg_mr` EFAULT ("Bad
  address [14]", `num_segments=0`). The Fulton fork's `_StagingSlotPool` instead D2H-copies
  GPU KV into pinned host slots and RDMA-writes from there" (`README.md:27-31`).
  The b12x image ships the upstream connector (worker md5 `b180493c…`, 2633 lines,
  `host_staging` count 0) which predates the staging series (`README.md:32-34`).
- Gates: `gates.verify_vendor()`, `gates.classify_installed()` → `stock` (`b180493c…`) |
  `already_ported` (`09388dea…`) | `unknown`; installer **fails closed** on `unknown`
  (`README.md:52-57`; md5 constants at `src/mooncake_worker/gates.py:14-16`).
  `envs_shim.inject()` adds a `vllm.envs` dict entry (not annotation) for
  `VLLM_PREFIX_CACHE_RETENTION_INTERVAL` (FES default `None`), anchor-count-gated,
  AST-validated, idempotent (`README.md:58-61`).
- Runtime deps shipped elsewhere (i.e. by `mods/pd-disagg/run.sh`, not this mod):
  `mooncake-transfer-engine-cuda13==0.3.12.post1` and `cupy-cuda13x<14.1.0`
  (`README.md:65-68`).
- Config consumed (set by `mods/pd-disagg/dispatch.sh`):
  `kv_connector_extra_config` = `{"host_staging": true, "staging_num_slots": 2,
  "staging_slot_size_mb": 2048}` — budget-neutral vs the default single 4 GiB buffer
  (`README.md:70-76`; default buffer sizes in the vendored connector:
  `src/mooncake_worker/vendor/store/worker.py:81-82`).
`Ports: none of its own — it never binds anything; it only installs files
  (`run.sh` above; pyproject at `mods/mooncake-worker/pyproject.toml:1-46`).

---

## 4. `mods/lmcache-offload/` — in-process offload path (NOT PD)

A separate, non-PD use of LMCache: "In-process LMCacheConnectorV1 KV offload + durability
mod" (`mods/lmcache-offload/run.sh:1-4`). "No PD, no MP server, no RDMA transfer"
(`run.sh:12`).

- Opt-in gate: inert unless `LMCACHE_OFFLOAD_ENABLED=1` (`run.sh:14-17`), which the recipe env
  cannot set directly because mods run in a separate process before the exec-script; the mod
  self-activates by grepping the exec-script for `LMCACHE_CONFIG_FILE`, with
  `LMCACHE_OFFLOAD_ENABLED=1` in the calling env as force-override (`run.sh:31-43`).
- Installs the same GB10 `lmcache` wheel (`run.sh:48-56`, wheel URL at `:52`) plus the filtered
  common requirements (`run.sh:58-77`), and `valkey-glide-sync>=2.3.0` because "the
  valkey:// L2 connector needs the synchronous GLIDE client (the async valkey-glide package
  is a different module namespace)" (`run.sh:79-85`).
- Renders a requester-only LMCache YAML: L1 = host DRAM (`local_cpu`), L2 = `fs_native` on
  shared NFS under a **per-node subdir** `/cluster-shared/lmcache-l2/${HOSTNAME}` (the bare
  `/cluster-shared/lmcache-l2` is deliberately NOT used — the PD MP-server fleet writes
  there), L2 remote = `valkey://10.10.10.53:6379` (`run.sh:87-96`, config keys at
  `:100-107`, `remote_url` at `:107`, final summary echo at `:232`). Caps:
  `LMCACHE_LOCAL_CPU_GIB` default 8, `LMCACHE_LOCAL_DISK_GIB` default 50 (`run.sh:20-23`,
  `:92-93`); config file `/tmp/lmcache-config.yaml` must equal the recipe env value
  (`run.sh:22-24`, `:94`).
- Two more vLLM fork patches ride along: mixed-window uniformity probe fix (same as
  pd-disagg, `run.sh:110-177`) and a GPU-device-ordinal clamp for cross-host TP=2
  (`local_worker_id % device_count`, otherwise `set_device(1)` raises on single-GPU nodes and
  degrades the rank to recompute; `run.sh:179-230`).
- Relationship to the MP path: **`lmcache-offload` is the in-process connector path** (the
  vLLM process owns the cache manager directly); **the PD lmcache backend is the MP-server
  path** (vLLM attaches to a separate `lmcache server` process over ZMQ). The offload mod
  shares the wheel + requirements with the pd-disagg lmcache branch
  (`run.sh:48-51` "mirrors mods/pd-disagg/run.sh:112-142") but configures a completely
  different topology: L1 local_cpu + fs_native NFS + valkey, no mooncake_store L2, no ZMQ,
  no PD roles.

`lmcache-requirements-common.txt` is byte-identical between the two mods
(`mods/lmcache-offload/lmcache-requirements-common.txt` == `mods/pd-disagg/
lmcache-requirements-common.txt`, verified by `diff`); it pins `numpy<=2.2.6` (numba
constraint), opentelemetry/prometheus client ranges, `pyzmq>=25.0.0`, `redis`, `torch`
(unpinned; comment block explains why), `transformers>=5.4` etc.
(`mods/lmcache-offload/lmcache-requirements-common.txt:1-50`).

---

## 5. L2 stack images & daemons (`infra/cluster-config`)

### 5.1 `docker/mooncake-store/` — the standalone-store daemon image

- Image identity: "dedicated slim image [for] mooncake_master and mooncake_client daemons
  (standalone-store KV pool). NOT vllm-gb10 image — minimal python:3.12-slim-trixie … No GPU,
  no vLLM, no --runtime nvidia. Ships mooncake_master and mooncake_client console scripts
  [that] os.execv [the] bundled C++ binaries directly — they never import vLLM"
  (`infra/cluster-config/docker/mooncake-store/Dockerfile:3-9`).
- Base pins: cudalibs stage `nvidia/cuda:13.0.3-devel-ubuntu22.04@sha256:3869b846…`; final
  stage `python:3.12-slim-trixie@sha256:d764629c…` — trixie (glibc 2.41) required because the
  aarch64 wheel is manylinux_2_39 (`Dockerfile:21-25`).
- RDMA runtime deps baked in: `libibverbs1 ibverbs-providers libnuma1 libcurl4t64
  libstdc++6 tini`; `engine.so`/`store.so` `DT_NEEDED libmlx5.so.1` (IBGDA) — this is the
  reason the vLLM ranks need no UCX (`Dockerfile:32-45`; cross-ref
  `mods/pd-disagg/mooncake-env.sh:14-16`).
- Wheel: `mooncake-transfer-engine-cuda13==${MOONCAKE_VERSION}` (default `0.3.12.post1`,
  `scripts/build-mooncake-store-image.sh:49`), with `ep_*.so`/`pg_*.so` Torch EP/PG
  extensions removed and package owned by uid 65532 (`Dockerfile:52-57`); stub
  `libcuda.so.1` + `libcudart.so.13` copied from the CUDA devel image and ldconfig'd
  (`Dockerfile:47-50`); tini PID 1, `USER 65532:65532` (`Dockerfile:59-68`).
- Built + smoke-tested (both `mooncake_master --help` and `mooncake_client --help` must print
  "Flags from") + pushed as `${LLM_REGISTRY_HOST}:${LLM_REGISTRY_PORT}/
  ${MOONCAKE_STORE_IMAGE_NAME}:${MOONCAKE_STORE_IMAGE_TAG}` with a `-<git-sha>` sha tag alias
  (`scripts/build-mooncake-store-image.sh:43-47`, `:56-85`, `:92-94`).
- Registry example: `MOONCAKE_STORE_IMAGE=10.100.170.3:5000/mooncake-store:v0.3.12.post1-1`
  (`docker/mooncake-store/compose.env.example:18-19`).

### 5.2 The compose cohort (`docker/mooncake-store/compose.yaml`)

Project `pd-disagg-mooncake`; compose covers only the long-lived daemons — "the vllm_node
engine stays with launch-cluster.sh" (`compose.yaml:1-7`).

| Service | Profile | Where | Command | Ports/devices |
|---|---|---|---|---|
| `mooncake-master` | `rank0` | rank-0 node only | `mooncake_master --port 50051 --enable_offload=true --promotion_on_hit=true --promotion_admission_threshold=2` (`compose.yaml:49-55`) | TCP :50051, `network_mode: host`, no GPUs/uverbs (`:44-47`; healthcheck TCP on 50051 `:56-61`) |
| `mooncake-client` | `client` | every node | `mooncake_client --global_segment_size ${MOONCAKE_SEGMENT_SIZE_GIB:-24}GB --master_server_address <rank0>:50051 --metadata_server P2PHANDSHAKE --protocol rdma --device_names roceP2p1s0f1 --host <host-ip>:50053 --enable_offload=true` (`compose.yaml:102-116`) | host net; `/dev/infiniband/uverbs1,uverbs3,rdma_cm`; `ulimits.memlock=-1`; NVMe bind-mount `/nvme/mooncake_offload:/mooncake-offload` (`:79-86`); env `MOONCAKE_OFFLOAD_FILE_STORAGE_PATH=/mooncake-offload`, `…_STORAGE_BACKEND_DESCRIPTOR=bucket_storage_backend`, `…_BUCKET_EVICTION_POLICY=lru`, `…_BUCKET_MAX_PHYSICAL_BYTES=${MOONCAKE_OFFLOAD_BUCKET_MAX_PHYSICAL_BYTES:?computed from MOONCAKE_OFFLOAD_MAX_GIB}`, `…_LOCAL_BUFFER_SIZE_BYTES=${MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES:-4294967296}` — 4 GiB, "one value across the template, this compose file and the docs" (`:87-95`) |

Profile gating is the lmcache-mode contract: "EVERY node in the default 'mooncake' KV
backend (serve.sh passes --profile client). Gated behind a profile so the 'lmcache' KV
backend (PD_KV_BACKEND=lmcache) can bring up ONLY the master (--profile rank0): there the
per-node LMCache MP server owns its own DRAM segment and the standalone client cohort must
NOT run" (`compose.yaml:68-73`; restated at
`mods/pd-disagg/README.md:137-139` and
`scripts/lib/spark-serve.sh:369-372` — the client "would double-book the unified-memory pool").

Healthchecks: master = TCP connect on :50051 (`compose.yaml:56-61`); client = process-table
probe for the `mooncake_client` binary because "in P2P-handshake mode the TransferEngine
binds an EPHEMERAL RPC port (observed 16800), so a TCP check on :50053 would fail forever"
(`compose.yaml:114-125`).

Per-node env rendered by `_mc_node_env()` (`scripts/lib/spark-serve.sh:256-302`): `NODE_RANK`,
`MOONCAKE_HOST_IP`, `MOONCAKE_MASTER_SERVER_ADDRESS`, `MOONCAKE_STORE_IMAGE`,
`MOONCAKE_MASTER_PORT=50051`, `MOONCAKE_CLIENT_PORT=50053`,
`MOONCAKE_SEGMENT_SIZE_GIB=${PD_MOONCAKE_SEGMENT_GIB:-${MOONCAKE_SEGMENT_SIZE_GIB:-24}}`,
`MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES=${PD_MOONCAKE_OFFLOAD_BUFFER_BYTES:-${MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES:-4294967296}}`,
`MOONCAKE_OFFLOAD_MAX_GIB=${MOONCAKE_OFFLOAD_MAX_GIB:-96}`,
`MOONCAKE_OFFLOAD_BUCKET_MAX_PHYSICAL_BYTES=$(( ${MOONCAKE_OFFLOAD_MAX_GIB:-96} * 1073741824 ))`
(`:284-294`).

The sizing chain is `operator launch override → cluster.env → literal default`: "PD_MOONCAKE_SEGMENT_GIB /
PD_MOONCAKE_OFFLOAD_BUFFER_BYTES are launch-time operator overrides only (no cluster.env sets
them), so the chain is override -> cluster.env -> literal default, and the compose and -e paths
land on the same numbers" (`scripts/lib/spark-serve.sh:278-283`). The fleet contract
(`infra/cluster-config/cluster.env:770-835`) is **per client node: a 24 GiB DRAM segment
(`MOONCAKE_SEGMENT_SIZE_GIB=24`, `:800`) + a 96 GiB local-NVMe offload cap
(`MOONCAKE_OFFLOAD_MAX_GIB=96`, `:829`; byte form
`MOONCAKE_OFFLOAD_BUCKET_MAX_PHYSICAL_BYTES=103079215104`, `:835`), plus a 4 GiB per-rank
TransferEngine local buffer (`MOONCAKE_LOCAL_BUFFER_GIB=4`, `:806`) and a 4 GiB per-client
offload staging buffer (`MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES=4294967296`, `:814`)**. The
compose fallbacks are the same literals — `${MOONCAKE_SEGMENT_SIZE_GIB:-24}` and
`${MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES:-4294967296}` — so "rendering the compose file
without a node env still lands on the fleet contract"
(`docker/mooncake-store/compose.yaml:92-105`).

Why 24 GiB and not "as large as the cache": GB10 unified memory is zero-sum, so the segment and
the buffers come out of the same pool vLLM's `gpu_memory_utilization` sizes against — "the
client DRAM segment + offload local buffer come out of the same pool vLLM's
gpu_memory_utilization sizes against, which is why the hot tier is capped at 24 GiB rather than
grown to cover the whole cache. Fleet contract (cluster.env): 24 GiB DRAM segment per client
node + 96 GiB local-NVMe cold tier + 4 GiB TransferEngine local buffer"
(`docker/mooncake-store/compose.env.example:25-30`). The segment math is in cluster.env: "At
max-model-len 262144 one full request is ~9.5-9.7 GiB of fp8 KV, so 24 GiB keeps ~2 full-length
requests resident before LRU eviction pushes further traffic down to the NVMe tier below. DRAM
is deliberately the hot tier only; the SSD tier absorbs the remainder instead of growing the
segment" (`cluster.env:794-799`); the 4 GiB per-rank buffer is "sized to match the template
budget comments (110/120 prefill, 104/120 decode at 24 GiB segment) so a rank's staging claim
on unified memory is bounded and the ~10 GiB prefill headroom stays clear of the co-tenant
embed reserve" (`cluster.env:801-806`).

That makes the engine GMU budget the derived quantity, not the store size: `gpu_memory_utilization`
is set *around* the reserved 24 GiB segment + 4 GiB rank buffers + 4 GiB staging buffer, and the
client's own cgroup is capped to match — 32 GiB high-water / 40 GiB kill line
(`MOONCAKE_CLIENT_OOM_HIGH_GIB=32`, `cluster.env:820`; `MOONCAKE_CLIENT_OOM_MAX_GIB=40`, `:821`),
because "the client cap must cover its DRAM segment (24 GiB) + the offload local buffer (4 GiB)
+ working overhead, hence ≈32 GiB high-water and a 40 GiB kill line — with a 24 GiB segment a
22/24 cap would OOM-kill the unit the moment the segment fills"
(`cluster.env:815-819`; the docker `--memory`/`--memory-swap` flags are on the unit,
`templates/mooncake-client.service.tmpl:32-34`). Re-budgeting GMU is therefore a per-recipe
operation, not a client-sizing one: the recipe knob `pd_mooncake_segment_size_gib`
(`infra/cluster-config/scripts/lib/spark-recipe.sh:173-179`) is consumed only by the LMCache MP
server's `mooncake_store` L2 segment sizing (MP-adapter-only: an unset recipe falls back to its
own 8 GiB literal there, not to cluster.env's 24), which converts that GiB value to bytes with a
1073741824 multiplier (`scripts/lib/spark-serve.sh:423-426`; knob and fallback at
`spark-recipe.sh:173-179`). It never sizes the mooncake-client cohort, which reads
`MOONCAKE_SEGMENT_SIZE_GIB` straight from cluster.env.

Launch flow: `launch_mooncake_daemons()` runs `docker compose -p pd-disagg-mooncake
--env-file <node.env> --profile rank0 --profile client up -d --wait` rank-ascending, master
first (`scripts/lib/spark-serve.sh:304-361`; rank0+client on rank 0 at `:327-328`);
teardown is `docker compose -p pd-disagg-mooncake down --remove-orphans --timeout 10`
degrading to `docker rm -f mooncake-master mooncake-client` (`:213-217`). The teardown names
`--profile client` explicitly because "mooncake-client is client-profile-gated and pins a 24 GiB
unified-memory segment; without the client started by a prior (non-skip) run [being included in
teardown it] survives and starves the next launch's request_memory guard" (`:208-212`).

### 5.3 `docker/lmcache-server/` — the LMCache MP-server image

- "Thin overlay on top of the validated `vllm-gb10:0.27.1-sm121-cu133` image that installs
  the LMCache wheel (GB10-optimized, sm_121 CUDA extensions) and its runtime dependencies.
  No vLLM, no Dynamo, no FlexTensor — just the LMCache multiprocess (MP) server daemon
  (`lmcache server`)" (`docker/lmcache-server/Dockerfile:1-7`).
- Base provides "torch 2.13.0+cu133 (from-source, sm_121) — the EXACT torch the lmcache wheel
  was compiled against" and "CUDA 13.3 runtime (for CUDA IPC between MP server and vLLM
  workers)" (`Dockerfile:8-12`) — the CUDA-IPC wording is about the dependency surface; the
  engine_driven mode below avoids relying on cross-container CUDA IPC.
- Wheel: the same dgx-spark-wheels `lmcache-mooncake-v0.5.5rc1-cu13.3` wheel (BUILD_MOONCAKE=1
  variant shipping `lmcache.lmcache_mooncake` for the mooncake_store L2 adapter),
  `--no-deps` to avoid clobbering the from-source torch (`Dockerfile:26-34`).
- Extra libs: `valkey-glide-sync>=2.3.0` (`Dockerfile:40-42`);
  `libyaml-cpp.so.0.8` copied from the lmcache-mooncake-wheel-builder image and
  `liburing.so.2` from the nixl-wheel-builder image; `/opt/mooncake/lib` ldconfig entry
  (`Dockerfile:75-84`); build-time static linkage gate on `lmcache_mooncake*.so` DT_NEEDED
  (`Dockerfile:85-97`).
- Smoke probes at build and at image-build time:
  `import lmcache.cuda_ops; import lmcache.lmcache_native; from
  lmcache.v1.multiprocess.server import MPCacheServer; import nixl_cu13`
  (`Dockerfile:57-63`; `scripts/build-lmcache-server-image.sh:63-68` adds
  `lmcache.lmcache_mooncake LMCacheMooncakeClient` to the probe). Image label:
  `cluster.gb10.image="lmcache-server"`, `cluster.gb10.lmcache="0.5.5rc1-sm121-cu133"`
  (`Dockerfile:100-101`).
- Build: `build-lmcache-server-image.sh` re-execs on `LLM_BUILD_HOST` (node2), builds the
  overlay, smoke-probes, pushes `${LLM_REGISTRY_HOST}:${LLM_REGISTRY_PORT}/
  ${LMCACHE_IMAGE_NAME}:${LMCACHE_IMAGE_TAG}` (`scripts/build-lmcache-server-image.sh:31-47`,
  `:53-77`).

### 5.4 `scripts/lmcache-server-entrypoint.sh` — MP-server launch contract

Host-side docker-run wrapper invoked by `templates/lmcache-server@.service.tmpl` ExecStart as
`lmcache-server-entrypoint.sh <instance>` where `<instance>` is the node hostname
(`lmcache-server-entrypoint.sh:1-5`); one MP server per GB10 node, "vLLM Dynamo workers
attach via LMCacheMPConnector over localhost ZMQ + CUDA IPC"
(`templates/lmcache-server@.service.tmpl:2-5`).

Container: `docker run --rm --name lmcache-server-${INSTANCE} --runtime nvidia --gpus all
--shm-size 8g --network host --device /dev/infiniband/uverbs3 /uverbs1 /rdma_cm
--ulimit memlock=-1 -v ${MOUNT_POINT}/lmcache-l2:/cluster-shared/lmcache-l2 --entrypoint
lmcache ${LMCACHE_IMAGE_REF} server ${LMCACHE_ARGS[@]}`
(`lmcache-server-entrypoint.sh:110-124`).

`lmcache server` args (`lmcache-server-entrypoint.sh:27-46`):

- `--instance-id <node> --host 0.0.0.0 --port ${LMCACHE_PORT:-5555} --http-port
  ${LMCACHE_HTTP_PORT:-8080} --l1-size-gb ${LMCACHE_L1_SIZE_GB:-1} --eviction-policy
  ${LMCACHE_EVICTION_POLICY:-LRU} --hash-algorithm ${LMCACHE_HASH_ALGORITHM:-blake3}
  --max-workers 4 --max-gpu-workers 2 --max-cpu-workers 2 --supported-transfer-mode
  ${LMCACHE_SUPPORTED_TRANSFER_MODE:-engine_driven}` (`:29-46`).
- The transfer-mode comment is the GB10 contract verbatim: "connector's
  kv_connector_extra_config sets lmcache.mp.mp_transfer_mode=engine_driven (GB10
  cross-container CUDA IPC buffer mapping fails); server's supported mode MUST match or no
  engine_driven handler registered and register_kv_caches times out. Server default
  lmcache_driven, so set explicitly" (`:40-45`).

L2 adapters (each appended only when its env var is set):

| Adapter | Gate env | Payload | Cite |
|---|---|---|---|
| `mooncake_store` | `LMCACHE_MOONCAKE_MASTER_ADDR` | `--no-l1-use-lazy` (RDMA preregistration needs the L1 pool fully allocated before registration) + `{"type":"mooncake_store","num_workers":4,"local_hostname":<node>,"metadata_server":"P2PHANDSHAKE","protocol":"rdma","rdma_devices":"roceP2p1s0f1","master_server_addr":<addr>,"global_segment_size":17179869184,"local_buffer_size":4294967296}` | `:48-58` |
| `fs_native` | `LMCACHE_FS_NATIVE_PATH` | `{"type":"fs_native","base_path":<path>,"num_workers":4,"max_capacity_gb":100}` | `:61-66` |
| `valkey` | `LMCACHE_VALKEY_NODES` | `{"type":"valkey","startup_nodes":<nodes>,"key_prefix":"gx10","max_capacity_gb":50,"num_workers":2}` — "Valkey on Dell (host networking, management IP). Separate machine, zero memory cost to GB10" | `:69-75` |
| P2P | `LMCACHE_P2P_ADVERTISE_URL` | `--p2p-advertise-url <url> --p2p-listen-url 0.0.0.0:6767 --p2p-transfer-engine nixl` (+ `--coordinator-url http://<host>:9300` when `LMCACHE_COORDINATOR_HOST` set) — "P2P KV sharing (Phase 3): cross-node RDMA over Rail B" | `:78-88` |

Container env (`:90-108`): `LMCACHE_PORT`, `LMCACHE_HTTP_PORT` (note the HTTP port env
default in the container env flag is 8082 while the `--http-port` arg default above is
8080; the fleet health gate keys on 8082 (`scripts/lib/spark-serve.sh:295`),
`PYTHONHASHSEED=0` ("Hash-seeding parity [with] vLLM ranks … mods/pd-disagg/README.md:74",
`:94-97`), plus NIXL/UCX RDMA env used by the P2P engine:
`UCX_TLS=cuda_copy,rc,sm,self,tcp`, `NIXL_DISABLE_CUDA_ADDR_WA=1`,
`NIXL_EP_USE_HOST_RDMA_BUFFER=1` with the GB10 caveat: "GB10 [has] no GPUDirect RDMA — UCX
must use cuda_copy (host-staged RDMA). Underscores in UCX_TLS mandatory (hyphens silently
fail → UCX falls back [to] rc which can't ibv_reg_mr GPU memory)" (`:100-108`).

Health gate for the fleet launch: `curl http://localhost:${LMCACHE_HTTP_PORT}/healthcheck`
within 60 s (`scripts/lib/spark-serve.sh:341-351`), image re-pull because
`LMCACHE_IMAGE_REF` is a mutable tag (`:327-333`), and the per-node drop-in
`20-l2.conf` writing `LMCACHE_MOONCAKE_MASTER_ADDR` (rank-0 master),
`LMCACHE_MOONCAKE_LOCAL_HOSTNAME=<node>`,
`LMCACHE_MOONCAKE_GLOBAL_SEGMENT_BYTES=<seg>` sized from the recipe's
`pd_mooncake_segment_size_gib` so the MP server does not over-register on GB10's unified
pool (`:423-426`, drop-in written at `:446-452`).

The systemd unit (`templates/lmcache-server@.service.tmpl:30-41`) is thin: container name
`lmcache-server-%i`, `Restart=on-failure` ("MP server failure does NOT take down vLLM —
workers degrade to recompute on cache miss and auto-recover", `:17-18`),
`TimeoutStartSec=60`.

### 5.5 `setup-mooncake-pd.sh` / `teardown-mooncake-pd.sh` — the systemd-path deployment

`setup-mooncake-pd.sh` is the older-but-alternative deployment shape (systemd units instead
of the compose cohort): "install/refresh Mooncake store-only P/D spike (prefill+decode vLLM
units, mooncake-master + mooncake-client store units on both ranks, pd-proxy router, OOM
drop-ins) [for] one placement instance. Idempotent" (`scripts/setup-mooncake-pd.sh:1-5`).

- Rank roles: rank 0 provisions `vllm-serve@<instance>-prefill`, `mooncake-master.service`,
  `mooncake-client.service`, `pd-proxy@<instance>`, OOM drop-ins, and registers the LiteLLM
  route `cluster/<served-model-name>`; ranks 1..N-1 provision
  `vllm-serve@<instance>-decode` + `mooncake-client.service` + drop-ins
  (`setup-mooncake-pd.sh:20-25`). Ordering contract: "RANK 0 FIRST — decode ranks' store
  connectors must find mooncake_master already listening" (`:8-12`; warn at `:557`).
- Image gates: vLLM image must be built with KV connectors default-on
  (`INSTALL_KV_CONNECTORS=true`, `:301-310`); store image gates on `mooncake_master --help`
  / `mooncake_client --help` gflags output (`:311-324`).
- Store JSON installed on both ranks at `${LLM_CONF_DIR}/mooncake/<preset>.json` from
  `templates/mooncake-store.json.tmpl` — same `global_segment_size=0` requester-only shape
  (`:230-236`, template at `infra/cluster-config/templates/mooncake-store.json.tmpl:1-10`,
  where `enable_offload: true` — note the template keeps the offload RPC available while the
  spark-vllm-docker render (`mooncake-env.sh:89`) sets it `false`; both cite `master_server_address
  `__RANK0_RAILB_IP__:…`, `protocol rdma`, `device_name roceP2p1s0f1`).
- The fork connector package is staged from a pinned FES/vllm tarball sha
  (`e93769fd9be0a0bcd1065eea20181e232ebedcf3`) into
  `${CONF_DIR}/mooncake/patches/mooncake` and every file compile-checked
  (`setup-mooncake-pd.sh:334-357`).
- Units: `mooncake-master.service` — `docker run --rm --name mooncake-master --network host
  mooncake_master --port ${MOONCAKE_MASTER_PORT} --enable_offload=true
  --promotion_on_hit=true --promotion_admission_threshold=2`, `Restart=on-failure`,
  `MemoryMax=${MOONCAKE_MASTER_OOM_MAX_GIB}G`, "metadata-only TCP (no RDMA)"
  (`templates/mooncake-master.service.tmpl:29-47`). `mooncake-client.service` —
  `docker run --rm --name mooncake-client --network host --device uverbs1/uverbs3/rdma_cm
  --ulimit memlock=-1 -v /nvme/mooncake_offload:/mooncake-offload … mooncake_client
  --global_segment_size ${MOONCAKE_SEGMENT_SIZE_GIB}GB --master_server_address
  __RANK0_RAILB_IP__:50051 --metadata_server P2PHANDSHAKE --protocol rdma --device_names
  roceP2p1s0f1 --host __THIS_RAILB_IP__:50053 --enable_offload=true`, offload env
  (`-e "MOONCAKE_OFFLOAD_BUCKET_MAX_PHYSICAL_BYTES=${MOONCAKE_OFFLOAD_BUCKET_MAX_PHYSICAL_BYTES}"` and
  `-e "MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES=${MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES}"` —
  the template hardcodes no byte values; both are rendered from cluster.env through `envsubst`,
  so the offload staging buffer is `4294967296` and the bucket cap `103079215104` on every node,
  the same 4 GiB / 96 GiB the compose cohort uses), `Restart=on-failure`, `MemoryHigh`/`MemoryMax` caps
  (`templates/mooncake-client.service.tmpl:25-57`); "the client registers its DRAM segment
  via ibv_reg_mr (host DRAM registers cleanly on GB10). The vLLM ranks use host_staging=true
  to D2H-copy GPU KV into a pinned staging buffer, then RDMA-write from staging to this
  client's registered segment" (`:10-13`).
- Health gates: client active + `docker ps` name match + rank-0 `ss -ltn :50051` within
  120 s (`setup-mooncake-pd.sh:417-438`); `/mooncake-offload` writable inside the container
  or fail (`:440-447`); prefill `/v1/models` gate 1560 s (`:482-497`); pd-proxy gated on
  "systemd unit active + pd-proxy container running" rather than `/health` because decode
  ranks come up after rank 0 (`:504-524`).
- `teardown-mooncake-pd.sh`: rank auto-detected from placements.d env files; rank 0 also
  stops `mooncake-master` and pd-proxy; `mooncake-client` torn down on both ranks;
  template-wide OOM drop-ins deliberately kept; SSD offload data left in place unless
  `--wipe-offload` ("bucket_backend metadata scan re-imports it on next bring-up");
  LiteLLM route NOT removed (`scripts/teardown-mooncake-pd.sh:2-24`, `:61-119`).

### 5.6 The pd-proxy being replaced

`mods/pd-disagg/pd_proxy.py` (byte-identical to
`infra/cluster-config/docker/pd-proxy/pd_proxy.py`, sha256
`fd2a5cb2…79154d`, `mods/pd-disagg/README.md:180-182`) is the FastAPI router launched on
rank 0 by dispatch.sh: `pd_proxy --model <model> --prefill <node0>:8100 --decode
<rank2>:8200 --port 8000 --verify-timeout 1800`, delay-gated `PD_ROUTER_START_DELAY=30`
(`mods/pd-disagg/dispatch.sh:233-257`). Its docstring: "KV path — supersedes both Nixl-style
kv_transfer_params extraction and the P2P MooncakeConnector transfer-id /
bootstrap-server contract" (`mods/pd-disagg/pd_proxy.py:16-17`, `--bootstrap-port`
retirement at `:33`). The Rust router cutover replaces exactly this component; the KV
backends and daemon cohorts documented above are its dependencies.

---

## 6. L2 adapter set + documented GB10 caveats

Adapter inventory as configured (where each appears):

- **mooncake_store** — MP-server L2 adapter (RDMA to the same rank-0 master) at
  `lmcache-server-entrypoint.sh:48-58`; production PD lmcache backend's L2
  (`mods/pd-disagg/README.md:96-97`, `:111-114`); requires the `lmcache-mooncake` wheel
  variant (`docker/lmcache-server/Dockerfile:26-30`, `:64-67`).
- **fs_native** — MP-server adapter (`lmcache-server-entrypoint.sh:61-66`) and the
  in-process offload path's L2 on per-node NFS subdirs
  (`mods/lmcache-offload/run.sh:88-96`).
- **valkey** — MP-server adapter to Dell (`lmcache-server-entrypoint.sh:69-75`) and the
  in-process offload path's remote tier (`mods/lmcache-offload/run.sh:107`).
- **P2P** — MP-server Phase-3 cross-node sharing over NIXL/Rail B
  (`lmcache-server-entrypoint.sh:78-88`); deployment drop-in sets
  `LMCACHE_P2P_ADVERTISE_URL=<railb>:6767`
  (`scripts/setup-dynamo-serve.sh:386-392`). Note the standalone-store JSON uses
  `metadata_server: P2PHANDSHAKE` — that is the mooncake master discovery mode, unrelated to
  the LMCache P2P adapter (`mods/pd-disagg/mooncake-env.sh:83`,
  `templates/mooncake-client.service.tmpl:44`).

Documented GB10 constraints (each with its citation):

1. **No GPUDirect RDMA / GPU-VA registration is a dead end.** `store.register_buffer(gpu_addr)`
   → `ibv_reg_mr` EFAULT ("Bad address [14]", `num_segments=0`)
   (`mods/mooncake-worker/README.md:27-29`); the fix is `_StagingSlotPool` host staging —
   D2H into pinned slots, RDMA from there (`README.md:30-31`; `README.md:21-23` in
   pd-disagg; `templates/mooncake-client.service.tmpl:10-13`). Same class as the retired
   NIXL-GPU failure (`mods/pd-disagg/README.md:22-23`).
2. **Cross-container CUDA IPC fails on GB10 unified memory** → transfer mode MUST be
   `engine_driven` on BOTH sides. The `auto`/`lmcache_driven` path shares KV via
   `_new_shared_cuda` → `cudaErrorMapBufferObjectFailed`, then the engine dies "LMCache
   server did not respond to register_kv_caches within 300s"; vLLM side sets
   `lmcache.mp.mp_transfer_mode=engine_driven` (dispatch.sh), MP server sets
   `--supported-transfer-mode engine_driven`; the server's own default is `lmcache_driven`
   (`mods/pd-disagg/README.md:123-132`; `dispatch.sh:191-196`;
   `lmcache-server-entrypoint.sh:40-45`; server default note
   `mods/pd-disagg/README.md:143-144`).
3. **Engine-driven + mixed KV groups**: `_single_group_block_ids()` hard-rejects multi-group
   block ids, so dispatch.sh patches identical groups to collapse
   (`mods/pd-disagg/run.sh:225-274`).
4. **Unified memory is zero-sum.** MP-server `mooncake_store` segment is sized from the
   recipe's `pd_mooncake_segment_size_gib`, not the entrypoint's 16 GiB default
   (`mods/pd-disagg/README.md:133-136`; sizing code
   `scripts/lib/spark-serve.sh:423-426`); the fleet `mooncake-client` cohort is the *other*
   budget line — 24 GiB DRAM segment + 96 GiB NVMe offload cap + 4 GiB per-rank buffer + 4 GiB
   staging buffer per client node (`cluster.env:770-835`), with the engine GMU sized around
   that reservation rather than the store grown to fit the engine
   (`compose.env.example:25-30`); the standalone `mooncake-client` cohort must NOT run in
   lmcache mode (`mods/pd-disagg/README.md:137-139`; `compose.yaml:68-73`).
5. **L1 CPU pool minimal on GB10**: `--l1-size-gb 1` default — "On unified-memory GB10, CPU
   and GPU share the same physical LPDDR5X pool — every GiB to LMCache L1 is a GiB taken
   from vLLM KV cache / FlexTensor weight blocks. The real offload value comes from L2
   backends … external storage, zero memory cost to GB10"
   (`docker/lmcache-server/Dockerfile:14-19`; same wording
   `lmcache-server-entrypoint.sh:11-15`).
6. **UCX transports on GB10**: `UCX_TLS=cuda_copy,rc,sm,self,tcp` mandatory because there is
   no GPUDirect RDMA — `rc` alone "can't ibv_reg_mr GPU memory"; underscores, not hyphens
   (`lmcache-server-entrypoint.sh:100-108`). NIXL env `NIXL_DISABLE_CUDA_ADDR_WA=1`,
   `NIXL_EP_USE_HOST_RDMA_BUFFER=1` ride along for the P2P engine.
7. **`PYTHONHASHSEED=0`** everywhere — hash-based KV dedup silently breaks otherwise
   (`mods/pd-disagg/mooncake-env.sh:18-20`, `README.md:74`, `:172`;
   `lmcache-server-entrypoint.sh:94-97`).
8. **Mooncake offload flags on GB10**: `MOONCAKE_OFFLOAD_USE_URING=false` (GB10 io_uring +
   DirectIO alignment — only set SSD restore surfaces alignment faults,
   `mods/pd-disagg/README.md:173-174`); SSD tier write-once/immutable bucket backend,
   `--offload_on_evict` stays false (spill-only) (`README.md:169-171`;
   `templates/mooncake-master.service.tmpl:12-14`).
9. **RDMA devices**: RoCE `roceP2p1s0f1` with uverbs1/uverbs3 + rdma_cm and
   `memlock=-1` everywhere (`compose.yaml:79-84`,
   `lmcache-server-entrypoint.sh:114-118`,
   `templates/mooncake-client.service.tmpl:30-33`).
10. **Client healthcheck shape**: no TCP check on :50053 — P2P-handshake mode binds an
    ephemeral RPC port (observed 16800); probe the process table instead
    (`compose.yaml:114-125`).

`[INFERENCE]` The engine_driven requirement interacts with item 2's wording in two places:
the Dockerfile comment says the CUDA 13.3 runtime exists "for CUDA IPC between MP server and
vLLM workers" (`docker/lmcache-server/Dockerfile:11`) while the dispatch/entrypoint comments
say cross-container CUDA IPC mapping fails on GB10 and must be avoided via engine_driven
(`dispatch.sh:191-195`, `lmcache-server-entrypoint.sh:40-45`). The consistent reading is
that the base image carries CUDA IPC support for in-container sharing, while the
cross-container zero-copy path is the one GB10 cannot do — the engine_driven mode is the
mandated workaround. No file states this reconciliation explicitly.

---

## 7. Contract summary table (per KV backend)

| | mooncake (production default) | lmcache MP (alternative PD path) | lmcache-offload (non-PD) |
|---|---|---|---|
| vLLM connector | `MooncakeStoreConnector` `kv_producer`/`kv_consumer` | `LMCacheMPConnector` + `--disable-hybrid-kv-cache-manager` | `LMCacheConnectorV1` in-process |
| extra_config | `host_staging=true, staging_num_slots=2, staging_slot_size_mb=2048` | `lmcache.mp.host=tcp://127.0.0.1, lmcache.mp.port=5555, mp_transfer_mode=engine_driven` | config YAML `/tmp/lmcache-config.yaml` |
| Sidecar containers | `mooncake-master` (rank-0, :50051 TCP, profile `rank0`) + `mooncake-client` (every node, :50053, RoCE, profile `client`), image `mooncake-store:v0.3.12.post1-1` | `mooncake-master` ONLY (rank 0, :50051) + `lmcache-server-<node>` per node (`lmcache server`, :5555 ZMQ, :8080/:8082 HTTP), image `lmcache-server` (vllm-gb10 overlay) | none |
| Boot-time pip installs (engine container) | `mooncake-transfer-engine-cuda13==0.3.12.post1`, `cupy-cuda13x<14.1.0`, `xxhash` | `lmcache` wheel (dgx-spark-wheels, `--no-deps`) + filtered requirements + `xxhash` | same lmcache wheel + requirements + `valkey-glide-sync` |
| Key env (vLLM ranks) | `MOONCAKE_CONFIG_PATH=/tmp/pd-disagg/mooncake-store.json`, `MOONCAKE_PREFERRED_SEGMENT=<ip>:50053`, `MOONCAKE_REQUESTER_LOCAL_HOSTNAME=<ip>`, master addr/port, `PYTHONHASHSEED=0` | `LMCACHE_MP_HOST=tcp://127.0.0.1`, `LMCACHE_MP_PORT=5555`, master addr/port, `PYTHONHASHSEED=0` | `LMCACHE_CONFIG_FILE`, `LMCACHE_LOCAL_CPU_GIB=8`, `LMCACHE_LOCAL_DISK_GIB=50` |
| Key env (daemons) | client: `MOONCAKE_SEGMENT_SIZE_GIB=24`, `MOONCAKE_OFFLOAD_MAX_GIB=96`, offload bucket env (4 GiB staging buffer `4294967296`, 96 GiB NVMe cap `103079215104`), cgroup 32 GiB / 40 GiB | server: `LMCACHE_MOONCAKE_MASTER_ADDR`, `LMCACHE_MOONCAKE_GLOBAL_SEGMENT_BYTES` (=recipe segment), `LMCACHE_SUPPORTED_TRANSFER_MODE=engine_driven`, `LMCACHE_L1_SIZE_GB=1` | — |
| L2 tiers | client DRAM segment (24 GiB) + NVMe SSD offload (`/nvme/mooncake_offload`, 96 GiB bucket) | `mooncake_store` (RDMA, 16 GiB default seg / 4 GiB buffer), `fs_native` (NFS), `valkey` (Dell), optional P2P (:6767) | `fs_native` per-node NFS + `valkey://10.10.10.53:6379` |
| Ports | 50051 master, 50053 client, 8000 pd-proxy, 8100/8200 APIs | 50051 master, 5555 MP ZMQ, 8080/8082 MP HTTP, 6767 P2P, 8000 router | none (in-process) |

Citations for each cell are in the sections above; port map line:
`mods/pd-disagg/README.md:56-57`.
