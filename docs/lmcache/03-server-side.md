# 03 - Server side: image, unit, and entrypoint

The MP server is a long-lived daemon per serving rank that owns the L1 host-DRAM
pool, the L2 adapters, eviction and prefetch. It is built as its own image, run as
`lmcache-server@<host>.service`, and configured entirely through one bash
entrypoint. Read this before changing anything under
`cluster-config/docker/lmcache-server/`, `cluster-config/templates/lmcache-server@.service.tmpl`,
or `cluster-config/scripts/lmcache-server-entrypoint.sh`.

All paths in this document are in the `cluster-config` repo.

## Why three files

| File | Owns | Changes when |
|---|---|---|
| `docker/lmcache-server/Dockerfile` | What is *inside* the container: lmcache wheel, valkey-glide, ctypes shim | Wheel version, dependency, or a new boot-time patch |
| `templates/lmcache-server@.service.tmpl` | The systemd unit: `ExecStart` only | Essentially never |
| `scripts/lmcache-server-entrypoint.sh` | The `lmcache server` argv, the docker run flags, all env plumbing | Almost every operational change |

The split exists because the unit is rendered through `envsubst`, and **envsubst and
systemd expand neither `${VAR:-default}` nor `$((...))`** - verified live on
systemd 255. Anything with a fallback or arithmetic therefore cannot live in the
template; it has to live in the entrypoint, which is a plain script invoked as
`ExecStart` and never passed through envsubst. That is why the unit is 41 lines and
the entrypoint is 154.

## Image

Thin overlay, not a rebuild:

```
ARG BASE_IMAGE=10.100.170.3:5000/vllm-gb10:0.28.0-sm121-cu133
FROM ${BASE_IMAGE}
```

**Base-image precedence matters.** The `ARG` default is a *fallback* for a manual
`docker build`. Every real build goes through
`scripts/build-lmcache-server-image.sh`, which passes

```
--build-arg "BASE_IMAGE=${LLM_REGISTRY_HOST}:${LLM_REGISTRY_PORT}/${LLM_IMAGE_NAME}:${LLM_IMAGE_TAG}"
```

so the base tracks `cluster.env`'s `LLM_IMAGE_TAG` and moves when that is bumped.
The fallback still has to stay in step, because the lmcache wheel is compiled
against one specific torch build: a stale fallback silently produces an image
whose torch does not match the wheel.

The base supplies the from-source torch 2.13.0+cu133 (sm_121) that the wheel was
built against, the CUDA 13.3 runtime needed for CUDA IPC, and the GB10/SM121
kernel workarounds. Nothing else is inherited.

Install order in the image:

1. The pinned wheel, `--no-deps` - `lmcache-0.5.5rc1+cu13.3torch2.13.glibc239-cp312-cp312-linux_aarch64.whl`
   from the `lmcache-mooncake-v0.5.5rc1-cu13.3` release. `--no-deps` because torch
   and the C++ extension runtimes are already in the base.
2. `valkey-glide-sync>=2.3.0`. The **sync** variant specifically: the plain
   `valkey-glide` package is async-only, and the Valkey L2 adapter's worker pool
   needs the synchronous GLIDE client.
3. The filtered `requirements-common.txt` (same filter as the engine side: drop
   torch, numpy, cufile-python, pytest, setuptools\*, nixl).
4. The **ctypes handle shim**, as a `RUN python3 - <<'PYEOF'` step.
5. A smoke probe that imports the CUDA and native extensions.

### Why the ctypes shim is baked into the image

The pinned cuda-python bindings expose `cudaIpcMemHandle_t` as an opaque object
with only `getPtr()` (the address of the internal 64-byte buffer) and no
`.reserved` byte-array field. lmcache touches `.reserved` directly at four sites
in two files, and this container runs the **export** side of those handles, so it
crashes at KV-cache registration with
`'cudaIpcMemHandle_t' object has no attribute 'reserved'`.

It cannot be patched from the entrypoint. The entrypoint runs on the **host**
before `docker run`, so:

- the container it would need to `docker exec` into does not exist yet, and
- racing the server's own import would be unsound even if it did.

Baking it into the image puts the patch in place before the process starts. The
engine side needs the same patch, applied by `mods/lmcache-mp-engine` on that
side; see `02-engine-side-mods.md` for the four rewrite pairs and the
`import ctypes` anchoring trap that makes a naive patch silently no-op.

### Mutable tag

The image tag is `LMCACHE_IMAGE_TAG=0.5.5rc1-sm121-cu133`, and the launcher
**re-pulls it on every unit start**. A rebuilt image is therefore picked up with no
unit or cluster.env change - but it also means "I rebuilt it" and "the running
container has it" are different facts. Confirm with
`docker inspect --format '{{.Image}}' lmcache-server-<host>` and compare against
the registry digest.

## Unit template

```ini
ExecStart=${CLUSTER_CONFIG_REPO_DIR}/scripts/lmcache-server-entrypoint.sh %i
```

That is the whole configuration surface. `%i` is the instance name, which by
convention is the node hostname (`gx10-cb11`, `gx10-f1d8`, ...), and it becomes the
container name `lmcache-server-<instance>` and the `LMCACHE_MOONCAKE_LOCAL_HOSTNAME`
default.

## Entrypoint

The entrypoint sources `lib.sh`/`cluster.env`, assembles the argv, then
`docker run`s the image with `--entrypoint lmcache ... server`.

### Always-present arguments

| Flag | Source | Note |
|---|---|---|
| `--instance-id` | `%i` | Also the container name suffix |
| `--chunk-size` | `LMCACHE_CHUNK_SIZE` | **argparse only** - see below |
| `--host` | literal `0.0.0.0` | ZMQ bind |
| `--port` | `LMCACHE_PORT` (5555) | ZMQ data/control plane |
| `--http-port` | `LMCACHE_HTTP_PORT` (8082) | `/metrics`, `/status`, `/healthcheck` |
| `--l1-size-gb` | `LMCACHE_L1_SIZE_GB` (1) | Host DRAM pool |
| `--eviction-policy` | `LMCACHE_EVICTION_POLICY` (LRU) | |
| `--hash-algorithm` | `LMCACHE_HASH_ALGORITHM` (blake3) | Must agree with the engine's hash seeding |
| `--max-workers`, `--max-gpu-workers`, `--max-cpu-workers` | `LMCACHE_MAX_*` | |
| `--supported-transfer-mode` | `LMCACHE_SUPPORTED_TRANSFER_MODE` | Default `engine_driven`; the lmcache-ext drop-in sets `auto` |

### Conditional arguments

| Flag | Emitted when |
|---|---|
| `--isolated-ipc` | `LMCACHE_ISOLATED_IPC` is non-empty |
| L2 adapter blocks | `LMCACHE_L2_DISABLED != 1` **and** the adapter's own env var is non-empty |

### Chunk size: the flag, not the env var

`--chunk-size "${LMCACHE_CHUNK_SIZE:-256}"`.

This is the single most-misdiagnosed knob in the whole stack, because the env var
reaches the container perfectly well and is still ignored. The server's config
object takes `chunk_size=args.chunk_size` - argparse only, no env fallback. An
`-e LMCACHE_CHUNK_SIZE=1648` on the server container therefore changes nothing.

The value must be a **multiple of the engine's resolved attention block size**.
For a mamba-align hybrid that resolved size is larger than the raw `--block-size`:
Qwen3_5 pads the attention block so its page covers the padded mamba page, landing
on 1648 tokens (103 × 16). Wheel default 256 fails at registration with
`LMCache chunk size 256 must be a multiple of 1648 (the vLLM block size scaled by
decode_context_parallel_size)`.

All three channels must agree: recipe env (engine config loader), launcher `-e`
(the engine container), and this flag (the server). See `05-failure-modes-and-patches.md`.

### `--isolated-ipc`

Selects `RawCudaIPCWrapper` on both sides - driver-level
`cudaIpcGetMemHandle`/`cudaIpcOpenMemHandle` instead of PyTorch's
`_share_cuda_`, which requires a `/dev/shm` the two containers do not share.

It must match the engine's `lmcache.mp.isolated_ipc` exactly. A mismatch means the
server registers no matching handler and `register_kv_caches` **times out** rather
than erroring usefully - the engine just hangs at startup.

### `--supported-transfer-mode`

| Value | Behaviour | Used by |
|---|---|---|
| `engine_driven` | Worker-side gather/scatter over ZMQ; no cross-container IPC. Single-group only. | PD-lmcache path (the default) |
| `lmcache_driven` | Server-pulled zero-copy over CUDA IPC. Multi-group capable. | Requires `--isolated-ipc` |
| `auto` | Registers **both** handler families; the engine picks via `lmcache.mp.mp_transfer_mode` | The `lmcache-ext` drop-in |

`auto` is the safe server-side default for a mixed fleet: the PD path's engines
still ask for `engine_driven` and get it, while the hybrid corridor asks for
`lmcache_driven`. Pinning the server to a single mode forces every consumer onto it.

### L1 sizing

`LMCACHE_L1_SIZE_GB=1` is deliberate. GB10 is unified-memory: the CPU pool and the
GPU KV pool draw from one LPDDR5X bank, so every GiB L1 holds is a GiB
`gpu_memory_utilization` cannot. At 1 GiB with 1648-token chunks the pool
constantly recycles, which is why `prefetch_failure_chunks_total{reason="l1_oom"}`
sits in the hundreds. That is steady state, not a fault: the chunk was fetched
from L2 and could not be retained, and the request falls through to recompute.

If you want the hit ratio up, raise L1 **and** lower the recipe's
`gpu_memory_utilization` by a matching amount - the same arithmetic as the
mooncake 24/96 sizing.

### Docker run shape

```
--network host            # so 5555/8082 are reachable on the node's mgmt IP
--runtime nvidia --gpus all
--shm-size 8g
--device /dev/infiniband/...   # RDMA for the tier that needs it
--ulimit memlock=-1
-v ${MOUNT_POINT}/lmcache-l2:/cluster-shared/lmcache-l2
```

Host networking is not optional: the engine dials `127.0.0.1:5555`, and
VictoriaMetrics scrapes `<mgmt-ip>:8082` - both assume the host namespace.

## L2 adapters

`--l2-adapter` takes one JSON object per adapter. Types are registered under
`lmcache/v1/distributed/l2_adapters/` in the wheel.

| Type | JSON shape | Semantics |
|---|---|---|
| `fs_native` | `{type, base_path, num_workers, max_capacity_gb}` | Shared NFS directory. Every rank points at the same store, so the directory is one object, not one per node. |
| `valkey` | `{type, startup_nodes, key_prefix, max_capacity_gb, num_workers}` | Remote KV store on the Dell. `startup_nodes` is a **comma-separated `host:port` string**, not a list. |
| `mooncake_store` | `{type, num_workers, local_hostname, metadata_server, protocol, rdma_devices, master_server_addr, global_segment_size, local_buffer_size}` | Only emitted when `LMCACHE_MOONCAKE_MASTER_ADDR` is set; also adds `--no-l1-use-lazy`. This is the PD corridor's tier. |
| `p2p` | - | Present in the metrics but unused in this deployment. |

Gating rule for all of them: emitted only if `LMCACHE_L2_DISABLED != 1` **and** the
adapter's own env var is non-empty. That makes L2 enable/disable a cluster.env edit
plus a unit restart, with no script change.

### Write fan-out, and the aggregation trap

Each stored chunk is written to every attached tier, so `fs_native` and `valkey`
normally report **identical** `l2_usage_bytes`. They are the same logical content in
two places, and they are each *shared across ranks*, so:

- read one tier to learn the footprint;
- aggregate across ranks with `max()`, never `sum()` - `sum()` counts the shared
  store once per rank;
- a tier reading `0` on one rank simply means that rank has not written it yet.

Per-process counters (lookup/prefetch/store rates, L1 residency) are *not* shared
and are legitimately summed.

## Operator endpoints

| Endpoint | Purpose |
|---|---|
| `:8082/metrics` | Prometheus text; the `lmcache_mp_*` family |
| `:8082/status` | JSON: `chunk_size`, attached adapters, prefetch controller health |
| `:8082/healthcheck` | The launch gate - `spark-serve.sh` polls this for 60 s before starting the engine |
| `:5555` | ZMQ, not HTTP - a TCP probe there will connect but never answer |

`/status` is the fastest way to confirm the chunk-size chain and whether the L2
adapters actually attached; see `06-verification-and-observability.md` for how to
read it and for the metric semantics.
