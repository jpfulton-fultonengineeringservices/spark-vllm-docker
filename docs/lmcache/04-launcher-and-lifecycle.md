# 04 - Launcher and lifecycle (cluster-config side)

The recipe only says *serve*: it pins `pd_kv_backend`, the model, and the `vllm serve` command. The cluster-config launcher (`infra/cluster-config/scripts/lib/spark-serve.sh`, referenced here as `cluster-config/scripts/lib/spark-serve.sh`) decides everything operational: which KV-transfer corridor activates, what sidecar daemons run on which ranks, which environment variables actually reach `docker run`, when the stack is healthy, and how every piece comes back down. This document covers the launcher mechanics for the non-PD `lmcache-ext` corridor (working recipe `recipes/qwen3.8-27b-coder390-2x-lmcache-l2.yaml`), the two MP-server launchers, the env-source matrix, teardown, and bring-up ordering plus the cluster-control-infra (CCI) placement notes. Corridor and connector semantics live in `01-architecture-and-corridors.md`; engine-side mods in `02-engine-side-mods.md`; server image/unit/entrypoint in `03-server-side.md`; fleet-wide context in `README.md`.

## The `lmcache-ext` backend id

The recipe declares the backend under `defaults:` (`gpu-cluster-forks/spark-vllm-docker/recipes/qwen3.8-27b-coder390-2x-lmcache-l2.yaml:86`):

```yaml
defaults:
  # ...
  pd_kv_backend: lmcache-ext
```

`cluster-config/scripts/lib/spark-recipe.sh` parses it with a generic `sed` extraction - no backend whitelist (`spark-recipe.sh:169-170`):

```bash
v="$(sed -n 's/^[[:space:]]*pd_kv_backend:[[:space:]]*//p' "$RECIPE_FILE" | head -1 | tr -d "\"'" | sed 's/[[:space:]]*#.*$//')"
SPARK_VLLM_DOCKER_KV_BACKEND="${v:-mooncake}"
```

A missing line resolves to `mooncake`, so pre-`pd_kv_backend` recipes keep the proven mooncake cohort path; `--kv-backend` CLI override wins over the recipe default (`spark-recipe.sh:173`). Because parsing is generic, a new id like `lmcache-ext` needs **no parser change** - the recipe resolved into `SPARK_VLLM_DOCKER_KV_BACKEND` is just a string that `spark-serve.sh` branches on, and it is threaded into the container template verbatim so the in-container `PD_KV_BACKEND` placeholder agrees with the launcher branch (`spark-serve.sh:777-779`).

Why a new id instead of reusing `lmcache`: the existing `lmcache` id is nested inside the **PD-recipe case arm** (`spark-serve.sh:672-678`). That arm exists because the PD MP server's `mooncake_store` L2 adapter needs the mooncake metadata master - it injects only `LMCACHE_MP_HOST`/`LMCACHE_MP_PORT` and then calls `launch_mooncake_master_only` (a mooncake master compose project, no client cohort) **plus** `launch_lmcache_mp_servers "${master_ip}"`. The non-PD corridor wants the opposite: MP servers with `fs_native` + `valkey` L2 adapters and **no mooncake master at all** (the 24 GiB DRAM segment + 96 GiB NVMe offload cohort is replaced by the MP pools; see `01-architecture-and-corridors.md`). Reusing `lmcache` would have forced the mooncake-master launch on a non-PD recipe or made the PD nesting conditional on topology - a new id keeps both arms flat and readable. The launcher enforces the pairing with a hard die: if the recipe pins `pd_kv_backend: lmcache-ext` but the resolved backend differs, launch aborts (`spark-serve.sh:754-755`).

## The non-PD case arm in `spark-serve.sh`

The arm lives in `cmd_launch`, outside the PD case (`spark-serve.sh:742-771`):

```bash
# 2x LMCache MP (non-PD, lmcache-ext backend): per-node lmcache-server
# units own the L1/L2 pools; vLLM attaches via LMCacheMPConnector
# (kv_both) over localhost ZMQ. NO mooncake cohort -- the ext backend
# replaces the store-l2 sibling's DRAM segment + NVMe offload with an
# MP-side L1 + NFS/valkey L2.
case "$SPARK_VLLM_DOCKER_RECIPE" in
qwen3.8-27b-coder390-2x-lmcache-l2*)
    [ "${SPARK_VLLM_DOCKER_KV_BACKEND:-mooncake}" = "lmcache-ext" ] \
        || die "$SPARK_VLLM_DOCKER_RECIPE: recipe pins pd_kv_backend: lmcache-ext but resolved backend is '${SPARK_VLLM_DOCKER_KV_BACKEND:-mooncake}'"
    args+=(
        "-e" "LMCACHE_MP_ENABLED=1"
        "-e" "LMCACHE_MP_HOST=tcp://127.0.0.1"
        "-e" "LMCACHE_MP_PORT=${LMCACHE_PORT:-5555}"
        # Chunk size must match MP server's (threaded into unit drop-in
        # from cluster.env value) and be a multiple of the model's
        # resolved attention block size for mamba-align hybrids
        # (Qwen3_5: 1648). The recipe also declares it, but the -e is
        # what actually lands in the container (recipe env cannot reach
        # the docker-run mod/serve process).
        "-e" "LMCACHE_CHUNK_SIZE=${LMCACHE_CHUNK_SIZE:-256}"
        "-e" "PYTHONHASHSEED=0"
    )
    launch_lmcache_mp_servers_external
    ;;
esac
```

Five `-e` pairs are injected into the `docker run` args that `run-recipe.sh`/`fes-eval.sh` forward to the fork's `launch-cluster.sh`. Each exists for a distinct reason:

| `-e` pair | Why it exists |
|---|---|
| `LMCACHE_MP_ENABLED=1` | Activates `mods/lmcache-mp-engine/run.sh`, which is opt-in and inert otherwise (`mods/lmcache-mp-engine/run.sh:17-19`, gate at `:47`). Without it the mod would skip wheel install and all four fork patches (device-ordinal clamp, block-id suppression, ctypes shims). |
| `LMCACHE_MP_HOST=tcp://127.0.0.1` | ZMQ endpoint the MP connector dials; mirrors the PD corridor's contract (`mods/pd-disagg/dispatch.sh:230-233`) so triage tooling sees the same variable either way. The engine's *actual* connection parameters travel in the recipe's `--kv-transfer-config` JSON (below) - this pair is corridor parity, not the live dial target. |
| `LMCACHE_MP_PORT=${LMCACHE_PORT:-5555}` | Same parity rationale; port must match the server unit's `--port` (cluster.env `LMCACHE_PORT=5555`, `cluster.env:1724`). |
| `LMCACHE_CHUNK_SIZE=${LMCACHE_CHUNK_SIZE:-256}` | The **only** copy of chunk size that reaches the container for the *engine/mod* side. Recipe env (`recipes/...lmcache-l2.yaml:139`) exports inside the exec-script that runs `vllm serve` - mods run BEFORE that exec-script, in a separate process, so they never see it (`mods/lmcache-mp-engine/run.sh:42-45`). |
| `PYTHONHASHSEED=0` | Hash-keyed KV chunk agreement: engine, mods, and MP server must share deterministic hashing (`recipes/...lmcache-l2.yaml:128-130`, server entrypoint `:127`). |

**Why `-e` is the only channel that reaches `docker run`:** recipe `env:` is turned into `export KEY="value"` lines inside the generated exec-script (`run-recipe.py:606-611`), which runs *inside* the container as the process that eventually execs `vllm serve`. `launch-cluster.sh` applies mods to the container first, in a separate process whose environment has no recipe exports. Launcher `-e` flags, by contrast, are part of the container creation command, so they are visible to mods, the exec-script, and everything downstream. This is the same mechanism the PD arm uses for `MOONCAKE_*` and the offload family uses for `LMCACHE_OFFLOAD_ENABLED` (`spark-serve.sh:737-741`).

The engine's connector configuration itself does **not** travel through env at all - the recipe `command:` pins it (`recipes/qwen3.8-27b-coder390-2x-lmcache-l2.yaml:173`):

```
--kv-transfer-config '{"kv_connector":"LMCacheMPConnector","kv_role":"kv_both",
  "kv_connector_extra_config":{"lmcache.mp.host":"tcp://127.0.0.1","lmcache.mp.port":5555,
  "lmcache.mp.mp_transfer_mode":"lmcache_driven","lmcache.mp.isolated_ipc":true}}'
```

After the arm, `cmd_launch` finally invokes `run-recipe.sh`/`fes-eval.sh` with the accumulated args (`spark-serve.sh:785-789`) - the MP servers were launched first (below).

## Two MP-server launchers compared

Both live in `cluster-config/scripts/lib/spark-serve.sh`: `launch_lmcache_mp_servers` (`:420-484`, PD) and `launch_lmcache_mp_servers_external` (`:496-549`, non-PD `lmcache-ext`).

| Aspect | `launch_lmcache_mp_servers` (PD) | `launch_lmcache_mp_servers_external` (non-PD) |
|---|---|---|
| Signature | Takes `master_ip` (`:421`) | No args; everything from cluster.env (`:497`) |
| Unit rendered | `lmcache-server@.service.tmpl`, base64-shipped (`:428`) | Same (`:499`) |
| `20-l2.conf` drop-in | Mooncake trio: `LMCACHE_MOONCAKE_MASTER_ADDR=<master_ip>:50051`, `LMCACHE_MOONCAKE_LOCAL_HOSTNAME=$host`, `LMCACHE_MOONCAKE_GLOBAL_SEGMENT_BYTES=<recipe segment>` (`:449`) | Ten-key ext set: `LMCACHE_L2_DISABLED`, `LMCACHE_FS_NATIVE_PATH`, `LMCACHE_VALKEY_NODES`, `LMCACHE_VALKEY_PREFIX`, `LMCACHE_VALKEY_CAPACITY_GB`, `LMCACHE_VALKEY_WORKERS`, `LMCACHE_FS_NATIVE_CAPACITY_GB`, `LMCACHE_CHUNK_SIZE`, `LMCACHE_SUPPORTED_TRANSFER_MODE` (literal `auto`), `LMCACHE_ISOLATED_IPC` (`:520`) |
| L2 adapter | `mooncake_store` (needs the rank-0 master) | `fs_native` + `valkey`, gated by `LMCACHE_L2_DISABLED` |
| Image re-pull | `docker pull $LMCACHE_IMAGE_REF`, fail = `LMCACHE_IMAGE_PULL_FAIL` (`:458-460`) | Same, fail = `LMCACHE_EXT_IMAGE_PULL_FAIL` (`:527-529`) |
| Start | `systemctl enable` (tolerated) + `restart` (`:465-467`) | Same (`:530-532`) |
| Health poll | 60 s loop: `is-failed` check then `curl /healthcheck`; emits `LMCACHE_FAIL` / `LMCACHE_HEALTH_TIMEOUT` (`:468-478`) | Same shape; emits `LMCACHE_EXT_FAIL` / `LMCACHE_EXT_OK` / `LMCACHE_EXT_HEALTH_TIMEOUT` (`:533-543`) |
| Failure tag prefix | `LMCACHE_*` (`LMCACHE_UNIT_INSTALL_FAIL`, `LMCACHE_DROPIN_INSTALL_FAIL`, ...) | `LMCACHE_EXT_*` (`LMCACHE_EXT_UNIT_INSTALL_FAIL`, `LMCACHE_EXT_DROPIN_INSTALL_FAIL`, ...) |

What **both** do, in order, per rank over `RANK_HOSTS` (skipping non-`THIS_HOST` under `--local-only`):

1. **Render the unit** from `lmcache-server@.service.tmpl` via `render_template` and base64 it into the remote body (`:428`/`:499`; `cluster-config/scripts/lib.sh:163`).
2. **Install it** at `/etc/systemd/system/lmcache-server@.service` (`:443-444`/`:514-515`).
3. **Write the `20-l2.conf` drop-in** into `/etc/systemd/system/lmcache-server@<host>.service.d/` - this is where the two cohorts diverge, because the entrypoint's L2-adapter blocks read exactly these env vars (`cluster-config/scripts/lmcache-server-entrypoint.sh:76-118`).
4. **`daemon-reload`** so systemd sees the new unit + drop-in (`:453`/`:524`).
5. **Re-pull the mutable image tag.** `LMCACHE_IMAGE_REF` (`cluster.env:1722`, tag `0.5.5rc1-sm121-cu133`) is mutable: a node that already cached the old tag would otherwise start a stale image whose L2 adapter mismatches the fleet. The pull is cheap when the digest is unchanged (`:454-460`/`:525-529`).
6. **`systemctl enable` then `systemctl restart` - never `enable --now`.** The launcher's own comment is the reason (`:461-464`): an already-active unit is a *no-op* for `enable --now`, so a changed drop-in (new segment size, flipped `LMCACHE_L2_DISABLED`) would silently never take effect on a re-launch. `restart` is idempotent and always applies the current drop-in env.
7. **Poll `/healthcheck` on `:8082` for 60 s** (2 s interval, `is-failed` short-circuit). MP-side health gate; engine-side is `/v1/models` (below).

The prefixes are deliberately distinct (`:495`: "tag failure markers with the cohort prefix (`LMCACHE_EXT_*`) so log triage can tell the two cohorts apart"). A mixed fleet - PD `lmcache` ranks and non-PD `lmcache-ext` ranks - produces unambiguous `journalctl`/deploy-log greps.

## Env-source matrix

The corridor's configuration arrives through four distinct channels, and mixing them up is the classic failure. `L` = launcher `-e` (docker run, reaches mods + exec-script), `R` = recipe `env:` (exec-script export only, invisible to mods), `D` = ext drop-in `20-l2.conf` (unit env → entrypoint), `C` = cluster.env (feeds L or D at render/launch time), `K` = recipe `command:` `--kv-transfer-config` JSON (engine args, not env at all).

| Variable / knob | Channels | Consumers | Notes |
|---|---|---|---|
| `LMCACHE_CHUNK_SIZE` | `R` (`recipes/...lmcache-l2.yaml:139`), `L` (`spark-serve.sh:766`), `C`→`D`→`--chunk-size` CLI (`entrypoint:38`), `C`→`L` default `256` | Engine connector + MP server | **THREE channels must agree.** The MP server reads chunk size ONLY from argparse - `MPServerConfig` takes `chunk_size=args.chunk_size`; the entrypoint comment is explicit: "CLI flag (`--chunk-size`), NOT env var - server's `MPServerConfig` only reads argparse" (`entrypoint:35-38`). A cluster.env edit alone fixes the server but not the engine; only the launcher `-e` (sourced from cluster.env) fixes both sides. Must be a multiple of the engine's resolved attention block size (1648 for Qwen3_5 mamba-align; see `01-architecture-and-corridors.md`). |
| `LMCACHE_MP_ENABLED` | `L` only (`spark-serve.sh:757`) | `mods/lmcache-mp-engine/run.sh` gate (`run.sh:47`) | Recipe env cannot activate the mod (separate process); drop-in is unit-scope and never reaches the engine container. |
| `LMCACHE_MP_HOST` / `LMCACHE_MP_PORT` | `L` (`:758-759`) + `K` (`recipes/...:173`) | Parity tooling (L) vs actual connector dial (K) | The connector dials what `--kv-transfer-config` says; the `-e` pairs keep PD/non-PD log and tooling parity. |
| `lmcache.mp.*` (`mp_transfer_mode`, `isolated_ipc`, ...) | `K` only | `LMCacheMPConnector` extra_config | NOT env, NOT drop-in. `lmcache_driven` + `isolated_ipc: true` is the working hybrid corridor (`01-architecture-and-corridors.md`). |
| `LMCACHE_ISOLATED_IPC` | `C`→`D` (`spark-serve.sh:520`) | Entrypoint → `${LMCACHE_ISOLATED_IPC:+--isolated-ipc}` server flag (`entrypoint:47-48`) | Server-side only; engine side gets `isolated_ipc` from `K`. cluster.env pins `LMCACHE_ISOLATED_IPC=1` (`cluster.env:1776`). |
| `LMCACHE_L2_DISABLED` | `C`→`D` (`:520`) | Entrypoint L2 gating (`entrypoint:76-118`); PD launcher's ok-line echoes it | cluster.env `LMCACHE_L2_DISABLED=0` (`cluster.env:1748`); `=1` is the proven L1-only fallback (`docs/pd-router/lmcache-parity-fixes.md`). |
| `LMCACHE_FS_NATIVE_PATH` / `_CAPACITY_GB`, `LMCACHE_VALKEY_NODES` / `_PREFIX` / `_CAPACITY_GB` / `_WORKERS` | `C`→`D` (`:520`) | Entrypoint adapter blocks | cluster.env values at `cluster.env:1750-1759` (NFS `/cluster-shared/lmcache-l2`, Dell `10.10.10.53:6379`). |
| `LMCACHE_SUPPORTED_TRANSFER_MODE` | `C`→`D`, literal `auto` in ext drop-in (`:520`) | Entrypoint `--supported-transfer-mode` (`entrypoint:68`) | `auto` registers both handlers; PD cohort keeps `engine_driven`. |
| `PYTHONHASHSEED=0` | `R` (`recipes/...:130`), `L` (`:767`), server `-e` in `DOCKER_ENV_FLAGS` (`entrypoint:127`) | All three processes | Belt-and-suspenders everywhere hashing matters. |

The trap in one sentence: **an env var's presence in cluster.env guarantees nothing** - each consumer reads exactly one channel, and the drop-in is frozen at unit-render time, so a cluster.env edit without a re-launch (or an `--isolated-ipc`-missing server) is a silent mismatch.

## Teardown

`cmd_stop` runs a per-rank body over `each_ordered_rank` (`spark-serve.sh:855-885`); the CCI teardown wrapper `teardown-spark-vllm-docker-serve.sh` reaches it via the impl's `stop` verb (`cluster-config/scripts/teardown-spark-vllm-docker-serve.sh:8-14`):

```bash
docker rm -f vllm_node >/dev/null 2>&1 || true
docker compose -p pd-disagg-mooncake -f '.../docker/mooncake-store/compose.yaml' down --remove-orphans >/dev/null 2>&1 || true
_h=$(hostname -s)
if systemctl cat "lmcache-server@${_h}.service" >/dev/null 2>&1; then
  sudo systemctl disable --now "lmcache-server@${_h}.service" >/dev/null 2>&1 || true
  sudo rm -f "/etc/systemd/system/lmcache-server@${_h}.service.d/20-l2.conf" 2>/dev/null || true
  sudo systemctl daemon-reload >/dev/null 2>&1 || true
  echo LMCACHE_MP_STOPPED host=${_h}
fi
```

Design points, all from the launcher's own commentary (`:856-871`):

- **Mooncake compose teardown is scoped to project `pd-disagg-mooncake`.** The mooncake/lmcache backends' launch path brings that cohort up, so stop tears it down - otherwise a `store-l2` → `lmcache-ext` replacement leaves 24 GiB of idle segments pinned on the ranks. The project-name scope means co-tenants (cadvisor, registry) can never be touched, and the next mooncake-backed launch rebuilds the cohort.
- **The per-node `lmcache-server@<host>` units ARE stopped.** They bind `:5555`/`:8082` and their container name (`lmcache-server-<host>`) is fixed, so leaving them installed across a profile swap means the next placement boots alongside a foreign MP server still holding the ports - with a `20-l2.conf` drop-in pointing at a dead mooncake master. Stop therefore does `disable --now`, **removes the drop-in**, and `daemon-reload`s, then emits `LMCACHE_MP_STOPPED host=<h>` for log triage.
- **Rollback hazard:** a `lmcache-ext` → `store-l2` rollback must re-run the mooncake-backed launch to rebuild the cohort *and* re-render the mooncake drop-in; a stop that skipped the drop-in removal (older tooling) would leave a stale `LMCACHE_MOONCAKE_MASTER_ADDR` pointed at a master that no longer exists. The current `cmd_stop` closes that hole by deleting the file.
- **L2 data survives.** `fs_native` lives on NFS (`/cluster-shared/lmcache-l2`) and `valkey` on Dell; both are persistent tiers outside the containers, so stop/restart cycles preserve cached KV and a relaunch warms from them (observed L2-read behavior in `06-verification-and-observability.md`).

## Ordering and gates

**Rank order - highest first.** `spark-topology.sh` builds `ORDERED_NODES` descending by rank (`cluster-config/scripts/lib/spark-topology.sh:76-82`, "start last for multi-node rendezvous"), and every `each_ordered_rank` loop walks that order. The fork's own launch path does the same: `launch-cluster.sh` starts workers before the head (`gpu-cluster-forks/spark-vllm-docker/launch-cluster.sh:1710-1717`, "Launch workers first (always background)"), and CCI's `setupInstance` fans out highest-rank-first for the same reason - rank 0 owns the TCPStore/master port and dies with `DistStoreError: Timed out after 601 seconds` if its workers are not up (`cluster-control-infra/src/lib/placement/orchestrator-setup.ts:114-123`).

**Only rank 0 launches.** The per-node setup wrapper no-ops on non-head hosts - `setup-spark-vllm-docker-serve.sh:84-87` prints `no-op on $this_host (head rank is $head_host; its launch reaches the workers over SSH)` and exits 0 - and `spark-serve.sh` itself refuses to run anywhere but the head (`spark-serve.sh:555-556`). So one SSH invocation on the head fans out to all ranks over SSH.

**MP servers come up BEFORE the engine.** In `cmd_launch`, the `lmcache-ext` arm calls `launch_lmcache_mp_servers_external` (`:769`) *before* the `run-recipe.sh`/`fes-eval.sh` invocation (`:785-789`). This ordering is required, not stylistic: the connector dials `tcp://127.0.0.1:5555` immediately at engine start and blocks on registration (launcher comment `:413-419`; flow diagram in `README.md`). A health-gated server cohort that answers `/healthcheck` before the engine process exists is the contract.

**Health gates.**

| Layer | Gate | Timeout / behavior |
|---|---|---|
| MP servers | `curl http://localhost:8082/healthcheck` per rank (`spark-serve.sh:533-543`) | 60 s poll; `LMCACHE_EXT_FAIL` on failed unit state, `LMCACHE_EXT_HEALTH_TIMEOUT` on expiry; any rank failure dies the launch (`:546-547`) |
| Engine | `curl http://<head>:8000/v1/models` requires `"id"` in body, with a `docker ps` liveness pre-check (`:796-816`) | `WAIT_SEC` budget; `HEALTH_WAIT reason=container_not_running` / `models_endpoint_not_ready`, then `SPARK_VLLM_DOCKER_FAIL` |

**CCI placement notes.** The preset catalog carries the corridor: `qwen38-27b-coder390-lmcache-l2` declares `engine: 'spark-vllm'`, `nodeCount: 2`, and `recipe: 'qwen3.8-27b-coder390-2x-lmcache-l2'` (`cluster-control-infra/src/lib/placement/presets-spark.ts:470-479`). The placement *profile* carries the topology: `rankOrder` (`"host:rank,host:rank"`) and `masterPort` (validated per-engine against `inventory.yaml`'s `sparkVllm` masterPortRange `[29571, 29580]`).

- `?recreate=1` on `POST /api/placement/apply` (`cluster-control-infra/src/app/api/placement/apply/route.ts:274`) threads into `applyPlacement(current, desired, dir, signal, { recreate })` (`:130`), where it forces a restart **only of ARRIVING instances** - `orchestrator.ts:116-123` pushes all `arriving` into `realArrivals` when `opts.recreate` is set, while unchanged already-serving instances are otherwise reused via the liveness probe. A same-id relaunch with *no* diff therefore needs the **`none`-profile-then-target sequence**: apply the `none` profile (an empty instance list is a valid profile shape - `src/app/api/placement/profiles/route.test.ts:74-77`), which makes every current instance "departing" (`orchestrator.ts:82`, teardown set at `:107`), then apply the target profile so it arrives fresh.
- **Warning:** the `none` profile tears down *every* listed instance in the current profile, not just the target - an embedding instance on a different node in the same profile departs with it. Sequence carefully or accept the co-tenant bounce.
- The apply route commits `current` = desired only on success (`route.ts:141-145`), so a failed apply leaves the diff intact and a retry re-runs the still-not-running instances instead of being diffed away as "unchanged".

## Related

- `README.md` - fleet overview and corridor index.
- `01-architecture-and-corridors.md` - why `lmcache_driven` + `isolated_ipc` is the working hybrid corridor.
- `02-engine-side-mods.md` - what the `-e LMCACHE_MP_ENABLED=1` arm activates engine-side.
- `03-server-side.md` - the unit template, entrypoint, and image the launchers render/install.
- `06-verification-and-observability.md` - how to prove the corridor is live and the metrics to watch.
- `docs/pd-router/lmcache-parity-fixes.md` - the two upstream L2-prefetch bugs behind `LMCACHE_L2_DISABLED`.