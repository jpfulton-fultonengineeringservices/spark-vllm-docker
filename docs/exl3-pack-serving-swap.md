# Playbook: Stop the MiMo 4x serve → run EXL3 pack build → restore

Audience: operator. Facts were probed live on 2026-10-07 (~04:27 UTC) via
fulton-remote-shell MCP; re-verify "Preflight" before executing.

## What is actually running (verified)

| Item | Value |
|---|---|
| Serve | `vllm serve XiaomiMiMo/MiMo-V2.6-Flash-RL`, **TP=4 across 4 nodes** (`--nnodes 4`, rank0 = node1, master `10.100.171.3:29571`) |
| Container (every node) | name `vllm_node`, image `vllm-node-b12x`; **PID1 `sleep infinity`**, serve is a `docker exec` child (`b12x-kcache exec vllm serve …`) |
| Orchestration | **NOT systemd `vllm-serve@*`** (no active unit). Driven by cluster-config's **`spark-vllm-docker-serve.sh`** wrapper on node1 → `run-recipe.sh` → `launch-cluster.sh`. `cluster.env` pins `SPARK_VLLM_DOCKER_RECIPE=mimo-v2.6-flash-4x`, `NODES="gx10-cb11 gx10-f1d8 gx10-becc gx10-9273"`, `MASTER_PORT=29571`, `SERVE_PORT=8000`. |
| Repo dir (node1) | `/cluster-shared/projects/spark-vllm-docker` (holds `.env`; `CLUSTER_NODES=10.100.171.3,.4,.2,.1`) |
| Config dir (node1) | `/cluster-shared/projects/cluster-config/scripts/spark-vllm-docker-serve.sh` |
| Recipe identity | **`recipes/mimo-v2.6-flash-4x.yaml`** — matches the live command (b12x, DFlash, fp8 KV, `--gpu-memory-utilization 0.68`, image `vllm-node-b12x`) |
| GPU footprint (per node) | `VLLM::Worker_TP*` ~75 GiB + `EngineCore` ~1.7 GiB |
| Co-tenant | `vllm-embed@bge-m3-embed-1.service` (systemd, healthy) — **leave running** |
| node1 root fs | **92% used, 155 GB free** — work dirs must NOT land on `/` |
| NAS | `/nas-1` 49T, 28T free — pack outputs go here |
| LiteLLM route | `cluster/mimo-v2.6-flash` → `http://10.10.10.165:8000`, reconciler-managed (auto re-registered via `/api/placement/reconcile` after restarts) |

## Pack targets (two models)

| Checkpoint | Node | Alias | Source (verified) |
|---|---|---|---|
| `mimo-v2.6-pro-rl-uncensored` | **node4** | `home-gx10-node4` | `/opt/llm/staging/mimo-v2.6-pro-rl-uncensored` (130 shards; NAS has `…-exl3-v1` already) |
| `mimo-v2.6-flash-rl-uncensored` | **node3** | `home-gx10-node3` | `/opt/llm/staging/mimo-v2.6-flash-rl-uncensored` (65 shards) |

Packing needs only ONE node per model, but the serve occupies all four → stop the fleet.

## Flow

```mermaid
flowchart LR
    A[Preflight] --> B["spark-vllm-docker-serve.sh stop (node1)"]
    B --> C[Verify GPUs freed on all 4 nodes]
    C --> D["detect geometry (node4 / node3)"]
    D --> E["pipeline: convert → repack → NAS"]
    E --> F["assemble servable checkpoint"]
    F --> G[Verify outputs on /nas-1]
    G --> H["spark-vllm-docker-serve.sh launch/health"]
```

## 1. Preflight (read-only)

On node1 confirm serve identity and wrapper, on the target node confirm the
source checkpoint and disk:

```text
shell_open host=gx10-node1
shell_exec: docker ps --format '{{.Names}} {{.Status}}' | grep vllm_node
shell_exec: ls /cluster-shared/projects/cluster-config/scripts/spark-vllm-docker-serve.sh
shell_exec: grep -E '^SPARK_VLLM_DOCKER_RECIPE=|^SPARK_VLLM_DOCKER_NODES=|^SPARK_VLLM_DOCKER_MASTER_PORT=' /cluster-shared/projects/cluster-config/cluster.env

shell_open host=gx10-node4        # or node3 for flash-rl-uncensored
shell_exec: ls /opt/llm/staging/<slug>/ | head
shell_exec: df -h /opt/llm /nas-1
```

## 2. Stop the serve fleet (node1)

```text
shell_open host=gx10-node1
shell_exec: cd /cluster-shared/projects/cluster-config && \
  ./scripts/spark-vllm-docker-serve.sh stop
```

- `stop` → `cmd_stop` → `docker rm -f vllm_node` on **every rank** (sequentially
  via `each_ordered_rank`) — this is the canonical, complete teardown and is
  idempotent (`--rm` keepalive container, no residue).
- Do NOT use `teardown-vllm-serve.sh <instance>` — that targets systemd
  `vllm-serve@*.service` units that are NOT active here.
- Leave `vllm-embed@bge-m3-embed-1.service` running.

### Consequences

- LiteLLM `cluster/mimo-v2.6-flash` (→ `10.10.10.165:8000`) fails (connection
  refused) for the window — expect gateway 5xx/alerts. The route is
  reconciler-managed and re-registers automatically on the next reconcile
  after the serve returns; a bare `POST /api/placement/reconcile` re-arms it
  immediately if needed. Announce the window.
- Containers recreated (`docker run`) on restore; no volume/state loss.

## 3. Verify GPUs freed (all four nodes)

```text
nvidia-smi | tail -8          # no VLLM:: processes
docker ps                     # only bge-m3 embed remains
ss -tlnp | grep -E ':8000'    # head rank 8000 closed
```

GB10 has unified memory: convert needs the full 128 GB pool on the target node.

## 4. Pack build (per model, driven from the WORKSTATION)

Run from the workstation fork checkout (`gpu-cluster-forks/spark-vllm-docker`).
The driver rsyncs only the build context, builds `exl3-pack:cu13.0` on the
node, and streams progress (stage/layers/ETA/Δ poll).

### 4a. node4 → `mimo-v2.6-pro-rl-uncensored`

```bash
# detect first — confirm geometry BEFORE spending GPU hours
./scripts/exl3-pack-drive.sh detect --host gx10-node4 \
  --model mimo-v2.6-pro-rl-uncensored /opt/llm/staging/mimo-v2.6-pro-rl-uncensored

# pipeline: convert → repack → NAS (foreground, streams progress)
./scripts/exl3-pack-drive.sh pipeline --host gx10-node4 \
  --model mimo-v2.6-pro-rl-uncensored \
  /opt/llm/staging/mimo-v2.6-pro-rl-uncensored \
  /opt/llm/fes-projects/exl3-mimo-build/mimo-v2.6-pro-rl-uncensored-work-k3 \
  /opt/llm/fes-projects/exl3-mimo-build/mimo-v2.6-pro-rl-uncensored-mcg-k3 \
  /nas-1/models/mimo/mimo-v2.6-pro-rl-uncensored-exl3-v1 3 mcg

# assemble the servable checkpoint
./scripts/exl3-pack-drive.sh assemble --host gx10-node4 \
  --model mimo-v2.6-pro-rl-uncensored \
  /opt/llm/staging/mimo-v2.6-pro-rl-uncensored \
  /nas-1/models/mimo/mimo-v2.6-pro-rl-uncensored-exl3-v1 \
  /nas-1/models/mimo/mimo-v2.6-pro-rl-uncensored-exl3-v1-serve
```

### 4b. node3 → `mimo-v2.6-flash-rl-uncensored`

Same three commands with `--host gx10-node3`, slug
`mimo-v2.6-flash-rl-uncensored`, source
`/opt/llm/staging/mimo-v2.6-flash-rl-uncensored`, work dirs
`mimo-v2.6-flash-rl-uncensored-{work,mcg}-k3`, outputs
`/nas-1/models/mimo/mimo-v2.6-flash-rl-uncensored-exl3-v1[-serve]`.

```bash
# detect first — confirm geometry BEFORE spending GPU hours
./scripts/exl3-pack-drive.sh detect --host gx10-node3 \
  --model mimo-v2.6-flash-rl-uncensored /opt/llm/staging/mimo-v2.6-flash-rl-uncensored

# pipeline: convert → repack → NAS (foreground, streams progress)
./scripts/exl3-pack-drive.sh pipeline --host gx10-node3 \
  --model mimo-v2.6-flash-rl-uncensored \
  /opt/llm/staging/mimo-v2.6-flash-rl-uncensored \
  /opt/llm/fes-projects/exl3-mimo-build/mimo-v2.6-flash-rl-uncensored-work-k3 \
  /opt/llm/fes-projects/exl3-mimo-build/mimo-v2.6-flash-rl-uncensored-mcg-k3 \
  /nas-1/models/mimo/mimo-v2.6-flash-rl-uncensored-exl3-v1 3 mcg

# assemble the servable checkpoint
./scripts/exl3-pack-drive.sh assemble --host gx10-node3 \
  --model mimo-v2.6-flash-rl-uncensored \
  /opt/llm/staging/mimo-v2.6-flash-rl-uncensored \
  /nas-1/models/mimo/mimo-v2.6-flash-rl-uncensored-exl3-v1 \
  /nas-1/models/mimo/mimo-v2.6-flash-rl-uncensored-exl3-v1-serve
```

Run 4a and 4b sequentially if both use the same node's GPU; they use different
nodes, so they can overlap — but both contend for the NAS. Sequential is safer.

- Bits/codebook come from the spec (`--model` slug); `mcg` required for b12x.
- Monitor: `--detach` then `…drve.sh watch --host <node> <exl3-out>`
  (status file `<output>/.pack-status.json` via `PACK_STATUS_FILE`).
- Failure modes (spec_parity mismatch, convert OOM on 128 GB unified memory,
  repack codebook mismatch) → `mods/exl3-pack/NEW_MODEL.md` §Failure modes.
- The driver sets `--ulimit nofile=1048576:1048576` on the container (override
  with `--nofile` or `EXL3_PACK_NOFILE`); without it, a 128-shard source dies
  with `Too many open files (errno=24)` during convert.
- Work/intermediate dirs live on node NVMe and are cleaned by the pipeline.

## 5. Verify outputs

```text
shell_open node4 → ls -la /nas-1/models/mimo/mimo-v2.6-pro-rl-uncensored-exl3-v1{,-serve}
shell_open node3 → ls -la /nas-1/models/mimo/mimo-v2.6-flash-rl-uncensored-exl3-v1{,-serve}
```

Expect `config.json` + safetensors with `quantization_config.codebook:"mcg"`.

## 6. Restore the serve (node1)

```text
shell_open host=gx10-node1
shell_exec: cd /cluster-shared/projects/cluster-config && \
  ./scripts/spark-vllm-docker-serve.sh launch --daemon
shell_exec: ./scripts/spark-vllm-docker-serve.sh health --wait 600
```

- `launch --daemon` reruns the pinned recipe `mimo-v2.6-flash-4x` and returns
  once containers start; `health --wait` polls rank0 `/v1/models`.
- To instead serve a NEW EXL3 pack, the EXL3 recipe family
  (`recipes/mimo-v2.6-flash-exl3-{1x,2x}.yaml`, mapped in
  `SPARK_VLLM_DOCKER_SLUGS`) registers under `cluster/mimo-v2.6-flash-exl3`.
- Verify readiness, not just container start:
  `curl --fail http://<head-ip>:8000/v1/models`. Then confirm
  `list_cluster_routes` shows `cluster/mimo-v2.6-flash` alive again; if not,
  `POST http://localhost:3000/api/placement/reconcile` re-registers it.

## Guardrails

- Do not prune images/caches; do not stop `bge-m3`.
- node1 root fs at 92% — never point work dirs at `/`.
- Use fulton-remote-shell MCP for remote ops, not raw `ssh` (project rule).
- If convert OOMs on the GB10, fall back to lower bits (`lut_e4m3` K2) — not a
  `--gpu-memory-utilization` hack.
- Restore uses `spark-vllm-docker-serve.sh launch` — do NOT hand-run bare
  `docker stop/run`.
