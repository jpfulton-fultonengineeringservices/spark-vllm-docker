# mods/mooncake-worker

Installs the **host-staged** Mooncake store connector package into the
`vllm-node-b12x` image's vLLM tree, from the Fulton-Engineering-Services vLLM
fork branch `cuda13.3-aarch64-gb10-glm5next-modular` (tip `4e79d6be6f`).

Packaged like `mods/exl3-pack`: uv-managed `pyproject.toml`, `src/mooncake_worker/`
library, CLI, ruff (line-100) + mypy strict + pytest (torch-free), `run.sh` as a
thin launcher.

```
mods/mooncake-worker/
├── pyproject.toml          ruff · mypy strict · pytest (pythonpath=src)
├── README.md
├── run.sh                  thin launcher → mooncake_worker.cli
├── src/mooncake_worker/
│   ├── gates.py            md5 gates + vendor integrity + state classify
│   ├── envs_shim.py        vllm/envs.py dict-entry injector (AST-validated)
│   ├── installer.py        matched-set atomic install + verify
│   ├── cli.py              install | verify | selftest | manifest
│   └── vendor/store/       the 8-file matched set (worker+data+…)
└── tests/                  test_gates · test_envs_shim · test_installer
```

## Why

GB10 has no GPUDirect RDMA for NIXL-style GPU-VA registration: the store
connector's direct `store.register_buffer(gpu_addr)` fails with `ibv_reg_mr`
EFAULT ("Bad address [14]", `num_segments=0`). The Fulton fork's
`_StagingSlotPool` instead D2H-copies GPU KV into pinned host slots and
RDMA-writes from there — the design proven in the qwen38 systemd deployments.
The b12x `vllm-node-b12x` image ships the **upstream** connector (worker md5
`b180493c…`, 2633 lines, `host_staging` count 0), which predates the whole
staging series.

## What it ships (the matched set)

`vendor/store/` — all eight files move together because the FES `worker.py`
reads `ReqMeta.partial_tail_offloads` while the stock image `data.py` defines
`boundary_state_offloads` (b12x/DiffKV lineage): a lone-worker swap would
`AttributeError`. `worker.py` (md5 `09388dea01fd935d7cb0335fb0cf736e` —
2026-10-11 lookup-client timeout hardening),
`data.py`, `connector.py`, `scheduler.py`, `coordinator.py`, `protocol.py`,
`metrics.py`, `__init__.py`.

The welded sibling files (`__init__.py`, `_worker_*.py`, `staging.py`,
`base.py`, `mooncake_connector.py`) are **not** shipped: they import
`get_kv_cache_layout`, which the b12x core renamed — irrelevant to the
store-only path.

## Gates

- `gates.verify_vendor()` — vendor set complete, every file parses, worker md5
  == branch tip.
- `gates.classify_installed()` — `stock` (any admitted image lineage:
  `b180493c…` vllm-node-b12x, `8cc1f01a…` derived coder390 image
  sha256:e9f6e32941ec…) | `already_ported` (`09388dea…`) | `unknown`.
  Each new pinned image lineage needs its stock md5 probed live and appended
  to `MD5_STOCK_WORKERS` — see `tests/test_gates.py` for the probe commands.
- Installer **fails closed** on `unknown` (image drift): re-port the vendored
  set, never blind-overwrite.
- `envs_shim.inject()` — `vllm.envs` resolves attributes through its
  `environment_variables` dict in `__getattr__`, so the shim adds a **dict
  entry** (not an annotation) for `VLLM_PREFIX_CACHE_RETENTION_INTERVAL`
  (FES default `None`). Anchor-count-gated, AST-validated, idempotent.

## Runtime requirements (shipped elsewhere)

- `mooncake-transfer-engine-cuda13==0.3.12.post1` — installed at boot by
  `mods/pd-disagg/run.sh` (the connector imports `mooncake.store` in-process).
- `cupy-cuda13x<14.1.0` — required by `_StagingSlotPool` (pinned slot allocs);
  installed by `mods/pd-disagg/run.sh`.

## Config (`kv_connector_extra_config`, set by `mods/pd-disagg/dispatch.sh`)

```json
{"host_staging": true, "staging_num_slots": 2, "staging_slot_size_mb": 2048}
```

Budget-neutral vs the default single 4 GiB buffer (2×2048 MiB).

## Usage

```
run.sh install     # default (mod apply)
run.sh verify      # confirm installed state
run.sh selftest    # offline integrity (no vLLM needed)
run.sh manifest    # gate constants
```

### Standalone store L2 (non-PD)

The store set is not PD-specific: `MooncakeStoreConnector` subclasses
`SupportsHMA` and runs standalone with `kv_role=kv_both`, giving a shared
prefix pool keyed by content hash (mamba hybrids included, align mode). The
store cohort owns the pool and the SSD tier, so KV survives an engine restart.

For that use, apply the store set only — pin `MOONCAKE_WORKER_P2P=skip` so the
P2P facade (irrelevant without a PD topology) is not installed:

```
MOONCAKE_WORKER_P2P=skip run.sh install
```

Recipe-env gotcha (bit the coder390 store-L2 recipes 2026-10-25): a recipe's
`env:` block CANNOT reach a mod — mods run in a separate process before the
exec-script exports recipe env, so `MOONCAKE_WORKER_P2P: "skip"` in the recipe
YAML is inert and the direct `mods/mooncake-worker` entry runs with the
default `auto` (and trips the P2P gate). Either omit the direct entry
(`mods/mooncake-store-l2` calls the installer itself with `P2P=skip` forced),
or inject the env as a container `-e` at docker-run time
(cluster-config's spark-serve.sh store-L2 branch does exactly that) —
`apply_mod_to_container`'s docker execs inherit the container env.

The P2P auto-add is idempotent and gate-fails-closed, so omitting the pin is
safe but installs unused files. `mods/mooncake-store-l2` is the recipe-side
companion: it installs the transfer-engine + cupy runtime deps and renders the
`standalone-store` `MOONCAKE_CONFIG_PATH` JSON this connector reads.

## Tests

```
PYTHONPATH=src python3 -m pytest -q      # 12 tests, torch-free
python3 -m ruff check . && python3 -m ruff format --check .
PYTHONPATH=src python3 -m mypy src/mooncake_worker
```
