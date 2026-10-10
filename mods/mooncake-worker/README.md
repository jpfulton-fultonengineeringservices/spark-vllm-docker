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
`AttributeError`. `worker.py` (md5 `b11bd659baf724298b2f347dad503b4e`),
`data.py`, `connector.py`, `scheduler.py`, `coordinator.py`, `protocol.py`,
`metrics.py`, `__init__.py`.

The welded sibling files (`__init__.py`, `_worker_*.py`, `staging.py`,
`base.py`, `mooncake_connector.py`) are **not** shipped: they import
`get_kv_cache_layout`, which the b12x core renamed — irrelevant to the
store-only path.

## Gates

- `gates.verify_vendor()` — vendor set complete, every file parses, worker md5
  == branch tip.
- `gates.classify_installed()` — `stock` (`b180493c…`) | `already_ported`
  (`b11bd659…`) | `unknown`.
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

## Tests

```
PYTHONPATH=src python3 -m pytest -q      # 12 tests, torch-free
python3 -m ruff check . && python3 -m ruff format --check .
PYTHONPATH=src python3 -m mypy src/mooncake_worker
```
