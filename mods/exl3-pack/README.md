# mods/exl3-pack — shared EXL3 pack builder

One shared library for every model's EXL3 pack build. The pack pipeline logic
(convert → repack → assemble) lives here; each model supplies only a thin spec
directory `mods/<model>/pack-build/spec.py`. A per-model `pack-build` shim
execs this library with that spec.

## Layout

```
mods/exl3-pack/
  pyproject.toml          uv-managed; ruff + mypy(strict) + pytest config
  src/exl3pack/
    geometry.py           detect(): model geometry from config.json + weight map (torch-free)
    spec.py               PackSpec / Geometry + loader
    paths.py              node-model-map load + local/NAS source resolution
    monitor.py            status-file writer + ETA (torch-free)
    status.py             status renderer / watch (torch-free)
    convert.py            exllamav3 convert_model orchestration (torch)
    repack.py             b12x exl3-v1 writer (torch + b12x)
    assemble.py           dense dequant + config patch (torch)
    pipeline.py           stage orchestration + cleanup
    cli.py                detect|convert|repack|assemble|pipeline|status|selftest|plan|help
  tests/                  host unit tests (torch-free)
  node-model-map.{json,md} node → local path → checkpoint map
  NEW_MODEL.md            onboarding a new model
```

## Model selection

Every invocation selects a model with `--model <slug>` (resolved under
`--specs-root`, default `/opt/exl3-specs` in the image) or `--spec <path>`.
Codebook/bits/ignored-layers come from the spec — the CLI no longer takes them
from the command line.

## Local source, NAS output

The builder reads the **source checkpoint from the node's local NVMe**
(resolved via `node-model-map.json`; falls back to `/nas-1`) and writes the
**final pack to the NAS** (`/nas-1/models/mimo/<slug>-exl3-v1`). The working
set (convert quantized state + intermediate exl3 pack) stays on node-local
NVMe and is removed after the final artifact exists (``--cleanup all``).

## Dry run

```
exl3-pack --model mimo-v2.6-pro-rl-uncensored --node home-gx10-node4 plan
```
prints the resolved source/work/output paths and the docker mount plan without
executing anything.

## Tests

Two tiers:

1. **Host, torch-free** — geometry/spec/paths/cli/status/monitor and the
   FP8-dispatch + assemble-e2e suites *skip* without torch:
   ```
   uv venv --python 3.12 .venv
   uv pip install --python .venv/bin/python -e ".[dev]"
   .venv/bin/pytest          # torch-free subset
   ```
2. **Host, with CPU torch** — unlocks the assemble end-to-end test and the FP8
   dispatch regression (both fully runnable without b12x/exllamav3):
   ```
   uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
   uv pip install --python .venv/bin/python safetensors numpy
   .venv/bin/pytest          # all suites
   ```
   (`.[hosttest]` lists the same set.)

`repack`/`convert` still cannot run on a host — they need the aarch64-CUDA
`b12x`/`exllamav3` wheels, which exist only in the image. `exl3-pack selftest`
(run in-image) covers the torch import surface plus the dispatch path.
