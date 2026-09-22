# FES weight staging for spark-vllm-docker (eval mode)

Companion integration that lets `run-recipe.sh` recipes serve **FES-staged
weights** instead of downloading from the hub. Weights are never copied or
rebaked: the FES cluster's staged copy (`/opt/llm/models/<slug>` on each gx10
node) is bind-mounted read-only, verified, and presented to vLLM in HF
hub-cache layout so the recipe's `vllm serve <org/model>` command resolves
offline.

## Components

| File | Runs where | Role |
|---|---|---|
| `mods/fes-weights/run.sh` | inside container (head + every worker, before serve) | verifies the mounted weights, builds the hub-cache shim |
| `mods/fes-weights/verify.py` | container or host | shard count vs `model.safetensors.index.json`, size parity (≤1.02x), `hf_quant_config.json` required for nvfp4 — same checks as cluster-config `scripts/model-weights.sh verify_location` |
| `scripts/fes-eval.sh` | host | pre-verifies the staged copy, then execs `run-recipe.sh` with the mount, mod, and offline env injected |

## Usage

```bash
# one-time per node (cluster-config, on a node or via --node):
scripts/model-weights.sh stage <slug>

# evaluation launch (on the lead node, from this repo checkout):
scripts/fes-eval.sh glm-5.3-flash glm-5.3-flash-nvfp4 --solo
```

`fes-eval.sh` expands to:

```bash
run-recipe.sh <recipe> \
    --apply-mod mods/fes-weights \
    -v /opt/llm/models/<slug>:/model:ro \
    -e FES_WEIGHTS_DIR=/model \
    -e FES_HUB_MODEL=<recipe model: field> \
    -e HF_HUB_OFFLINE=1 \
    <your args>
```

All remaining args are passed to `run-recipe.sh` verbatim (`--solo`, `-n`,
`--port`, and the recipe overrides all work as usual).

## How the hub shim works

`vllm serve <org/model>` in every recipe resolves through the HF hub cache.
The mod builds, inside the already-mounted cache dir
(`$HF_HOME:-/root/.cache/huggingface/hub`):

```
hub/models--<ORG>--<MODEL>/
├── refs/main                    # contains "fes-staged"
├── .fes-shim-fp                 # fingerprint of the staged file set
└── snapshots/fes-staged/        # one symlink per file -> /model/<file>
```

`huggingface_hub` in offline mode resolves `refs/main` → `snapshots/fes-staged`
→ the symlinked files, so the recipe command needs no edits. The fingerprint
(`name + size` of every staged file) makes the shim idempotent and
self-healing if the slug is restaged.

## Contracts

- Weights root defaults to `/opt/llm/models` (FES `LLM_MODELS_DIR`); override
  with `--weights-root`.
- Multi-node recipes need the slug staged on **every** node — `launch-cluster.sh`
  applies the mount on each node's container, and the mod verifies per node
  (fail-fast before serve).
- Log lines follow the FES convention: `FES_WEIGHTS_OK op=verify dir=... slug=...`
  / `FES_WEIGHTS_FAIL op=verify dir=... reason=...`.
- Draft/speculative models: set `FES_DRAFT_DIR` (default `/drafter`) and
  `FES_DRAFT_HUB_MODEL` via extra `-e` args to verify and shim a second model.

## Recipe ↔ FES slug pairs to try first

| Recipe | Recipe model id | FES slug |
|---|---|---|
| `glm-5.3-flash` | `local-inference-lab/GLM-5.3-Flash-NVFP4-Spark` | `glm-5.3-flash-nvfp4` |
| `qwen3.8-27b-nvfp4-w4a4` | Qwen3.8-27B NVFP4 W4A4 | `qwen3.8-27b-nvfp4-w4a4` |

Confirm the exact slug names/availability with
`cluster-config scripts/model-weights.sh status` before staging; pass
`--hub-model` explicitly if a recipe's `model:` id differs from the served
checkpoint's hub id.