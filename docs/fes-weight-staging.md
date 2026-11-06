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

Stage the selected slug on every node first (cluster-config, on a node or via
`--node`):


```bash
scripts/model-weights.sh stage <slug>
```

A recipe opts into direct staged launches with a literal declaration:


```yaml
model: org/model
env:
  FES_WEIGHTS_SLUG: "org-model-revision"
```

Then use the normal runner:


```bash
./run-recipe.sh <recipe> --solo
```

The runner resolves the slug in this order:


1. Explicit `--slug <slug>`.
2. Recipe `env.FES_WEIGHTS_SLUG`.
3. The last `--default-env FES_WEIGHTS_SLUG=<slug>` (cluster/orchestrator
   compatibility default).
4. No staged integration; normal hub-cache behavior.


`--no-fes-weights` overrides all of these for a hub launch. It adds no staged
mount or FES environment and retains normal model preflight/download. An
inert `mods/fes-weights` entry already in the recipe remains inert because
`FES_WEIGHTS_ENABLED` is not injected; its literal slug may remain visible for
traceability. Set `HF_HUB_OFFLINE=0` in a hub-mode recipe if the image's offline
default would otherwise block hub access. `--no-fes-weights` does not override
an independently supplied explicit `-e FES_WEIGHTS_ENABLED=1`.

The staged launch passes these values to `launch-cluster.sh`:


```bash
--apply-mod <repo>/mods/fes-weights \
-v <weights-root>/<slug>:/model:ro \
-e FES_WEIGHTS_ENABLED=1 \
-e FES_WEIGHTS_DIR=/model \
-e FES_HUB_MODEL=<recipe model: field> \
-e FES_WEIGHTS_SLUG=<slug> \
-e HF_HUB_OFFLINE=1
```


Only the mount is host-specific. The mod is copied into each container, where
these environment values are available before the generated serve script runs.
`FES_HUB_MODEL` and the other standard mod settings can still be overridden
with explicit `-e` values; a missing `model:` requires an explicit/default
`FES_HUB_MODEL`. The selected slug is recorded in the container environment.
`FES_WEIGHTS_SLUG` is not itself the activation switch for the mod. In staged
mode the mod is always ordered before all other mods, including when the recipe
omits it, so dependent mods see the verified shim.

Staged launches require `FES_WEIGHTS_ENABLED=1`; empty, bare, false, or other
values are rejected rather than disabling verification. A recipe/default
`FES_WEIGHTS_DIR` other than `/model` requires an operator-supplied read-only
`-v <weights-root>/<slug>:<FES_WEIGHTS_DIR>:ro` mount of the selected directory.
An explicit `-e FES_WEIGHTS_DIR=<absolute-dir>` creates that matching read-only
mount automatically. Missing, duplicate, writable, or mismatched mounts are
rejected before build, download, or launch instead of silently weakening
recipe environment precedence.

`--weights-root DIR` changes `/opt/llm/models`. Slugs match the cluster contract
`^[A-Za-z0-9][A-Za-z0-9._-]*$`; empty values, slashes, whitespace, shell
metacharacters, and traversal are rejected before any build/download/launch.
Arguments are not evaluated as shell snippets. Use a whitespace-free weights
root (the standard `/opt/llm/models` is supported); the existing launcher's
string-based Docker volume handling does not preserve custom paths containing
whitespace. Docker creates a missing mount source as an empty directory, but
`mods/fes-weights` then fails verification and stops before serve; there is no
hub fallback. Stage every rank and do not treat a successful `docker run` as
proof that weights were verified.

With staged weights selected, model cache preflight and model download/copy
are skipped, including under `--setup`, `--force-download`, and
`--download-only`. Image setup and distribution remain unchanged. Use a dry run
to inspect the generated mount, environment, and mod arguments without staging
or launching:


```bash
./run-recipe.sh <recipe> --solo --dry-run
./run-recipe.sh <recipe> --solo --slug <other-slug>
./run-recipe.sh <recipe> --solo --no-fes-weights
```

### Legacy wrapper and cluster handoff


```bash
scripts/fes-eval.sh glm-5.3-flash glm-5.3-flash-nvfp4 --solo
scripts/fes-eval.sh --no-fes-weights glm-5.3-flash ignored --solo
```


`fes-eval.sh` retains its positional `<recipe> <slug>` interface, host-side
verification, `--weights-root`, and `--hub-model`. It now delegates with
`--slug <slug> --weights-root <root> --default-env FES_HUB_MODEL=<id>` rather
than separately injecting the mount and mod. Therefore a recipe declaration
cannot cause duplicate FES mods, and an explicit later `--slug` still wins.
All remaining args pass through verbatim. In `--no-fes-weights` mode the
positional slug remains required for compatibility but is ignored and no
staged verification or mount occurs.


The cluster wrapper resolves its own profile/map/default selection and then
calls staged launches as:


```bash
scripts/fes-eval.sh <recipe> <resolved-slug> <launch-args...>
```


The fork's precedence at that boundary is explicit later `--slug` > recipe
`FES_WEIGHTS_SLUG` > compatibility `--default-env FES_WEIGHTS_SLUG`. The cluster
wrapper must resolve `SPARK_VLLM_DOCKER_SLUGS[recipe]` and its global default
before calling, so a mapped/default cluster choice is passed explicitly rather
than being confused with the fork's recipe tier. For hub calls it passes:


```bash
scripts/fes-eval.sh --no-fes-weights <recipe> <unused-positional-slug> <launch-args...>
```


This guarantees recipe slug declarations cannot turn a cluster hub launch into
staged mode.
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