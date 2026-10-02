# Adding a new model to the EXL3 pack builder

The `pack-build` tooling in this directory is model-agnostic. Geometry is
auto-detected from the source model's `config.json`, so most MoE transformers
onboard with no per-model code or config file. This document is the full
approach: what auto-detection reads, when a config override is required, and
the step-by-step onboarding runbook.

## Model geometry auto-detection

`pack-model-info.py <source-dir>` reads the model and emits a JSON geometry
record:

```json
{
  "source": "/nas-1/models/mimo-v2.6-flash-rl",
  "architecture": "MimoV2ForCausalLM",
  "hidden_size": 4096,
  "intermediate_size": 2048,
  "num_experts": 256,
  "num_slots": 64,
  "moe_layer_count": 47,
  "moe_layers": [1, 2, 3, "...", 47],
  "dense_layers": []
}
```

What it reads, in order:

1. **`config.json`** — `architectures`, plus the model config via a
   `text_config` wrapper when present (`MimoV2`, `DeepSeek-V4`, etc. nest their
   transformer config under `text_config`).
2. **Geometry fields** — `hidden_size`, `moe_intermediate_size` (falls back to
   `intermediate_size`), and `n_routed_experts` (falls back to `num_experts`
   then `num_local_experts`).
3. **`num_slots`** — `intermediate_size / 32`. The b12x `exl3-v1` container
   requires `slot_channels = 32`; a model whose intermediate size is not a
   multiple of 32 emits a warning.
4. **MoE layer indices** — scans the weight map (from
   `model.safetensors.index.json`, or the safetensors file headers directly
   for single-shard models) for keys containing `.mlp.experts.` and extracts
   the layer index from the `model.layers.<N>.mlp.experts.` path.

If `hidden_size`, `num_experts`, or `intermediate_size` are zero, or no MoE
layers are found, the script emits a `_warning` in the output. Treat any
warning as "auto-detection failed" and add an override (below).

## Onboarding runbook

All paths are node-side paths (the source model and the work/output dirs all
live under `/nas-1`, mounted on every cluster node). On the local workstation,
drive this with `scripts/exl3-pack-drive.sh`.

1. **Detect geometry** — confirm auto-detection before committing hours of GPU:

   ```bash
   ./scripts/exl3-pack-drive.sh detect --host home-gx10-node1 \
     /nas-1/models/<org>/<model>
   ```

   Expected: `architecture`, non-zero `hidden_size`/`intermediate_size`/
   `num_experts`, and `moe_layer_count` > 0, no `_warning`.

2. **Convert** — run the exllamav3 quantizer to produce an EXL3 pack:

   ```bash
   ./scripts/exl3-pack-drive.sh convert --host home-gx10-node1 \
     /nas-1/models/<org>/<model> \
     /nas-1/fes-projects/exl3-mimo-build/<model>-mcg-k3 \
     /nas-1/fes-projects/exl3-mimo-build/<model>-work-k3 \
     3 mcg
   ```

   Extra `convert_model` arguments (calibration dataset, rope alpha, sequence
   length) pass through after the codebook.

3. **Repack** — reshape the EXL3 pack into the b12x `exl3-v1` container:

   ```bash
   ./scripts/exl3-pack-drive.sh repack --host home-gx10-node1 \
     /nas-1/fes-projects/exl3-mimo-build/<model>-mcg-k3 \
     /nas-1/fes-projects/exl3-mimo-build/<model>-exl3-v1 \
     3 --self-check
   ```

4. **Assemble the serving checkpoint** — the `exl3-v1` container is experts-only;
   the b12x runtime also needs the ordinary dense checkpoint (attention / dense
   MLP / embeddings / head / router) with the routed experts removed, plus a
   fork-shaped `quantization_config`. `assemble` writes that directory:

   ```bash
   ./scripts/exl3-pack-drive.sh assemble --host home-gx10-node1 \
     /nas-1/models/<org>/<model> \
     /nas-1/fes-projects/exl3-mimo-build/<model>-exl3-v1 \
     /nas-1/models/<family>/<model>-exl3-v1
   ```

   The output is a normal-looking HF checkpoint (config.json + re-sharded dense
   weights + index + tokenizer/modeling files) plus `exl3-manifest.json` and the
   `exl3-layer-*.safetensors` container, so the FES weight-staging path
   (`model-weights.sh stage` / `verify`) accepts it directly.

5. **Or run the full pipeline** (convert → repack, one command):

   ```bash
   ./scripts/exl3-pack-drive.sh pipeline --host home-gx10-node1 \
     /nas-1/models/<org>/<model> \
     /nas-1/fes-projects/exl3-mimo-build/<model>-work-k3 \
     /nas-1/fes-projects/exl3-mimo-build/<model>-mcg-k3 \
     /nas-1/fes-projects/exl3-mimo-build/<model>-exl3-v1 \
     3 mcg
   ```

6. **Verify the container** — confirm the manifest and per-layer safetensors:

   ```bash
   ssh home-gx10-node1 \
     "ls /nas-1/fes-projects/exl3-mimo-build/<model>-exl3-v1 | head"
   # expects exl3-manifest.json + exl3-layer-<NNNNN>.safetensors per MoE layer
   ```

7. **Boot the runtime** — reference the pack from a recipe YAML (see
   `recipes/` and the `exl3-mimo` mod README) and run greedy-parity + PPL
   validation.

## When a config override is required

Auto-detection handles standard `transformers` MoE layouts. Add an override
when any of these is true:

- The weight map uses non-standard key naming for experts (detection finds no
  `.mlp.experts.` keys).
- Expert count / sizes are stored under an unusual config key (a `_warning`
  reports zero geometry).
- The model needs a calibration corpus, custom RoPE alpha, or sequence length
  that should always be passed to `convert_model`.

The override is a YAML file under `models/<name>.yaml` (sparse — only the
fields that differ from auto-detection):

```yaml
source: /nas-1/models/<org>/<model>
codebook: mcg
bits: 3
cal_dataset: /nas-1/data/c4-calibration
rope_alpha: 2.0
geometry:
  hidden_size: 4096
  intermediate_size: 2048
  num_experts: 256
```

`pack-model-info.py` merges declared `geometry` over detected values. The
driver and pipeline read the same file, so one override covers both.

## Codebook and bitrate constraints

The b12x `exl3-v1` runtime path (what `repack` targets) accepts only these
exllamav3 codebooks:

| Codebook | Supported bits | Notes |
|----------|----------------|-------|
| `mcg`    | K3–K6          | K2 rejected at runtime |
| `lut_e4m3` | K2–K4        | — |

- `convert_model` can emit `mul1` / `mcg` / `3inst`; **use `-cb mcg`** for
  anything you intend to `repack` to b12x.
- Bits are **uniform** — the b12x adapter does not accept `per_expert_pair`
  rates. A mixed K3/K4 pack is not representable without extending the fork.
- Repack fails loudly if the pack's `quantization_config.codebook` is not
  `mcg`/`lut_e4m3`/`lut_fp16`.

## Failure modes and diagnosis

| Symptom | Cause | Fix |
|---------|-------|-----|
| `_warning`: intermediate_size not divisible by 32 | model's MoE intermediate size unsupported | model not packable at current b12x layout; revisit |
| `_warning`: no MoE layers found | dense model or non-standard key naming | add `models/<name>.yaml` override with explicit layer list |
| repack: `codebook is 'mul1'` | pack converted with the wrong codebook | re-convert with `-cb mcg` |
| convert OOM on GB10 | bitrate too high for 128 GB unified memory | drop bits (K3 → K2 via `lut_e4m3`) or pick a smaller model |
| `g_sc`/`proxy_err` NaN in status | degenerate calibration | use a larger calibration corpus |

## Progress and ETA

Every stage writes structured status to `<output>/.pack-status.json`
(override with `--status-file`). The driver streams it live:

```bash
./scripts/exl3-pack-drive.sh watch --host home-gx10-node1 \
  /nas-1/fes-projects/exl3-mimo-build/<model>-exl3-v1
```

The status record carries `stage`, `phase`, `current_layer`, `layers_total`,
`layers_completed`, `eta_seconds`, `elapsed_seconds`, `gpu_memory_used_mb`,
and `disk_free_gb`. ETA is derived from elapsed time per completed layer.
