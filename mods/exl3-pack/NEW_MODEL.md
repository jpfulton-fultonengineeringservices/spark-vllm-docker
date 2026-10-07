# Adding a new model to the EXL3 pack builder

The pack builder is model-agnostic: geometry is auto-detected from the source
model's `config.json`, and all pipeline logic lives in the shared library
`mods/exl3-pack/`. A new model onboards by adding **only a thin spec
directory**; no pipeline code changes.

## Model geometry auto-detection

`exl3-pack detect <source-dir>` reads the model and emits a JSON geometry
record (`exl3pack.geometry.detect`):

```json
{
  "source": "/opt/llm/staging/<model>",
  "architecture": "MiMoV2ForCausalLM",
  "hidden_size": 4096,
  "intermediate_size": 2048,
  "num_experts": 256,
  "num_slots": 64,
  "moe_layer_count": 47,
  "num_hidden_layers": 48,
  "moe_layers": [1, 2, 3, "..."],
  "dense_layers": [0]
}
```

What it reads, in order:

1. **`config.json`** — `architectures`, plus the model config via a
   `text_config` wrapper when present.
2. **Geometry fields** — `hidden_size`, `moe_intermediate_size` (falls back to
   `intermediate_size`), `n_routed_experts` (falls back to `num_experts` then
   `num_local_experts`), `num_hidden_layers`.
3. **`num_slots`** — `intermediate_size / 32`. The b12x `exl3-v1` container
   requires `slot_channels = 32`; a non-multiple-of-32 intermediate size emits
   a `_warning`.
4. **MoE layer indices** — scans the weight map (`model.safetensors.index.json`,
   or the safetensors headers for single-shard models) for `.mlp.experts.` keys.

A `_warning` means auto-detection failed; fix the spec geometry (below).

## Onboarding a new model

1. **Add the spec directory** `mods/<model>/pack-build/` with:
   - `spec.py` — a `PackSpec` (slug, `Geometry`, codebook, bits, dense_format,
     ignored_layers, `node_map_key`). Copy an existing spec and adjust the
     numbers from the `detect` output.
   - `pack-build` — the thin shim (copy verbatim; it execs `exl3pack.cli`
     with this dir's `spec.py`):
     ```bash
     #!/usr/bin/env bash
     set -euo pipefail
     SPEC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
     exec python3 -m exl3pack.cli --spec "${SPEC_DIR}/spec.py" "$@"
     ```
   - `expected-geometry.json` — the captured `detect` output, used by the
     detect-parity check and the host unit tests.
2. **Add it to the image + driver**: append the model dir to the
   `COPY mods/<model>/pack-build /opt/exl3-specs/<model>/pack-build` lines in
   `Dockerfile.exl3-pack`, and to the loop in `scripts/exl3-pack-drive.sh`
   `do_sync`.
3. **Add a node-map entry** (if the source lives on a node) to
   `mods/exl3-pack/node-model-map.json`.

## Runbook (node-local source → NAS output)

The builder reads the source from the node's **local NVMe** (resolved via the
node map; falls back to `/nas-1`) and writes the final pack to the **NAS**.
Work/intermediate dirs stay on node-local NVMe and are cleaned up after.

```bash
# 1. detect geometry (confirm before committing GPU hours)
./scripts/exl3-pack-drive.sh detect --host home-gx10-node4 \
  --model <model> /opt/llm/staging/<model>

# 2. full pipeline: convert -> repack, output to the NAS
./scripts/exl3-pack-drive.sh pipeline --host home-gx10-node4 \
  --model <model> \
  /opt/llm/staging/<model> \
  /opt/llm/fes-projects/exl3-mimo-build/<model>-work-k3 \
  /opt/llm/fes-projects/exl3-mimo-build/<model>-mcg-k3 \
  /nas-1/models/mimo/<model>-exl3-v1 3 mcg

# 3. assemble the servable checkpoint (dense-minus-experts + exl3 container)
./scripts/exl3-pack-drive.sh assemble --host home-gx10-node4 \
  --model <model> \
  /opt/llm/staging/<model> \
  /nas-1/models/mimo/<model>-exl3-v1 \
  /nas-1/models/mimo/<model>-exl3-v1-serve
```

`--codebook`/`--bits` come from the spec; extra `convert_model` args can still
be passed through.

## Codebook and bitrate constraints

The b12x `exl3-v1` runtime path accepts only these exllamav3 codebooks:

| Codebook | Supported bits | Notes |
|----------|----------------|-------|
| `mcg`    | K3–K6          | K2 rejected at runtime |
| `lut_e4m3` | K2–K4        | — |

- Use `codebook="mcg"` for anything you intend to repack to b12x.
- Bits are **uniform** — mixed K3/K4 is not representable without fork changes.
- `PackSpec.__post_init__` rejects a non-b12x codebook up front; repack fails
  loudly if the pack's `quantization_config.codebook` is not one of the three.

## Failure modes

| Symptom | Cause | Fix |
|---------|-------|-----|
| `_warning`: intermediate_size not divisible by 32 | unsupported MoE intermediate | model not packable at current b12x layout |
| `_warning`: no MoE layers found | dense model or non-standard key naming | fix spec / detection |
| `spec_parity` mismatch on detect | spec geometry ≠ checkpoint | correct the spec numbers |
| repack: codebook not mcg/lut_* | pack converted with the wrong codebook | re-convert with `codebook="mcg"` |
| convert OOM on GB10 | bitrate too high for 128 GB unified memory | drop bits (K2 via `lut_e4m3`) or smaller model |

## Tests

- **Host (torch-free):** `cd mods/exl3-pack && uv run pytest` — geometry,
  spec, paths, cli, status, monitor.
- **In-image integration:** `exl3-pack selftest` asserts the torch family
  imports (run inside the image; not a host requirement).
