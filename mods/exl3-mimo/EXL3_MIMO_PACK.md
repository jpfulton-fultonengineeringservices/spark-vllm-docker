# Building the MiMo-V2.6 EXL3 pack (`exl3-v1`)

`mods/exl3-mimo` keeps the runtime on the **`exl3-v1`** container the
`vllm-node-b12x` image reads and writes. This documents producing that pack.
**Operational** (GPU + large disk). Confirmed host: `gx10-node1` (GB10, CUDA
13.0, `vllm-node-b12x:latest`, B12X 1.3.0). Stage inputs/outputs on `/nas-1`
(30 TB) — local `/` has ~358 GB.

## Container: `exl3-v1`

- `<root>/exl3-manifest.json` (`kind` `exl3-manifest`, `schema` `exl3-v1`) +
  one `<root>/exl3-layer-<NNNNN>.safetensors` per MoE layer.
- Per-layer tensors: `codes` (u8 `[num_slots, row_stride]`), `rotations`
  (fp16 `[num_slots, E, 3, slot_channels]`), `gate_suh`/`up_suh`/`down_svh`
  (fp16 `[H]` or `[E,H]`); optional `rates_fc1`/`rates_fc2` (per-expert-pair),
  `sign_pattern` (intermediate Hadamard).
- Geometry (MiMo): `num_experts` 256, `hidden_size` 4096, `intermediate_size`
  2048 → `slot_channels` 32, `num_slots` 64, `moe_layer_indices` = the 47 MoE
  layers.
- Rates: codebook `mcg`, **uniform** bits (mcg K3–K6; K2 needs `sqg_e4m3`).
  Target uniform K3 first. Hadamard: uncoupled for the first cut.

## Serving assembly

The container is experts-only. The b12x runtime (`Exl3MoEMethod`) reads
`exl3-manifest.json` + `exl3-layer-*.safetensors` from the **served model
directory** and loads everything else (attention / dense-MLP / embeddings /
head / router) from the ordinary HF checkpoint. A servable checkpoint is
therefore assembled from the source model (with the routed experts removed),
this container, and a `quantization_config` that declares the exl3 method:

```bash
pack-build assemble <src> <exl3-v1> <serve-out>
```

Output = `config.json` (`quant_method: exl3`, `exl3.manifest`,
`dense_format: bf16`, `ignored_layers`), the non-routed weights re-emitted as
BF16 (`weight_scale_inv` dropped), `model.safetensors.index.json`, copied
tokenizer/modeling files, and the `exl3-v1` container. This artifact is what the
FES weight-staging path stages and serves (see `NEW_MODEL.md`, and the fork's
`docs/fes-weight-staging.md`).

## In-image writer (B12X 1.3.0)

`b12x/moe/_shared/kernels/w4a16/exl3_synth.py`:

- `Exl3SynthConfig` — geometry + rates + hadamard + layout declarations.
- `Exl3LayerPayloads` — the logical per-layer content: `planes[(expert, slot,
  matrix)] = (low_plane, high_plane)`, each int16 `[hidden/16, 16*bits]`,
  matrices 0=gate 1=up 2=down; plus `rotations`, `gate_suh`/`up_suh`/
  `down_svh`, optional `sign_pattern`/`rates_*`.
- `assemble_code_rows(config, layer, payloads)` — packs planes into the padded
  expert-major `codes` tensor.
- `write_exl3_checkpoint(root, config)` — in-tree writer, but it calls
  `synth_layer_payloads` internally, so it emits **synthetic (random)
  content**. For real weights, compose `Exl3LayerPayloads` + `assemble_code_rows`
  + the manifest/layer-metadata helpers with encoder output.

Reader (verify on CPU): `b12x.moe.checkpoints.exl3.read_exl3_manifest` /
`read_exl3_layer`.

## Encoder

`shihanqu/shapleymcg` `reproducibility/r10/` (corrected EXL3/MCG closure):

```python
from quant_pipeline.codecs.exl3_mcg import Exl3MCGCodec
codec = Exl3MCGCodec(source_root=<r10 bundle>,
                     numeric_core=<r10>/lineage/encode_tr3_v31.py,
                     extension=<exllamav3_ext.so>, device="cuda:0")
```

`encode_candidates(unit_id="L<l>.E<e>.(gate_proj|up_proj|down_proj)",
weight_hf, covariance, bits, input_vector, output_vector)` — K3/K4/K5; K,N
% 128 == 0 (MiMo experts qualify). The adapter hash-binds the closure +
numeric core + extension.

## Build steps

1. Stage MiMo-V2.6 source + calibration corpus on `/nas-1`.
2. Calibrate routed experts (Hessians / input scales) for the 47 MoE layers.
3. Encode each expert projection at uniform `mcg` K3 (scale to K4/K5 if
   quality needs it).
4. Convert encoder output → `Exl3LayerPayloads` planes (low/high int16
   `[hidden/16, 16*bits]`), `rotations` (fp16 `[num_slots,E,3,32]`),
   `gate_suh`/`up_suh`/`down_svh`; assemble `codes` with `assemble_code_rows`;
   write the manifest + per-layer safetensors (either reuse the in-tree writer
   helpers or reproduce the exact layout).
5. Validate on CPU: `read_exl3_manifest` + `read_exl3_layer` (geometry/extent
   legality), `pack-build/test_dispatch_fix.py` (dense dequant grid convention),
   then boot `recipes/mimo-v2.6-flash-exl3-2x.yaml` and run greedy parity + PPL
   (`README.md`). Note the fork's `plan_exl3_extent` rejects TP=1 outright
   (`EXL3 experts require TP in 2..24`), so the `-1x` recipe cannot serve this
   pack — validate on the 2-node recipe.

## Dense / non-routed

Only routed experts are EXL3. Attention / dense-MLP / embeddings / head are
dequantized from the source FP8-block format to plain BF16, so the checkpoint
`quantization_config` declares `exl3` with `dense_format: bf16`, the manifest,
and an `ignored_layers` list naming the non-routed modules (the mod reads these).
Fused projections must be dequantized per interleaved group — see
`_dequant_fp8_block_dispatch` and `prove_dequant_bug.py`.

## Licensing

shapleymcg is source-available with attribution (not OSI); B12X-derived code
Apache-2.0 where marked. Review `LICENSE` / `THIRD_PARTY_NOTICES.md` before
redistributing derived artifacts.

---

# Option 1 — from-scratch calibration pack (documented; NOT this run)

This is the faithful route: encode MiMo experts with the shapleymcg r10
EXL3/MCG codec from source weights + calibration. It is **not** executed by the
current pack build (which converts the existing calibrated pack instead), but
is documented here so it can be run later for a custom bitrate/quality target.

## Hard constraint discovered on-node

`Exl3MCGCodec.encode_candidates(...)` → `R10TrellisCodec.encode_bits(
tensor_id, weight_hf, covariance, bits, suh, svh, sigma_reg, provenance)`
**requires a real per-matrix covariance** `[K,K]` and `suh`/`svh` scaling
vectors. The codec explicitly raises `"degenerate covariance; identity-H
fallback is forbidden"` — there is no calibration-free path. So a from-scratch
pack needs the full calibration pipeline, not just the encoder.

## Pipeline (port shapleymcg's Qwen campaign to MiMo)

shapleymcg exposes the stages as scripts under `scripts/` (Qwen-specific); a
MiMo port means adapting the model-facing pieces and running the same stage
order:

1. **Freeze identities** — MiMo-V2.6 revision, tokenizer, calibration corpus,
   r10 closure, `exllamav3_ext` (the FES prebuilt cu13.0 wheel), sigma_reg.
2. **Calibration capture** (`calibration/qwen_capture.py` analogue) — for each
   of the 47 MoE layers, capture routed observations and the full uncentered
   second moment `H_p = Σ route_weight^p · x xᵀ` for p=0,1,2 (float64). Down
   projections need the *conditional* (post-SiLU gate·up) state.
3. **Fit** (`calibration/fitter.py`) — per-expert component Hessians; keep
   cross-coordinate terms (full matrix, not diagonal).
4. **Normalization / scales** (`normalization/absolute_v31.py`) — derive the
   incoherence/`suh`/`svh` (absolute-v31 normalization + GSS scalar search) per
   matrix.
5. **Encode** — `encode_candidates(unit_id="L<l>.E<e>.<gate_proj|up_proj|down_proj>",
   weight_hf, covariance, bits=(3,| 4,| 5,), input_vector=suh, output_vector=svh)`.
   MiMo gate/up are `[N=2048,K=4096]`, down `[N=4096,K=2048]` — all 128-clean.
6. **Allocate** (`allocation/global_dp.py`) — optional; a uniform K3 pack needs
   none. Mixed K3/K4 would use the exact-byte multiple-choice knapsack.
7. **Container emit** — convert the encoder's per-matrix `trellis` + `suh`/
   `svh` into the `exl3-v1` container (same `assemble_code_rows` path as the
   conversion route in the main doc): planes `[hidden/16, 16*bits]`, `rotations`
   `[num_slots,E,3,32]`, `gate_suh`/`up_suh`/`down_svh`.
8. **Validate + boot** — CPU `read_exl3_manifest`/`read_exl3_layer`; then
   `recipes/mimo-v2.6-flash-exl3-1x.yaml` + greedy parity + PPL.

## Notes / cost

- Requires a MiMo capture adapter (no Qwen path applies) and a corpus with
  enough routed coverage; their scientific gate wants ≥25 final windows.
- Compute is real: expert Hessians for 302.8 B routed params plus calibration
  forward passes; feasible on GB10 but hours–days, and the codec's covariance
  is float32 `[K,K]` per matrix (memory-planned by the codec's paging).
- Optional quality work: attention/shared-expert promotion, K3/K4 allocation,
  coupled Hadamard (`intermediate_hadamard`) — see shapleymcg method docs.
- Generated by options: uniform `mcg` K3 is the sensible first target;
  `bits=(3,)` only.

## Why option 2 instead

`benthecarman/MiMo-V2.6-Flash-RL-exl3` is already a calibrated, measured EXL3
trellis pack (experts 2.0/2.5, attention 4.0, …) built with the same exllamav3
trellis math the r10 codec is derived from. Reshaping its existing
`trellis`/`suh`/`svh` into the `exl3-v1` container needs **no re-encoding** and
preserves its measured quality — hence option 2 (conversion) is the current
run.

---

> **Superseded (2026-10-01):** option 2 is **dead**. benthecarman's pack is
> `codebook: mul1`, and b12x `CODEBOOKS = ('mcg','lut_e4m3','lut_fp16')` has no
> `mul1`; the two are different procedural codebooks, so the trellis codes are
> not interchangeable and a relabel cannot work. A **`mcg`** pack must be built.
> The revised route — exllamav3 `convert.py -cb mcg` (MiMo merged upstream, PR
> #399) then repack to `exl3-v1` — plus the full choices/tradeoffs/runbook and
> the calibration-corpus survey are in
> [`tmp/perf-concepts/exl3-prior-art/12-exl3-mcg-conversion-and-calibration.md`](../../tmp/perf-concepts/exl3-prior-art/12-exl3-mcg-conversion-and-calibration.md)
> and
> [`13-code-domain-calibration-corpora.md`](../../tmp/perf-concepts/exl3-prior-art/13-code-domain-calibration-corpora.md).
> Headline constraint: `mcg` forces integer bits and the b12x adapter forces
> **uniform** bits, so the pack is uniform **K2** (fits one GB10) or uniform K3
> (does not fit); the 2.25 bpw size is not representable without extending the
> fork to accept `per_expert_pair` rates.
