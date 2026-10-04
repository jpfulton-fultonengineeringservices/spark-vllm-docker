# mods/exl3-mimo

Serves a MiMo-V2.6 **EXL3** pack on the b12x vLLM runtime. The runtime ships
an EXL3 routed-expert path specialised for DeepSeek-V4.1 / Kimi-K3; this mod
relaxes the two gates that reject MiMo. No container repoint is needed: the
image's B12X reads and writes the **`exl3-v1`** container that the fork's
`Exl3MoEMethod` already uses.

## What the fork/image already has

- vLLM fork (`local-inference-lab/vllm@dev/karmic-kraken`, in the
  `vllm-node-b12x` image): `"exl3"` registered in
  `quantization/__init__.py`; `Exl3Config` / `Exl3MoEMethod` in
  `quantization/exl3.py`; reads `exl3-v1` via `b12x.moe.checkpoints.exl3`.
- B12X 1.3.0 (in the image) ships the matching **exl3-v1** tooling:
  reader `b12x/moe/checkpoints/exl3.py`; schema
  `b12x/moe/_shared/exl3_schema.py`; prep
  `b12x/moe/_shared/kernels/w4a16/exl3.py`; writer helpers
  `b12x/moe/_shared/kernels/w4a16/exl3_synth.py`
  (`Exl3LayerPayloads`, `assemble_code_rows`, `write_exl3_checkpoint`).
- `b12x.moe.fused_moe` exposes `plan_weights`/`prepare_weights`/
  `ActivationSpec`/`MoEGeometry`/`PreparedExperts` (verified in-image).

## What this mod changes (anchored, idempotent, fail-loud)

0. `Exl3Config.__init__` — extends `exclude_modules` with the fused dense names
   (`gate_up_proj`, `qkv_proj`) whenever their unfused checkpoint parts appear in
   `ignored_layers`. Required because `is_layer_skipped` can only expand a fused
   name via `quant_config.packed_modules_mapping`, and the resolved top-level
   class (`MiMoV2OmniForCausalLM`) defines no mapping — so at runtime that
   mapping is empty and `gate_up_proj` matched nothing. Without this, MiMo's
   dense layer-0 MLP is handed the MXFP8 method over BF16 weights that have no
   `weight_scale_inv`, and its output comes back exactly zero.
1. `Exl3Config.from_config` — drops the DeepSeek KDA/MLA arrangement checks
   (`dense_format == "mxfp8"`, the KDA ignore-list, the `q/k/v_proj`
   exclusion); keeps the `exl3-manifest.json` requirement and a fail-closed
   `ignored_layers` list check; accepts `bf16`, `fp8`, or `mxfp8` dense.
2. `Exl3MoEMethod.__init__` — admits `MoEActivation.SILU` (MiMo) alongside
   DeepSeek's SiTU(4/25); BF16 + bias-free still enforced.
3. The fused-MoE weight plan is built with `nonlinearity="silu"`.
4. `mimo_v2.py` — adds `MiMoV2Model._try_load_bf16_qkv_proj` and intercepts the
   fused `qkv_proj` before the generic fused loader. The assembled checkpoint
   keeps the source's interleaved `[Q_i|K_i|V_i]` layout but stores it in BF16,
   so the fork's FP8-only `_shard_fp8_qkv_proj` never fires and the generic
   loader assumes a simple `[Q|K|V]` layout — at TP>1 that selects half the
   wrong rows. This mirrors `_shard_fp8_qkv_proj`'s de-interleave without its
   FP8 dequant/requant steps (verified row-exact against it).

## Pack contract (`EXL3_MIMO_PACK.md`)

`exl3-v1`: `exl3-manifest.json` + one `exl3-layer-<NNNNN>.safetensors` per MoE
layer (`codes`/`rotations`/`gate_suh`/`up_suh`/`down_svh`). Geometry: 256
experts, hidden 4096, intermediate 2048 → `num_slots` 64 (`slot_channels` 32);
codebook `mcg`, uniform bits, `scaled_hadamard` block 128, uncoupled for the
first cut.

> Note: the newer upstream **`btx-atoms-v1`** container generation is *not*
> present in this image's B12X — do not repoint at it. Build `exl3-v1`.

## Serving

The pack is experts-only. Before serving, assemble a servable checkpoint from
the source model (minus routed experts) + the `exl3-v1` container + an
`exl3` `quantization_config`:

```bash
pack-build assemble <src> <exl3-v1> <serve-out>
```

The output is a normal-looking HF checkpoint that the FES weight-staging path
(`model-weights.sh stage` → `/opt/llm/models/<slug>` → `mods/fes-weights` hub
shim) stages and the recipe serves. See `EXL3_MIMO_PACK.md` and
`pack-build/NEW_MODEL.md`.

## Provenance / licensing

- The fork's `exl3.py` is Apache-2.0 (vLLM project / Local Inference Lab);
  this mod patches it in place. No third-party code vendored.
- The encoder used to build the pack (`shihanqu/shapleymcg`) is
  **source-available with attribution, not OSI**; B12X-derived parts
  Apache-2.0. Check its `LICENSE` / `THIRD_PARTY_NOTICES.md` before
  redistributing derived artifacts.

## First-launch verification

- startup log shows EXL3 layers planned and the fused trellis MoE engaged —
  not a silent fallback;
- greedy output token-identical to the TabbyAPI/benthecarman baseline;
- wikitext-2 PPL ≈ 5.4 for a 2.25 bpw-class pack;
- `moe.activation` resolves to `MoEActivation.SILU`.
