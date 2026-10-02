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

1. `Exl3Config.from_config` — drops the DeepSeek KDA/MLA arrangement checks
   (`dense_format == "mxfp8"`, the KDA ignore-list, the `q/k/v_proj`
   exclusion); keeps the `exl3-manifest.json` requirement and a fail-closed
   `ignored_layers` list check; accepts `fp8` or `mxfp8` dense.
2. `Exl3MoEMethod.__init__` — admits `MoEActivation.SILU` (MiMo) alongside
   DeepSeek's SiTU(4/25); BF16 + bias-free still enforced.
3. The fused-MoE weight plan is built with `nonlinearity="silu"`.

## Pack contract (`EXL3_MIMO_PACK.md`)

`exl3-v1`: `exl3-manifest.json` + one `exl3-layer-<NNNNN>.safetensors` per MoE
layer (`codes`/`rotations`/`gate_suh`/`up_suh`/`down_svh`). Geometry: 256
experts, hidden 4096, intermediate 2048 → `num_slots` 64 (`slot_channels` 32);
codebook `mcg`, uniform bits, `scaled_hadamard` block 128, uncoupled for the
first cut.

> Note: the newer upstream **`btx-atoms-v1`** container generation is *not*
> present in this image's B12X — do not repoint at it. Build `exl3-v1`.

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
