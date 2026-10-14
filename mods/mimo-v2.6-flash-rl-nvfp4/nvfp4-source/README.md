# MiMo-V2.6-Flash-RL NVFP4 — source structure capture

Gathered 2026-10-13 on `gx10-node1` (via fulton-remote-shell MCP) while adapting the
EXL3 pack/serve system to a **non-EXL3** quant path. This is the pack-free twin of
`mods/exl3-mimo/uncensored-source/`: a structural + quantization analysis of the
NVFP4 checkpoint, **not** an EXL3 pack build.

**Source truth (on-node, NFS — mounted on every cluster node):**

`/nas-1/models/mimo/mimo-v2.6-flash-nvfp4`

> Upstream provenance: this is `primitive-ai/MiMo-V2.6-Flash-RL-NVFP4` (see the HF
> card mirrored in the on-node `README.md`), an **exact transcode** of
> `XiaomiMiMo/MiMo-V2.6-Flash-RL`'s routed experts from MXFP4 to NVFP4 — every
> routed-expert weight value is preserved; only the expert *format* and the
> per-layer FP4 activation scales are new. Not HF-downloadable in this deployment;
> staged from the NAS archive under FES slug `mimo-v2.6-flash-nvfp4`.

## Contents

| Path | What it is |
|---|---|
| `geometry.json` | Geometry record equivalent to `pack-model-info.py detect()`: hidden/intermediate/expert counts, MoE layer indices, quantization_config (compacted), index metadata |
| `summary.json` | Per-shard tensor counts + byte sizes, total tensors, index metadata, tensor dtype histogram |
| `structure.txt` | Human-readable dump: geometry + shard summary + expert→shard map + key tensor templates |
| `weight_map.json.gz` | Complete `model.safetensors.index.json` (145 273 tensors → 65 shards) |
| `tensor_inventory.jsonl.gz` | Full per-tensor inventory: `name`, `dtype`, `shape`, `shard` (from the safetensors headers) |
| `dflash/dflash_structure.json` | DFlash drafter structure (config + tensor list), from `dflash/` |
| `vllm_patch/mimo_v2.diff` | The **exact shipped 21-line ModelOpt load bridge** (see "Serving" below) |
| `capture_nvfp4.py` | The capture tool that produced the above (reads safetensors headers only — never materializes tensors) |

`model-metadata/` is **not committed** (large / LFS). On-node, uncommitted, under the source dir:
`config.json`, `generation_config.json`, `preprocessor_config.json`, `tokenizer.json`,
`vocab.json`, `merges.txt`, `tokenizer_config.json`, `chat_template.jinja`,
`modeling_mimo_v2.py`, `configuration_mimo_v2.py`, `MiMo_V2_6_technical_report.pdf`,
`audio_tokenizer/` (its own `model.safetensors` ~1.87 GB), `assets/architecture.png`.

## Geometry (authoritative, from `config.json`)

`MiMoV2ForCausalLM`; `hidden_size` **4096**; `moe_intermediate_size` **2048**
(→ `num_slots` 64, `slot_channels` 32); `intermediate_size` 16384 (dense MLP);
`n_routed_experts` **256**; **47 MoE layers** (layers 1–47; layer 0 dense);
`num_hidden_layers` 48; `num_nextn_predict_layers` 3 (MTP); `num_experts_per_tok` 8;
`num_attention_heads` 64, `num_key_value_heads` 4 (SWA: 8 kv, `head_dim` 192 / v 128);
fused QKV layout; `vocab_size` 152576; `max_position_embeddings` 1048576;
`sliding_window` 128; rope θ 1e7, `partial_rotary_factor` 0.334; `hidden_act` **silu**;
`dtype` bfloat16.

**Geometry matches the Flash EXL3 source** — the 9 MoE/core fields captured in
`mods/exl3-mimo/uncensored-source/geometry.json` (hidden 4096, moe-intermediate 2048,
256 experts, 48 layers / 47 MoE, layer 0 dense, 3 MTP layers, topk 8) are identical; the
attention geometry above matches the shared MiMo-V2.6 Flash config (the EXL3 capture JSON
omits attention fields, so attention equivalence is asserted against the live config, not
that file). Only the routed-expert **storage format** differs.

## Storage layout

- **EP-sharded**, 65 shards / 145 273 tensors / `total_size` **182 403 280 256 B**
  (170.0 GiB; 187 GB on disk). `model_pp0_ep0..63_shard0.safetensors` hold the
  routed experts (4 experts/shard); `model_pp0_ep0_shard0.safetensors` (13.5 GB, 3097
  tensors) additionally holds all non-expert tensors (embed, attention, norms, router,
  visual/audio towers, speech_embeddings) under the `model.` prefix;
  `model_mtp.safetensors` holds the 3 MTP layers.
- Index metadata: `{"total_size": 182403280256, "save_format": "modelopt_mixed"}`.
- Tensor dtype histogram: **BF16 716** (o_proj, embeddings, lm_head, routers, vision/audio
  encoders), **U8 36 096** (NVFP4 expert codes), **F8_E4M3 36 159** (expert + FP8-block
  scales + qkv), **F32 72 302** (per-expert/per-layer FP32 scales).

## Quantization (authoritative, `quantization_config`)

```json
{"quant_method": "modelopt", "quant_algo": "MIXED_PRECISION", "kv_cache_quant_algo": null,
 "exclude_modules": [], "quantized_layers": <36159 entries>,
 "producer": {"name": "primitive-ai mimo_nvfp4.py (exact transcode)", "source_quantization": "fp8"}}
```

`quantized_layers` (compacted in `geometry.json`):

| module | count | quant_algo | tensors |
|---|---|---|---|
| `*.mlp.experts.{gate,up,down}_proj` | 36 096 | **NVFP4** (E2M1 codes, E4M3 scale per 16, FP32 per-tensor global, W4A4) | `.weight` (U8) + `.weight_scale` (F8_E4M3, per-16) + `.weight_scale_2` (F32, per-tensor) + `.input_scale` |
| `*.self_attn.qkv_proj` (51, incl. 3 MTP) + MTP MLP (9) + layer-0 dense MLP (3) | 63 | **FP8_PB_WO** (FP8 block 128×128) | `.weight` (F8_E4M3) + **`.weight_scale_inv` (F32, 2-D `[out_blk, in_blk]`)** |
| `o_proj`, embeddings, `lm_head`, routers, vision/audio encoders | — | BF16 | `.weight` (BF16) |

The counts and algos above are read directly from the checkpoint's
`quantization_config.quantized_layers`. The *producer's* card (on-node `README.md`)
claims the FP8 qkv and BF16 tensors are "byte-identical to the release" and the
NVFP4 experts "reproduce the released MXFP4 values exactly" — this capture does
not independently verify those byte-level claims.

**Naming caveat (critical for serving):** on disk this checkpoint is self-consistent —
non-expert FP8_PB_WO layers store `weight_scale_inv` (2-D), exactly like the release;
only NVFP4 experts use `weight_scale`/`weight_scale_2`. The conflict is on the **vLLM
param side**: the b12x image's FP8_PB_WO scheme (`modelopt.py::KFp8Block128`) registers
its module scale param as **`weight_scale`** (4-D `[out_blk,1,in_blk,1]`), and the
fork's MiMo fused-QKV loader (`mimo_v2.py::_try_load_fp8_qkv_proj`) recognizes only the
`weight_scale_inv` *suffix/param*. So the shipped load bridge is needed to reconcile the
`weight_scale_inv` on-disk name with the `weight_scale` param name.

## DFlash drafter

`dflash/` is carried in the checkpoint (as in the EXL3 source). Capture in
`dflash/dflash_structure.json`: `DFlashDraftModel`, **5 layers**, hidden **4096**,
intermediate 16384, 64 q-heads / 8 kv-heads, head_dim 128, `block_size` 8, `sliding_window`
1024, `is_causal: false`, `num_target_layers` 48, `target_layer_ids` **[0,11,23,35,47]**,
`mask_token_id` 151675, `attention_value_scale` 0.612; 63 BF16 tensors,
`dflash_draft_model.safetensors` ~2.9 GB (+ `mask_embedding.pt`).

This is **identical to the EXL3 source drafter**, so `mods/mimo-v2.6-flash` stages it
unchanged (from `MIMO_V26_MODEL_ID`'s `dflash/`).

## Serving (the load bridge)

The b12x image's `modelopt.py` natively resolves `MIXED_PRECISION` → `modelopt_mixed`,
NVFP4 routed experts → `ModelOptNvFp4FusedMoE`, and the 63 FP8_PB_WO layers →
`KFp8Block128` (which registers the module scale param as **`weight_scale`**, 4-D
`[out_blk,1,in_blk,1]`). The gap is in `models/mimo_v2.py`: its fused-QKV loader
(`_try_load_fp8_qkv_proj`) keys on the `weight_scale_inv` **on-disk** name and the
`weight_scale_inv` **param** name, so the ModelOpt 4-D `weight_scale` block scale
(and the fused-qkv weight it guards) is silently dropped → garbage qkv output.
The generic per-tensor tail likewise looks the param up by the on-disk name.

The checkpoint ships `vllm_patch/mimo_v2.diff` (copied verbatim here) — the 21-line
ModelOpt naming bridge (detect `_mopt_ws`; map `weight_scale_inv`→`weight_scale`;
reshape the 4-D block scale to the loader's expected shape; rewrite the tail param name).
`mods/mimo-v2.6-flash-rl-nvfp4` ports this diff into the image's `mimo_v2.py` as an
anchor-gated, idempotent patcher (verified applying cleanly + idempotently against
`vllm-node-b12x:latest`'s `mimo_v2.py`, sha256 `fc310f21…`). The EXL3 mod
(`mods/exl3-mimo`) is **not** applied for NVFP4.

Serve flags differ from the EXL3 path: `--quantization modelopt_mixed` (not `exl3`),
`--load-format b12x`, `--moe-backend b12x`; the DFlash drafter, P2P Mooncake PD
disaggregation, and `mods/mimo-diffkv-fp8-kv` KV layout are unchanged. See
`recipes/pd-disagg-mimo-nvfp4-4x.yaml`, `recipes/mimo-v2.6-flash-rl-nvfp4-2x.yaml`,
`recipes/mimo-v2.6-flash-rl-nvfp4-4x.yaml`.

## Sizing — why TP=1 per PD stage is infeasible

Routed experts are ~4.5 bpw (NVFP4) vs EXL3's ~3 bpw, so the weight set is larger:
**~170 GiB** of weights vs the EXL3 pack's ~110 GB. A TP=1 stage loads the **entire**
weight set into one node's **~122 GiB** UMA — it cannot fit (GB10 UMA has no weight
offload; all experts must be resident). The upstream card itself states 2×96 GB cards.
Byte math: ~170 GiB weights > ~122 GiB UMA ⇒ TP=1 OOMs at load.

- **TP=2** → ~85 GiB weights/rank (+ towers/KV/activations): fits, tight.
- **TP=4** → ~43 GiB weights/rank: comfortable.

`mods/pd-disagg/dispatch.sh` fails closed at `NNODES!=4`/`TP!=2`, matching this TP=2
sub-group floor. The supported NVFP4 configurations are therefore PD TP=2/TP=2 (4 nodes),
non-PD TP=2 (2 nodes), and non-PD TP=4 (4 nodes).

## All on `/nas-1` (49 TB; 25 TB free at capture)
