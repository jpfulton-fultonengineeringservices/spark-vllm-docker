# MiMo-V2.6-Flash-RL-uncensored — source structure capture

Gathered 2026-10-07 from `gx10-node4` (via fulton-remote-shell MCP) for the next
phase of adapting the EXL3 pack system to this new source checkpoint.

**Source of truth (on-node, NFS — mounted on every cluster node):**
`/nas-1/models/mimo/mimo-v2.6-flash-rl-uncensored`

> Note: the requested path `/opt/llm/staging/mimo-v2.6-flash-rl-uncensored ->
> /nas-1/models/...` does **not** exist on node4. The staging dir there contains
> only `mimo-v2.6-pro-rl-uncensored`, and `/opt/llm/staging` has no symlinks.
> The real uncensored **flash** checkpoint lives directly at the `/nas-1` path
> above (created 2026-10-03), and that is what was captured.

## Contents of this directory

| Path | What it is |
|---|---|
| `geometry.json` | Geometry record equivalent to `pack-model-info.py detect()`: hidden/intermediate/expert counts, MoE layer indices, quantization_config, index metadata |
| `summary.json` | Aggregate: shard list + byte sizes, expert→shard map, total tensors, config-identical-to-working flag |
| `structure.txt` | Human-readable dump of the above (geometry + shard summary + expert map + key templates) |
| `weight_map.json.gz` | Complete `model.safetensors.index.json` from the source (73 081 tensors → 65 shards) |
| `tensor_inventory.jsonl.gz` | Full per-tensor inventory: `name`, `dtype`, `shape`, `shard` (from safetensors headers) |
| `model-metadata/` | **Not committed** — derived copy of the checkpoint's small config/code/tokenizer files (`config.json`, `generation_config.json`, `preprocessor_config.json`, `tokenizer.json`, `vocab.json`, `merges.txt`, `tokenizer_config.json`, `chat_template.jinja(.orig)`, `modeling_mimo_v2.py`, `configuration_mimo_v2.py`, tech-report PDF, `dflash/` config + `dflash.py` + draft index + `mask_embedding.pt`, `audio_tokenizer/` configs). Reproduce from the on-node source; gitignored (LFS-tracked, unpushable to the public fork). |
| `packref/` | Target-format reference from the **existing** working pack: `v1-exl3-manifest.json`, `v1-index.json` (assembled dense-checkpoint index), `v1-config.json`, `v2-config.json` |
| `dflash/dflash_structure.json` | Structural capture of the standalone DFlash drafter across three checkpoints (`working`, `uncensored`, `exl3-v1-pack`): config, index, all 63 tensors (name/dtype/shape), and file sizes + sha256 |

## Key findings for the pack-system adaptation

### 1. Source packaging = same mxfp4 / EP-sharded layout as the working 4x source
`config.json` is **byte-identical** to `/nas-1/models/mimo/mimo-v2.6-flash-rl`
(the checkpoint `mimo-v2.6-flash-4x.yaml` serves). Index metadata is identical
too: `{"save_format":"mxfp4","total_size":172923364096,"tp_size":4}`.

- 66 safetensors, ~161 GiB (172 932 505 288 B): `model_pp0_ep0..63_shard0.safetensors`
  (64 expert-parallel shards) + `model_mtp.safetensors`.
- Each `ep` shard holds exactly **4 experts** for **all 47 MoE layers**:
  `ep0`→experts 0–3 … `ep63`→experts 252–255.
- `model_pp0_ep0_shard0.safetensors` additionally holds all non-expert tensors
  (embed, attention, norms, router, visual/audio towers, MTP) — 841 of them under
  the `model.` prefix.
- Unlike MiMo-V2.6-Flash-RL, this bundle is **mxfp4-stored** (U8 blocks + F32
  `weight_scale`): experts are `U8 (4096,1024)/(2048,2048)` gate/up + `U8 (2048,128)/(4096,64)`
  down with per-expert `weight_scale`; non-expert routed weights use FP8 (`F8_E4M3`
  + `weight_scale_inv`). There is **no** `model_token_embed.safetensors`-style
  sharding and no separate MTP shards beyond `model_mtp.safetensors`.

### 2. Geometry (authoritative, from config.json)
`MimoV2ForCausalLM`; `hidden_size` 4096; `moe_intermediate_size` 2048 (→
`num_slots` 64, `slot_channels` 32); `intermediate_size` 16384 (dense MLP);
`n_routed_experts` 256; **47 MoE layers** = layers 1–47 (`moe_layer_count` 47,
`moe_layers` 1..47, `dense_layers` [0]); `num_hidden_layers` 48;
`num_nextn_predict_layers` 3 (MTP); `num_experts_per_tok` 8;
`num_attention_heads` 64, `num_key_value_heads` 4 (SWA: 8 kv, head_dim 192 / v 128);
fused QKV layout; `vocab_size` 152576; rope θ 1e7, partial_rotary_factor 0.334.

### 3. Existing pack target format (what we must adapt toward)
`packref/v1-config.json` (the working EXL3 pack's config) shows the assembled
serving checkpoint shape:
- `quantization_config.quant_method = "exl3"`, `codebook = "mcg"`, `bits = 3`,
  `exl3.manifest = "exl3-manifest.json"`, `dense_format = "bf16"`,
  `ignored_layers = [down_proj, eh_proj, gate, gate_proj, k_proj, lm_head, o_proj,
  q_proj, qkv_proj, up_proj, v_proj]`, and the original source FP8 config nested
  under `original_quantization_config`.
- Dense checkpoint is **4 shards** `model-0000{1..4}-of-00004.safetensors`
  (826 tensors) with **all `experts.*` tensors removed** — experts live only in
  the `exl3-manifest.json` + `exl3-layer-<NNNNN>.safetensors` container.
- `packref/v1-exl3-manifest.json`: `schema exl3-v1`, `codebook mcg`,
  geometry block (same numbers as above), `rates.structure uniform / bits 3`,
  `hadamard` flags, `layout` (row_alignment 4096, extent_alignment_slots 4,
  extent_barriers [32]), and a per-layer map of file + sha256.

### 4. What differs vs. the working pack's source
The working pack was built from `mimo-v2.6-flash-rl` (Sep 22). The new source
checkpoint `mimo-v2.6-flash-rl-uncensored` (Oct 3) is a **dealignai
"uncensored" weight-level refusal-removal** of `XiaomiMiMo/MiMo-V2.6-Flash-RL`
(HarmBench-320: 99.69% compliance thinking-off / 90.94% thinking-on, per its
README). Vision, audio and the DFlash speculative head are preserved. The source
*packaging* is identical to the working one, so the pack pipeline can be driven
the same way; the deliverable will be an uncensored EXL3 pack.

### 4a. DFlash drafter (draft model used by the TP4 recipe)
There is **no separate NAS drafter model**. The DFlash speculative-decoding head
is the `dflash/` subdirectory of the checkpoint and is staged at runtime by
`mods/mimo-v2.6-flash` to `/workspace/MiMo-V2.6-Flash-RL-dflash` (vLLM cannot
address a Hub subdir; the dir also carries `mask_embedding.pt` at its root, and
the shipped `config.json` has a trailing comma that transformers rejects, so the
mod sanitizes it). Structural capture is in `dflash/dflash_structure.json`:

- `DFlashDraftModel`, 5 layers, hidden 4096, intermediate 16384, 64 q-heads /
  8 kv-heads, head_dim 128 (v 128), `block_size` 8, `sliding_window` 1024,
  `is_causal: false`, `num_target_layers` 48, `target_layer_ids` [0,11,23,35,47]
  (the 5 anchors), `mask_token_id` 151675, `attention_value_scale` 0.612.
- 63 tensors, all BF16, one file `dflash_draft_model.safetensors`
  (~6.3 GiB, `total_size` 6 734 920 760 in the index).
- The drafter is **byte-identical (sha256) and config-identical across the
  working checkpoint, the uncensored checkpoint, and the existing exl3-v1 pack**
  — only the target model's weights differ, so the DFlash head needs no rebuild
  for the uncensored pack. Note the TP4 recipe stages it
  `--speculative-config '{"method":"dflash","model":"/workspace/MiMo-V2.6-Flash-RL-dflash",
  "num_speculative_tokens":7,"attention_backend":"B12X"}'`.

## How the pack build consumes this (from `pack-build/NEW_MODEL.md`)
1. `pack-build detect  <source-dir>` — auto-detection of geometry (this capture
   is the equivalent output; expect no `_warning`).
2. `pack-build convert <src> <mcg-out> <work> 3 mcg` — exllamav3 quantizer.
3. `pack-build repack  <mcg-out> <exl3-v1-out> 3 --self-check`.
4. `pack-build assemble <src> <exl3-v1-out> <serve-out>` — dense checkpoint
   (experts removed) + fork-shaped `quantization_config`.
5. Boot via `recipes/mimo-v2.6-flash-exl3-2x.yaml` (TP=2) / `-1x.yaml`.

All staged under `/nas-1` (30 TB; ~28 TB free as of capture).
