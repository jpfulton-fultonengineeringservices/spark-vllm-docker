# MiMo-V2.6-Pro-RL-uncensored — source structure capture

Gathered 2026-10-07 from `gx10-node4` (via fulton-remote-shell MCP + scp) for the
next phase of adapting the EXL3 pack system to the Pro source checkpoint.

**Source of truth (on-node, NFS — mounted on every cluster node):**
`/nas-1/models/mimo/mimo-v2.6-pro-rl-uncensored` (~535 GiB, created 2026-10-03)
`/opt/llm/staging/mimo-v2.6-pro-rl-uncensored` is a plain copy (same 130 shards,
same index metadata). This is the only Pro checkpoint on the NAS.

> **Pro set is complete.** A wider sweep (`find /nas-1 -maxdepth 4 -iname '*pro*'`,
> plus `/nas-1/models/{checkpoints,hf_cache,vllm}`, plus every
> `model_pp0_ep*_shard0.safetensors` root) found exactly one Pro checkpoint —
> this one — and only three mxfp4/EP-sharded checkpoints total, all under
> `/nas-1/models/mimo/` (`flash-rl`, `flash-rl-uncensored`, `pro-rl-uncensored`).
> Nothing else to capture for the Pro tier.

## Contents of this directory

| Path | What it is |
|---|---|
| `geometry.json` | Geometry record equivalent to `pack-model-info.py detect()` + extra fields (heads, layer patterns, quantization_config, index metadata) |
| `summary.json` | Aggregate: per-shard tensor counts + byte sizes, expert→shard map, total tensors |
| `structure.txt` | Human-readable dump (geometry + shard summary + expert map + key templates) |
| `weight_map.json.gz` | Complete `model.safetensors.index.json` (160 040 tensors → 130 shards) |
| `tensor_inventory.jsonl.gz` | Full per-tensor inventory: `name`, `dtype`, `shape`, `shard` |
| `model-metadata/` | **Not committed** — derived copy of the checkpoint's config/tokenizer/code (`config.json`, tokenizers, `modeling_mimo_v2.py`, `configuration_mimo_v2.py`, tech-report PDF, `dflash/` + `audio_tokenizer/` configs, drafter `mask_embedding.pt`). Reproduce from the on-node source; gitignored (LFS-tracked, unpushable to the public fork). |
| `dflash/dflash_structure.json` | DFlash drafter structural capture (config, index, 63 tensors, file sizes + sha256) |

## Key findings

### 1. Same mxfp4 packaging family as Flash, but a larger model and different EP layout
`save_format: mxfp4`, but **`tp_size: 8`** (Flash was 4) and **EP=128**:

- 130 safetensors, ~527 GiB (566 048 103 928 B):
  `model_pp0_ep0..127_shard0.safetensors` (128 EP shards) +
  `model_pp0_ep0_shard1.safetensors` (the one `shard1` overflow file) +
  `model_mtp.safetensors`.
- Each `ep<n>_shard0` holds exactly **3 experts** for **all 69 MoE layers**
  (`ep0`→experts 0–2 … `ep127`→experts 381–383; 128 × 3 = **384 experts**).
- Non-expert tensors are split across **two files**: `ep0_shard0` (575: embed,
  lm_head, all attention, router, norms, patch-embed + part of the visual tower)
  and `ep0_shard1` (441: the audio encoder, `speech_embeddings`, and the rest of
  the visual tower). `model_mtp.safetensors` = 48 (3 MTP layers).
  The `ep0_shard0` file also carries the two large visual buffer tensors
  (`1280×3×2×16×16`, `5120×5120` in Flash; scaled for Pro).

### 2. Geometry (authoritative, from config.json)
`MiMoV2ForCausalLM`; `hidden_size` **6144**; `moe_intermediate_size` 2048
(→ `num_slots` 64, `slot_channels` 32); `intermediate_size` 16384;
`n_routed_experts` **384**; `num_hidden_layers` **70** with **69 MoE layers**
(layers 1–69; layer 0 dense); `num_experts_per_tok` 8; `num_attention_heads`
**128**, `num_key_value_heads` 8 (SWA: 8 kv, head_dim 192 / v 128);
`vocab_size` 152576; `max_position_embeddings` 1048576; rope θ 1e7,
`partial_rotary_factor` 0.334; `hybrid_layer_pattern` and `moe_layer_freq` are
both length 70; `layernorm_epsilon` 1e-5. `num_nextn_predict_layers` is **absent**
(Flash had 3) though `model_mtp.*` weights are present.

**Hard exl3-v1 constraint satisfied:** `moe_intermediate_size` 2048 % 32 == 0 →
`num_slots` 64, `slot_channels` 32 (identical to Flash). `pack-model-info.py`
emits no `_warning` for Pro.

> Distinct from Flash (hidden 4096, 256 experts, 48 layers, 47 MoE, tp_size 4).
> Auto-detection (`pack-model-info.py`) handles this with no override —
> `hidden_size`/`n_routed_experts`/`moe_intermediate_size` are all standard keys.

### 3. DFlash drafter — Pro-specific, differs from Flash
`DFlashDraftModel`: 5 layers, **hidden 6144**, intermediate 16384,
**128 q-heads / 8 kv-heads**, head_dim 128, `block_size` 8, `sliding_window`
1024, `is_causal: false`, `num_target_layers` **70**, `target_layer_ids`
**[0, 15, 31, 47, 69]** (the 5 anchors — Flash used [0,11,23,35,47]),
`mask_token_id` 151675, `attention_value_scale` 0.612. 63 BF16 tensors, one
`dflash_draft_model.safetensors` (~5.16 GiB; see `dflash/dflash_structure.json`
for exact byte size / `total_size` index metadata).

- `dflash_draft_model.safetensors` sha256 `39208e3d…`, `mask_embedding.pt`
  sha256 `436ad088…` — **different from the Flash drafter** (which was sha
  `94d9c02c…` / `b35b379f…`), so the Pro pack **must use its own drafter**, not
  the Flash one. It ships embedded as the checkpoint's `dflash/` subdir and is
  staged at runtime by `mods/mimo-v2.6-flash` (vLLM cannot address a Hub subdir).

### 4. Sibling of the working pack
The existing EXL3 pack (`/nas-1/models/mimo/mimo-v2.6-flash-rl-exl3-v1`) is a
Flash pack; Pro is the next tier up (6144 hidden / 384 experts / 70 layers).
The pack pipeline (`detect` → `convert` → `repack` → `assemble`) is
geometry-agnostic and should carry over directly; the numeric sizes differ enough
that staging/budget assumptions (EP=128, tp_size 8, ~527 GiB source) must be
re-derived.

## How the pack build consumes this (from `pack-build/NEW_MODEL.md`)
1. `pack-build detect  /nas-1/models/mimo/mimo-v2.6-pro-rl-uncensored` — expect
   no `_warning` (this capture is the equivalent output).
2. `pack-build convert <src> <mcg-out> <work> 3 mcg`.
3. `pack-build repack  <mcg-out> <exl3-v1-out> 3 --self-check`.
4. `pack-build assemble <src> <exl3-v1-out> <serve-out>`.
5. Boot via a recipe referencing the assembled dir.

All staged under `/nas-1` (30 TB; ~28 TB free as of capture).
