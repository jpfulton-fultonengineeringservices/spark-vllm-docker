# Qwen3.8-27B-Coder390-EThink-DFlash2 — source structure capture

Gathered 2026-10-14 on `gx10-node1` (via fulton-remote-shell MCP) for EXL3 pack/serve
planning. Pack-free capture: structural analysis of a dense Qwen3.8-27B-lineage
checkpoint that ships an FP8 DFlash2 drafter subdirectory. **Not** an EXL3 pack build.

**Source truth (on-node, NFS — mounted on every cluster node):**

`/nas-1/models/qwen3.8/qwen3.8-27b-coder390-ethink-dflash2`

> Uniqueness: `/nas-1/models/qwen3.8/` contains only this directory, four
> `qwen3.8-27b-nvfp4-*` variants, `qwen3.8-flash-next` (captured separately), and
> `qwen-agentworld-35b-a3b-nvfp4-w4a16`. No other coder390 / DFlash2 twin exists.
> The earlier long-named directory `Qwen3.8-27B-Coder390-EfficientThink-…-DFlash2-BF16`
> does **not** exist — the short name above is the real one.

## Contents

| Path | What it is |
|---|---|
| `geometry.json` | Geometry record equivalent to `exl3pack.geometry.detect()`: hidden/intermediate counts, layer_types, `text_config`/`vision_config` (raw), index metadata |
| `summary.json` | Per-shard tensor counts + byte sizes, total tensors, index metadata, dtype histogram |
| `structure.txt` | Human-readable dump: geometry + shard summary + key tensor templates (**no expert shard map — dense model**) |
| `weight_map.json.gz` | Complete `model.safetensors.index.json` (1199 tensors, 3 shards) |
| `tensor_inventory.jsonl.gz` | Per-tensor `{name, dtype, shape, shard}` inventory (1199 rows) |
| `drafter/DFlash2-FP8/structure.json` | Drafter config + 101-tensor inventory + file sizes/sha256 |
| `drafter/DFlash2-FP8/manifest.json` | Verbatim copy of the drafter's manifest (schema_version 1, FP8 E4M3 + BF16 keeps) |
| `drafter/DFlash2-FP8/SHA256SUMS` | Verbatim copy of the drafter's checksum manifest |
| `capture_source.py` | Generator used for this capture (shared with `qwen3.8-flash-next`) |

`model-metadata/` **not committed** (gitignored, LFS-tracked and unpushable to the
public fork). Derived copy of the on-node checkpoint's small text files.

All values in this README were read from the verified on-node `config.json`
(3799 bytes, md5 `7267333c…`, sha256 `b25e4020…` — the sha256 is listed in this
directory's own `SHA256SUMS`), which is the source for everything the capture
records. The six metadata copies currently present are byte-identical to the
on-node files (md5-verified): `generation_config.json` `7a45161e…`,
`preprocessor_config.json` `7bcaedf7…`, `video_preprocessor_config.json`
`41bc84d7…`, `chat_template.jinja` `519239a4…` (identical to checkpoint A's),
`tokenizer_config.json` `71ae0856…`, `SHA256SUMS`.

**Not transferred** (exceed the MCP `shell_read_file` cap): `config.json`
(3799 B), `tokenizer.json` (19.9 MB). Regenerate from
`/nas-1/models/qwen3.8/qwen3.8-27b-coder390-ethink-dflash2` when needed.

`geometry.json`/`summary.json`/`structure.txt`/`weight_map.json.gz` and the
`drafter/` capture are the verified artifacts and are independent of this
directory.

## Geometry (`config.json`) — dense, no MoE

`Qwen3_5ForConditionalGeneration` (top-level `model_type` **`qwen3_5`**); geometry
nested under **`text_config`** (`model_type` **`qwen3_5_text`**), vision under
`vision_config` (`qwen3_5_vision`).

| field | value |
|---|---|
| `hidden_size` | **5120** |
| `intermediate_size` | **17408** (dense MLP — no `moe_intermediate_size`) |
| `num_experts` | **0** (no `num_experts`/`n_routed_experts`/`moe_intermediate_size`) |
| `num_slots` | **N/A** (dense model) |
| `moe_layer_count` / `dense_layers` | **0** / all 64 layers |
| `num_hidden_layers` | **64** |
| `num_attention_heads` / `num_key_value_heads` | 24 / 4 |
| `head_dim` | 256 |
| `attn_output_gate` / `output_gate_type` | true / **`"swish"`** |
| linear-attn (Gated DeltaNet) | 16 QK heads, 48 V heads, head dim 128, conv kernel 4 |
| `layer_types` | explicit **64-entry array**: `full_attention` at indices 3, 7, …, 63 (**16 layers**), `linear_attention` elsewhere (**48 layers**); `full_attention_interval` 4 |
| `mtp_num_hidden_layers` | 1 |
| `vocab_size` | 248 320 |
| `max_position_embeddings` | 262 144 |
| `dtype` | bfloat16 |
| `quantization_config` | **null** at top level (the only quantized artifact is the drafter subdir) |

## Storage layout

- **3 shards / 1199 tensors**:
  - `model-00001-of-00002.safetensors` — 49 825 162 976 B (49.8 GB), **784** tensors
    (`lm_head`, `language_model.embed_tokens`, 782 layer tensors)
  - `model-00002-of-00002.safetensors` — 4 888 445 168 B (4.9 GB), **400** tensors
    (66 layer tensors, `language_model.norm`, 333 visual tensors)
  - **`mtp-bf16.safetensors`** — 849 400 424 B (811 MiB), **15** `mtp.*` tensors.
    A **standalone index shard, not a `model-NNNNN` file** — it is named in the same
    `model.safetensors.index.json` weight map, which is why the shard count is 3.
- Index metadata: `metadata.total_size` **55 562 855 904 B** (~55.5 GB). No `save_format`.
- Tensor dtype histogram: **BF16 1199** (all).
- The dir also carries a `@eaDir/` (Synology metadata directory) — the capture skips
  `.cache` and `@eaDir` when discovering subdirectories.

## Key naming caveats

- **Nesting is `model.language_model.layers.<N>.…`**; the MTP block uses the top-level
  `mtp.…` prefix (`mtp.fc_embedding.weight`, `mtp.layers.0.…`). The capture's
  `layer_of()` scans for any `layers.<N>` segment, so both resolve.
- **Dense MLP**, not MoE: `model.language_model.layers.<N>.mlp.{gate,up}_proj.weight`
  (17408, 5120) and `…mlp.down_proj.weight` (5120, 17408) — 64 templates. The
  `.mlp.experts.` marker never matches, so `summary.json` carries **no
  `expert_shard_map` key** and `structure.txt` omits the `=== EXPERT SHARD MAP ===`
  section entirely (deliberately not emitting an empty section).

## DFlash2 drafter (`drafter/DFlash2-FP8/`)

`DFlash2DraftModel`, `model_type` **`qwen3`**, **5 layers** all `sliding_attention`,
`sliding_window` 2048, `max_window_layers` 5, hidden 5120, intermediate 17408,
`head_dim` 128, 32 heads / 8 KV heads, `num_target_layers` 64,
`dflash_config.target_layer_ids` **[5, 19, 33, 47, 61]**, `mask_token_id` 248070,
`block_size` 8, `conv_group_size` 16, `conv_kernel_size` 2, `selector_rank` 256,
`selector_top_k` 16; vocab 248 320, rope θ 1e7, `max_position_embeddings` 262 144.

- `model.safetensors` — 2 407 379 992 B (2.4 GB), **101 tensors**:
  **20 `F8_E4M3`** weights, **20 `F32` `weight_scale_inv`**, **61 `BF16`**
  (`weight_block_size` [128, 128], `modules_to_not_convert` includes
  q_proj/k_proj/v_proj/qkv_proj/fc/lm_head).
- Tensor keys live under bare `layers.<N>…`, plus `fc.weight`,
  `candidate_selector.*`, `norm.weight`, `hidden_norm.weight`.
- The drafter's `manifest.json` (schema_version 1, `artifact_role`
  "DFlash2 speculative draft model", `tensor_counts` {float8_e4m3fn 20,
  float32_block_scale 20, bfloat16 61, total 101}) and `SHA256SUMS` are copied
  verbatim into the capture.

**Distinct from the `mtp.*` in-body block.** The drafter is a separate subdirectory
artifact; `mtp.*` tensors live in `mtp-bf16.safetensors` in the main weight map.

## Serving context

The Qwen3.8-27B-family DFlash2 consumers in this repo are
`recipes/qwen3.8-27b-nvfp4-dflash2.yaml` (TP=1 solo, TP=2 cluster) and the README's
Qwen3.8-27B NVFP4 + DFlash2 section, both of which reference
`z-lab/Qwen3.8-27B-DFlash2` as an *external* draft model via
`--speculative-config {"method":"dflash","model":"z-lab/Qwen3.8-27B-DFlash2",...}`.
**This checkpoint's bundled `DFlash2-FP8/` drafter is its own FP8 artifact**, not that
external draft model — do not conflate the two when wiring speculative decoding.

## All on `/nas-1`

`/nas-1/models/qwen3.8/` also holds `qwen3.8-flash-next` (captured separately), four
`qwen3.8-27b-nvfp4-*` variants, and `qwen-agentworld-35b-a3b-nvfp4-w4a16`.
