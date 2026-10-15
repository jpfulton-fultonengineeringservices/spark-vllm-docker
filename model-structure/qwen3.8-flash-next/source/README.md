# Qwen3.8-Flash-Next — source structure capture

Gathered 2026-10-14 on `gx10-node1` (via fulton-remote-shell MCP) for EXL3 pack/serve
planning. This is a pack-free capture: structural analysis of the BF16 base checkpoint,
**not** an EXL3 pack build. The served checkpoint is the NVFP4 variant
`local-inference-lab/Qwen3.8-Flash-Next-NVFP4` (no NVFP4 twin exists on `/nas-1`).

**Source truth (on-node, NFS — mounted on every cluster node):**

`/nas-1/models/qwen3.8/qwen3.8-flash-next`

> Uniqueness: `find /nas-1/models -maxdepth 3 -iname '*flash*next*'` returns exactly
> this one directory. There is no node-local or HF-cache twin.

## Contents

| Path | What it is |
|---|---|
| `geometry.json` | Geometry record equivalent to `exl3pack.geometry.detect()`: hidden/intermediate/expert counts, MoE layer indices, `text_config`/`vision_config` (raw), index metadata |
| `summary.json` | Per-shard tensor counts + byte sizes, total tensors, index metadata, dtype histogram, expert→shard map |
| `structure.txt` | Human-readable dump: geometry + shard summary + expert shard map + key tensor templates |
| `weight_map.json.gz` | Complete `model.safetensors.index.json` (1658 tensors, 131 shards) |
| `tensor_inventory.jsonl.gz` | Per-tensor `{name, dtype, shape, shard}` inventory (1658 rows) |
| `capture_source.py` | Generator used for this capture (shared with `qwen3.8-27b-coder390-ethink-dflash2`) |

`model-metadata/` **not committed** (gitignored, LFS-tracked and unpushable to the
public fork). Derived copy of the on-node checkpoint's small text files.

All values in this README were read from the verified on-node `config.json`
(md5 `6775ee0c…`, 4745 bytes), which is the source for everything the capture
records. The six metadata copies currently present are byte-identical to the
on-node files (md5-verified): `generation_config.json` `efeaa00f…`,
`preprocessor_config.json` `7bcaedf7…`, `video_preprocessor_config.json`
`41bc84d7…`, `chat_template.jinja` `519239a4…`, `tokenizer_config.json`
`e8436422…`, `LICENSE` `74d74256…`.

**Not transferred** (exceed the MCP `shell_read_file` cap, or failed checksum):
`config.json` (4745 B), `tokenizer.json` (12.8 MB), `vocab.json` (6.7 MB),
`merges.txt` (3.35 MB), `README.md` (65 KB). Regenerate from
`/nas-1/models/qwen3.8/qwen3.8-flash-next` when needed.

`geometry.json`/`summary.json`/`structure.txt`/`weight_map.json.gz` are the
verified artifacts and are independent of this directory.

## Geometry (`config.json`)

`Qwen4ExpForConditionalGeneration` (top-level `model_type` **`qwen4_exp`**); all model
geometry is nested under **`text_config`** (`model_type` **`qwen4_exp_text`**) — the
capture reads that sub-object, exactly as `exl3pack.geometry._text_config` does.

| field | value |
|---|---|
| `hidden_size` | **2560** |
| `moe_intermediate_size` | **640** (→ `num_slots` **20**, `slot_channels` 32) |
| `intermediate_size` | **absent** (no top-level or `text_config` key) |
| `num_experts` | **512** (`num_experts` key — there is no `n_routed_experts`) |
| `num_experts_per_tok` | 10 |
| `shared_expert_intermediate_size` | 640 |
| `moe_layer_count` / `dense_layers` | **48** / **`[]`** |
| `num_hidden_layers` | **48** |
| `num_attention_heads` / `num_key_value_heads` | 24 / 2 |
| `head_dim` | 256 |
| linear-attn (Gated DeltaNet) | 16 QK heads, 48 V heads, head dim 128, conv kernel 4 |
| `layer_types` | 48-entry array; **12 `full_attention`** at indices 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47 (`full_attention_interval` 4) |
| `mtp_num_hidden_layers` | 1 (MTP block is MoE too) |
| `hc_count` / `hc_lowrank` | 4 / 320 |
| PLE | `ple_layer_ids` [2], `ple_embed_dim` 2560, conv kernel 4 |
| n-gram | `ngram_size` 3, `ngram_vocab_size_base` 20 000 000, `split_ngram_parts` 128 |
| `vocab_size` | 248 320 |
| `max_position_embeddings` | 262 144 |
| `dtype` | bfloat16 |
| `quantization_config` | **null** (BF16 base; the served NVFP4 variant is a separate artifact) |
| `num_nextn_predict_layers` | absent → null |

## Storage layout

- **131 shards / 1658 tensors**, `model-00001-of-00131.safetensors` …
  `model-00131-of-00131.safetensors`. Shard packing is **non-uniform**:
 - shard **1** = all 333 `visual.*` tensors + `hyper_connection_mixer.weight`
 - shard **130** = `embed_tokens`
 - shard **131** = `lm_head.weight` alone
 - Expert tensors are **packed one per shard**, not scattered: 97 shards carry
   98 expert tensors (49 `gate_up_proj` + 49 `down_proj`) — 96 shards hold exactly
   one, and the single exception is shard **62**, which holds two
   (`layers.20…mlp.experts.down_proj` and the MTP `mtp.layers.0…mlp.experts.down_proj`).
  The MTP layer's two expert tensors sit in shards **60** (`gate_up_proj`) and
  **62** (`down_proj`). Expert shard ids form two clusters: **00002–00005** and
  **00038–00130**; shards 6–37 carry no expert tensors at all. `summary.json`'s
  `expert_shard_map` is what records this.
- Index metadata: no `save_format`; `metadata.total_size` **359 999 963 128 B**.
- **Two size figures, not reconciled** — they measure different things:
  `du -sh` reports **336 GiB** on disk; the index `total_size` sums parameter bytes
  (~360 GB).
- Tensor dtype histogram: **BF16 1655**, **I64 3**. The three I64 tensors are the PLE
  index tensors `…ple.ple_embedding.layer_multipliers` (3,),
  `…ngram_heads_offsets` (16,), `…ngram_heads_vocab_sizes` (16,). No quantized tensors.

## Key naming caveats

- **Nesting is `model.language_model.layers.<N>.…`** — not `model.layers.<N>`. The
  capture's `layer_of()` mirrors `exl3pack.geometry._layer_index` and scans for any
  `layers.<N>` segment, so this and `mtp.layers.<N>` both resolve.
- **Routed experts are fused stacked tensors**, one key per module per layer —
  `…mlp.experts.down_proj` (512, 2560, 640) and `…mlp.experts.gate_up_proj`
  (512, 1280, 2560). There is **no per-expert index** (`experts.<N>.`), so the
  expert→shard map counts expert *tensors* per shard rather than parsing expert
  indices.
- **The MTP block is itself MoE** — `mtp.layers.0.mlp.experts.{down_proj,
  gate_up_proj}`, one layer, so 48 body MoE layers + 1 MTP layer. Hence
  49 `gate_up_proj` + 49 `down_proj` = **98** expert tensors over 48 body
  layers (`expert_layer_total` 48).
- **PLE n-gram embedding is 128-way split at layer 2** —
  `…ple.ple_embedding.ngram_embedding.shard_<N>.weight`, `<N>` 0…127. The capture's
  numeric→`<N>` template rewrite collapses these into one template (count 128).
- Router `…mlp.gate.weight` (512, 2560); shared expert
  `…mlp.shared_expert.{down,gate,up}_proj.weight` and
  `…mlp.shared_expert_gate.weight` (1, 2560).

## `num_slots` 20 — unverified, confirm first

`moe_intermediate_size` 640 → `num_slots` **20** (640 % 32 == 0, `slot_channels` 32).
The served path (`--quantization modelopt_mixed --load-format b12x`) is noted here only.
**Only 32-slot positions have been exercised** in b12x `exl3-v1` acceptance to date —
a 20-slot position is untested. Confirm before any pack acceptance run.

## Serving context

Consumed by `recipes/qwen3.8-flash-next-nvfp4-solo.yaml` /
`recipes/qwen3.8-flash-next-nvfp4-cluster.yaml`, which serve the NVFP4 variant
(`local-inference-lab/Qwen3.8-Flash-Next-NVFP4`) and are sourced from z-lab upstream.

## All on `/nas-1`

`/nas-1/models/qwen3.8/` also holds `qwen3.8-27b-coder390-ethink-dflash2` (captured
separately), four `qwen3.8-27b-nvfp4-*` variants, and `qwen-agentworld-35b-a3b-nvfp4-w4a16`.
None of those are Flash-Next variants.
