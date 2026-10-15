# `model-structure/` — checkpoint source-structure captures

Pack-free structural captures of on-cluster checkpoints, in the convention established by
`mods/exl3-mimo/*-source/` and `mods/mimo-v2.6-flash-rl-nvfp4/nvfp4-source/`. Each
capture lets EXL3 pack/serve planning proceed without downloading multi-hundred-GB
weights: geometry, shard layout, full weight map, per-tensor inventory, and the
embedded drafter — **no weights are ever copied**. Reads are safetensors-header-only.

## Captures

| Capture | Checkpoint | Shape |
|---|---|---|
| [`qwen3.8-flash-next/`](qwen3.8-flash-next/source/README.md) | `/nas-1/models/qwen3.8/qwen3.8-flash-next` | BF16 MoE base, 131 shards / 1658 tensors |
| [`qwen3.8-27b-coder390-ethink-dflash2/`](qwen3.8-27b-coder390-ethink-dflash2/source/README.md) | `/nas-1/models/qwen3.8/qwen3.8-27b-coder390-ethink-dflash2` | BF16 dense + FP8 DFlash2 drafter, 3 shards / 1199 tensors |

## Generator

[`capture_source.py`](capture_source.py) — CLI `capture_source.py <checkpoint-dir> <out-dir>`.
Generalized over `mods/mimo-v2.6-flash-rl-nvfp4/nvfp4-source/capture_nvfp4.py` in four
data-driven ways (no hard-coded model names):

1. `layer_of()` scans for any `layers.<N>` segment, so `model.layers.<N>`,
   `model.language_model.layers.<N>`, `mtp.layers.<N>` and drafter-side `layers.<N>` all
   resolve — mirrors `exl3pack.geometry._layer_index`.
2. Geometry reads the `text_config` sub-object when present (`exl3pack.geometry._text_config`),
   which Qwen3.8/Qwen4 text-in-VLM checkpoints nest under the top level.
3. The expert→shard map counts expert *tensors* per shard instead of parsing a per-expert
   index (fused experts are single stacked keys, no `experts.<N>.` segment), and the map
   plus its `structure.txt` section are **omitted entirely for dense models** rather than
   emitting an empty one.
4. Drafter subdirectories are discovered generically — any immediate child dir holding a
   `*.safetensors`, skipping `.cache` and `@eaDir` — replacing the hard-coded
   `dflash/dflash_draft_model.safetensors` path.

Copies in each `source/` are the exact script that produced that capture.

## Regenerating

Run on any cluster node (NFS-mounted `/nas-1`):

```bash
python3 model-structure/capture_source.py <checkpoint-dir> <out-dir>
```

Reads only safetensors headers (8-byte length + JSON); never materializes tensors.

## `model-metadata/` is not committed

Each `source/` also carries a gitignored `model-metadata/` directory holding a derived
copy of the checkpoint's small text files (`config.json`, tokenizer/generation configs,
`chat_template.jinja`, `SHA256SUMS`, README/LICENSE). It is LFS-tracked and unpushable to
the public fork, so it stays untracked; the captures above are self-describing without it.
