# node-model-map — MiMo checkpoints on cluster-node local NVMe

Machine-readable form: `node-model-map.json` (loaded by `mods/exl3-pack/paths.py`
via `load_node_map` / `resolve_source`). This file is the human companion.
Probed live 2026-10-07 from each node via the fulton-remote-shell MCP.

Purpose: the pack builder reads its **source checkpoint from a node's local
NVMe** instead of the NAS, to avoid moving 166–535 GB across the NAS link.
`local_root` is the base under which local checkpoints live (currently
`/opt/llm`). `role: source` = unquantized serving checkpoint usable as a pack
input; `role: pack` = an existing EXL3 pack. Only **complete** checkpoints are
listed as resolvable entries.

## Human summary

| Node | Alias | Source checkpoints (local NVMe) | Packs |
|---|---|---|---|
| node3 `gx10-becc` | `home-gx10-node3` | **flash-rl-uncensored** (65 shards), flash-rl | flash-rl-exl3-v1 |
| node4 `gx10-9273` | `home-gx10-node4` | **pro-rl-uncensored** (130 shards), flash-rl | flash-rl-exl3-v1 |
| node1 `gx10-cb11` | `home-gx10-node1` | flash-rl | flash-rl-exl3-v1 (+ exl3-v2) |
| node2 `gx10-f1d8` | `home-gx10-node2` | flash-rl | flash-rl-exl3-v1 (+ exl3-v2) |

The two **new** sources to pack (`flash-rl-uncensored`, `pro-rl-uncensored`)
each live on exactly one node: **node3** and **node4** respectively.

## Per-node paths

**node3 `gx10-becc`** (`local_root` `/opt/llm`)
- source `mimo-v2.6-flash-rl-uncensored` → `/opt/llm/staging/mimo-v2.6-flash-rl-uncensored` (65 shards, mxfp4, tp4)
- source `mimo-v2.6-flash-rl` → `/opt/llm/models/mimo-v2.6-flash-rl`
- pack `mimo-v2.6-flash-rl-exl3-v1` → `/opt/llm/models/mimo-v2.6-flash-rl-exl3-v1`

**node4 `gx10-9273`** (`local_root` `/opt/llm`)
- source `mimo-v2.6-pro-rl-uncensored` → `/opt/llm/staging/mimo-v2.6-pro-rl-uncensored` (130 shards, mxfp4, tp8)
- source `mimo-v2.6-flash-rl` → `/opt/llm/models/mimo-v2.6-flash-rl`
- pack `mimo-v2.6-flash-rl-exl3-v1` → `/opt/llm/models/mimo-v2.6-flash-rl-exl3-v1`

**node1 `gx10-cb11` / node2 `gx10-f1d8`** (`local_root` `/opt/llm`)
- source `mimo-v2.6-flash-rl` → `/opt/llm/models/mimo-v2.6-flash-rl`
- pack `mimo-v2.6-flash-rl-exl3-v1` → `/opt/llm/models/mimo-v2.6-flash-rl-exl3-v1`
- (node1 also has a **2.3 GB partial** `mimo-v2.6-flash-rl-uncensored` under
  `/opt/llm/staging/` — excluded from the map as incomplete; node2 also has
  `mimo-v2.6-flash-rl-exl3-v2`.)

## Not present anywhere local

No `pro-rl-*` checkpoint exists on node1/node2, nor beyond node4's copy (and its
`/nas-1` twin).

## NAS twins (fallback / copy-back)

The same checkpoints also exist under `/nas-1/models/mimo/` (mounted on every
node). The builder prefers the local NVMe copy and falls back to `/nas-1` when
no local entry is present. Final pack artifacts are written back under
`/nas-1/models/mimo/<slug>-exl3-v1` for the FES weight-staging path.
