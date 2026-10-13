# P2P host-staging port: pooled-view KV geometry fix

**Scope:** `mods/mooncake-worker` `vendor/p2p/` (MooncakeConnector host-staged
KV path) + `pd-disagg-mimo-*` recipes on the 4-node GB10 cluster.
**Status:** root cause fixed and verified live (2026-10-13).

## Symptom

PD TP=2 boot failed at first cross-role contact with:

```
Mooncake KV region length mismatch for homogeneous TP at region 0:
local=737280, remote=1310720.
```

Both roles registered KV caches successfully and logged identical cache
layouts (`HND` + `BLHNC`), identical model, TP=2, fp8 KV with
`--kv-cache-dtype-skip-layers sliding_window`. Only the registered
`block_lens`/`kv_block_lens` differed:

| role | layers | registered `block_len` |
|---|---|---|
| prefill (ranks 0–1) | 48 | 737280 (full-attn) / 368640 (SWA) |
| decode (ranks 2–3) | 53 | 1310720 (uniform) |

## Root cause

`register_kv_caches` (`vendor/p2p/_worker_setup.py`) derived

```python
block_len = cache.stride(0) * cache.element_size()
```

and fell back to `kv_block_len = block_len` for non-MLA layers.
Under MiMo's **hybrid pooled KV layout** each per-layer KV tensor is a
strided **view** into a shared pool, so `stride(0)` is the *address pitch
between consecutive kernel blocks*, **not** the number of bytes in one
block. The pitch differs per role by construction (it depends on how many
groups share the pool row), so an equality check across roles can never
pass, and the per-block copy size (`kv_block_len`) would have been wrong
even if both sides coincidentally matched.

The per-block byte count is the KV-cache spec's `page_size_bytes`, which is
**identical across roles** (verified with a registration-time spec dump,
2026-10-13):

| layer family | spec | `page_size_bytes` | `block_size` | count |
|---|---|---|---|---|
| prefill & decode full-attention | `FullAttentionSpec` | 81920 | 128 | 9 |
| prefill & decode SWA | `SlidingWindowSpec` | 163840 | 64 | 39 |
| decode-only DFlash drafter | `SlidingWindowSpec` | 262144 | 128 | 5 |

Layer-count delta (decode 53 = 48 + 5) is the DFlash drafter KV — decode
layers `model.layers.48..52` (note the `model.` prefix vs the target model's
`language_model.model.` prefix). These need no exclusion logic: transfer
regions are aligned by layer name and iterated over the **producer's**
registered layers, so consumer-only drafter layers are simply never
requested (drafter KV is computed decode-local).

## Fix

`vendor/p2p/_worker_setup.py`: derive `kv_block_len` from the spec
(`layer_spec.page_size_bytes`), keeping `block_len` as the addressing pitch.

```python
page_size = getattr(layer_spec, "page_size_bytes", None)
block_bytes = block_len if page_size is None else page_size
if self.transfer_topo.virtually_split_kv_in_blocks and not isinstance(layer_spec, MambaSpec):
    kv_block_len = block_bytes // 2
else:
    kv_block_len = block_bytes
```

Addressing is unchanged and remains self-consistent: the host-staging window
carve (`_setup_host_staging`) and the H2D copy base (`_staging_h2d_copy_request`)
both use each side's own `block_len` pitch; only the copied length
(`kv_block_len`) and the cross-role equality check use the spec page size.

## Instrumentation used

Two permanent/incidental INFO logs in `vendor/p2p/_worker_setup.py`:

- `Mooncake register view layer=... shape=... stride=...` — the per-layer
  tensor view (`_log_debug_cache_registration`, promoted DEBUG→INFO).
- `Mooncake register spec layer=... spec=... page_size_bytes=... block_size=...`
  — the per-layer spec geometry (added during this diagnosis).

Both are low-volume (one line per KV layer at boot) and are what made the
pooled-view vs per-block divergence observable without a live debugger.

## Evidence (live, relaunch 9 → 10)

- relaunch 9: spec dump shows identical specs both roles; `block_len`
  diverges → `kv_block_len` mismatch reproduced.
- relaunch 10 (fix): both roles register matching `kv_block_len`; the
  region-length error is gone and transfer regions align.

## Related commits

- `af30b6c` vendor P2P host-staged package
- `0ea4876` P2P host-staged KV path for `mooncake` backend
- `7afe890` patch `TransferTopology.virtually_split_kv_in_blocks`
- `f12a13a` / `9a5dbf5` promote registration logs to INFO
- `56c1a02` dump per-layer spec geometry
- `8de1104` **fix: use spec page size for `kv_block_len`**
