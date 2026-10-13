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

`vendor/p2p/_worker_setup.py`: derive `kv_block_len` from the contiguous
per-block content; keep `block_len` as the addressing pitch.

```python
content = 1
for dim in cache.shape[1:]:
    content *= int(dim)
kv_block_len = content * cache.element_size()   # contiguous per-block bytes
if self.transfer_topo.virtually_split_kv_in_blocks and not isinstance(layer_spec, MambaSpec):
    kv_block_len = kv_block_len // 2
```

**Correction (2026-10-13, after the split test).** An intermediate revision
used `layer_spec.page_size_bytes`. That is the *padded* pool page, not the
block content: FullAttention pages equal the content (81920 B) but
SlidingWindow pages are padded to **2× content** (163840 vs 81920). Copying
`page_size_bytes` over-copied every SWA layer with padding, so the PD path
emitted garbage while prefill-direct stayed coherent. The authoritative
per-block copy length is the **contiguous content** `prod(shape[1:]) * elem`
(inner dims are contiguous in the strided pooled view: `(N,2,128,320)` fp8 →
81920 B; `(N,4,64,320)` → 81920 B), identical across roles and families.
`block_len = stride(0)*elem` remains the addressing pitch and legitimately
differs per role (prefill full/SWA 737280/368640; decode 1310720/655360).

**How it was caught:** serving prefill directly (`:8100`) with the same
weights answers coherently ("The capital of France is Paris.") while the PD
path (`:8000`) returns repeated-token garbage — proving the pack/weights are
fine and the corruption is in the KV transfer.

Addressing unchanged and remains self-consistent: host-staging window
carve (`_setup_host_staging`) and H2D copy base (`_staging_h2d_copy_request`)
both each side's own `block_len` pitch; only copied length
(`kv_block_len`) uses the contiguous content size.

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
- `4e7e9ef` **fix: allow consumer-only trailing KV groups**
- cluster-config `0013010` **fix: tear down profile-gated services on relaunch**

---

## Follow-on defect 1: consumer-only KV group (decode DFlash drafter)

With the geometry fixed, the live request advanced to
`KV group count mismatch: local=11, remote=12`. Prefill registers 11
KV-cache groups; decode registers 12 — the decode-only DFlash drafter adds
a trailing group. `metadata.reqs_to_send` (producer) and `req_blocks`
(consumer) are group-indexed vectors, and the sender required exact count
equality before zipping them, so any consumer-only group aborted every
transfer.

**Fix** (`_worker_send.py`): reject only a *producer* group the consumer
lacks (`len(local) > len(remote)`); extra trailing consumer groups are
dropped by the positional zip. Safe because the consumer-only group is
computed decode-local and never transferred, and `_align_transfer_regions`
still fail-closes on any real per-group divergence for shared layers
(it asserts aligned regions share a `group_index`).

## Follow-on defect 2: stale vllm-router Mooncake engine_id

The router injects the prefill `engine_id` into each decode leg, read once
from `GET {prefill}:8998/query` at the **router's** boot. `teardown_stack`
ran `docker compose down` without profiles, so the profile-gated
`vllm-router` survived every relaunch and kept injecting the engine_id from
the previous engine generation. Decode then logged, on every request:

```
ERROR _worker_receive.py:286 Failed to find remote engine_id
0274aaae-... from bootstrap server http://<prefill>:8998
```

while the live bootstrap server advertised the current id. Decode hung
until the router was restarted.

**Fix** (cluster-config `spark-serve.sh`): name the profiles in the teardown
(`--profile rank0 --profile router down --remove-orphans`) and fall back to
an explicit `docker rm -f ... vllm-router`, so the router is recreated each
relaunch and re-queries the live bootstrap server. Verified: after the fix
the router's container age resets with the stack and adopts the live
engine_id automatically.
