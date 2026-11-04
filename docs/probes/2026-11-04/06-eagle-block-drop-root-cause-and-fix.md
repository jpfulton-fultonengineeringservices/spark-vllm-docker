# Root cause: DFlash treated as EAGLE-family for the prefix-cache block drop

Status: **root-caused, fixed, and deployed.** The `ductduct` reasoning loop is a
**server-side KV/SSM state corruption** caused by the served vLLM build
predating upstream PR #54163: DFlash is classified as EAGLE-family, so every
prefix-cache hit drops the trailing matched block and forces a boundary
recompute — which, under `--mamba-cache-mode align`, corrupts the GDN SSM
resume boundary for sequences admitted against a live sibling's speculative
tail. The fix is a one-field recipe change
(`"disable_eagle_block_drop": true`), already supported by the installed
build. Deployed as `78f7b8c132` and verified clean — see
[07-post-fix-verification.md](07-post-fix-verification.md).

Applies to image `vllm-node-b12x` as pinned by the recipe, engine
`vllm 0.32.0rc1.dev301+g1ac942555.d20261016` (commit `1ac942555`, cut
2026-10-16, `eugr/spark-vllm:nightly-20261016` lineage plus installed-stage
patches), recipe `qwen3.8-27b-coder390-4x-store-l2` (TP=4, durable Mooncake
store L2).

## 1. Why the replay probes were clean

Probes [01](01-onset-probe.md), [03](03-gateway-replay.md),
[04](04-concurrent-replay.md), and [05](05-long-gen-replay.md) replayed the
genuine failing conversation — direct to the engine, through the gateway,
8-way concurrent, and at an 8192-token cap — and were clean in every run.
[02](02-gateway-sse-delta-check.md) exonerated the gateway's SSE framing.

That is the expected result for this defect: the trigger is not prompt
content but **live engine state at admission time**. A replayed prompt on a
quiescent engine has no sibling decoding, so the corrupting geometry cannot
arise. The probes' value is negative evidence: they eliminate prompt shape,
sampling regime, gateway framing, concurrency *per se*, and generation depth,
leaving engine-side prefix-cache/SSM state as the only surviving class of
explanation.

## 2. The defect

Upstream fix **PR #54163** (commit `0bfeb8d0`, merged 2026-10-23) — *"DFlash/
DSpark no longer drop the last prefix-cache block (mamba align context
recompute)"* — is **absent** from the served build, which was cut 2026-10-16.

Verified **live in the running container** (`vllm_node` on `gx10-node1`):

| Check | Live result |
|---|---|
| `grep needs_eagle_block_drop` in the installed tree | **0 matches** (fix absent) |
| `speculative.py:1982-1990` | `use_eagle()` returns True for method `"dflash"`; `use_eagle_block_drop()` = `use_eagle() and not disable_eagle_block_drop` |
| `speculative.py:450` | `disable_eagle_block_drop: bool = False` — **field exists in the installed build** |
| `scheduler.py:319` | consumes `speculative_config.use_eagle_block_drop()` |
| `scheduler.py:343` | passes it as `use_eagle` into the KV cache manager |
| `kv_cache_coordinator.py:867` | `drop_eagle_block = use_eagle and idx not in eagle_verified` |
| `kv_cache_coordinator.py:875` | mamba groups carved out of the drop — but the **attention** hit boundary is still shortened |

The mamba carve-out at `:875` is why this looks safe on inspection. It is
not: shortening the **attention** hit boundary is what shifts the resume
point that the mamba-align SSM state is keyed to.

## 3. Mechanism

1. Every prefix-cache hit unnecessarily **drops the trailing matched block**
   and forces a boundary recompute. Live hit rates: **83.9% GPU**,
   **24.9% Mooncake** — so this path runs on the overwhelming majority of
   agent requests.
2. Under `--mamba-cache-mode align`, the resume boundary is exactly where the
   48-layer GDN SSM state checkpoint lives.
3. A sequence admitted after a long waiting-queue stay — observed
   **ttft = 97 s** under `max_num_seqs=8` saturation — whose hit boundary
   lands on a **still-decoding sibling's volatile trailing block** gets
   *valid attention KV* but a **mismatched SSM state**.
4. The recurrent channel then contradicts attention from token 0. The model
   is **born looping**: an exact 4-character cycle (`ductduct…`).
5. Retries re-present the same prefix → same cached blocks + same drop
   geometry → **deterministic**. Retry storms of 8–12+ observed,
   `schedulerToken` up to 33.
6. A sibling whose state was honestly computed stays coherent — matching the
   user's two-simultaneous-sessions observation (one session loops, the other
   is fine).

## 4. Client-side signature mechanics

The pi-ai guard (`thinking-loop.ts`) scans every
`EXACT_CHECK_STRIDE = 128` characters. Its short-unit rule (unit ≤ 60 chars)
requires ≥ 4 repetitions and ≥ 180 characters. For a 4-character cycle the
**first checkpoint past the threshold is 256 characters = 64 repetitions**.
Hence every cluster-route fire reads *"exact 4-character cycle 64×
back-to-back"*, and LiteLLM bills `completion_tokens = 64`.

This is why the client-side signature is always 64: it is the guard's
sampling grid, not a property of the model's degeneration.

## 5. Server-side billing proof

`SpendLogs`, last 4 days, cluster model, `completion_tokens ≤ 512`:

| completion_tokens | rows |
|---|---|
| 0 | 1570 |
| **64** | **703** |
| **65** | **231** |
| 146 | 24 |

The 64/65 spike is **934 looped turns** — a distribution no healthy workload
produces (a real answer is hundreds of tokens; a guard abort is exactly one
checkpoint).

Storm rows, today 14:25–14:35 UTC — one retry ladder, all `completion_tokens`
64–65, all `status = success` (the guard abort is client-side; LiteLLM sees a
completed stream):

| # | prompt_tokens | completion_tokens |
|---|---|---|
| 1 | 77,438 | 64–65 |
| 2 | 77,577 | 64–65 |
| 3 | 77,716 | 64–65 |
| 4 | 77,855 | 64–65 |
| 5 | 77,994 | 64–65 |
| 6 | 78,133 | 64–65 |
| 7 | 78,272 | 64–65 |
| 8 | 78,411 | 64–65 |
| 9 | 78,550 | 64–65 |
| 10 | 78,689 | 64–65 |

Prompt growth is ~139 tokens per retry — the accumulating failed turn is
appended to the conversation each time, which is also why the prefix keeps
hitting the same cached blocks and the failure stays deterministic.

## 6. Ranked suspects

| # | Suspect | Verdict |
|---|---|---|
| S1 | **DFlash classified as EAGLE-family → trailing block drop → mamba-align SSM boundary corruption** (this document) | **PRIME — confirmed** |
| S2 | `mamba-spec-relocate-guard` silently substituting state | alternative, **unconfirmed** |
| S3 | Upstream #62841 prompt-tail boundary re-mark | **absent from build**; companion fix |
| S4 | #58668 spec-token `num_computed_tokens` miscount | borderline inclusion |
| S5 | Mooncake store path | **largely refuted** — lineage `cache_prefix` present, zero transfer errors |
| S6 | DFlash drafter desync | **low** — costs acceptance, not correctness |
| S7 | `--async-scheduling` | **amplifier, not mechanism** — zero preemptions in 6 h |

## 7. The fix

Add one field to the recipe's `--speculative-config` JSON
(`recipes/qwen3.8-27b-coder390-4x-store-l2.yaml`):

```json
{"method":"dflash","model":"/workspace/Qwen3.8-27B-Coder390-dflash",
 "num_speculative_tokens":{num_speculative_tokens},
 "draft_tensor_parallel_size":{draft_tensor_parallel_size},
 "disable_eagle_block_drop":true}
```

Why this is correct and sufficient:

- The field **already exists** at `speculative.py:450` in the installed
  build, so it is accepted in `--speculative-config` today — **no code patch
  needed**.
- It is **exactly equivalent to #54163 for DFlash**: the drop was never
  needed, because DFlash builds draft KV from the target hidden state alone.
- **Zero acceptance-rate impact** — the drop only ever affected the
  prefix-cache hit boundary, not draft quality.

Follow-up at the next image rebuild: cherry-pick `0bfeb8d0` (#54163) plus
companions `d561af5d` (#62841) and `56e993a6` (#65319, *"Start capped async
KV loads while the batch is full"* — the 97 s-queue regime).

## 8. Exposure amplifier

`max_num_seqs` was raised **6 → 8** the same day the incident escalated
(recipe comment dated 2026-11-04). More concurrent sequences means more
siblings holding live speculative tails, more waiting-queue admissions, and
therefore a higher probability that an admitted sequence's hit boundary lands
on a volatile trailing block. The defect predates the change; the change
raised its hit rate.

## 9. Live-serve corroboration

| Observation | Reading |
|---|---|
| Mooncake KV transfer metrics error-free throughout (`lookup`/`load`/`save` `failed_keys = 0`) | store path not implicated (S5) |
| `load_get` p90 up to **778 ms** (host-staging, `staging_num_slots=2`) | long admission stalls → the 97 s ttft regime |
| **12:46:35** Triton JIT of `_prepare_dflash_inputs_kernel` mid-inference | concurrency-dependent draft shape absent from warmup — first-hit latency spike, not correctness |
| `B12X_SKIP_RECLAIM=1` | job-end reclamation sweeps disabled; keeps blocks resident longer, widening the live-tail window |

## 10. Verification and open items

1. **Relaunch — done.** A recipe change does not take effect on
   `docker restart` (the vLLM argv is baked at container create time), so the
   4-node TP=4 serve was recreated via the staged launcher; the flag is live
   on all 4 ranks and `/v1/models` returned 200 at 2026-11-04T15:00:08Z. See
   [07-post-fix-verification.md](07-post-fix-verification.md) §2.
2. **Post-fix acceptance check — done, clean.** The storm-shaped workload
   (mixed prompts sharing long prefixes, concurrent admission) produced zero
   cycles, and `SpendLogs` shows zero `completion_tokens ∈ {64,65}` rows
   since relaunch. See
   [07-post-fix-verification.md](07-post-fix-verification.md) §3.
3. **Image rebuild follow-up** (§7) so the fix survives without the recipe
   field.
4. **S2 remains open** — `mamba-spec-relocate-guard` state substitution was
   not disproven, only deprioritised; if loops persist after the fix, it is
   the next suspect.
