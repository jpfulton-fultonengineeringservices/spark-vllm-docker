# Post-fix verification: coder390 DFlash block-drop fix deployed and clean

Status: **DEPLOYED and verified.** The config-level equivalent of upstream
PR #54163 (`disable_eagle_block_drop: true`) is live on all 4 ranks of the
coder390 serve. Post-fix: **zero** exact-cycle events in every verification
arm, and **zero** `completion_tokens ∈ {64,65}` rows in SpendLogs since
relaunch — the defect's billing signature (703 rows at exactly 64 in the 4
days prior) is gone.

Root cause: [06-eagle-block-drop-root-cause-and-fix.md](06-eagle-block-drop-root-cause-and-fix.md).

## 1. The change

Recipe `recipes/qwen3.8-27b-coder390-4x-store-l2.yaml`: added
`"disable_eagle_block_drop":true` to the `--speculative-config` JSON.

Commit lineage:

| Commit | What |
|---|---|
| `2cc05233f1` | First fix commit — comment block misplaced between shell line-continuations, truncating the rendered command. **Caught by `run-recipe.py --dry-run` before deploy.** |
| `78f7b8c132` (deployed SHA `78f7b8c132d36e76f10d5831d1251ae43d99761a`) | Comment relocated above `command:`; the fix that went live. |

This is the config-level equivalent of upstream vLLM PR #54163 (`0bfeb8d0`)
for DFlash, per the root-cause analysis in doc 06: the trailing-block drop
was never needed (DFlash builds draft KV from target hidden state alone), so
disabling it restores correct mamba-align SSM resume boundaries with zero
acceptance-rate impact.

## 2. Relaunch

Canonical launcher on node1, coordinated 4-node staged bring-up:

```bash
spark-vllm-docker-serve.sh launch --mode staged --daemon --force \
  --recipe qwen3.8-27b-coder390-4x-store-l2
```

- **Log routing preserved**: `exec-script.sh >> /proc/1/fd/1` →
  `docker logs vllm_node` (see `b12x-serve-log-routing-and-orphan-risk`).
- Healthy `/v1/models` 200 at **2026-11-04T15:00:08Z**.
- Live cmdline on **all 4 ranks** shows `disable_eagle_block_drop":true`.
- Mooncake `cache_prefix` lineage re-derived `…-9ac7bf65eeef` →
  `…-962bbc315095` (fresh L2 namespace — the spec-config change alters the
  cache identity, as designed; **all other args byte-identical**).

## 3. Verification arms

### (a) Direct 8-way mixed storm

Rank-0 `:8000`, 15:00:17–15:02:13Z. 4× `replay_msgs.json` (the genuine
18.4k-token failing conversation) + 4× math-reasoning prompts, cap 1024,
streaming, cycle detector at the pi-ai exact thresholds.

Verbatim rows:

```
replay run=0 finish=stop reason_chars=916 content_chars=145 onset=None cycle=None
replay run=1 finish=stop reason_chars=261 content_chars=223 onset=None cycle=None
replay run=2 finish=stop reason_chars=2928 content_chars=0 onset=None cycle=None
replay run=3 finish=stop reason_chars=3251 content_chars=0 onset=None cycle=None
math runs 4-7 finish=stop reason_chars 256-375 content_chars 141-221 onset=None cycle=None
```

| arm | runs | finish | onset | cycle |
|---|---|---|---|---|
| replay (genuine failing conversation) | 4/4 | stop | None | None |
| math reasoning | 4/4 | stop | None | None |

**8/8 clean** under the exact production trigger geometry — mixed prompts
sharing long prefixes, concurrent admission, streaming.

Note on replay runs 2–3 (`content_chars=0`, `finish=stop`): that is the
previously documented **budget-starvation** shape (the model closes thinking
at the cap and emits no visible content), *not* the loop defect — no cycle
was present in the reasoning channel, and generation stopped naturally.
Readers should not misread an empty content field as residual defect.

### (b) Gateway 4-way storm

Via dell LiteLLM `cluster/qwen3.8-27b-coder390` (the OMP production route),
replay conversation, cap 2048: **all 4 `finish=stop`, 1570 completion tokens
each** per SpendLogs.

**Caveat:** this gateway route does not forward reasoning deltas (SSE delta
keys are `content`/`role` only), so gateway-leg cycle detection was
**content-only**. The direct leg (arm a) carried the reasoning-channel check.
The gateway leg still verifies end-to-end request success, completion-token
sanity, and billing shape on the production path.

### (c) Engine health

| Metric | Post-fix value | Reading |
|---|---|---|
| Spec acceptance | ≈**1.62** accepted/draft-step (per-token rate 23.2%, +1 ≈ 2.62) | consistent with pre-fix 2.31–2.67 — **no acceptance-rate impact**, as predicted |
| Prefix-cache queries | 174,644, with 9,888 external (Mooncake) hits | accumulating under the new lineage — prefix caching healthy after re-derivation |

### (d) SpendLogs billing signature

`litellm-db`, since relaunch T0 **14:53:26Z**:

| Check | Result |
|---|---|
| rows with `completion_tokens ∈ {64,65}` | **ZERO** (defect's billing signature; 703 rows at exactly 64 in the 4 days prior) |
| failure rows ≤ 14:58:40 | 16, all **inside the boot window pre-health** (connection-refused class from real OMP traffic) — not the defect signature |

## 4. Anomalies during the operation (for operators)

1. **First fix commit broke boot.** `2cc05233f1` placed the comment block
   between shell line-continuations, truncating the rendered command. Caught
   by `run-recipe.py --dry-run` **before** deploy; fixed in `78f7b8c132` by
   relocating the comment above `command:`. Lesson: always dry-run-render a
   recipe edit that adds comments near the command chain.
2. **First relaunch attempt defaulted to the mimo recipe.** `launch` without
   `--recipe` uses `SPARK_VLLM_DOCKER_RECIPE`; the attempt was torn down and
   hard-failed the mimo weight gate, then immediately relaunched with
   explicit `--recipe`. **~5.5 min of user-facing 5xx** during the
   coordinated relaunch (16 logged failures, all pre-health — the same rows
   counted in §3(d)).
3. **L2 prefix pool starts cold** under the new lineage digest. Expected and
   correctness-first: the spec-config change invalidates the old cache
   identity; Mooncake hits re-accumulate from zero (§3(c)).

## 5. Conclusion

- **Residual cycles post-fix: NONE.** The born-looping defect is closed by
  the config-level #54163 equivalent.
- **Fallback ready:** the staged MTP variant
  (`qwen3.8-27b-coder390-4x-store-l2-mtp.yaml`, commit `60b04d6591`;
  publisher-validated vLLM drafter per HF discussion #8) remains available as
  a one-command cutover if any 64/65-token rows reappear.
- **Monitoring:** periodic SpendLogs query for `completion_tokens ∈ {64,65}`
  on the coder390 model. Any row is a regression signal; zero rows is the
  healthy steady state.
