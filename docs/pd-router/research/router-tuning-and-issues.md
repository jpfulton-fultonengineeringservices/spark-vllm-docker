# vllm-project/router — Performance Tuning & Open Issues Audit

**Date:** 2026-10-11 · **Scope:** research only (no code/config edits)
**Purpose:** Summarize router performance-tuning parameters and audit open/closed
issues relevant to our PD-disaggregation + Mooncake pivot.

**Router pin:** `vllm/vllm-router:v0.1.16`
(`infra/cluster-config/cluster.env:681-682`; `VLLM_ROUTER_IMAGE_NAME=vllm/vllm-router`,
`VLLM_ROUTER_IMAGE_TAG=v0.1.16`). Deployed as rank-0 sibling container by
`infra/cluster-config/docker/mooncake-store/compose.yaml:132-198` (`compose.yaml:143`).
Codekeeper-indexed revision of `vllm-project/router`: `91b65ee4bdfbe98863a32a2205dced5dbb249ffe`
(`gpu-cluster-forks/spark-vllm-docker/docs/pd-router/upstream-router-contracts.md:4`).

---

## 1. Performance Tuning

### 1.1 DeepWiki fetch — 43 retries exhausted

The DeepWiki page
`https://deepwiki.com/vllm-project/router/10.4-performance-tuning` was
attempted three times (2026-10-11) and returned **HTTP 429** on every attempt:

1. `read https://deepwiki.com/vllm-project/router/10.4-performance-tuning` →
   `failed: HTTP 429` (first call)
2. `read` (retry) → `failed: HTTP 429` (second call)
3. `xd://mcp__searxng_mcp_fetch_page` with the same URL →
   `[http_error] ... returned HTTP 429` (third call)

**Fallback applied:** all tuning parameters below are sourced from the
codekeeper-indexed router source revision `91b65ee4…` via the existing in-repo
contract map
`gpu-cluster-forks/spark-vllm-docker/docs/pd-router/upstream-router-contracts.md`,
which cites every parameter as `src/main.rs:<line>` or
`src/config/types.rs:<line>` from that revision. The `upstream-router-contracts.md`
file was read this session (lines 1–300 and 301–468) and cross-checked against the
live `compose.yaml:132-198` and `cluster.env:674-682`.

### 1.2 Tuning parameters (CLI flags, defaults, citations)

All defaults from `CliArgs` in `src/main.rs` of the codekeeper-indexed revision
(`upstream-router-contracts.md:93-178`).

#### Worker startup / health

| Flag | Default | Cite |
|---|---|---|
| `--worker-startup-timeout-secs` | `600` | `src/main.rs:155-156` |
| `--worker-startup-check-interval` | `30` | `src/main.rs:159-160` |
| `--health-failure-threshold` | `3` | `src/main.rs:339-340` |
| `--health-success-threshold` | `2` | `src/main.rs:343-344` |
| `--health-check-timeout-secs` | `5` | `src/main.rs:347-348` |
| `--health-check-interval-secs` | `60` | `src/main.rs:351-352` |
| `--health-check-endpoint` | `/health` | `src/main.rs:355-356` |

#### Request / concurrency / queue

| Flag | Default | Cite |
|---|---|---|
| `--request-timeout-secs` | `1800` | `src/main.rs:280-281` |
| `--max-concurrent-requests` | `32768` | `src/main.rs:284-285` |
| `--queue-size` (fixed default) | `100` | `src/main.rs:600-660` |
| `--queue-timeout-secs` (fixed default) | `60` | `src/main.rs:600-660` |
| `--rate-limit-tokens-per-second` (fixed default) | none (unset) | `src/main.rs:600-660` |
| `--max-payload-size` | `536870912` (512 MiB) | `src/main.rs:183-184` |

#### Retry / circuit breaker

| Flag | Default | Cite |
|---|---|---|
| `--retry-max-retries` | `5` | `src/main.rs:293-294` |
| `--retry-initial-backoff-ms` | `50` | `src/main.rs:297-298` |
| `--retry-max-backoff-ms` | `30000` | `src/main.rs:301-302` |
| `--retry-backoff-multiplier` | `1.5` | `src/main.rs:305-306` |
| `--retry-jitter-factor` | `0.2` | `src/main.rs:309-310` |
| `--disable-retries` | `false` | `src/main.rs:313-314` |
| `--cb-failure-threshold` | `10` | `src/main.rs:318-319` |
| `--cb-success-threshold` | `3` | `src/main.rs:322-323` |
| `--cb-timeout-duration-secs` | `60` | `src/main.rs:326-327` |
| `--cb-window-duration-secs` | `120` | `src/main.rs:330-331` |
| `--disable-circuit-breaker` | `false` | `src/main.rs:334-335` |

#### Load-balancing (cache_aware policy)

| Flag | Default | Cite |
|---|---|---|
| `--cache-threshold` | `0.3` | `src/main.rs:163-164` |
| `--balance-abs-threshold` | `64` | `src/main.rs:167-168` |
| `--balance-rel-threshold` | `1.5` | `src/main.rs:171-172` |
| `--eviction-interval` | `120` | `src/main.rs:175-176` |
| `--max-tree-size` | `67108864` (2^26) | `src/main.rs:179-180` |

#### PD-specific

| Flag | Default | Cite |
|---|---|---|
| `--vllm-pd-disaggregation` | `false` | `src/main.rs:134-135` |
| `--vllm-discovery-address` | none | `src/main.rs:139-140` |
| `--prefill` (repeatable) | none | `src/main.rs:143-144` |
| `--decode` (repeatable) | none | `src/main.rs:145-146` |
| `--prefill-policy` | none | `src/main.rs:147-148` |
| `--decode-policy` | none | `src/main.rs:151-152` |
| `--kv-connector` | `nixl` (enum: `nixl`/`mooncake`/`moriio`) | `src/config/types.rs:536-547`; `src/main.rs:372-373` |
| `--intra-node-data-parallel-size` | `1` | `src/main.rs:200-201` |
| `--profile` | none | `src/main.rs:368-369` |

`enum KvConnector` (`src/config/types.rs:536-547`): `nixl` (default) / `mooncake`
/ `moriio`. The `mooncake` variant queries each prefill node's Mooncake bootstrap
server (port 8998) at startup, generates a per-request `transfer_id`, and injects
Mooncake-specific `kv_transfer_params` (`transfer_id`, `remote_bootstrap_addr`,
`remote_engine_id`) into both prefill and decode requests
(`upstream-router-contracts.md:310-317`; PR #151 body).

### 1.3 Our deployed values

From `infra/cluster-config/docker/mooncake-store/compose.yaml:159-177` (read this
session):

```yaml
- --vllm-pd-disaggregation
- --kv-connector
- nixl                          # ← compose currently pins nixl, NOT mooncake
- --prefill
- http://${MOONCAKE_HOST_IP}:${PD_PREFILL_PORT:-8100}
- --decode
- http://${PD_ROUTER_DECODE_IP}:${PD_DECODE_PORT:-8200}
- --worker-startup-timeout-secs
- "1800"                        # 3× default 600 — EXL3 pack loads take 10–20 min
- --worker-startup-check-interval
- "30"                          # default
- --prometheus-host
- "0.0.0.0"                     # overridden: default 127.0.0.1 hides :29000 from VM scrape
- --prometheus-port
- ${PD_ROUTER_METRICS_PORT:-29000}
```

Cross-referenced in `cutover-vllm-router.md:38-46`:
`--host 0.0.0.0 --port ${PD_ROUTER_PORT:-8000}`,
`--vllm-pd-disaggregation`, `--kv-connector nixl` (note: cutover doc line 39 says
`nixl`; the **pivot to `mooncake`** described in the task header has NOT yet been
applied to `compose.yaml` — the file still reads `nixl` at `compose.yaml:161`).

**Key gap:** the compose currently uses `--kv-connector nixl`, which means the
router injects NIXL-style `kv_transfer_params` (no `transfer_id`,
`remote_bootstrap_addr`, or `remote_engine_id`). The engines, however, are
configured for `MooncakeStoreConnector` (store-only) by `dispatch.sh`. The router
is KV-backend-agnostic and does not participate in the KV handshake
(`cutover-vllm-router.md:47-51`), so the `--kv-connector` flag only controls what
metadata the router injects into the request body — it does NOT change which
connector the engines use. The intended pivot to `--kv-connector mooncake` would
make the router query the prefill's bootstrap server (port 8998) and inject
P2P `MooncakeConnector` parameters, which is the upstream `vllm-project/router`
mode the task targets.

### 1.4 Health-gate interaction

`compose.yaml:188-193` healthcheck uses a `/proc` walk for process liveness
(`compose.yaml:189`), `interval: 15s`, `timeout: 10s`, `retries: 6`,
`start_period: 15s`. The router starts listening on `:8000` only after
`VllmPDRouter::new` completes its `--worker-startup-timeout-secs 1800` wait
(`compose.yaml:178-180`). `serve.sh` then gates on real `/v1/models` readiness
after the compose `health` verb reports up
(`infra/cluster-config/docs/serve-lifecycle-integration.md:410-412,504-505`).

---

## 2. Issues & PRs Audit — `vllm-project/router`

Searched the GitHub Issues API
(`https://api.github.com/search/issues?q=repo:vllm-project/router+KEYWORD`)
for each keyword: `mooncake`, `pd`, `prefill`, `decode`, `bootstrap`,
`kv_transfer_params`, `hang`, `timeout`, `stuck`. Results below are the
PD/Mooncake/KV-transfer-relevant subset. `stuck` returned 0 results.

### 2.1 Mooncake / bootstrap / kv_transfer_params

| # | Title | State | URL | Relevance |
|---|---|---|---|---|
| **#209** | fix: bootstrap dynamically discovered Mooncake prefill workers | **open** | https://github.com/vllm-project/router/pull/209 | Mooncake bootstrap metadata initialized only for static `--prefill` URLs at `VllmPDRouter` startup; workers added via K8s discovery get no bootstrap query → "No Mooncake bootstrap info for prefill … decode will proceed without kv_transfer_params". Directly relevant to `--kv-connector mooncake` pivot. |
| #151 | feat: add Mooncake connector support PD disaggregation | closed | https://github.com/vllm-project/router/pull/151 | Original PR that added `--kv-connector mooncake`: queries bootstrap server (port 8998), generates `transfer_id`, injects `kv_transfer_params` (transfer_id, remote_bootstrap_addr, remote_engine_id). Shipped in v0.1.16. |
| #338 | Send transfer_id to prefill and decode for the NIXL connector | open | https://github.com/vllm-project/router/pull/338 | NIXL transfer_id propagation (not Mooncake, but same `kv_transfer_params` injection path). |
| #130 | feat: add KV events-aware precise cache routing for P/D disaggregation | open | https://github.com/vllm-project/router/pull/130 | KV-event-aware routing for PD. |
| #229 | [Bugfix]: Cached_tokens always returns -1 | open | https://github.com/vllm-project/router/pull/229 | PD cached_tokens accounting bug. |
| #317 | [RFC]: Preserve Cached-Token Usage and Upstream HTTP Errors in vLLM PD Routing | open | https://github.com/vllm-project/router/issues/317 | RFC on preserving prefill cache usage in PD responses. |
| #255 | [Bug]: re-registering worker URL leaves stale registry indexes | open | https://github.com/vllm-project/router/issues/255 | Re-registering same URL with different model/type leaves stale secondary-index entries → `get_all()` returns wrong workers. Relevant if workers re-register mid-run. |
| #326 | [Feature] Add multi-tier KV cache placement index from engine events | closed | https://github.com/vllm-project/router/pull/326 | KV-block placement index (merged into #320). |

### 2.2 Prefill / decode / PD dispatch

| # | Title | State | URL | Relevance |
|---|---|---|---|---|
| **#360** | [Bugfix] Prevent premature stop in prefill-only requests | **open** | https://github.com/vllm-project/router/pull/360 | Router sets `max_tokens=1` on prefill but `min_tokens` could remain 0; if first token is EOS/stop, prefill finishes `FINISHED_STOPPED` instead of `FINISHED_LENGTH_CAPPED` → no KV handoff parameters → decode recomputes locally. Created 2026-10-10 (yesterday). Very relevant to PD prefill KV transfer. |
| #207 | feat: concurrent P/D dispatch for NIXL (decode prepares while prefill runs) | open | https://github.com/vllm-project/router/pull/207 | Concurrent dispatch (NIXL only, `VLLM_PD_CONCURRENT_DISPATCH=1`); companion to vllm-project/vllm#51919. Reduces serial two-stage TTFT. |
| #184 | Merge routed_experts across prefill/decode in P/D disaggregation | open | https://github.com/vllm-project/router/pull/184 | Expert-parallel PD. |
| #337 | feat: configure prefill and decode data parallel sizes independently | open | https://github.com/vllm-project/router/pull/337 | Separate DP sizes for P and D. |
| #345 | [Bugfix] Reconcile prefill cache usage and preserve Decode errors in vLLM PD routing | open | https://github.com/vllm-project/router/pull/345 | Fixes cache accounting returning decode's cache as prefill's; preserves decode error bodies instead of turning 429 into router 500. |
| #311 | Fix persistent worker state for discovered P/D routing (#299) | open | https://github.com/vllm-project/router/pull/311 | Persistent prefill/decode worker state across routing selections; reconcile registration lifetimes. |
| #313 | fix(vllm-pd): per-DP-rank prefill expansion in service-discovery mode | open | https://github.com/vllm-project/router/pull/313 | Per-DP-rank prefill expansion in discovery mode. |
| #210 | fix: address DP-aware workers by global DP rank | open | https://github.com/vllm-project/router/pull/210 | DP-aware worker addressing. |
| #263 | [Bug]: PD logprobs use byte offsets for non-ASCII prompt tokens | open | https://github.com/vllm-project/router/issues/263 | PD logprobs byte-offset bug. |
| #193 | [Bug]: cached_tokens = prompt_tokens in prefill-decode disagg with stream=true | open | https://github.com/vllm-project/router/issues/193 | Cache accounting bug in streaming PD. |
| #222 | [Feature]: Support different DP sizes for prefill and decode | open | https://github.com/vllm-project/router/issues/222 | RFC for independent P/D DP sizes. |
| #296 | [RFC] E/P/D follow-up: contract alignment, static Encode pool, and staged productionization | open | https://github.com/vllm-project/router/issues/296 | E/P/D productionization RFC (builds on #266). |
| #266 | [Feature] Add EPD routing with JSON metadata | open | https://github.com/vllm-project/router/pull/266 | EPD routing support. |

### 2.3 Timeout / queue / hang

| # | Title | State | URL | Relevance |
|---|---|---|---|---|
| **#247** | [Bug]: queued token acquisition expires before configured timeout | **open** | https://github.com/vllm-project/router/issues/247 | `TokenBucket::acquire_timeout` wraps `acquire` with a second (shorter) timeout → `QueueProcessor` can HTTP 408 despite enough time remaining. Directly relevant to queue tuning. |
| #248 | [Bugfix] Respect explicit token acquisition timeout | open | https://github.com/vllm-project/router/pull/248 | Fix for #247: honors caller's explicit acquisition deadline under contention. |
| #171 | feat: export queue metrics and make queue size/timeout configurable | open | https://github.com/vllm-project/router/pull/171 | Exposes `--queue-size`, `--queue-timeout-secs`, `--rate-limit-tokens-per-second` as CLI flags (currently fixed defaults at `src/main.rs:600-660`). |
| #185 | Expose --queue-size, --queue-timeout-secs, and --rate-limit-tokens-per-second CLI flags | open | https://github.com/vllm-project/router/pull/185 | Same as #171 — makes queue params configurable. |
| #146 | fix: add timeout to streaming load tracking to prevent phantom requests | closed | https://github.com/vllm-project/router/pull/146 | Adds 300s per-chunk timeout to streaming load tracking; prevents phantom requests when vLLM hangs mid-stream (e.g. GPU error). Shipped in v0.1.16. |
| #114 | feat: support stream response in process_vllm_two_stage_request_discovered | closed | https://github.com/vllm-project/router/pull/114 | Streaming support for discovered PD. Shipped in v0.1.16. |

### 2.4 Other relevant

| # | Title | State | URL | Relevance |
|---|---|---|---|---|
| #174 | [Bug]: vllm-router power_of_two policy queries /get_load but vLLM doesn't expose it | open | https://github.com/vllm-project/router/issues/174 | power_of_two policy breaks if vLLM doesn't expose `/get_load`. |
| #175 | fix: align load endpoint with vLLM API (/load + server_load) | open | https://github.com/vllm-project/router/pull/175 | Aligns load endpoint with vLLM's `/load`. |
| #178 | fix(power_of_two): combine cached load with local counter to avoid thundering herd | open | https://github.com/vllm-project/router/pull/178 | Thundering-herd mitigation for power_of_two. |
| #159 | [Bug]: `--api-key` not applied to worker health checks | open | https://github.com/vllm-project/router/issues/159 | API key not forwarded to worker health checks. |

---

## 3. Most Relevant Open Issue

### Flagged: **#209** — fix: bootstrap dynamically discovered Mooncake prefill workers

**URL:** https://github.com/vllm-project/router/pull/209
**State:** open (created 2026-08-13, last updated 2026-09-14)
**Labels:** `bug`

**Why it's the single most relevant open issue for our pivot:**

Our task is pivoting from the store-only `MooncakeStoreConnector` path (which
wedges the prefill engine — see
`gpu-cluster-forks/spark-vllm-docker/docs/pd-router/pd-failure-evidence.md`
§1.2–§2.2) to the upstream `vllm-project/router` `--kv-connector mooncake` mode,
which uses the **P2P `MooncakeConnector`** with a bootstrap-server handshake.
PR #209 is the **only open issue** in `vllm-project/router` that directly
addresses the Mooncake bootstrap lifecycle in PD disaggregation.

**What #209 fixes (from PR body, fetched via GitHub API this session):**

> Mooncake bootstrap metadata [is] currently initialized only [when] static
> `--prefill` URLs [are] present when `VllmPDRouter` starts. Prefill workers
> added later by Kubernetes service discovery [are] published to router without
> querying Mooncake bootstrap server. As result, decode requests selected through
> service discovery can log:
>
> ```text
> No Mooncake bootstrap info for prefill ...
> decode will proceed without kv_transfer_params
> ```

**Impact on our deployment:**

- Our current `compose.yaml:161` uses `--kv-connector nixl` (not `mooncake`),
  so #209 does **not** affect us today.
- If we switch to `--kv-connector mooncake` **with static `--prefill` URLs**
  (our current mode — `compose.yaml:162-163`), the bootstrap query runs at
  `VllmPDRouter::new` startup for each static URL, so #209's bug (dynamic
  discovery only) would not fire. This is the path #151 (closed, shipped in
  v0.1.16) implements.
- If we later adopt Kubernetes service discovery
  (`--service-discovery` + `--prefill-selector`/`--decode-selector`,
  `upstream-router-contracts.md:325`), #209's fix becomes **required** — without
  it, dynamically discovered prefill workers would never get their bootstrap
  metadata queried, and decode would "proceed without kv_transfer_params",
  silently skipping KV transfer.

**Version implication:** #209 is unmerged as of v0.1.16 (our pinned tag). Its
fix is on `main` but not in any released tag. If we adopt K8s discovery for the
Mooncake P2P path, we must either (a) wait for #209 to merge and release, (b)
pin to a `main` SHA that includes it, or (c) stay on static `--prefill` URLs
where the existing #151 path suffices.

### Secondary concern: **#360** — Prevent premature stop in prefill-only requests

**URL:** https://github.com/vllm-project/router/pull/360
**State:** open (created 2026-10-10 — yesterday)

If we do get the P2P Mooncake path working, #360 is a latent landmine: the
router sets `max_tokens=1` on prefill but may leave `min_tokens=0`; if the first
generated token is EOS, the prefill finishes `FINISHED_STOPPED` instead of
`FINISHED_LENGTH_CAPPED`, and the prefill response carries **no KV handoff
parameters** — decode would recompute locally, defeating the PD split. This is
not our current symptom (our prefill deadlocks before generating any token), but
it would manifest immediately after fixing the deadlock if the first token
happens to be a stop token.

### Why our exact symptom is NOT a router issue

Our symptom — router routes fine (`pd_requests_total > 0`), but the prefill
engine never admits / 0 completions / deferred `WAITING_FOR_REMOTE_KVS` — is an
**engine-side** `MooncakeStoreConnector` deadlock, not a router defect. Evidence:
`gpu-cluster-forks/spark-vllm-docker/docs/pd-router/diagnosis-prefill-mooncake-lookup-deadlock.md:4`
("traffic *does* reach cluster; prefill rank's EngineCore is deadlocked inside
`MooncakeStoreConnector`'s KV-store lookup … Not litellm, router, or network
fault"). The router correctly forwards requests to both prefill (`:8100`) and
decode (`:8200`); the prefill engine's scheduler busy-spins at
`vllm/v1/engine/core.py:1591` with `model_executed: False` because the
LMCache-MP / MooncakeStore lookup returns `None` forever
(`pd-failure-evidence.md:46-58,85-115`). No `vllm-project/router` issue matches
this engine-side deadlock — it's in the vLLM fork
(`Fulton-Engineering-Services/vllm @ cuda13.3-aarch64-gb10-glm5next-modular`)
and the Mooncake store image (`10.100.170.3:5000/mooncake-store:v0.3.12.post1-1`),
not the router.

---

## Sources

| Claim | Source |
|---|---|
| Router pin `v0.1.16` | `infra/cluster-config/cluster.env:681-682` |
| Compose vllm-router service flags | `infra/cluster-config/docker/mooncake-store/compose.yaml:132-198` |
| All CLI defaults & `file:line` citations | `gpu-cluster-forks/spark-vllm-docker/docs/pd-router/upstream-router-contracts.md:93-178` (codekeeper-indexed revision `91b65ee4…`) |
| `KvConnector` enum (nixl/mooncake/moriio) | `upstream-router-contracts.md:200` (`src/config/types.rs:536-547`) |
| Mooncake bootstrap + `kv_transfer_params` injection | `upstream-router-contracts.md:310-317`; PR #151 body |
| Cutover context (`--kv-connector nixl`, startup timeout) | `gpu-cluster-forks/spark-vllm-docker/docs/pd-router/cutover-vllm-router.md:38-51` |
| Health-gate lifecycle | `infra/cluster-config/docs/serve-lifecycle-integration.md:410-412,504-505` |
| Store-path wedge symptom & root cause | `gpu-cluster-forks/spark-vllm-docker/docs/pd-router/pd-failure-evidence.md:46-58,85-115` |
| Diagnosis (router exonerated) | `gpu-cluster-forks/spark-vllm-docker/docs/pd-router/diagnosis-prefill-mooncake-lookup-deadlock.md:4` |
| All issues/PRs (#151, #209, #247, #248, #360, #207, #311, #146, #114, #345, #337, #255, #326, #338, #130, #229, #317, #171, #185, #184, #313, #210, #263, #193, #222, #296, #266, #174, #175, #178, #159) | GitHub Issues API `https://api.github.com/repos/vllm-project/router/issues/<n>` and Search API `https://api.github.com/search/issues?q=repo:vllm-project/router+<keyword>` (fetched 2026-10-11) |
| DeepWiki 429 (3 attempts) | `read` + `xd://mcp__searxng_mcp_fetch_page` on `https://deepwiki.com/vllm-project/router/10.4-performance-tuning` — all returned HTTP 429 |