# 06 - Verification and observability (LMCache MP L2)

This document separates the two evidence classes the LMCache MP corridor needs: **PROVE** (acceptance gates - one-shot, per-change evidence that the corridor is actually wired and moving data) and **WATCH** (steady-state observation - continuous Prometheus-scraped series on the `gx10-lmcache-mp` dashboard that answer "is it still healthy" without re-running gates). Boot log lines, `/healthcheck`, `/status`, `/metrics`, NFS directory listings, and valkey scans are PROVE instruments; the dashboard's 16 panels are WATCH instruments. Corridor internals live in [01-architecture-and-corridors.md](01-architecture-and-corridors.md), engine-side wiring in [02-engine-side-mods.md](02-engine-side-mods.md), server-side deployment in [03-server-side.md](03-server-side.md); the two upstream L2-prefetch bugs referenced throughout are documented in [../pd-router/lmcache-parity-fixes.md](../pd-router/lmcache-parity-fixes.md).

## PROVE vs WATCH

| | PROVE (acceptance) | WATCH (steady state) |
|---|---|---|
| Question answered | "Is the corridor wired and does data actually flow end-to-end?" | "Is it degrading, wedged, or mis-aggregated right now?" |
| Instruments | Engine/MP-server logs, `:8082/healthcheck`, `:8082/status`, `:8082/metrics` deltas, `ls` on the NFS tier, `valkey-cli` scans, request TTFT | `gx10-lmcache-mp` dashboard (scrape job `gx10-lmcache`, 30 s refresh) |
| When | After boot, recipe change, wheel bump, or L2 enable/disable flip | Continuously; trends and velocities, not absolute snapshots |
| Failure meaning | Gate fails → do not trust serving; fix before load | Panel anomaly → consult the quick-reference table below |
| Counter semantics | Raw deltas over a controlled single-request experiment | `rate()` over 5 m windows (except where a rule forbids it) |

A gate passing once says nothing about an hour later; a clean dashboard says nothing about whether a store ever landed on disk. Run the gates after any change, then watch.

## Gate 1 - Boot integrity (connector + server up, HMA intact)

Three greps against the engine container on each serving rank, plus two probes against each rank's MP server. Container names: `vllm_node` (engine), `lmcache-server-<instance>` (server; instance = node hostname, e.g. `gx10-cb11`).

```console
# 1a. The MP connector attached (one or more hits expected; TWO ranks -> check both nodes)
$ docker logs vllm_node 2>&1 | grep -c 'Using external LMCacheMPConnector lmcache.integration.vllm.lmcache_mp_connector'
1

# 1b. HMA must stay ON - these must return NOTHING (exit 1)
$ docker logs vllm_node 2>&1 | grep 'Turning off hybrid kv cache manager'
$ docker logs vllm_node 2>&1 | grep 'Failed to promote local KV cache specs'

# 1c. MP server alive on this rank
$ curl -sf http://localhost:8082/healthcheck && echo OK
OK

# 1d. Chunk size threaded end-to-end (argparse only - see 03-server-side.md)
$ curl -s http://localhost:8082/status | jq '.chunk_size'
1648
```

Expected observables:

1. Grep 1a returns at least 1 per rank. This is the `LMCacheMPConnector (KVConnectorBase_V1, SupportsHMA)` factory line - its presence with `kv_role kv_both` proves the MP corridor attached rather than the in-process `LMCacheConnectorV1` fallback (which cannot serve hybrid; see 01-architecture-and-corridors.md).
2. Greps 1b return zero matches. Either line means HMA was disabled - `Turning off hybrid kv cache manager` is the silent degradation, `Failed to promote local KV cache specs` is the hard failure (`ValueError: Failed to promote local KV cache specs to one unified type.`). Never pass `--disable-hybrid-kv-cache-manager` anywhere in this corridor.
3. `/healthcheck` exits 0 (the same poll `spark-serve.sh` uses at bring-up: `curl -sf http://localhost:<http_port>/healthcheck` emitting `LMCACHE_EXT_OK`, `infra/cluster-config/scripts/lib/spark-serve.sh:538-539`).
4. `/status` reports `chunk_size` equal to the engine's RESOLVED attention block size - 1648 for Qwen3_5 mamba-align (103 x 16). Mismatch here is the chunk-size defect surfacing at its final checkpoint: recipe env `LMCACHE_CHUNK_SIZE: "1648"` (`recipes/qwen3.8-27b-coder390-2x-lmcache-l2.yaml:139`) does NOT reach `docker run` for the serve process, so the launcher re-injects `-e LMCACHE_CHUNK_SIZE=` (`spark-serve.sh:766`), and the server reads it ONLY from the `--chunk-size` argparse flag (`cluster-config/scripts/lmcache-server-entrypoint.sh:38`; `MPServerConfig` takes `chunk_size=args.chunk_size`, never env - comment at `:31-37`). A `256` here means the drop-in/`-e` chain broke; the engine would later hard-fail `LMCache chunk size 256 must be a multiple of 1648`.

## Gate 2 - L1 write (stores submitted AND finished, residency non-zero)

Send one prompt comfortably larger than one chunk: ~5k tokens spans ~3 full 1648-token chunks (3 x 1648 = 4944). Then diff the MP server's store counters across the request.

```console
# Fire a ~5k-token completion (any client; keep max_tokens small)
$ curl -s http://localhost:8000/v1/completions -H 'Content-Type: application/json' \
    -d '{"model":"<served-model>","prompt":"'"$(repeat-words 2500)"'","max_tokens":1}' > /dev/null

# Counter deltas on the rank that served it
$ curl -s http://localhost:8082/metrics | grep -E '^lmcache_mp_num_(submitted|finished)_stores_total'
lmcache_mp_num_submitted_stores_total{...} 12.0
lmcache_mp_num_finished_stores_total{...} 12.0
$ curl -s http://localhost:8082/metrics | grep '^lmcache_mp_l1_memory_usage_bytes'
lmcache_mp_l1_memory_usage_bytes{...} 6.8e+08
```

Expected observables:

1. `lmcache_mp_num_submitted_stores_total` AND `lmcache_mp_num_finished_stores_total` both advance (dashboard panel 216 plots their rates). BOTH must move: submitted-without-finished is a stuck store path (event-bus or L0->L1 transfer hang - see panel 219 and 217).
2. `lmcache_mp_l1_memory_usage_bytes` becomes non-zero (panels 202 target B and 210 target A).
3. **Warning:** `save_unfull_chunk: false` means a sub-chunk prompt stores nothing. A prompt that does not cross at least one full chunk boundary produces zero stores and leaves L1 at zero - that is correct behaviour, not a fault. The pinned config value is visible in the in-process corridor's rendered config (`mods/lmcache-offload/run.sh:105`); the same LMCache default governs the MP corridor, so Gate 2's prompt must be several thousand tokens, not a sentence. (A 5k-token prompt fills exactly 3 full chunks; the ~56-token remainder is an unfull chunk and is dropped.)

## Gate 3 - L2 write (chunk files on NFS, keys in valkey)

With `LMCACHE_L2_DISABLED=0` (re-enablement procedure in [../pd-router/lmcache-parity-fixes.md](../pd-router/lmcache-parity-fixes.md) §3.3; current cluster.env value `LMCACHE_L2_DISABLED=0`), the MP server fans every L1 eviction/write out to BOTH tiers: `fs_native` on the shared NFS (`LMCACHE_FS_NATIVE_PATH=/cluster-shared/lmcache-l2`, bind-mounted into the server container at `lmcache-server-entrypoint.sh:149`) and `valkey` on Dell (`LMCACHE_VALKEY_NODES=10.10.10.53:6379`, key prefix `gx10` from `:103`).

```console
# 3a. fs_native: chunk files appear under an escaped-model directory
$ ls /cluster-shared/lmcache-l2 | grep -c .                     # >= 1 model directory
$ ls /cluster-shared/lmcache-l2 | grep -i coder390              # escaped model dir name
$ find /cluster-shared/lmcache-l2 -type f | wc -l               # total chunk files
126

# 3b. valkey: keys under the gx10 prefix, named after the model
$ valkey-cli -h 10.10.10.53 -p 6379 --scan --pattern 'gx10*Coder390*' | wc -l
248
$ valkey-cli -h 10.10.10.53 -p 6379 DBSIZE
248
```

Expected observables:

1. The fs_native base path contains one directory per stored model in escaped form (the HF repo path with separators escaped - the directory name matches the model, not a UUID), and `. vs full-chunk` chunk files accumulate under it after Gate 2's request. Observed live: 126 chunk files / 9.6 GB on the NFS directory.
2. valkey holds `gx10*<Model>*` keys - the prefix comes from `--l2-adapter ... key_prefix` (`LMCACHE_VALKEY_PREFIX=gx10`). Observed live: 248 keys for the coder390 model.
3. **Pass requires BOTH tiers** (the write path fans out to both by design): `find ... | wc -l` grows AND `DBSIZE` grows. Files but flat `DBSIZE` = dead valkey adapter; keys but an empty NFS dir = fs_native adapter dead or `LMCACHE_FS_NATIVE_PATH` unset in the unit drop-in (`spark-serve.sh:519-522` writes `20-l2.conf`). Confirm adapters are attached on the dashboard first: panel 205 (store/prefetch adapters active) and per-tier series on panels 203/211.
4. Dashboard corroboration: `lmcache_mp_l2_usage_bytes{l2_name="fs_native"}` and `{l2_name="valkey"}` advance together (panels 203/211) - they report the SAME shared stores (see aggregation rules below).

## Gate 4 - L2 read (the test everyone gets wrong)

**An IDENTICAL-prompt repeat proves NOTHING.** vLLM's own prefix caching (APC, `--enable-prefix-caching` in the recipe) answers the repeat in full, LMCache is never consulted, `lookup_hit_tokens` stays flat, and TTFT looks like a cache hit regardless of LMCache health. A green TTFT on a repeated prompt is vLLM APC evidence, not L2-read evidence.

The correct test: **same long prefix, different tail.** Build a prompt whose ~25k-token prefix is byte-identical to a previously stored prompt but whose last few hundred tokens are new, then fire it.

```console
# Trial 1 (cold, fresh prefix+tail): baseline TTFT
$ time curl -s http://localhost:8000/v1/completions -d @req-prefix-tailA.json | jq -r '.usage'

# Trial 2 (same prefix, DIFFERENT tail)
$ curl -s http://localhost:8082/metrics | grep '^lmcache_mp_lookup_hit_tokens_total'   # before
lmcache_mp_lookup_hit_tokens_total{...} 9888.0
$ curl -s http://localhost:8000/v1/completions -d @req-prefix-tailB.json > /dev/null
$ curl -s http://localhost:8082/metrics | grep '^lmcache_mp_lookup_hit_tokens_total'   # after
lmcache_mp_lookup_hit_tokens_total{...} 19776.0

# Engine-side confirmation on the rank that served trial 2
$ docker logs vllm_node 2>&1 | grep 'Prefetch request completed'
Prefetch request completed (L1+L2): 4/4 retained keys (0 L1, 4 L2) in 812 ms
```

Expected observables:

1. Engine log line `Prefetch request completed (L1+L2): N/N retained keys (0 L1, N L2) in <ms> ms` - all retained keys sourced from L2. Observed live: `4/4 retained keys (0 L1, 4 L2) in 812 ms`.
2. `lmcache_mp_lookup_hit_tokens_total` advances monotonically in chunk-size multiples. Observed live: 3296 -> 9888 -> 19776 (steps of 3296 = 2 chunks x 1648). Flat counter with fast TTFT means APC answered the request - vary the tail further or restart the engine between trials.
3. TTFT win versus the cold baseline on the same token count. Observed live: 3.7 s warm vs 20.6 s cold on a 27.9k-token prompt. If the prefix was served by the other rank/node, the win is larger because APC cannot pre-empt the lookup.
4. Dashboard corroboration: panel 214 (`prefetch_lookup_requests_total`, `prefetch_hit_chunks_total`, `prefetch_load_completed_chunks_total` rates) and panel 212 (requested vs HIT tok/min).

## Pass criteria, resets, and re-attach

| Signal | Pass | Fail / investigate |
|---|---|---|
| Store path | `num_submitted_stores_total` AND `num_finished_stores_total` both advance; submitted minus finished does not grow unbounded | Submitted climbs, finished flat = stuck store (event bus, L0->L1) |
| L2 write | fs_native files AND valkey keys both grow (fan-out) | Only one tier moves = one adapter dead; check panel 205 |
| L2 read | `prefetch_hit_chunks_total` / `lookup_hit_tokens_total` advance; `Prefetch request completed (L1+L2)` appears with `0 L1, N L2` | Lookups advance, hits flat = misses (`l1_oom` on panel 215 is the usual reason); lookups flat = connector never consulted (APC answered, or layout/model-name mismatch - see parity fixes §1.3) |
| Counter resets | Cumulative counters restart at 0 after `systemctl restart lmcache-server@<host>` - `spark-vllm-docker-serve.sh` re-attaches (restarts) the MP units on every recipe relaunch (`launch_lmcache_mp_servers_external`, `spark-serve.sh:486-549`), so ANY relaunch zeroes that rank's counters | A reset with NO relaunch = the server process died and restarted; check `process_start_time_seconds{job="gx10-lmcache"}` and unit journal |
| Tier gauges | `l2_usage_bytes` (fs_native/valkey) reflects the shared stores and does NOT drop when a rank's server restarts - NFS dir and Dell valkey persist | One tier's gauge collapses to 0 while files/keys exist = adapter registration failed, not data loss |

Distinguish "reset because relaunched" from "reset because broken" with `curl -s :8082/metrics | grep process_start_time_seconds` against the systemd unit's `ActiveEnterTimestamp` (`systemctl show lmcache-server@<host> -p ActiveEnterTimestamp`). Cumulative dashboard series legitimately step to zero at relaunch; rates on panels 210-217 read as dips, not faults.

## Steady state (WATCH): `gx10-lmcache-mp`

One dashboard, `uid gx10-lmcache-mp` ("GX10 LMCache MP Store (KV Cache L1/L2)"), refresh 30 s, default window now-1h, `$node` template multi-select from `label_values(up{job="gx10-lmcache"}, node)`. The scrape target is `job="gx10-lmcache"`; there is no separate `gx10-lmcache.json` dashboard file. All panels are described below.

| Panel | Type | Watches | Steady state |
|---|---|---|---|
| 201 | stat | MP Servers Up (`count(up == 1) OR on() vector(0)`) | 2; the ONLY red-on-purpose tile |
| 202 | stat | L1 fill ratio (worst-rank, percentunit) + summed `l1_memory_usage_bytes` | Fill near 1.0 is HEALTHY (cache full by design; 1 GiB on GB10) |
| 203 | stat | `l2_usage_bytes` per tier, `max()` | fs_native ~ valkey, grows with corpus |
| 204 | stat | Lookup hit ratio (diagnostic) | Expected LOW (1 GiB L1 vs 1648-token fp8 chunks) |
| 205 | stat | Store/prefetch adapters active | 2 tiers x ranks; p2p = third store adapter, zero usage |
| 210 | ts | L1 usage per rank + evicted chunks/min | Eviction at capacity is the design working |
| 211 | ts | L2 footprint by tier (`max()`) | Tiers mirror; a rank can show fs_native 0 before first file write |
| 212 | ts | Requested vs HIT tok/min | Hits correlate with repeated-prefix traffic |
| 213 | ts | L2 store/load req/min by tier | Both tiers move together |
| 214 | ts | Prefetch lookups / L2 hit chunks / loaded chunks | The L2 read path; flat = nobody consulting L2 |
| 215 | ts | Prefetch failures by `reason` | `l1_oom` ~162 cumulative is STEADY STATE; `l1_contended` also exists |
| 216 | ts | Store/retrieve submit+finish rates, chunks loaded | Submit/finish pairs track together |
| 217 | ts | L0->L1 and L2 store throughput GB/s (p50/p95) | Sanity of copy bandwidth |
| 218 | ts | L1 chunk lifetime p50/p95 | Cumulative only - see aggregation rules |
| 219 | ts | Event bus depth, drops/min, drain lag | Depth ~0, no sustained drops |
| 220 | ts | Active prefetch jobs, p2p lookups, RSS | Jobs return to 0; wedge = jobs stuck > 0 |

Noise not to alarm on: `l1_oom` prefetch failures (L1 too small for a whole chunk is the intended tiering), `rdma_context.cpp ... register memory ... Bad address [14]` in server logs (documented harmless), and `l2_usage_bytes{l2_name="p2p"} = 0` while the p2p store counter advances (p2p carries no usage series).

## Aggregation rules that will bite

1. **`l2_usage_bytes`: `max()`, never `sum()`.** fs_native (one NFS directory) and valkey (one Dell instance) are SHARED stores; every rank's MP server reports the same tier totals, and the write path fans out so the two tiers report identical values (`scrape.yml:75-76` encodes this rule in the scrape config comment). Summing across ranks doubles the footprint; per-tier `max()` (panels 203/211) is correct. Per-process counters - lookup/prefetch/store rates, `l1_memory_usage_bytes` residency - are NOT shared and legitimately sum (panel 202 target B) or plot per rank.
2. **`l1_chunk_lifetime_seconds` must be queried CUMULATIVELY - no `rate()`.** Evictions are sparse; a 5-10 m rate window is usually empty and the panel reads "No data" despite a populated histogram (panel 218 queries the raw histogram: `histogram_quantile(... sum by (le) (..._bucket))`). Live check: 4 lifetime observations total, all older than 10 m.
3. **Never merge ratio and bytes into one stat tile.** `l1_usage_ratio` is a ratio (`percentunit`), `l1_memory_usage_bytes` is bytes. Panel 202 keeps them as separate targets with a `byFrameRefID` override carrying the `bytes` unit on the second series - a single unit setting would mislabel one of them.

## Stat-tile colour semantics (do not regress)

Red means an actionable fault. On this dashboard exactly ONE tile may alarm: **201 MP Servers Up** (red step at 0, green at 2, `colorMode: "background"`). Every informational tile - L1 residency, L2 footprint, lookup hit ratio, adapter count - must be colourless: a full L1 or a growing L2 footprint is the design working, and a permanently red tile trains operators to ignore the row.

Absence of a `thresholds` block is NOT "no colour" - Grafana falls back to an implicit red step. Colourlessness must be explicit on all three surfaces: `options.colorMode: "none"`, `fieldConfig.defaults.color: {mode: "fixed", fixedColor: "text"}`, and a single `text` thresholds step. Verified shape (panel 203, L2 Offload Footprint):

```json
{
  "id": 203, "type": "stat", "title": "L2 Offload Footprint",
  "options": { "colorMode": "none", "textMode": "value_and_name" },
  "fieldConfig": { "defaults": {
      "color": { "mode": "fixed", "fixedColor": "text" },
      "thresholds": { "mode": "absolute",
                      "steps": [ { "color": "text", "value": null } ] },
      "unit": "bytes" }, "overrides": [] }
}
```

Verify colour by RENDERING and counting pixels, not by reading JSON: `metrics_render` (action `render`, dashboard `gx10-lmcache-mp`, `panelId` 203) returns a PNG; count the red-channel band. A correct colourless tile shows ~0.01% red pixels (anti-aliasing noise); tens of percent red means a threshold step or `colorMode` regressed. Panel 201 is the control: it renders with a green background at value 2 and red at 0/1.

## Quick reference: symptom -> panel/metric

| Operator question | Look at | Healthy reads as |
|---|---|---|
| Is L2 filling? | Panel 203 / 211 (`l2_usage_bytes` by tier, `max()`); ground truth = Gate 3 `find`/`DBSIZE` | Both tiers grow together |
| Are lookups hitting? | Panel 212 (tok/min), 214 (hit chunks/min), 204 (ratio); per-request delta on `lookup_hit_tokens_total` | Counter advances in chunk multiples |
| Is L1 too small? | Panel 210 (evicted/min), 215 (`l1_oom`/min), 218 (lifetime) | `l1_oom` steady, lifetime short - by design; raise `LMCACHE_L1_SIZE_GB` only with unified-memory headroom |
| Is the MP server sick? | Panel 201 (up count), 219 (event bus), 220 (RSS); `:8082/healthcheck`, `:8082/metrics` `process_*` | up = expected ranks; no drops; RSS stable |
| Did the prefetch wedge? | Panel 220 (`active_prefetch_jobs` stuck > 0) + vLLM `num_requests_running=0` while `num_requests_waiting` grows; `:8082/status` `prefetch_controller` counters (`submission_queue_size`, pending, `max_in_flight`, in-flight/completed) | Two documented upstream bugs produce exactly this scheduler spin: (A) L2 prefetch job vanishes from the PrefetchController so the engine spins `WAITING_FOR_REMOTE_KVS` with zero running requests (`lmcache-parity-fixes.md` §1 - engine `core.py:1591`, observed `num_requests_waiting 13` while `request_success_total 0`); (B) heartbeat lazy-start -> 3600 s registration-grace reap, server log `Reaped non-GPU instance ... silent for 3626.8s (pinged=False)` (§2). Both fire only with L2 adapters active; the shipped fallback `LMCACHE_L2_DISABLED=1` sidesteps both (§3) - check which state cluster.env is in before debugging |
| Store path stuck? | Panel 216 submitted vs finished; panel 219 event bus | Divergence = stuck L0->L1 transfer |
| One L2 tier missing? | Panel 205 (adapters active), 203 (tier series flat) | Adapter count 2 x ranks; drop-in `20-l2.conf` env check |
| Did a relaunch reset counters? | `process_start_time_seconds` vs unit `ActiveEnterTimestamp`; pass-criteria table above | Reset coincides with relaunch = benign |

Metric names for direct PromQL are exactly the dashboard's `lmcache_mp_*` family on `job="gx10-lmcache"` (raw list: `curl -s http://localhost:8082/metrics | grep '^lmcache_mp'`); the scrape job and target inventory live in `infra/home-infrastructure/groups/metrics/config/victoriametrics/scrape.yml:66-89`.