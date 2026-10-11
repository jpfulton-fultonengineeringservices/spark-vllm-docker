# Phase 5 live verification — mooncake twin with vllm-router cutover

**Date:** 2026-10-11 20:08–20:27Z · **Plan:** `.omp/plans/pd-disagg-vllm-router.md` (Phase 5, steps 17–18) · **Operator-visible via this session's cluster probes.**

## What was launched

`spark-vllm-docker-serve.sh launch --recipe pd-disagg-mimo-uncensored-exl3-4x --rank-order gx10-cb11:0,gx10-f1d8:1,gx10-becc:2,gx10-9273:3 --mode staged --daemon --recreate` on the head rank (node1/gx10-cb11), after all four submodule change sets were pulled onto every node (cluster-config `6f05a0d`, fork `1c5dbbe`).

## Result: PASS — every acceptance criterion

| # | Criterion (plan step 17) | Observed |
|---|---|---|
| 1 | Compose cohort up | `mooncake-master` + `mooncake-client` + `vllm-router` all `Up (healthy)` (53s after compose up) |
| 2 | Router binds :8000 after engines | `Router ready \| workers: ["http://10.100.171.2:8200","http://10.100.171.3:8100"]` at 20:24:35Z (`vllm-router` logs); `GET /v1/models` → 200 |
| 3 | E2E completion through router | `POST :8000/v1/chat/completions` → **200, 1.4 s**, real multi-token content ("PD router cutover works."), two-stage request id `___prefill_addr_10.100.171.3:8100___decode_addr_10.100.171.2:8200_022a0f83…` |
| 4 | Sustained load (3 more) | 3× 200, 0.85–1.02 s each |
| 5 | Router PD counters | `vllm_router_pd_requests_total{route} 4`, `pd_prefill_requests_total{worker=8100} 4`, `pd_decode_requests_total{worker=8200} 4`, `pd_request_duration_count 4` — each request counted on both legs |
| 6 | `pd_errors_total` = 0 | Only 8 non-`#` router families present; **no `pd_errors_total`/`requests_errors_total` series at all** (zero-count series are not emitted) — 4/4 requests succeeded, no error counter instantiation |
| 7 | Engine health | prefill `:8100` 200 (boot ~8 min: 17 s weights, b12x prepare, flashinfer autotune cache hit, KV 5.72 GiB); decode `:8200` 200 (~15 min incl. DFlash speculator capture, KV 5.52 GiB) |
| 8 | KV path active | MooncakeStoreConnector booted on both roles: `Mooncake mode=standalone-store (global_segment_size=0, local_buffer 4 GiB, preferred_segment 10.100.171.3:50053)`, host_staging put/get pools `2×2048 MiB pinned`, KV-load receive thread started |

## KV-contract probe (plan step 18) — resolved without intervention

The pre-registered contingency (strip `kv_transfer_params` in the fork connector) was **not needed**:

- The router ran `--kv-connector nixl`, injecting prefill `kv_transfer_params {do_remote_decode:true,…nulls}` (`vllm_pd_router.rs:186-205`).
- The fork's MooncakeStoreConnector **tolerated** the injected params (prefill leg completed; engine logged `kv_transfer_config KVTransferConfig(kv_connector='MooncakeStoreConnector', …, kv_role='kv_producer', …host_staging…)` and served).
- Decode leg ran hint-free hash lookup as designed (nixl `build_decode_kv_transfer_params` returns None when the prefill response carries no kv params — store-only flow).
- No EADDRINUSE: the in-engine `pd_proxy.py` is deleted (fork `cde9338`); only the router container binds :8000.

## Known cosmetic findings (not blockers)

- `register_buffer failed … -600` (rdma_context `Bad address`) appears once per role at KV registration on the 3.3–3.6 GiB **KV buffer itself** — GB10 GPU-VA registration is the documented dead end; the connector's host_staging path is what carries data, and KV get/put pools pin successfully after. Same signature as pre-cutover boots; no functional impact observed (4/4 completions).
- Decode-side KV concurrency for 1M-token contexts remains 0.15x (KV 5.52 GiB at GMU 0.68) — pre-existing sizing, unchanged by this cutover.
- `num_groups=0` in `Registered KV caches` (both roles): the HMA-disable path collapses groups; behavior identical to pre-cutover (five KV-group blockers re-verified live by this boot).

## LMCache MP twin (Phase 4 evidence, separate relaunch)

Separate `--recreate` relaunch with `LMCACHE_L2_DISABLED=1`: 6 successful completions (was 0), 0 deferred, 0 L2 adapters, no core.py:1591 spin (py-spy idle). Details + root-cause citations: `docs/pd-router/lmcache-parity-fixes.md` (fork `1c5dbbe`).

## VictoriaMetrics scrape (Phase 3) — pending one control-plane pass

`up{job="vllm-router"}` is empty because the router target document is written by the cluster-control reconciler's next Apply/Reconcile (file_sd refreshes only on a placement pass — scrape.yml "KNOWN LIMITATION"). The writer (`buildRouterTargets` → `ROUTER_SD_OUTPUT_PATH`) and job are deployed (home-infrastructure `f9a28b6`, control-infra `31b6aae`); the dell control-plane container needs a redeploy/roll to pick up the new code, then one `POST /api/placement/reconcile` materializes `vllm-router.json`.
