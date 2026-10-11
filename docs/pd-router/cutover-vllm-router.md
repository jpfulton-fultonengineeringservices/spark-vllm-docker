# Cutover: Python pd_proxy → upstream vllm-router container (rank-0 sibling)

**Date:** 2026-10-11 · **Plan:** `.omp/plans/pd-disagg-vllm-router.md` (Phase 2,
steps 6–8) · **Repo scope:** `gpu-cluster-forks/spark-vllm-docker/`

## What was removed

- `mods/pd-disagg/pd_proxy.py` — deleted. It was a byte-identical vendored copy of
  `infra/cluster-config/docker/pd-proxy/pd_proxy.py` (sha256
  `fd2a5cb23c3bafcd6f009d369ed0114c5ab50eb441f09fa15e73ac9e8d79154d`),
  a FastAPI round-robin proxy launched in-engine on rank 0 by `dispatch.sh`
  with a sleep-gated `PD_ROUTER_START_DELAY=30` and `--verify-timeout 1800`.
- `mods/pd-disagg/dispatch.sh` rank-0 router block (was ~lines 232–261): the
  `PD_ROUTER_ENABLED` gate, `PD_ROUTER_SCRIPT`/`PROXY_SCRIPT` path resolution,
  `router_decoder_host` derivation, and the `sleep … exec python3 pd_proxy.py …`
  subshell that wrote `/tmp/pd-proxy.log`. Header comment now states the
  router is a rank-0 sibling `vllm-router` container, not an in-engine process.
- `recipes/pd-disagg-mimo-uncensored-exl3-4x.yaml` and
  `…-4x-lmcache.yaml`: `PD_ROUTER_ENABLED: "1"` env lines and
  `toy_proxy_server` / `PD_ROUTER_START_DELAY` comment prose removed/rewritten.
- `recipes/pd-disagg-mimo-exl3.yaml` (legacy 3-node NIXL research recipe):
  `toy_proxy_server` comment prose + `PD_ROUTER_ENABLED: "1"` env line updated
  to describe the `vllm-router` sibling container.
- `mods/pd-disagg/README.md`: topology, mermaid, env-var table, deployment
  health-gate, and provenance sections rewritten; all `pd_proxy` / `pd-proxy`
  / `toy_proxy` / `PD_ROUTER_ENABLED` / `PD_ROUTER_START_DELAY` /
  `PD_ROUTER_SCRIPT` references removed.

## New contract (pointer)

The PD-disaggregation request router is now an **upstream Rust
`vllm-project/router` container** (`vllm/vllm-router` image, pinned in
`infra/cluster-config/cluster.env`), run as a **rank-0 sibling container** by
cluster-config serve.sh compose (`--profile router` on both the mooncake and
lmcache KV-backend branches of `spark-serve.sh`'s PD case). It is **NOT** an
in-engine process inside `dispatch.sh`.

- **OpenAI API:** `:8000` (`--host 0.0.0.0 --port ${PD_ROUTER_PORT:-8000}`,
  `--vllm-pd-disaggregation`, `--kv-connector nixl`).
- **Prometheus `/metrics`:** `:29000` (`--prometheus-host 0.0.0.0
  --prometheus-port ${PD_ROUTER_METRICS_PORT:-29000}`).
- **Prefill/decode backends:** `--prefill http://<rank0>:${PD_PREFILL_PORT:-8100}`,
  `--decode http://<rank2>:${PD_DECODE_PORT:-8200}`.
- **Worker startup:** `--worker-startup-timeout-secs 1800
  --worker-startup-check-interval 30` (same 30-min EXL3 load budget the old
  `--verify-timeout 1800` provided).
- **KV-transfer contract UNCHANGED.** Both backends stay as-is —
  `MooncakeStoreConnector` (store-only, host-staged RDMA) for the mooncake
  path and `LMCacheMPConnector` (per-node MP server, engine_driven) for the
  lmcache path. The router is KV-backend-agnostic; it does not participate in
  the KV handshake. See `kv-backends-l2.md` and `upstream-router-contracts.md`.

## Port / env defaults kept

| Var | Value | Owned by |
|---|---|---|
| `PD_ROUTER_PORT` | `8000` | serve.sh compose (recipe `pd_router_port` default feeds it) |
| `PD_ROUTER_METRICS_PORT` | `29000` | serve.sh compose |
| `PD_PREFILL_PORT` | `8100` | dispatch.sh per-role assembly |
| `PD_DECODE_PORT` | `8200` | dispatch.sh per-role assembly |

Recipe `pd_router_port` / `pd_prefill_port` / `pd_decode_port` defaults and the
`PD_ROUTER_PORT={pd_router_port}` token on the `command:` lines are
intentionally kept — `dispatch.sh` no longer consumes `PD_ROUTER_PORT` itself,
but serve.sh compose uses it to configure the router container.

## Metrics evidence (acceptance gate)

After cutover, the PD-disaggregation stack is healthy when:

- `docker ps` shows `mooncake-master` (rank-0 only) + `mooncake-client` (4×) +
  `vllm-router` (rank-0 only);
- `:8000/v1/chat/completions` returns 200 (multi-token, not `max_tokens=1`);
- `:29000/metrics` → 200 and exposes `pd_*` counters +
  `vllm_router_pd_requests_total`;
- per-role engine `/metrics` on `:8100` (prefill) and `:8200` (decode) → 200,
  with `request_success_total > 0` under sustained load and zero
  `nixl_num_failed_transfers` / store KV get/put errors;
- VictoriaMetrics shows `up{job="vllm-router"}=1` and
  `up{job="vllm-serve",role="prefill"}` / `role="decode" = 1`.

## Historical docs (intentionally kept, superseded)

The following `docs/pd-router/` records still reference `pd_proxy.py` and the
old in-engine proxy. They are the **research/failure-analysis trail** that
motivated this cutover and are kept verbatim as history; this note supersedes
their present-tense claims about the proxy being the live front end:

- `kv-backends-l2.md` §5.6 — documents the `pd_proxy.py` component being
  replaced (see the "Superseded 2026-10-11" pointer added to that section).
- `pd-failure-evidence.md` §4 — the in-session proxy lineage and exonerated
  `pd_proxy.py` investigation.
- `upstream-router-contracts.md` — the `vllm-router` contract ground truth this
  cutover implements.

## Verification (this phase)

- `bash -n mods/pd-disagg/dispatch.sh` → exit 0 (367 lines after block removal;
  399 before).
- `bash -n mods/pd-disagg/run.sh` → exit 0 (unchanged; no proxy references).
- `grep` for `pd_proxy\.py` / `PD_ROUTER_SCRIPT` / `toy_proxy` / `pd-proxy`
  across `mods/` + `recipes/` + `tests/` → 0 hits (the README env-var table and
  provenance bullet now describe the container; the only surviving
  `pd_proxy.py` / `toy_proxy` strings are the historical citations in
  `docs/pd-router/` above, and the `pd_proxy.py` sha256 retained in this note
  and the README provenance bullet as the retired artifact's fingerprint).
- `tests/` sweep: no test references `pd_proxy`, `PD_ROUTER`, `toy_proxy`,
  `pd-disagg`, `pd_disagg`, or `dispatch.sh`. No PD-disagg test subset exists to
  run; the fork's test suite does not cover `dispatch.sh` command assembly.
  (End-to-end boot verification is Phase 5 of the plan, out of this fork-scoped
  phase.)