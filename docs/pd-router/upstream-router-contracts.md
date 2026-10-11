# Upstream vLLM Router — Contract Map (what it IS, and what integration points it exposes)

**Repo:** `vllm-router` (codekeeper registry) — clone `https://github.com/vllm-project/router.git`
**Indexed commit:** `91b65ee4bdfbe98863a32a2205dced5dbb249ffe` (revision `HEAD`, status `READY`, `lastIndexedAt` 2026-10-11T18:54:30.727Z`; latest run status PARTIAL — 353/353 files, 9102 symbols, 39167 relationships). Source: codekeeper `getRepositoryStatus`.

Every claim below was read this session from that indexed revision. Citations are `path:line` (file lines) as returned by codekeeper `readFile`/`searchForPattern`, or a codekeeper `namePath`. Where a fact comes from a repo doc file rather than code, the doc path is cited and, where both were read, the code site is cited alongside.

**Codekeeper enumeration notes (recorded honestly):** (a) `findFile` with a `src/**/*.rs` glob returned an empty list against this revision (server glob quirk), and `listDir` truncates a listing at ~12 entries — the §2 module tree therefore also relies on the `pub mod` declarations in `src/lib.rs`, which are authoritative for the crate. (b) `searchForPattern` with a **file** (not directory) `relativePath` fails with `ENOTDIR`; directory scopes work.

---

## 1. What the router is / architectural role

Self-description, `README.md:9`:

> "A high-performance and light-weight request forwarding system for vLLM large scale deployments, providing advanced load balancing methods and prefill/decode disaggregation support."

Key features (`README.md:13-19`): request-routing framework + async processing; load balancing (cache-aware, power of two, consistent hashing, random, round robin); Program scheduling; Prefill-Decode disaggregation; Kubernetes-native worker management and health monitoring; circuit breakers / retry logic / metrics; gRPC workers via `grpc://` URLs. It is a **fork of SGLang Model Gateway** (`README.md:340`).

Answer to "PD-aware? KV-aware? just a gateway?": **all three, layered.**

- **Gateway / reverse proxy.** It reverse-proxies OpenAI HTTP requests and transparently forwards any unmatched path/body to a selected worker. Transparent handler `transparent_proxy_handler` `src/server.rs:114-158`; trait method `RouterTrait::route_transparent` `src/routers/mod.rs:169-183`. Transparent proxy is hard-enabled at startup: `let enable_transparent_proxy = true;` `src/server.rs:1169`.
- **PD-aware (first-class).** A dedicated `VllmPDRouter` does two-stage prefill→decode dispatch: struct `VllmPDRouter` `src/routers/http/vllm_pd_router.rs:38`; `impl RouterTrait for VllmPDRouter` at `src/routers/http/vllm_pd_router.rs:2052` (`namePath` `VllmPDRouter`).
- **Cache/prefix-aware — approximate, NOT engine KV.** The `cache_aware` policy keeps a per-worker approximate radix tree built from request *text* prefixes to maximize prefix-cache hits, with LRU eviction and load-balancing thresholds (`src/policies/cache_aware.rs:1-60`). `consistent_hash` gives session affinity for KV-cache reuse (`docs/load_balancing/README.md:14-17`). This is a router-side approximation; it does **not** read engine KV state.
- **Engine-KV-event-driven index exists, but as a separate subsystem.** `src/kv_index/` is a multi-tier KV-block placement index fed by vLLM engine KV events (module doc `src/kv_index/mod.rs:1`; discovery via `/get_server_info` + `/kv_event_sources` `src/kv_index/discovery.rs:1-2`). See §5.

It is **not** a model server: it performs no inference and launches no backends; it only routes. (It can, however, issue profiling control calls to workers when `--profile` is set — `src/routers/http/pd_router.rs:135-200`.)

---

## 2. Crate / module layout

Cargo package `vllm_router_rs`, version `0.1.16`, lib `crate-type = ["cdylib", "rlib"]`, binary `vllm-router` at `src/main.rs` (`Cargo.toml:1-19`). The Python extension exposes `PolicyType` and `Router` classes (`src/lib.rs:23-47`).

Module tree from the `pub mod` declarations in `src/lib.rs:1-21`:

| Module | Role |
|---|---|
| `backend` | gRPC (`grpc://`) worker backend — token_ids path, control, health, convert (`src/backend/{mod,control,convert,detect,frontend,grpc,health,openai,preprocess,vllm_frontend}.rs`) |
| `config` | `RouterConfig` and friends (`src/config/{mod,types,validation}.rs`) |
| `core` | `worker`, `worker_registry`, `circuit_breaker`, `retry`, `token_bucket`, `error` |
| `data_connector` | response store: `MemoryResponseStorage` / `NoOpResponseStorage` |
| `kv_hash` | vLLM-compatible `sha256_cbor` prefix-cache sequence hashing (`src/kv_hash.rs:1`) |
| `kv_index` | multi-tier KV-block placement index fed by engine events |
| `metrics` | Prometheus metrics (`src/metrics.rs`) |
| `middleware` | request-id, concurrency limiter, trace layer |
| `otel_http`, `otel_trace` | OpenTelemetry HTTP client spans + provider wiring |
| `policies` | LB policies: `random, round_robin, cache_aware, power_of_two, consistent_hash, rendezvous_hash` |
| `program_scheduling` | optional Program identity / RequestPool / Global Queue scheduling |
| `protocols` | request/response specs (`protocols::spec`, `protocols::worker_spec`) |
| `routers` | `factory`, `router_manager`, `http/*` routers |
| `server` | axum app + route table + startup (`src/server.rs`) |
| `service_discovery` | Kubernetes pod watcher (`src/service_discovery.rs`) |
| `tokenizer` | HF / tiktoken tokenizers + chat templates |
| `tree` | radix tree used by cache-aware policy |
| `wasm_middleware` | WASM Component-Model OnRequest middleware (`src/wasm_middleware.rs`) |
| `logging` | logging init (`src/logging.rs`) |
| `token_estimator` (private) | prompt-token estimation |

CLI entrypoint and config assembly: `src/main.rs`. Router implementations: `src/routers/factory.rs` (`RouterFactory`), `src/routers/http/router.rs` (`Router`, regular HTTP), `src/routers/http/openai_router.rs` (`OpenAIRouter`), `src/routers/http/pd_router.rs` (`PdRouterBase`), `src/routers/http/vllm_pd_router.rs` (`VllmPDRouter`), `src/routers/router_manager.rs` (`RouterManager`, multi-router/IGW mode).

---

## 3. Configuration surface

### 3a. Config files

**There is no startup config file for the binary.** Configuration is CLI flags, a `.env` file, and environment variables. Evidence:

- `src/main.rs` loads `.env` with `dotenvy::dotenv().ok()` (`src/main.rs:718`) and otherwise parses only clap args. A `searchForPattern` for config-file loading found no `--config` flag and no router-config file read; the only `from_file`/`read_to_string` hits were tokenizer/chat-template loaders and the legacy `src/routes/*` tree loader.
- `RouterConfig` derives `Serialize/Deserialize` (`src/config/types.rs:12-13`) and is also constructible programmatically by the Python binding (`src/lib.rs:95-300`), so a JSON object *can* be used by a library caller — but the shipped CLI never reads one from disk.
- `examples/configs/round_robin_config.json` and `single_server_config.json` belong to the **legacy** `RouteHandle` tree (`src/routes/interface.rs:1-11`, `src/routes/routing_tree_builder.rs:16-18`), not the current router binary; their `{"route":{"type":"RoundRobinRoute"}}` shape does not appear in `CliArgs`.

Config inputs that are file-like at runtime: the WASM middleware artifact path (`--wasm-middleware`), and a JSON **string** (not file) for Program scheduling (`--program-scheduling-config-json`).

### 3b. Environment variables (every read site found this session)

| Env var | Meaning | Cite |
|---|---|---|
| `API_KEY_VALIDATION_URLS` | Comma-separated validation URLs; when set, all endpoints require `Authorization: Bearer <token>` | `src/main.rs:525-534` |
| `OPENAI_API_KEY` | Bearer token the router sends on outbound prefill/decode requests | `src/routers/http/vllm_pd_router.rs:1388,1538,1733` |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | Honored when `--otlp-traces-endpoint` is unset (SDK default host:port) | `src/config/types.rs:838`; `src/otel_trace.rs:154` |
| `VLLM_ROUTER_STAGES` | Backend staging toggle; truthy = `1`/`true`/`TRUE`/`yes`/`YES` | `src/backend/mod.rs:31-32` |
| `VLLM_ROUTER_TOKENIZER` | Tokenizer load-key override | `src/backend/preprocess.rs:102` |
| `VLLM_ROUTER_MODEL` | Model/tokenizer override (load key + model dir) | `src/backend/preprocess.rs:122,458` |
| `HF_TOKEN` | HuggingFace auth token for tokenizer downloads (`const HF_TOKEN_ENV_VAR = "HF_TOKEN"`) | `src/tokenizer/hub.rs:13,49,139` |
| `HOSTNAME` / `HOST` / `COMPUTERNAME` | Derive OTel `service.instance.id` (`{host}-{pid}`) | `src/otel_trace.rs:116-124` |
| `PYTHON`, `VIRTUAL_ENV` | Test helper only | `src/backend/preprocess.rs:468,473` |
| `CI` | Test skip guard | `src/tokenizer/factory.rs:297` |

### 3c. CLI flags — exact names and defaults

All flags are fields of `struct CliArgs` (starts `src/main.rs:106`), clap derive; defaults are the attribute on the line above each field. The Python launcher mirrors them in `RouterArgs` (`py_src/vllm_router/router_args.py:9-95`) with an optional `router-` prefix (`router_args.py:106-118`). Note: CLI `--log-level` accepts `debug|info|warn|error` (`src/main.rs:220`); the Python launcher accepts `debug|info|warning|error|critical` (`router_args.py:322-328`).

**Server / mode**

| Flag | Default | Cite (src/main.rs) |
|---|---|---|
| `--host` | `127.0.0.1` | 108-109 |
| `--port` | `30000` | 112-113 |
| `--worker-urls` (`num_args=0..`) | `[]` | 116-117 |
| `--policy` | `cache_aware` (choices: random, round_robin, cache_aware, power_of_two, consistent_hash, rendezvous_hash) | 120-121 |
| `--enable-program-scheduling` | `false` | 125-126 |
| `--program-scheduling-config-json` | none | 130-131 |
| `--backend` (alias `--runtime`) | `vllm` (enum `vllm,trtllm,openai,anthropic`, `Backend` 54-83); trtllm/anthropic warn "not implemented yet; falling back" and route as regular | 211-213 (warn 753-762) |
| `--enable-igw` | `false` | 360-361 |
| `--history-backend` | `memory` (`memory|none`) | 364-365 |

**PD disaggregation**

| Flag | Default | Cite (src/main.rs) |
|---|---|---|
| `--vllm-pd-disaggregation` | `false` | 134-135 |
| `--vllm-discovery-address` | none | 139-140 |
| `--decode` (repeatable) | `[]` | 143-144 |
| `--prefill <url> [port\|none]` | — | custom-appended, parsed before clap by `split_prefill_args_from_others` 15-52 |
| `--prefill-policy` | none | 147-148 |
| `--decode-policy` | none | 151-152 |
| `--kv-connector` | `nixl` (`nixl|mooncake|moriio`, enum `KvConnector` 536) | 372-373 |
| `--intra-node-data-parallel-size` | `1` | 200-201 |
| `--profile` | `false` | 368-369 |

**Service discovery**

| Flag | Default | Cite (src/main.rs) |
|---|---|---|
| `--service-discovery` | `false` | 244-245 |
| `--selector` `key=v1 key2=v2` | `[]` | 248-249 |
| `--service-discovery-port` | `80` | 252-253 |
| `--service-discovery-namespace` | none | 256-257 |
| `--prefill-selector` | `[]` | 260-261 |
| `--decode-selector` | `[]` | 264-265 |

**Load-balancing / payload / middleware**

| Flag | Default | Cite (src/main.rs) |
|---|---|---|
| `--cache-threshold` | `0.3` | 163-164 |
| `--balance-abs-threshold` | `64` | 167-168 |
| `--balance-rel-threshold` | `1.5` | 171-172 |
| `--eviction-interval` | `120` | 175-176 |
| `--max-tree-size` | `67108864` (2^26) | 179-180 |
| `--max-payload-size` | `536870912` (512 MiB) | 183-184 |
| `--wasm-middleware` | none | 187-188 |
| `--wasm-middleware-sha256` | none | 191-192 |
| `--wasm-middleware-route` (repeatable) | default `/v1/chat/completions` when `--wasm-middleware` is set | 196-197 |
| `--api-key` | none | 204-205 |
| `--api-key-validation-urls` | env fallback `API_KEY_VALIDATION_URLS` | 208-209 |

**Timeouts / concurrency / retry / circuit breaker / health**

| Flag | Default | Cite (src/main.rs) |
|---|---|---|
| `--worker-startup-timeout-secs` | `600` | 155-156 |
| `--worker-startup-check-interval` | `30` | 159-160 |
| `--request-timeout-secs` | `1800` | 280-281 |
| `--max-concurrent-requests` | `32768` | 284-285 |
| `--cors-allowed-origins` | `[]` (empty = allow all) | 288-289 |
| `--retry-max-retries` | `5` | 293-294 |
| `--retry-initial-backoff-ms` | `50` | 297-298 |
| `--retry-max-backoff-ms` | `30000` | 301-302 |
| `--retry-backoff-multiplier` | `1.5` | 305-306 |
| `--retry-jitter-factor` | `0.2` | 309-310 |
| `--disable-retries` | `false` | 313-314 |
| `--cb-failure-threshold` | `10` | 318-319 |
| `--cb-success-threshold` | `3` | 322-323 |
| `--cb-timeout-duration-secs` | `60` | 326-327 |
| `--cb-window-duration-secs` | `120` | 330-331 |
| `--disable-circuit-breaker` | `false` | 334-335 |
| `--health-failure-threshold` | `3` | 339-340 |
| `--health-success-threshold` | `2` | 343-344 |
| `--health-check-timeout-secs` | `5` | 347-348 |
| `--health-check-interval-secs` | `60` | 351-352 |
| `--health-check-endpoint` | `/health` | 355-356 |
| `--request-id-headers` | see defaults below | 276-277 |
| `--log-dir` | none (stdout only) | 216-217 |
| `--log-level` | `info` | 220-221 |

Default request-ID headers: `x-request-id`, `x-correlation-id`, `x-trace-id`, `request-id` (`src/server.rs:1146-1153`; `README.md:172`).

**Prometheus / tracing**

| Flag | Default | Cite (src/main.rs) |
|---|---|---|
| `--prometheus-port` | `29000` | 268-269 |
| `--prometheus-host` | `127.0.0.1` | 272-273 |
| `--enable-trace` | `false` | 224-229 |
| `--otlp-traces-endpoint` | none (→ `OTEL_EXPORTER_OTLP_ENDPOINT`) | 232-233 |
| `--otel-sampling-ratio` | `1.0` | 236-237 |
| `--otel-excluded-paths` | `[/health,/health_generate,/liveness,/readiness]` | 240-241 |

Fixed (non-CLI) defaults assembled in `to_router_config`: `queue_size = 100`, `queue_timeout_secs = 60` (`src/main.rs:600-660` region), `profile_timeout_secs = 10` and `rate_limit_tokens_per_second = None` (`src/main.rs:655-660`), `connection_mode = ConnectionMode::Http` (`src/main.rs:604`). `health_check` and CB/retry come from the CLI values above.

### 3d. `RouterConfig` JSON keys (library / Python binding)

`struct RouterConfig` (`src/config/types.rs:13-93`) fields: `mode`, `connection_mode`, `policy`, `host`, `port`, `max_payload_size`, `request_timeout_secs`, `worker_startup_timeout_secs`, `worker_startup_check_interval_secs`, `intra_node_data_parallel_size`, `api_key`, `api_key_validation_urls`, `discovery`, `metrics`, `log_dir`, `log_level`, `request_id_headers`, `max_concurrent_requests`, `queue_size`, `queue_timeout_secs`, `rate_limit_tokens_per_second`, `cors_allowed_origins`, `retry`, `circuit_breaker`, `disable_retries`, `disable_circuit_breaker`, `health_check`, `enable_igw`, `history_backend`, `enable_profiling`, `profile_timeout_secs`, `kv_connector`, `program_scheduling`.

- `enum RoutingMode` (`src/config/types.rs:566-593`): `regular{worker_urls}`, `openai{worker_urls}`, `vllm_prefill_decode{prefill_urls,decode_urls,prefill_policy?,decode_policy?,discovery_address?}`.
- `enum PolicyConfig` (`src/config/types.rs:645-690`): `random`, `round_robin`, `cache_aware{cache_threshold,balance_abs_threshold,balance_rel_threshold,eviction_interval_secs,max_tree_size}`, `power_of_two{load_check_interval_secs}`, `consistent_hash{virtual_nodes}`, `rendezvous_hash`.
- `enum KvConnector` (`src/config/types.rs:536-547`): `nixl` (default) / `mooncake` / `moriio`.
- `enum HistoryBackend` (`src/config/types.rs:526-533`): `memory` (default) / `none`.
- `struct DiscoveryConfig` (`src/config/types.rs:695-728`) + `Default` (718-728): `port=8000`, `check_interval_secs=120`, empty selectors, `bootstrap_port_annotation="vllm.ai/bootstrap-port"`.
- `struct RetryConfig` (`src/config/types.rs:731-760`): defaults `max_retries=5, initial_backoff_ms=50, max_backoff_ms=30000, backoff_multiplier=1.5, jitter_factor=0.2`.
- `struct HealthCheckConfig` 764, `struct CircuitBreakerConfig` 791, `struct MetricsConfig` 815 (defaults port 29000, host 127.0.0.1 in CLI; `PrometheusConfig::default` `src/metrics.rs:15-21`).
- `ProgramSchedulingConfig` (`src/config/types.rs:107-180`) — see §5/§6.

---

## 4. HTTP API surface

Route table built in `build_app_with_wasm_middleware` (`src/server.rs:797-930`). Every handler calls `authorize_request` first when `api_key_validation_urls` is non-empty (`authorize_request` `src/server.rs:337-405`).

### Health / readiness / introspection (public)

| Method | Path | Handler / cite |
|---|---|---|
| GET | `/liveness` | `src/server.rs:867`; handler 160-168; body `200 OK "OK"` (`src/routers/mod.rs:160-162`) |
| GET | `/readiness` | `src/server.rs:868` |
| GET | `/health` | `src/server.rs:869` |
| GET | `/health_generate` | `src/server.rs:870` (actually generates tokens; tests one P/D pair — `src/routers/http/pd_router.rs:823-907`) |
| GET | `/v1/models` | `src/server.rs:871` |
| GET | `/get_model_info` | `src/server.rs:872` |
| GET | `/get_server_info` | `src/server.rs:873` |

Response shapes (all read this session):

- **Regular `/readiness`** → `200 {"status":"ready","healthy_workers":N,"total_workers":M}`, else `503` (`src/routers/http/router.rs:2387-2411`; `namePath` `Router/readiness`).
- **Regular `/health`** → `200 "All servers healthy"`, else `503 "Unhealthy servers: [...]"` (`src/routers/http/router.rs:2098-2115`).
- **PD `/readiness`** → `200 {"status":"ready","prefill":{"healthy":..,"total":..},"decode":{...}}`; else `503 {"status":"not_ready","reason":"...", ...}` (`src/routers/http/pd_router.rs:1044-1095`; `namePath` `PdRouterBase/readiness`).
- **PD `/health`** direct-URL mode → `200 "All servers healthy"` / `503 "Unhealthy servers: [...]"` (`src/routers/http/pd_router.rs:793-821`). **PD discovery mode overrides `/health`:** until at least one discovered prefill and one decode exist it returns `503 "Waiting for discovered workers: {n} prefill, {m} decode"` (`src/routers/http/vllm_pd_router.rs:2058-2074`; guard `discovery_is_ready` `vllm_pd_router.rs:74-76`).
- **`/v1/models`** → OpenAI list object. For `grpc://` workers it is built from gRPC Control `GetModelInfo` (`src/routers/http/router.rs:2159-2180`; `src/backend/control.rs:113-135` → `{"object":"list","data":[{"id",...,"object":"model","created":0,"owned_by":"vllm"}]}`).

### Inference (protected) — `src/server.rs:800-866`

| Method | Path | Router call |
|---|---|---|
| POST | `/generate` | `route_generate` |
| POST | `/inference/v1/generate` | `route_inference_generate` (vLLM disaggregated generate) |
| POST | `/v1/chat/completions` | `route_chat` |
| POST | `/v1/completions` | `route_completion` |
| POST | `/rerank` | `route_rerank` |
| POST | `/v1/rerank` | `route_rerank` |
| POST | `/v1/responses` | `route_responses` (transparent) |
| POST | `/v1/embeddings` | `route_embeddings` |
| GET | `/v1/responses/{response_id}` | `get_response` |
| POST | `/v1/responses/{response_id}/cancel` | `cancel_response` |
| DELETE | `/v1/responses/{response_id}` | `delete_response` |
| GET | `/v1/responses/{response_id}/input` | `list_response_input_items` |

Handlers `src/server.rs:250-330`. The interface for these is `RouterTrait` (`src/routers/mod.rs:57-183`). `delete_response` / `list_response_input_items` default to `501 Not Implemented` unless the router overrides them (`src/routers/mod.rs:110-128`).

### Admin / control (protected) — `src/server.rs:875-882`

| Method | Path | Handler / response |
|---|---|---|
| POST | `/add_worker?url=<url>` | `add_worker` (407-421) |
| POST | `/remove_worker?url=<url>` | `remove_worker` (436-452) |
| GET | `/list_workers` | `list_workers` → `{"urls":[...]}` (423-434) |
| POST | `/flush_cache` | `flush_cache` (454-461) |
| GET | `/get_loads` | `get_loads` → `{"workers":[{"worker","load"}]}` (463-470; impl `src/routers/http/router.rs:2364-2381`) |
| GET | `/scheduling/diagnostics` | `scheduling_diagnostics`; `404 {"enabled":false,"error":"This Router does not own Program scheduling"}` when not owned (472-489) |

### RESTful worker management (protected) — `src/server.rs:885-890`

| Method | Path | Notes |
|---|---|---|
| POST | `/workers` | body `WorkerConfigRequest`; returns `WorkerApiResponse` or `WorkerErrorResponse` (`src/server.rs:495-535`) |
| GET | `/workers` | `WorkerListResponse`; in single-router mode `{"workers":[...],"total":N,"stats":{"prefill_count","decode_count","regular_count"}}` (`src/server.rs:537-590`) |
| GET | `/workers/{url}` | worker info, or `404 {"error":...,"code":"WORKER_NOT_FOUND"}` (`src/server.rs:592-635`) |
| DELETE | `/workers/{url}` | remove (`src/server.rs:637-667`) |

`struct WorkerConfigRequest` fields (`src/protocols/worker_spec.rs:9-36`): `url` (required), `model_id?`, `priority?` (comment: default 50), `cost?` (comment: default 1.0), `worker_type?` (`"regular"|"prefill"|"decode"`), `bootstrap_port?`, `labels` (map).

### Fallback / unmatched paths

Unmatched paths route to `transparent_proxy_handler` because transparent proxy is hard-enabled (`src/server.rs:1169`; handler `src/server.rs:114-158`), which authorizes, parses the JSON body, and calls `route_transparent`. With transparent proxy disabled the fallback would be `sink_handler` → `404` (`src/server.rs:106-108`).

### Cross-cutting

- Body limits: `DefaultBodyLimit::max(max_payload_size)` + `RequestBodyLimitLayer::new(max_payload_size)` (`src/server.rs:912-917`); default 512 MiB.
- CORS: empty `cors_allowed_origins` → allow all origins/methods/headers/exposed; non-empty → allow-listed `GET/POST/OPTIONS` with content-type+authorization headers, `max_age` 3600 (`create_cors_layer` `src/server.rs:1233-1260`).
- Auth: `Authorization: Bearer <token>`; each configured validation URL is GET'd with the bearer token; HTTP `200` = valid; results cached in memory (`src/server.rs:337-405`). 401 body: `"You must provide a valid API key. Obtain one from http://helmholtz.cloud"` (`src/server.rs:335-336`).
- Listener sets `TCP_NODELAY` on accepted connections to avoid Nagle-induced TTFT delay (`bind_listener` `src/server.rs:1186-1204`).
- Graceful shutdown on Ctrl-C / SIGTERM (`shutdown_signal` `src/server.rs:1206-1230`).

---

## 5. PD-disaggregation behavior

### Mode selection

`--vllm-pd-disaggregation` routes into `RoutingMode::VllmPrefillDecode`. Exactly one of three worker sources must be present or startup fails with a validation error: `--vllm-discovery-address`, or static `--prefill`/`--decode`, or `--service-discovery` together with `--prefill-selector`/`--decode-selector` (`src/main.rs:434-467`). Static URLs plus ZMQ discovery may be combined ("hybrid mode", static as fallback) (`src/main.rs:460-478`).

### Two-stage dispatch (the load-bearing algorithm)

`process_vllm_two_stage_request` (`src/routers/http/vllm_pd_router.rs:1278-1677`; direct-URL mode) and its discovery twin `process_vllm_two_stage_request_discovered` (`:877-1271`):

1. Select a prefill worker and a decode worker via the prefill/decode policies using header-aware `select_worker_with_headers` (`vllm_pd_router.rs:2280-2300`). If either side is empty → `503 "No workers available: {p} prefill, {d} decode"` and `RouterMetrics::record_pd_error("server_selection")` (`vllm_pd_router.rs:2296-2310`).
2. Build a request id `___prefill_addr_{addr}___decode_addr_{addr}_{uuid}` (`generate_vllm_request_id` `vllm_pd_router.rs:484-491`).
3. **Prefill request** is rewritten by `prepare_prefill_request` (`vllm_pd_router.rs:577-618`):
   - `/inference/v1/generate` → set `sampling_params.max_tokens = 1` (and clamp `sampling_params.min_tokens` to ≤1); create `sampling_params{max_tokens:1,min_tokens:1}` if absent.
   - `/v1/responses` → set `max_output_tokens = 1`.
   - other OpenAI endpoints → set `max_tokens = 1` (and `max_completion_tokens = 1` if present), clamp `min_tokens` to ≤1.
   - always set `stream = false` and remove `stream_options`.
4. Inject `kv_transfer_params` on the prefill request (per connector, below) and `POST` to `prefill_worker.endpoint_url(path)` with `Authorization: Bearer $OPENAI_API_KEY` and `X-Request-Id`, plus `X-data-parallel-rank` when DP-aware (`vllm_pd_router.rs:1379-1420`).
5. Parse the prefill JSON response, extract `kv_transfer_params`, then **decode request** = original request + decode-leg `kv_transfer_params`, `POST` to the decode worker (`vllm_pd_router.rs:1500-1560`).

Profiling: with `--profile`, the router calls `POST {worker}/start_profile` then `/stop_profile` around each leg, with a timeout task (`src/routers/http/pd_router.rs:135-200`; `vllm_pd_router.rs:508-566`).

### KV-transfer parameters per connector

Built by `build_prefill_kv_transfer_params` / `build_decode_kv_transfer_params` (`vllm_pd_router.rs:186-320`):

- **NIXL** (default). Prefill params: `{"do_remote_decode":true,"do_remote_prefill":false,"remote_engine_id":null,"remote_block_ids":null,"remote_host":null,"remote_port":null}`. Decode params are copied from the prefill response's `kv_transfer_params`; a push mode (`transfer_mode == "push"`) is supported, where the router caches per-prefill-URL/rank identity `{remote_engine_id, remote_host, remote_port, tp_size, transfer_mode:"push"}` and builds decode params `{do_remote_prefill:true, do_remote_decode:false, remote_engine_id, remote_host, remote_port, tp_size, remote_request_id}` (`vllm_pd_router.rs:186-205,226-300`).
- **Mooncake.** Prefill params: `{"do_remote_decode":true,"do_remote_prefill":false,"transfer_id":..}`. At startup the router queries each prefill node's Mooncake bootstrap server `GET {bootstrap}/query` (bootstrap port default **8998** when not given via `--prefill URL PORT`) to learn `engine_id` per DP rank (`query_mooncake_bootstrap` `vllm_pd_router.rs:96-160`; struct `MooncakePrefillInfo` `:34-38`; startup loop `:1938-1966`). Decode params: `{"do_remote_decode":false,"do_remote_prefill":true,"transfer_id","remote_bootstrap_addr","remote_engine_id"}` (`build_mooncake_decode_kv_transfer_params` `:300-320`).
- **MoRI-IO.** Transfer `READ`/`WRITE` mode is learned from the first service registration (`MoriIOTransferMode` `src/routers/http/vllm_service_discovery.rs:39-45`). Prefill params carry `transfer_id` (prefix constant `MORIIO_TRANSFER_PREFIX = "tx"`, `vllm_pd_router.rs:63`) plus `remote_dp_size`/`remote_tp_size` (WRITE mode; `remote_dp_rank` only when DP>1) — `vllm_pd_router.rs:167-225`. Dispatching with no registered mode is an **error**, not a silent default (`:180-186`).
- Transfer-id generation: Mooncake `xfer-<uuid>`, MoRI-IO `tx-<uuid-simple>`, NIXL none (`generate_transfer_id` `vllm_pd_router.rs:161-174`).

### Worker registration / discovery protocol

Three selectable mechanisms (`src/main.rs:434-467`), combinable with static URLs as fallback:

1. **Static config** — `--prefill`/`--decode` at startup (`src/main.rs:434-467`).
2. **ZMQ push registration (vLLM P2P NCCL path).** The router binds a ZMQ `ROUTER` socket on `tcp://{--vllm-discovery-address}` and receives msgpack multipart registrations (`start_listener` `src/routers/http/vllm_service_discovery.rs:165-230`). Payload `ServiceRegistration { "type": "P"|"D", http_address, zmq_address }` (`:23-31`); MoRI-IO adds `{transfer_mode:"READ"|"WRITE", tp_size, dp_size}` (`:47-64`). Registrations expire after `DEFAULT_PING_SECONDS = 5` (`:15`) and expired entries are pruned each loop (`:334-380`). This is how vLLM's NCCL connector coordinates prefill/decode. The router's request id embeds the discovered ZMQ addresses (`get_zmq_address` `vllm_pd_router.rs:493-516`).
3. **Kubernetes pod watch.** `--service-discovery` + selectors; the router watches `Pod` objects (`kube::runtime::watcher`) and classifies pods prefill/decode/regular by label selector; the prefill **bootstrap port** comes from the `vllm.ai/bootstrap-port` pod annotation (config key `bootstrap_port_annotation`, `DiscoveryConfig::default()` `src/config/types.rs:718-728`; watcher `src/service_discovery.rs`, `ServiceDiscoveryConfig` `:18-51`, `PodInfo::from_pod` `:108-160`). `ServiceDiscoveryConfig.pd_mode` distinguishes PD vs regular (`src/service_discovery.rs:25`). Discovery metrics exist (`vllm_router_discovery_*`, §6).

**Pure discovery mode:** `VllmPDRouter::new` with a discovery address constructs a `PdRouterBase` with empty static worker lists, starts the ZMQ listener, and sets `use_discovery: true` (`vllm_pd_router.rs:1880-1920`); `/health` then depends on `service_registry.get_instance_counts()`.

### DP-aware routing (used for both prefill and decode)

Worker URLs may carry an optional `@rank` suffix (e.g. `http://host:8000@3`); `--intra-node-data-parallel-size > 1` expands each URL into one worker per rank (`get_dp_aware_workers` `src/routers/http/dp_utils.rs:108-138`; `parse_worker_url`/`extract_dp_rank` `:180-215`). Outbound requests to a ranked worker add `X-data-parallel-rank: <n>` (`add_dp_rank_header` `:214-225`). DP size can be auto-discovered from a worker's `/metrics` by counting distinct `engine="..."` labels on `vllm:` series, falling back to `--intra-node-data-parallel-size` when absent (`discover_dp_size`/`count_engine_labels` `dp_utils.rs:13-71`).

### KV index (engine KV events) — separate subsystem

`src/kv_index/` maintains a multi-tier (Device/Host/External) KV-block placement index fed by vLLM engine KV events. Discovery queries `GET {worker}/get_server_info` and `GET {worker}/kv_event_sources`, spawning one ZMQ subscriber per (worker, rank) and one indexer per cache identity (`src/kv_index/discovery.rs:1-2,116-170,360-377`; supervisor `KvIndexSupervisor` `:74`). Block keys use vLLM-compatible `sha256_cbor` sequence hashing (`src/kv_hash.rs:1-60`). `TieredMatchProvider` (`src/kv_index/mod.rs:19-24`) is the read contract the policy/cost model consumes. This subsystem is independent of the `cache_aware` policy tree.

---

## 6. Metrics surface

Prometheus exporter via `metrics-exporter-prometheus` (`start_prometheus` `src/metrics.rs:298-330`), started at boot when `prometheus_config` is present (`src/server.rs:1084-1087`) — the CLI always sets it (`src/main.rs:508-511`). Default listener host `127.0.0.1`, port `29000` (`PrometheusConfig::default` `src/metrics.rs:15-21`; CLI `--prometheus-host`/`--prometheus-port`). Histograms whose name matches `*duration_seconds` get one shared bucket set (`src/metrics.rs:298-306`). The exporter's path is `/metrics` (the `with_http_listener` default — **not asserted in source; verify on a live instance**).

Metric names are registered in `init_metrics` (`src/metrics.rs:23-296`); label sets are read from the `RouterMetrics::record_*`/`set_*` functions (`src/metrics.rs:487-880`).

**Request**

| Metric | Type | Labels |
|---|---|---|
| `vllm_router_requests_total` | counter | `route` |
| `vllm_router_request_duration_seconds` | histogram | `route` |
| `vllm_router_request_errors_total` | counter | `route`,`error_type` |
| `vllm_router_retries_total` | counter | `route` |
| `vllm_router_retry_backoff_duration_seconds` | histogram | `attempt` |
| `vllm_router_retries_exhausted_total` | counter | `route` |

**Circuit breaker**

| Metric | Type | Labels |
|---|---|---|
| `vllm_router_cb_state` | gauge | `worker` (0=closed,1=open,2=half_open) |
| `vllm_router_cb_state_transitions_total` | counter | `worker`,`from`,`to` |
| `vllm_router_cb_outcomes_total` | counter | `worker`,`outcome` |

**Worker / policy**

| Metric | Type | Labels |
|---|---|---|
| `vllm_router_active_workers` | gauge | — |
| `vllm_router_worker_health` | gauge | `worker` |
| `vllm_router_worker_load` | gauge | `worker` |
| `vllm_router_processed_requests_total` | counter | `worker` |
| `vllm_router_policy_decisions_total` | counter | `policy`,`worker` |
| `vllm_router_cache_hits_total` / `vllm_router_cache_misses_total` | counter | — |
| `vllm_router_tree_size` | gauge | `worker` |
| `vllm_router_load_balancing_events_total` | counter | — |
| `vllm_router_max_load`, `vllm_router_min_load` | gauge | — |
| `vllm_router_running_requests` | gauge | `worker` |

**PD-specific** (`record_pd_*` `src/metrics.rs:792-855`)

| Metric | Type | Labels |
|---|---|---|
| `vllm_router_pd_requests_total` | counter | `route` |
| `vllm_router_pd_request_duration_seconds` | histogram | `route` |
| `vllm_router_pd_prefill_requests_total` | counter | `worker` |
| `vllm_router_pd_decode_requests_total` | counter | `worker` |
| `vllm_router_pd_errors_total` | counter | `error_type` |
| `vllm_router_pd_prefill_errors_total` | counter | `worker` |
| `vllm_router_pd_decode_errors_total` | counter | `worker` |
| `vllm_router_pd_stream_errors_total` | counter | `worker` |

**Service discovery** (`record_discovery_update` `src/metrics.rs:857-862`): `vllm_router_discovery_updates_total` (counter), `vllm_router_discovery_workers_added` (gauge), `vllm_router_discovery_workers_removed` (gauge).

**Endpoint / embeddings / generate**: `vllm_router_generate_duration_seconds` (histogram), `vllm_router_embeddings_total` (counter), `vllm_router_embeddings_duration_seconds` (histogram), `vllm_router_embeddings_errors_total` (counter,`error_type`), `vllm_router_embeddings_queue_size` (gauge) (`src/metrics.rs:244-268`).

**Tokenizer** (prefix `vllm_tokenizer_`): `encode_duration_seconds`, `decode_duration_seconds`, `encode_batch_duration_seconds`(`batch_size`), `encode_requests_total`(`tokenizer_type`), `decode_requests_total`(`tokenizer_type`), `encode_errors_total`(`error_type`), `decode_errors_total`(`error_type`), `tokens_per_encode`, `chars_per_encode`, `tokens_per_decode`, `vocab_size`, `stop_sequences_detected_total`(`type`), `partial_matches_total`, `stop_detection_duration_seconds`, `stream_tokens_total`, `stream_incomplete_utf8_total`, `stream_step_duration_seconds`, `factory_loads_total`, `factory_errors_total`, `factory_load_duration_seconds`, `cache_hits_total`, `cache_misses_total`, `cache_evictions_total`, `cache_oversized_total`, `cache_entries`, `cache_bytes` (`src/metrics.rs:270-296`; recorders `:880-1000`).

**Program scheduling** (prefix `vllm_router_agent_aware_`): `rank_queue_size`(`target`), `rank_active_programs`(`target`), `transitions_total`(`target`,`reason`), `queue_wait_seconds`(`target`), `fitted_ttl_seconds`(`target`), `context_growth_tokens`(`target`), `rolling_samples`(`target`,`window`), `token_estimate_coefficient`(`model_pool`,`endpoint`), `token_estimate_feedback_total`(`model_pool`,`endpoint`,`outcome`), `cache_miss_impact_seconds`(`target`), `cache_miss_impact_rolling_mean_seconds`(`target`), `capacity_observation_fresh`(`target`), `request_interval_seconds`(`target`), `armed_ttl_seconds`(`target`,`source`), `shared_prefix_observed_tokens`(`target`), and `admission_*` histograms (`used_tokens`, `required_tokens`, `growth_reserve_tokens`, `capacity_pressure_ratio`, `projected_pressure_ratio`, `backend_kv_usage_ratio`, `backend_running_requests`, `backend_waiting_requests`, `capacity_observation_age_seconds`) — registered `src/metrics.rs:100-200`, recorded `:594-790`.

### Tracing (OTLP)

- Enabled with `--enable-trace`; exporter is OTLP gRPC (tonic) — `prepare_otel` `src/otel_trace.rs:143-200`.
- Resource attributes: `service.name="vllm-router"`, `service.version=<crate version>`, `service.instance.id={host|pid}-{pid}` (`src/otel_trace.rs:178-180`, `service_instance_id` `:116-124`).
- Sampler: parent-based; ratio ≤0 → AlwaysOff, ≥1 → AlwaysOn, else `TraceIdRatioBased` (`parent_based_sampler` `src/otel_trace.rs:134-141`).
- BatchSpanProcessor: scheduled delay 500 ms, max export batch 64 (`src/otel_trace.rs:165-170`).
- Propagators: W3C `TraceContext` + `Baggage` (`src/otel_trace.rs:207-210`).
- Default excluded paths: `/health`, `/health_generate`, `/liveness`, `/readiness` (`TraceConfig::default_excluded_paths` `src/config/types.rs:853-860`).
- OTel is shut down after the server exits so the final batch flushes (`src/main.rs:823-827`).

---

## 7. Container / packaging

**Dockerfile** `Dockerfile.router` (61 lines):

- Builder `rustlang/rust:nightly-bookworm`; installs `build-essential pkg-config libssl-dev`; copies `Cargo.toml Cargo.lock src wit`; `cargo build --release` (`Dockerfile.router:1-19`).
- Runtime `python:3.12-slim-bookworm`; copies the binary to `/usr/local/bin/vllm-router`; copies `py_src` for the Python launcher; installs `requests`, then uninstalls `pip` and removes `ensurepip` (`Dockerfile.router:21-45`).
- `EXPOSE 8080` and `EXPOSE 29000` (`Dockerfile.router:47-51`).
- `CMD ["vllm-router", "--host", "0.0.0.0", "--port", "8080"]` (`Dockerfile.router:58-60`).

**Published image** (`vllm/vllm-router` on DockerHub), from the Buildkite release pipeline:

- Nightly (`NIGHTLY=1`): `vllm/vllm-router:nightly-x86_64`, `:nightly-aarch64`, then a multi-arch `:nightly` manifest (`.buildkite/release-pipeline.yml` nightly groups; scripts `.buildkite/scripts/publish-release-images.sh`, `cleanup-nightly-router-builds.sh`).
- Versioned release (manual input `X.Y.Z`): `vllm/vllm-router:vX.Y.Z-x86_64`, `:vX.Y.Z-aarch64`, then `:vX.Y.Z` + `:latest` (`.buildkite/release-pipeline.yml` versioned groups).
- Premerge CI only builds locally: `docker build -f Dockerfile.router -t vllm-router:${BUILDKITE_COMMIT}` + `:latest` (`.buildkite/pipeline.yml`, Docker Build step).
- `README.md` says images are published using `/docker/Dockerfile.router`; the file present in the indexed tree is root `Dockerfile.router` (see §Gaps).

**PyPI / Python**: package `vllm-router` version `0.1.16`, console script `vllm-router = vllm_router.launch_router:main`, `requires-python >=3.8`, deps `setproctitle,aiohttp,orjson,uvicorn,fastapi,requests>=2.25.0` (`pyproject.toml:11-36`). Python launcher uses the PyO3 `Router` (`py_src/vllm_router/launch_router.py:1-110`).

**Binary name**: `vllm-router` (`Cargo.toml:13-15`). **Ports:** client `30000` default (`src/main.rs:112-113`), `8080` in the container CMD; metrics `29000`.

---

## 8. How workers (vLLM instances) register / are configured

| Mechanism | Contract | Cite |
|---|---|---|
| Static URLs | `--worker-urls` (regular) or `--prefill`/`--decode` (PD) | `src/main.rs:116-117,434-467` |
| ZMQ self-registration | workers push msgpack `{type:"P"\|"D", http_address, zmq_address[, transfer_mode, tp_size, dp_size]}` to the router's `ROUTER` socket bound at `--vllm-discovery-address`; TTL 5 s | `src/routers/http/vllm_service_discovery.rs:15,23-64,165-330` |
| Kubernetes | pod watch by label selector; prefill bootstrap port via the `vllm.ai/bootstrap-port` annotation | `src/service_discovery.rs:18-160`; `src/config/types.rs:718-728` |
| REST | `POST /workers` with `WorkerConfigRequest` | `src/server.rs:495-535`; `src/protocols/worker_spec.rs:9-36` |
| gRPC workers | `grpc://host:port` → vLLM Rust `Inference` API; liveness via `grpc.health.v1` Check with empty service name | `docs/backend/grpc.md:22-25`; `src/backend/detect.rs:1-20` |

Worker pools are homogeneous: all-HTTP or all-gRPC; mixed schemes are rejected at init / `add_worker`, and `grpcs://` is rejected until tonic TLS is enabled (`classify_worker_urls` `src/backend/detect.rs:27-60`). HTTP workers are transparent OpenAI reverse proxies; gRPC workers receive router-rendered `token_ids` and the router adapts the stream back to OpenAI responses (`docs/backend/grpc.md:1-16`). On the gRPC path, active tool-calling and multimodal content currently return `400` (a deliberate capability boundary, `docs/backend/grpc.md:38-52`).

---

## 9. Admin / control API (add / remove worker)

See the Admin and RESTful tables in §4. Load-bearing caveats:

- **The PD router does not implement the generic worker-admin path.** `PdRouterBase::add_worker` returns `Err("PD router requires specific add_prefill_server or add_decode_server methods")` (`src/routers/http/pd_router.rs:762-768`; `namePath` `PdRouterBase/add_worker`). So `POST /add_worker?url=` is ineffective for a PD router; PD workers are added at startup or via discovery. Specific methods exist: `PdRouterBase::add_prefill_server(url, bootstrap_port)` / `add_decode_server` (`pd_router.rs:269-276`), and `VllmPDRouter::add_prefill_server` / `add_decode_server` / `remove_prefill_server` wrap them (`vllm_pd_router.rs:2025-2046`).
- `VllmPDRouter` delegates `add_worker`/`remove_worker`/`get_worker_urls` to the base and evicts cached NIXL push identity on removal (`impl WorkerManagement for VllmPDRouter` `src/routers/http/vllm_pd_router.rs:2702-2715`).
- In IGW mode (`--enable-igw`), a `RouterManager` owns the worker registry and registers routers `http-regular` and `http-pd`; `POST /workers` is served by `RouterManager::add_worker` (`src/server.rs:1095-1140`; `src/routers/router_manager.rs:1-8`).
- Legacy vs RESTful naming coexist: `/add_worker`,`/remove_worker`,`/list_workers` (query-param, plain-text-ish) alongside `/workers` (JSON). Single-router `/workers` reports per worker `url, model_id, worker_type, is_healthy, load, connection_mode, priority, cost` (`src/server.rs:537-590`).
- `POST /flush_cache` proxies to workers' cache-flush endpoint; `GET /get_loads` returns `{"workers":[{"worker","load"}]}` (`src/routers/http/router.rs:2364-2381`).

---

## Gaps / not confirmed

1. **deepwiki cross-check not performed.** The task allows it only as a cross-check, and every assertion here is grounded in the indexed revision, so no deepwiki text was used or copied.
2. **Prometheus HTTP path** (`/metrics`) is the `metrics-exporter-prometheus` `with_http_listener` default, not asserted in source (`src/metrics.rs:298-330`). Confirm against a live instance.
3. **Codekeeper enumeration quirks:** `findFile` with a `src/**/*.rs` glob returned empty against this revision, and `listDir` truncates at ~12 rows; §2's module list is anchored on `src/lib.rs`. `searchForPattern` rejects a file `relativePath` (`ENOTDIR`).
4. **`to_router_config` fixed-default line numbers** (`queue_size=100`, `queue_timeout_secs=60`, `profile_timeout_secs=10`, `connection_mode=Http`) are cited as the `src/main.rs:600-660` region rather than exact per-line, because that block was read as part of the full-file read.
5. **Program scheduling** is summarized (enable switch, `ProgramSchedulingConfig` field/default surface `src/config/types.rs:107-180`, enable-key contract `docs/program_scheduling.md:5-20`, metric names). Its scheduler internals (`src/program_scheduling/*`) were not exhaustively read; out of scope for this contract map.
6. **gRPC tool-calling / multimodal `400`** is current design (`docs/backend/grpc.md:38-52`), a capability boundary not a defect.
7. The `README.md`-claimed `/docker/Dockerfile.router` path is not present in the indexed tree; the actual file is root `Dockerfile.router` (both cited in §7).
