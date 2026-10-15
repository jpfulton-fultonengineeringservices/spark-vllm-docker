-- eval rig → console Postgres schema (shape the Python ingester targets).
-- Authoritative migration lives in the console repo (agents/lm-eval-console
-- migrations/0001-initial.ts); this file documents the same shape for the rig
-- and can be applied standalone. Idempotent.

CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS eval_run (
    id                uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    run_name          text NOT NULL,
    model             text NOT NULL,
    model_name        text,
    backend           text,
    base_url          text,
    harness_git       text,
    lm_eval_version   text,
    transformers_version text,
    started_at        timestamptz,
    total_seconds     numeric,
    task_count        integer,
    sample_count      integer,
    config_snapshot   jsonb,
    source_path       text NOT NULL UNIQUE,
    ingested_at       timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS eval_task (
    id                 uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id             uuid NOT NULL REFERENCES eval_run(id) ON DELETE CASCADE,
    task               text NOT NULL,
    output_type        text,
    num_fewshot        integer,
    repeats            integer,
    dataset_path       text,
    test_split         text,
    generation_kwargs  jsonb,
    filter_names       jsonb,
    description        text,
    unsafe_code        boolean DEFAULT false,
    coverage_original  integer,
    coverage_effective integer,
    counts             jsonb,
    UNIQUE (run_id, task)
);

CREATE TABLE IF NOT EXISTS eval_metric (
    id                 uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    task_id            uuid NOT NULL REFERENCES eval_task(id) ON DELETE CASCADE,
    metric             text NOT NULL,
    grp                text NOT NULL DEFAULT 'none',
    value              double precision,
    stderr             double precision,
    higher_is_better   boolean
);

CREATE TABLE IF NOT EXISTS eval_sample (
    id           uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    task_id      uuid NOT NULL REFERENCES eval_task(id) ON DELETE CASCADE,
    doc_id       integer NOT NULL,
    filter       text NOT NULL DEFAULT 'none',
    prompt_hash  text,
    doc_hash     text,
    target_hash  text,
    outcome      text,
    scores       jsonb,
    prompt       text,
    output       text,
    reference    text,
    choices      jsonb,
    picked       integer,
    correct      integer,
    prompt_text  text,
    UNIQUE (task_id, doc_id, filter)
);

CREATE TABLE IF NOT EXISTS eval_sample_telemetry (
    id                  uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    sample_id           uuid NOT NULL UNIQUE REFERENCES eval_sample(id) ON DELETE CASCADE,
    request_hash        text,
    ordinal             integer,
    wall_ms             double precision,
    ttfb_ms             double precision,
    ttft_ms             double precision,
    prompt_tokens       integer,
    completion_tokens   integer,
    total_tokens        integer,
    cache_read_tokens   integer,
    cache_write_tokens  integer,
    reasoning_content   text,
    content             text,
    status              text,
    images              jsonb
);

CREATE INDEX IF NOT EXISTS eval_run_run_name_idx ON eval_run (run_name);
CREATE INDEX IF NOT EXISTS eval_task_run_idx ON eval_task (run_id);
CREATE INDEX IF NOT EXISTS eval_metric_task_idx ON eval_metric (task_id);
CREATE INDEX IF NOT EXISTS eval_sample_task_idx ON eval_sample (task_id);
CREATE INDEX IF NOT EXISTS eval_sample_prompt_text_idx ON eval_sample (prompt_text);
CREATE INDEX IF NOT EXISTS eval_sample_prompt_hash_idx ON eval_sample (prompt_hash);
CREATE INDEX IF NOT EXISTS eval_sample_telemetry_ordinal_idx ON eval_sample_telemetry (ordinal);

-- Orphan telemetry lane: rows that could not be correlated to a sample are
-- stored here (not dropped), so the UI can surface uncorrelated counts.
CREATE TABLE IF NOT EXISTS eval_telemetry_orphan (
    id                 uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id             uuid NOT NULL REFERENCES eval_run(id) ON DELETE CASCADE,
    task               text,
    request_hash       text,
    prompt_hash        text,
    ordinal            integer,
    wall_ms            double precision,
    ttfb_ms            double precision,
    ttft_ms            double precision,
    prompt_tokens      integer,
    completion_tokens  integer,
    total_tokens       integer,
    cache_read_tokens  integer,
    cache_write_tokens integer,
    reasoning_content  text,
    content            text,
    status             text,
    error              text,
    images             jsonb,
    created_at         timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS eval_telemetry_orphan_run_idx ON eval_telemetry_orphan (run_id);
