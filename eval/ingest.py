#!/usr/bin/env python3
"""Ingest lm-eval run output into the console Postgres DB.

Discovers ``results_*.json`` (+ sibling ``samples_*.jsonl``) under a results
dir, normalizes each via ``report.load_run``, and upserts into the ``eval_*``
tables (see ``db/schema.sql`` / the console repo's migration). Also attaches
per-sample telemetry from ``telemetry*.jsonl`` sidecars written by the
``tel_eval`` plugin, correlating on an extracted prompt text (falling back to
request ordinal).

Idempotent: re-ingesting the same results dir inserts nothing new.

Usage (normally via ``eval.sh ingest``)::

    python ingest.py --results results [--watch] [--dsn postgresql://...]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import psycopg  # noqa: E402

from report import discover_runs, load_run, sample_record  # noqa: E402

DEFAULT_DSN = "postgresql://postgres:postgres@localhost:5442/lmeval"


def extract_prompt_text(arg0: Any) -> str:
    """First text content of a chat-templated argument, or the raw string.

    Handles the harness shapes for ``arguments.gen_args_0.arg_0``: a plain
    string (completions), a JSON string of a chat list, or a list wrapping that
    JSON string (chat path). Mirrors the plugin's ``_prompt_text_from_payload``
    so telemetry rows join on this key.
    """
    value = arg0
    # Unwrap single-element lists (chat path stores List[JsonChatStr]).
    while isinstance(value, list) and len(value) == 1:
        value = value[0]
    if isinstance(value, list):
        return _first_text(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value
        if isinstance(parsed, list):
            return _first_text(parsed)
        return value
    return str(value or "")


def _first_text(messages: list) -> str:
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    return str(part.get("text") or "")
    return ""


def load_telemetry(samples_dir: Path, timestamp: str) -> list[dict]:
    """Read the telemetry rows for one run.

    ``cmd_run`` renames the run's ``telemetry.jsonl`` to
    ``telemetry_<timestamp>.jsonl`` in the run output dir on completion (samples
    live in the model-slug subdir), so prefer the timestamp-matched file and
    fall back to the unmatched ``telemetry.jsonl`` for live/in-flight runs.
    """
    candidates = [
        samples_dir / f"telemetry_{timestamp}.jsonl",
        samples_dir.parent / f"telemetry_{timestamp}.jsonl",
        samples_dir / "telemetry.jsonl",
        samples_dir.parent / "telemetry.jsonl",
    ]
    rows: list[dict] = []
    for path in candidates:
        if not path.exists():
            continue
        try:
            with path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
        except (OSError, json.JSONDecodeError):
            continue
        if rows:
            break
    return rows


def _match_telemetry(
    samples: list[dict], telemetry: list[dict], *, allow_ordinal: bool
) -> tuple[dict[int, dict], int]:
    """Map sample index -> telemetry row. Returns (mapping, unmatched_count).

    Primary key: exact ``prompt_text`` match. Fallback: ordinal order, ONLY when
    ``allow_ordinal`` (``num_concurrent == 1``) — with concurrency the harness
    length-sorts requests before dispatch, so ordinal order is not doc order and
    pairing would attach the wrong telemetry.
    """
    by_prompt: dict[str, list[dict]] = {}
    for row in telemetry:
        by_prompt.setdefault(row.get("prompt_text") or "", []).append(row)

    mapping: dict[int, dict] = {}
    used: set[int] = set()
    for i, s in enumerate(samples):
        key = s["_prompt_text"]
        bucket = by_prompt.get(key)
        if bucket:
            for j, row in enumerate(bucket):
                if id(row) not in used:
                    mapping[i] = row
                    used.add(id(row))
                    break
    unmatched = len(telemetry) - len(mapping)
    if not mapping and telemetry and samples and allow_ordinal:
        ordered = sorted(telemetry, key=lambda r: r.get("ordinal") or 0)
        for i, row in enumerate(ordered[: len(samples)]):
            mapping[i] = row
        unmatched = len(telemetry) - len(mapping)
    return mapping, unmatched


def _num_concurrent(ref) -> int:
    """Read ``model_args.num_concurrent`` from the run's results JSON (default 1)."""
    try:
        data = json.loads(ref.results_file.read_text())
        margs = (data.get("config") or {}).get("model_args") or {}
        return int(margs.get("num_concurrent") or 1)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return 1


def ingest_run(conn: psycopg.Connection, ref, run) -> bool:
    """Upsert one run. Returns True when newly inserted, False if it existed."""
    source_path = str(ref.results_file.resolve())
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM eval_run WHERE source_path = %s", (source_path,))
        row = cur.fetchone()
        if row:
            return False
        started_at = None
        if ref.date:
            from datetime import datetime, timezone

            started_at = datetime.fromtimestamp(ref.date, tz=timezone.utc)

        snapshot_path = ref.results_file.parent / "config_snapshot.json"
        snapshot = json.loads(snapshot_path.read_text()) if snapshot_path.exists() else None

        cur.execute(
            """
            INSERT INTO eval_run (run_name, model, model_name, backend, base_url,
                harness_git, lm_eval_version, transformers_version, started_at,
                total_seconds, task_count, sample_count, config_snapshot, source_path)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            RETURNING id
            """,
            (
                run.meta["run_name"],
                run.meta["model"],
                run.meta["model_name"],
                run.meta["backend"],
                run.meta["base_url"],
                run.meta["harness_git"],
                run.meta["lm_eval_version"],
                run.meta["transformers_version"],
                started_at,
                run.meta["total_seconds"],
                run.meta["task_count"],
                run.meta["sample_count"],
                json.dumps(snapshot) if snapshot is not None else None,
                source_path,
            ),
        )
        run_id = cur.fetchone()[0]

        for t in run.tasks:
            cur.execute(
                """
                INSERT INTO eval_task (run_id, task, output_type, num_fewshot, repeats,
                    dataset_path, test_split, generation_kwargs, filter_names, description,
                    unsafe_code, coverage_original, coverage_effective, counts)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                RETURNING id
                """,
                (
                    run_id,
                    t.task,
                    str(t.config.get("output_type") or ""),
                    int(t.config.get("num_fewshot") or 0),
                    int(t.config.get("repeats") or 1),
                    str(t.config.get("dataset_path") or ""),
                    str(t.config.get("test_split") or ""),
                    json.dumps(t.config.get("generation_kwargs") or {}),
                    json.dumps(
                        [str(f.get("name")) for f in (t.config.get("filter_list") or [])]
                    ),
                    str(t.config.get("description") or ""),
                    bool(t.unsafe_code),
                    int(t.coverage.get("original") or 0),
                    int(t.coverage.get("effective") or 0),
                    json.dumps(_outcome_counts(t)),
                ),
            )
            task_id = cur.fetchone()[0]

            for m in t.rows:
                cur.execute(
                    """
                    INSERT INTO eval_metric (task_id, metric, grp, value, stderr, higher_is_better)
                    VALUES (%s,%s,%s,%s,%s,%s)
                    """,
                    (task_id, m.metric, m.group, m.value, m.stderr, m.higher_is_better),
                )

            records = [sample_record(s, t.config) for s in t.samples]
            sample_ids: list[Any] = []
            prompt_texts: list[str] = []
            for s, rec in zip(t.samples, records):
                arg0 = (s.get("arguments") or {}).get("gen_args_0", {}).get("arg_0", "")
                prompt_text = extract_prompt_text(arg0)
                cur.execute(
                    """
                    INSERT INTO eval_sample (task_id, doc_id, filter, prompt_hash, doc_hash,
                        target_hash, outcome, scores, prompt, output, reference, choices,
                        picked, correct, prompt_text)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    RETURNING id
                    """,
                    (
                        task_id,
                        int(s.get("doc_id") or 0),
                        str(s.get("filter") or "none"),
                        s.get("prompt_hash"),
                        s.get("doc_hash"),
                        s.get("target_hash"),
                        rec["outcome"],
                        json.dumps(rec["scores"]),
                        rec["prompt"],
                        rec["output"],
                        rec["reference"],
                        json.dumps(rec["choices"]),
                        rec["picked"],
                        rec["correct"],
                        prompt_text,
                    ),
                )
                sample_ids.append(cur.fetchone()[0])
                prompt_texts.append(prompt_text)

            # Telemetry attach (uses the run's samples dir sidecar).
            samples_dir = ref.results_file.parent
            telemetry = load_telemetry(samples_dir, ref.timestamp)
            if telemetry and sample_ids:
                annotated = [{"_prompt_text": pt} for pt in prompt_texts]
                mapping, unmatched = _match_telemetry(
                    annotated, telemetry, allow_ordinal=_num_concurrent(ref) <= 1
                )
                for i, tel in mapping.items():
                    if i >= len(sample_ids):
                        continue
                    _insert_telemetry(cur, sample_ids[i], tel, ref.run_name)
                if unmatched:
                    print(
                        f"[ingest] {t.task}: {unmatched} telemetry row(s) uncorrelated",
                        file=sys.stderr,
                    )
    conn.commit()
    return True


def _normalize_image_paths(images: list, run_name: str) -> list:
    """Rewrite run-dir-relative image paths to results-root-relative.

    The console serves images from the results root, so a stored path must be
    ``<run-name>/images/<file>``. Older sidecars wrote ``images/<file>``; the
    current plugin already writes the run-name-prefixed form, so only prefix
    when missing.
    """
    out = []
    for img in images:
        if not isinstance(img, dict):
            continue
        path = img.get("path")
        if isinstance(path, str) and path and not path.startswith(f"{run_name}/"):
            img = {**img, "path": f"{run_name}/{path.lstrip('/')}"}
        out.append(img)
    return out


def _insert_telemetry(cur: psycopg.Cursor, sample_id, tel: dict, run_name: str) -> None:
    images = _normalize_image_paths(tel.get("images") or [], run_name)
    cur.execute(
        """
        INSERT INTO eval_sample_telemetry (sample_id, request_hash, ordinal, wall_ms,
            ttfb_ms, ttft_ms, prompt_tokens, completion_tokens, total_tokens,
            cache_read_tokens, cache_write_tokens, reasoning_content, content, status, images)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        ON CONFLICT (sample_id) DO NOTHING
        """,
        (
            sample_id,
            tel.get("request_hash"),
            tel.get("ordinal"),
            tel.get("wall_ms"),
            tel.get("ttfb_ms"),
            tel.get("ttft_ms"),
            tel.get("prompt_tokens"),
            tel.get("completion_tokens"),
            tel.get("total_tokens"),
            tel.get("cache_read_tokens"),
            tel.get("cache_write_tokens"),
            tel.get("reasoning_content"),
            tel.get("content"),
            tel.get("status"),
            json.dumps(images),
        ),
    )


def _outcome_counts(task) -> dict[str, int]:
    counts = {"pass": 0, "fail": 0, "partial": 0}
    for s in task.samples:
        rec = sample_record(s, task.config)
        counts[rec["outcome"]] += 1
    return counts


def discover_and_ingest(results_dir: Path | str, dsn: str) -> int:
    """Ingest every run under ``results_dir``. Returns count newly inserted."""
    rdir = Path(results_dir)
    if not rdir.is_absolute():
        rdir = SCRIPT_DIR / rdir
    # If pointed straight at a run dir (results/<run-name>), walk up so run_name
    # derives from the run-name dir, not the model slug subdir below it.
    if (rdir / "config_snapshot.json").exists() or (rdir / "telemetry.jsonl").exists():
        rdir = rdir.parent
    refs = discover_runs(rdir)
    inserted = 0
    with psycopg.connect(dsn, autocommit=False) as conn:
        for ref in refs:
            run = load_run(ref)
            if ingest_run(conn, ref, run):
                inserted += 1
    return inserted


def watch(results_dir: Path | str, dsn: str, interval: float = 5.0) -> int:
    """Poll the results dir and ingest new runs until interrupted (LIVE)."""
    print(f"[ingest] watching {results_dir} every {interval}s (Ctrl-C to stop)")
    total = 0
    try:
        while True:
            try:
                total += discover_and_ingest(results_dir, dsn)
            except psycopg.Error as exc:
                print(f"[ingest] DB error: {exc}", file=sys.stderr)
            time.sleep(interval)
    except KeyboardInterrupt:
        print(f"\n[ingest] stopped; {total} run(s) ingested this session")
    return total


def main() -> int:
    p = argparse.ArgumentParser(description="ingest eval results into Postgres")
    p.add_argument("--results", default="results")
    p.add_argument("--watch", action="store_true")
    p.add_argument("--interval", type=float, default=5.0)
    p.add_argument("--dsn", default=None)
    args = p.parse_args()
    dsn = args.dsn or os.environ.get("EVAL_DB_DSN") or DEFAULT_DSN
    if args.watch:
        watch(args.results, dsn, args.interval)
        return 0
    n = discover_and_ingest(args.results, dsn)
    print(f"[ingest] inserted {n} run(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
