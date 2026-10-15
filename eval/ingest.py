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
import re
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


def canonical_prompt(arg0: Any, fallback: str) -> str:
    """Clean, JSON-parseable prompt for storage.

    The harness stores ``arguments.gen_args_0.arg_0`` as a Python ``str()`` of a
    list wrapping a JSON chat string (single-quoted, with ``\\'`` escapes for
    apostrophes) — unreadable and not valid JSON. Recover the chat message list
    and re-serialize as clean JSON so the UI can render it directly; fall back
    to the raw string when it isn't a chat prompt (completions).
    """
    value = arg0
    while isinstance(value, list) and len(value) == 1:
        value = value[0]
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return fallback
    if isinstance(value, list) and all(isinstance(m, dict) for m in value):
        return json.dumps(value, ensure_ascii=False)
    return fallback


def _last_run_segment(rows: list[dict]) -> list[dict]:
    """Keep only the final run's rows from a possibly concatenated sidecar.

    ``cmd_run`` renames ``telemetry.jsonl`` to ``telemetry_<timestamp>.jsonl``
    when a run completes (``cli.py::_stamp_telemetry``). If a run CRASHES before
    that rename, the file is left in place and the NEXT run appends its rows to
    it. ``tel_eval/sink.py::next_ordinal()`` is a per-process counter starting
    at 1, so a row whose numeric ``ordinal`` is strictly lower than the previous
    row's numeric ``ordinal`` marks the start of a new run. Without dropping the
    earlier segment, the dead run's rows (same ``prompt_hash`` values, erroring)
    would win ``_match_telemetry``'s first-unconsumed-row matching over the live
    run's rows.

    Rows without a numeric ``ordinal`` never trigger a reset and stay in the
    current segment. Returns the whole list when there is no reset.
    """
    start = 0
    prev: float | None = None
    for i, row in enumerate(rows):
        ordinal = row.get("ordinal")
        if isinstance(ordinal, bool) or not isinstance(ordinal, (int, float)):
            continue
        if prev is not None and ordinal < prev:
            start = i
        prev = ordinal
    return rows[start:]


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
    # A crash skips the rename, so the next run appends to the same
    # ``telemetry.jsonl``; keep only the current (last) run's rows so the dead
    # run's erroring rows cannot win matching over the live run's rows.
    rows = _last_run_segment(rows)
    return rows


def _match_telemetry(
    samples: list[dict], telemetry: list[dict], *, allow_ordinal: bool
) -> tuple[dict[int, dict], int]:
    """Map sample index -> telemetry row. Returns (mapping, unmatched_count).

    Matches across the WHOLE run (caller passes all samples of every task with a
    globally-unique index) so repeated prompts cannot steal rows across tasks.
    Primary key: the harness ``prompt_hash``; secondary: exact ``prompt_text``
    (for sidecars written before the plugin emitted ``prompt_hash``). Fallback:
    ordinal order, ONLY when ``allow_ordinal`` (``num_concurrent == 1``) — with
    concurrency the harness length-sorts requests before dispatch, so ordinal
    order is not doc order and pairing would attach the wrong telemetry.
    """
    def bucketize(tel_key: str) -> dict[str, list[dict]]:
        out: dict[str, list[dict]] = {}
        for row in telemetry:
            # Outage rows are mid-flight retry markers: they share the
            # request's prompt_hash but carry no completion, and precede the
            # request's final ok row in file order. Excluding them here stops a
            # retried-then-succeeded request from attaching its *outage* row
            # (empty content) instead of its real result. They fall through to
            # the orphan lane below, preserving the event.
            if row.get("status") == "outage":
                continue
            k = row.get(tel_key) or ""
            if k:
                out.setdefault(k, []).append(row)
        return out

    used: set[int] = set()
    mapping: dict[int, dict] = {}

    def consume(sample_key: str, tel_key: str) -> None:
        by_key = bucketize(tel_key)
        for i, s in enumerate(samples):
            if i in mapping:
                continue
            key = s.get(sample_key) or ""
            if not key:
                continue
            for row in by_key.get(key, []):
                if id(row) not in used:
                    mapping[i] = row
                    used.add(id(row))
                    break

    # prompt_hash is per-request exact; prompt_text is the legacy fallback.
    consume("_prompt_hash", "prompt_hash")
    consume("_prompt_text", "prompt_text")
    # Outage rows are excluded from matching (see bucketize); count only the
    # correlation-eligible rows so `unmatched` is not inflated by them.
    eligible = [r for r in telemetry if r.get("status") != "outage"]
    unmatched = len(eligible) - len(mapping)
    if not mapping and eligible and samples and allow_ordinal:
        ordered = sorted(eligible, key=lambda r: r.get("ordinal") or 0)
        for i, s in enumerate(samples):
            if i < len(ordered):
                mapping[i] = ordered[i]
        unmatched = len(eligible) - len(mapping)
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
    """Insert (or replace) one run from its results JSON.

    Returns True when written, False when the stored run is already current.
    The harness writes ``results_*.json`` only at run end, so a ``--watch``
    pass re-discovers completed runs every poll; skipping unchanged ones keeps
    the watcher cheap and stops row-id churn (which would reset the Live
    window and dangle open sample URLs).
    """
    source_path = str(ref.results_file.resolve())
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, started_at FROM eval_run WHERE source_path = %s", (source_path,)
        )
        existing = cur.fetchone()
        if existing:
            from datetime import datetime, timezone

            stored = existing[1]
            incoming = (
                datetime.fromtimestamp(ref.date, tz=timezone.utc) if ref.date else None
            )
            if stored == incoming:
                return False  # same results file already ingested
            # Results advanced (re-run extended the file): replace wholesale.
            # Children cascade-delete.
            cur.execute("DELETE FROM eval_run WHERE id = %s", (existing[0],))
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

        # Collect every inserted sample across ALL tasks, then attach telemetry
        # once per run (global match) so repeated prompts can't steal rows
        # across tasks via a fresh per-task `used` set.
        all_samples: list[dict] = []
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
            for s, rec in zip(t.samples, records):
                arg0 = (s.get("arguments") or {}).get("gen_args_0", {}).get("arg_0", "")
                prompt_text = extract_prompt_text(arg0)
                prompt_json = canonical_prompt(arg0, rec["prompt"])
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
                        prompt_json,
                        rec["output"],
                        rec["reference"],
                        json.dumps(rec["choices"]),
                        rec["picked"],
                        rec["correct"],
                        prompt_text,
                    ),
                )
                all_samples.append(
                    {
                        "_sample_id": cur.fetchone()[0],
                        "_task": t.task,
                        "_prompt_text": prompt_text,
                        "_prompt_hash": str(s.get("prompt_hash") or ""),
                    }
                )

        # Telemetry attach: one global match per run.
        samples_dir = ref.results_file.parent
        telemetry = load_telemetry(samples_dir, ref.timestamp)
        if telemetry and all_samples:
            mapping, unmatched = _match_telemetry(
                all_samples, telemetry, allow_ordinal=_num_concurrent(ref) <= 1
            )
            for i, tel in mapping.items():
                _insert_telemetry(cur, all_samples[i]["_sample_id"], tel, ref.run_name)
            if unmatched:
                _insert_orphan_telemetry(cur, run_id, telemetry, set(map(id, mapping.values())))
                print(
                    f"[ingest] {ref.run_name}: {unmatched} telemetry row(s) uncorrelated",
                    file=sys.stderr,
                )
    conn.commit()
    return True


def _insert_orphan_telemetry(
    cur: psycopg.Cursor, run_id: Any, telemetry: list[dict], used: set[int]
) -> None:
    """Persist telemetry rows that matched no sample into the orphan lane."""
    for tel in telemetry:
        if id(tel) in used:
            continue
        cur.execute(
            """
            INSERT INTO eval_telemetry_orphan (run_id, task, request_hash, prompt_hash,
                ordinal, wall_ms, ttfb_ms, ttft_ms, prompt_tokens, completion_tokens,
                total_tokens, cache_read_tokens, cache_write_tokens, reasoning_content,
                content, status, error, images)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """,
            (
                run_id,
                tel.get("task"),
                tel.get("request_hash"),
                tel.get("prompt_hash"),
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
                tel.get("error"),
                json.dumps(tel.get("images") or []),
            ),
        )


def _normalize_image_paths(images: list, run_name: str) -> list:
    """Rewrite run-dir-relative image paths to results-root-relative.

    The console serves images from the results root, so a stored path must be
    ``<run-name>/images/<file>``. Older sidecars wrote ``images/<file>``; the
    current plugin already writes the run-name-prefixed form, so only prefix a
    path that doesn't already match the expected shape (validated structurally,
    not by run-name prefix match — a run literally named ``images`` would
    otherwise double-prefix).
    """
    shaped = re.compile(rf"^{re.escape(run_name)}/images/[A-Za-z0-9_.-]+$")
    out = []
    for img in images:
        if not isinstance(img, dict):
            continue
        path = img.get("path")
        if isinstance(path, str) and path and not shaped.match(path):
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
    """Ingest new/changed runs under ``results_dir``.

    Returns the number of runs actually written. Runs whose stored
    ``started_at`` already matches the results file's ``date`` are skipped, so
    repeated polling (``--watch``) is cheap and doesn't churn row ids.
    """
    rdir = Path(results_dir)
    if not rdir.is_absolute() and not rdir.exists():
        # CWD-relative first (path already exists); else fall back to eval/ dir.
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
    # Resolve once so the startup line proves which absolute dir is watched
    # (--results resolution is CWD-first, then eval dir).
    rdir = Path(results_dir)
    if not rdir.is_absolute() and not rdir.exists():
        rdir = SCRIPT_DIR / rdir
    shown = rdir.resolve()
    print(f"[ingest] watching {shown} every {interval}s (Ctrl-C to stop)", flush=True)
    total = 0
    passes = 0
    try:
        while True:
            passes += 1
            try:
                n = discover_and_ingest(results_dir, dsn)
                total += n
                if n:
                    print(f"[ingest] pass {passes}: wrote {n} run(s)", flush=True)
                elif passes % 12 == 0:
                    # Heartbeat every ~interval*12s so a quiet log still proves
                    # the watcher is alive (not wedged).
                    print(
                        f"[ingest] pass {passes}: idle ({total} written this session)",
                        flush=True,
                    )
            except psycopg.Error as exc:
                print(f"[ingest] DB error: {exc}", file=sys.stderr, flush=True)
            time.sleep(interval)
    except KeyboardInterrupt:
        print(f"\n[ingest] stopped; {total} run(s) ingested this session", flush=True)
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
