from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from exl3pack.cli import build_parser
from exl3pack.logconfig import JsonlFormatter, get_logger, log_event


def _parse(argv: list[str]):
    parser = build_parser()
    return parser.parse_args(argv)


def test_jsonl_formatter_required_keys(tmp_path: Path) -> None:
    log = get_logger("coord-test", log_dir=tmp_path)
    log_event(
        log,
        logging.INFO,
        "worker.shard_done",
        fields={"shard": 3, "module": "model.layers.0"},
    )
    line = (tmp_path / "coord-test.jsonl").read_text(encoding="utf-8").splitlines()[-1]
    record = json.loads(line)
    assert set(record) >= {"ts", "level", "event", "node", "pid", "fields"}
    assert record["level"] == "INFO"
    assert record["event"] == "worker.shard_done"
    assert isinstance(record["pid"], int)
    assert record["pid"] == os.getpid()
    assert record["fields"] == {"shard": 3, "module": "model.layers.0"}


def test_jsonl_ts_is_iso8601_utc(tmp_path: Path) -> None:
    log = get_logger("ts-test", log_dir=tmp_path)
    log_event(log, logging.WARNING, "worker.bad_spec", fields={"spec": "x.json"})
    line = (tmp_path / "ts-test.jsonl").read_text(encoding="utf-8").splitlines()[-1]
    record = json.loads(line)
    import datetime

    parsed = datetime.datetime.fromisoformat(record["ts"])
    assert parsed.tzinfo == datetime.UTC


def test_jsonl_non_serializable_fields_fallback(tmp_path: Path) -> None:
    log = get_logger("ser-test", log_dir=tmp_path)
    log_event(log, logging.INFO, "worker.start", fields={"exotic": {1, 2}})
    line = (tmp_path / "ser-test.jsonl").read_text(encoding="utf-8").splitlines()[-1]
    record = json.loads(line)
    assert record["fields"]["exotic"]  # str() fallback, still JSONL-valid


def test_log_event_without_log_dir_writes_stream_only(tmp_path: Path, capsys) -> None:
    log = get_logger("stream-only", log_dir=None)
    log_event(log, logging.INFO, "worker.start", fields={"device": 0})
    # No file should exist.
    assert not (tmp_path / "stream-only.jsonl").exists()
    err = capsys.readouterr().err
    assert "[stream-only] worker.start" in err
    assert "device=0" in err


def test_get_logger_appends_multiple_events(tmp_path: Path) -> None:
    log = get_logger("append-test", log_dir=tmp_path)
    for i in range(3):
        log_event(log, logging.INFO, "worker.shard_done", fields={"shard": i})
    lines = (tmp_path / "append-test.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    records = [json.loads(ln) for ln in lines]
    assert [r["fields"]["shard"] for r in records] == [0, 1, 2]


def test_jsonl_formatter_direct_formatrecord(tmp_path: Path) -> None:
    record = logging.LogRecord(
        name="exl3pack.coord",
        level=logging.ERROR,
        pathname="",
        lineno=0,
        msg="coordinator.error",
        args=(),
        exc_info=None,
    )
    record.event = "coordinator.error"  # type: ignore[attr-defined]
    record.fields = {"detail": "boom"}  # type: ignore[attr-defined]
    formatter = JsonlFormatter()
    out = json.loads(formatter.format(record))
    assert out["level"] == "ERROR"
    assert out["event"] == "coordinator.error"
    assert out["fields"] == {"detail": "boom"}


def test_dist_coordinator_parses_log_dir() -> None:
    args = _parse(
        [
            "dist-coordinator",
            "--source",
            "/nas/src",
            "--work",
            "/nas/work",
            "--exl3-out",
            "/nas/out",
            "--recipe",
            "/nas/work/recipe.yaml",
            "--nodes",
            "gx10-1",
            "--log-dir",
            "/nas/work/dist/logs/run-42",
        ]
    )
    assert args.log_dir == Path("/nas/work/dist/logs/run-42")


def test_dist_coordinator_log_dir_defaults_none() -> None:
    args = _parse(
        [
            "dist-coordinator",
            "--source",
            "/nas/src",
            "--work",
            "/nas/work",
            "--exl3-out",
            "/nas/out",
            "--recipe",
            "/nas/work/recipe.yaml",
            "--nodes",
            "gx10-1",
        ]
    )
    assert args.log_dir is None


def test_dist_worker_parses_log_dir() -> None:
    args = _parse(
        [
            "dist-worker",
            "--inbox",
            "/nas/work/dist/inbox/gx10-1",
            "--shared",
            "/nas/work",
            "--device",
            "0",
            "--stop",
            "/nas/work/dist/stop-gx10-1",
            "--log-dir",
            "/nas/work/dist/logs/run-42",
        ]
    )
    assert args.log_dir == Path("/nas/work/dist/logs/run-42")


def test_dist_worker_log_dir_defaults_none() -> None:
    args = _parse(
        [
            "dist-worker",
            "--inbox",
            "/nas/work/dist/inbox/gx10-1",
            "--shared",
            "/nas/work",
            "--device",
            "0",
            "--stop",
            "/nas/work/dist/stop-gx10-1",
        ]
    )
    assert args.log_dir is None
