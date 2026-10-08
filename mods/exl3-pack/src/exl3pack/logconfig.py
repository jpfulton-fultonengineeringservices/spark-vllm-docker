"""Structured JSONL logging for the dist pipeline (coordinator + workers).

One JSON object per line in ``<log_dir>/<name>.jsonl``, produced by stdlib
``logging`` with ``python-json-logger`` — the standard OSS JSON formatter
for stdlib logging. Every line carries at minimum:

- ``ts``: ISO-8601 UTC timestamp
- ``level``: ``INFO`` / ``WARNING`` / ``ERROR`` / ...
- ``event``: namespaced event name (e.g. ``worker.shard_done``)
- ``node``: node slug (hostname when unset)
- ``pid``: process id
- ``fields``: arbitrary structured payload

A human-readable stderr ``StreamHandler`` stays attached in every mode, so
without ``--log-dir`` output is unchanged from the previous ``print`` lines.
"""

from __future__ import annotations

import contextlib
import datetime
import json
import logging
import os
import socket
import sys
from pathlib import Path
from typing import Any

from pythonjsonlogger.json import JsonFormatter

__all__ = ["JsonlFormatter", "get_logger", "log_event"]

_RESERVED = frozenset({"ts", "level", "event", "node", "pid", "fields", "name", "message"})


class JsonlFormatter(JsonFormatter):
    """JSON formatter emitting the required JSONL envelope keys."""

    def add_fields(
        self,
        log_record: dict[str, Any],
        record: logging.LogRecord,
        message_dict: dict[str, Any],
    ) -> None:
        log_record["ts"] = datetime.datetime.now(datetime.UTC).isoformat()
        log_record["level"] = record.levelname
        log_record["event"] = getattr(record, "event", record.getMessage())
        log_record["node"] = getattr(record, "node", None) or socket.gethostname()
        log_record["pid"] = record.process if record.process is not None else 0
        fields = getattr(record, "fields", None)
        if fields is None:
            fields = {k: v for k, v in message_dict.items() if k not in _RESERVED}
        log_record["fields"] = fields
        for key in set(log_record) - _RESERVED:
            del log_record[key]


def get_logger(name: str, log_dir: Path | None = None, node: str | None = None) -> logging.Logger:
    """Return a logger emitting JSONL events to ``<log_dir>/<name>.jsonl``.

    Always attaches a plain stderr StreamHandler so human-readable output is
    preserved. With ``log_dir`` set, also appends structured JSONL. NFS
    root-squash means containers run as a non-root UID and cannot chown, so
    files are opened plain with a best-effort 0664 chmod for host+container
    read/append.
    """
    logger = logging.getLogger(f"exl3pack.{name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(stream)
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_path = log_dir / f"{name}.jsonl"
        handler = logging.FileHandler(file_path, mode="a", encoding="utf-8")
        handler.setFormatter(JsonlFormatter())
        logger.addHandler(handler)
        # root-squash NFS may refuse the chmod; plain 0664 umask suffices
        with contextlib.suppress(OSError):
            os.chmod(file_path, 0o664)
    logger._exl3_node = node  # type: ignore[attr-defined]
    return logger


def log_event(
    logger: logging.Logger,
    level: int,
    event: str,
    fields: dict[str, Any] | None = None,
) -> None:
    """Emit one structured event through *logger* at *level*.

    The human-readable stderr line keeps the ``[component] event: k=v ...``
    shape the pipeline prints today.
    """
    payload = fields or {}
    human = " ".join(
        f"{k}={v if isinstance(v, str) else json.dumps(v, default=str)}" for k, v in payload.items()
    )
    prefix = f"[{logger.name.rsplit('.', 1)[-1]}] {event}"
    message = f"{prefix}: {human}" if human else prefix
    record = logging.LogRecord(
        name=logger.name,
        level=level,
        pathname="",
        lineno=0,
        msg=message,
        args=(),
        exc_info=None,
    )
    record.event = event
    record.fields = payload
    node = getattr(logger, "_exl3_node", None)
    if node:
        record.node = node
    logger.handle(record)
