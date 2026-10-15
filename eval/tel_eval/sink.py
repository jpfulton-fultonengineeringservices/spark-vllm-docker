"""Thread-safe append-only JSONL telemetry sink for the eval rig plugin.

One process-wide writer guarded by a lock (the harness issues concurrent async
requests). Records are plain dicts; extras are passed through ``sanitize`` to keep
very large strings bounded.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

_LOCK = threading.Lock()
_ORDINAL = 0
_MAX_STR = 65536


def _sanitize(value: Any) -> Any:
    if isinstance(value, str):
        if len(value) > _MAX_STR:
            return value[:_MAX_STR] + f"...[truncated {len(value) - _MAX_STR} chars]"
        return value
    if isinstance(value, dict):
        return {str(k): _sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(v) for v in value]
    return value


def next_ordinal() -> int:
    """Return a process-wide monotonically increasing request ordinal."""
    global _ORDINAL
    with _LOCK:
        _ORDINAL += 1
        return _ORDINAL


def telemetry_path() -> Path | None:
    raw = os.environ.get("EVAL_TELEMETRY_PATH")
    return Path(raw) if raw else None


def images_dir() -> Path | None:
    raw = os.environ.get("EVAL_TELEMETRY_IMAGES_DIR")
    return Path(raw) if raw else None


def results_root() -> Path | None:
    """The results dir the console mounts, for relativizing image paths."""
    raw = os.environ.get("EVAL_RESULTS_ROOT")
    return Path(raw) if raw else None


def record(entry: dict[str, Any]) -> None:
    """Append one telemetry record to the sidecar (no-op when unconfigured)."""
    path = telemetry_path()
    if path is None:
        return
    line = json.dumps(_sanitize(entry), ensure_ascii=False) + "\n"
    with _LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line)


def reset_ordinal() -> None:
    """Reset the ordinal counter (test hook)."""
    global _ORDINAL
    with _LOCK:
        _ORDINAL = 0
