"""Pack-build progress status: JSON status file + renderer.

Ports ``pack-monitor.sh``'s status-file writer and ``pack-status.py``'s
renderer. The status JSON keys are a contract observed by the driver's
``poll()`` and by ``pack-status.py``; keep them stable.
"""

from __future__ import annotations

import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path

STATUS_KEYS = (
    "stage",
    "phase",
    "model",
    "codebook",
    "bits",
    "layers_total",
    "layers_completed",
    "current_layer",
    "eta_seconds",
    "elapsed_seconds",
    "gpu_memory_used_mb",
    "disk_free_gb",
    "errors",
    "output_size",
    "convert_duration",
    "repack_duration",
)


def status_path(explicit: str | None = None) -> Path:
    if explicit:
        return Path(explicit)
    return Path(os.environ.get("PACK_STATUS_FILE", "/work/.pack-status.json"))


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _gpu_memory_mb() -> tuple[int, int] | None:
    """Best-effort (used_mb, total_mb) GPU sample; None when unavailable."""
    if os.path.exists("/proc/driver/nvidia/version"):
        try:
            import subprocess

            out = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if out.returncode == 0 and out.stdout.strip():
                used = total = 0
                for line in out.stdout.strip().splitlines():
                    parts = [p.strip() for p in line.split(",")]
                    used += int(parts[0])
                    total += int(parts[1])
                return used, total
        except Exception:
            return None
    return None


def _disk_free_gb(path: Path) -> int | None:
    try:
        st = os.statvfs(str(path)) if hasattr(os, "statvfs") else None
        if st is not None:
            return int(st.f_bavail * st.f_frsize / 1e9)
    except Exception:
        pass
    try:
        import shutil

        return int(shutil.disk_usage(str(path)).free / 1e9)
    except Exception:
        return None


def write_status(
    path: Path,
    started_at: str,
    **fields: object,
) -> None:
    """Atomically merge ``fields`` into the JSON status file at ``path``.

    Mirrors the original ``_monitor_write_status``: always emits ``started_at``,
    ``last_update``, and (when available) ``gpu_memory_used_mb`` /
    ``gpu_memory_total_mb`` / ``disk_free_gb``; computes ``elapsed_seconds``
    from ``started_at`` when not supplied. Unknown keys are preserved.
    """
    current: dict[str, object] = {}
    if path.exists():
        try:
            current = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            current = {}
    current.update(fields)
    current["started_at"] = started_at
    current["last_update"] = _utc_now()
    mem = _gpu_memory_mb()
    if mem is not None:
        used, total = mem
        current.setdefault("gpu_memory_used_mb", used)
        current.setdefault("gpu_memory_total_mb", total)
    disk = _disk_free_gb(Path(os.environ.get("PACK_WORK_DIR", "/work")))
    if disk is not None:
        current.setdefault("disk_free_gb", disk)
    if "elapsed_seconds" not in fields and started_at:
        try:
            t0 = datetime.strptime(started_at, "%Y-%m-%dT%H:%M:%SZ").replace(
                tzinfo=UTC
            )
            current["elapsed_seconds"] = int(
                (datetime.now(UTC) - t0).total_seconds()
            )
        except ValueError:
            pass
    tmp = Path(str(path) + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps(current, indent=2) + "\n")
    tmp.replace(path)


def eta_seconds(elapsed: float, current: int, total: int) -> int:
    """Linear per-layer ETA (the pack-pipeline.sh formula)."""
    if current <= 0 or total <= 0:
        return 0
    return int(elapsed * (total - current) / current)


def timestamp() -> str:
    return _utc_now()


def now_epoch() -> int:
    return int(time.time())
