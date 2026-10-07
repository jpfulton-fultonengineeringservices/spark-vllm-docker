"""Read and display the pack-build progress status file.

Faithful port of the original ``pack-status.py``.

Usage::

    status [--json] [--watch [INTERVAL]] [--delta N] [--interval S] [<status-file>]

Default status file: ``/work/.pack-status.json`` (env ``PACK_STATUS_FILE`` or a
positional argument). ``--watch`` refreshes every INTERVAL seconds (default 2).
The driver's ``poll()`` calls this as
``status.py --delta <n> --interval <s> <file>``, so those flags MUST be parsed.
"""

from __future__ import annotations

import json
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ``datetime.UTC`` is 3.11+; this renderer runs on the operator's workstation
# (may be an older Python). ``timezone.utc`` is equivalent and always present.
UTC = timezone.utc


def _status_path(pos_args: list[str]) -> Path:
    for a in pos_args:
        if not a.startswith("-"):
            return Path(a)
    return Path(os.environ.get("PACK_STATUS_FILE", "/work/.pack-status.json"))


def _human_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        m, s = divmod(int(seconds), 60)
        return f"{m}m{s:02d}s"
    h, r = divmod(int(seconds), 3600)
    m, s = divmod(r, 60)
    return f"{h}h{m:02d}m{s:02d}s"


def _live_elapsed(started: str) -> float | None:
    if not started:
        return None
    try:
        t = datetime.strptime(started, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    except ValueError:
        return None
    return max(0.0, (datetime.now(UTC) - t).total_seconds())


def _progress_bar(done: int, total: int, width: int = 30) -> str:
    if not total or total <= 0:
        return ""
    pct = min(done / total, 1.0)
    filled = int(width * pct)
    bar = "█" * filled + "░" * (width - filled)
    return f"{bar} {done}/{total} {pct * 100:.0f}%"


def _coerce_float(value: object) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0


def _coerce_int(value: object) -> int:
    return int(_coerce_float(value))


def _render(st: dict[str, object], delta: int = 0, interval: float = 0) -> str:
    B = "\033[1m"
    G = "\033[32m"
    Y = "\033[33m"
    R = "\033[31m"
    C = "\033[36m"
    N = "\033[0m"

    phase = st.get("phase", "")
    if phase == "done":
        color = G
    elif phase == "error":
        color = R
    elif phase in ("quantizing", "assembling"):
        color = Y
    else:
        color = C

    lines: list[str] = []
    stage = st.get("stage", "unknown")
    lines.append(
        f"{B}    stage:{N} {color}{stage}" + (f" / {phase}" if phase else "") + f"{N}"
    )

    started = str(st.get("started_at", ""))
    if started:
        lines.append(f"  started: {started}")
    last = st.get("last_update", "")
    if last:
        lines.append(f"  updated: {last}")

    elapsed: float | None = (
        _coerce_float(st["elapsed_seconds"])
        if st.get("elapsed_seconds") is not None
        else None
    )
    live = _live_elapsed(started)
    if live is not None:
        elapsed = live
    if elapsed is not None:
        lines.append(f" elapsed: {_human_duration(elapsed)}")

    total = st.get("layers_total")
    done = st.get("layers_completed")
    if elapsed and elapsed > 0 and total and done is not None:
        lpm = _coerce_int(done) / (elapsed / 60)
        lines.append(f"    rate: {lpm:.1f} layers/min")

    if total and done is not None:
        pbar = _progress_bar(_coerce_int(done), _coerce_int(total))
        lines.append(f"progress: {pbar}")

    eta = st.get("eta_seconds")
    if eta is not None and _coerce_float(eta) > 0:
        lines.append(f"     ETA: {_human_duration(_coerce_float(eta))} remaining")

    if delta > 0 and interval > 0:
        rate = delta / (interval / 60)
        lines.append(f"  Δ poll: +{delta} layers ({rate:.1f} layers/min)")

    model = st.get("model")
    if model:
        lines.append(f"\n   model: {model}")
    cb = st.get("codebook")
    bits = st.get("bits")
    if cb or bits is not None:
        lines.append(f"codebook: {cb} {f'K{bits}' if bits is not None else ''}")

    cur = st.get("current_layer")
    if cur is not None and not total:
        lines.append(f"   layer: {cur}")

    ld = st.get("layer_detail")
    if ld:
        lines.append(f"  detail: {ld}")

    gpu = st.get("gpu_memory_used_mb")
    gpu_tot = st.get("gpu_memory_total_mb")
    if gpu is not None:
        pct_g = f"({(_coerce_float(gpu) / _coerce_float(gpu_tot) * 100):.0f}%)" if gpu_tot else ""
        lines.append(f" GPU mem: {gpu} MB / {gpu_tot or '?'} MB {pct_g}")
    disk = st.get("disk_free_gb")
    if disk is not None:
        lines.append(f"    disk: {disk} GB free")

    errs = st.get("errors", [])
    if errs:
        lines.append(f"\n{R}errors:{N}")
        for e in errs if isinstance(errs, list) else [errs]:
            lines.append(f"  - {e}")

    model_info = st.get("model_info")
    if isinstance(model_info, dict):
        lines.append("")
        lines.append("detected model geometry:")
        lines.append(f"  architecture:  {model_info.get('architecture', '?')}")
        lines.append(f"  hidden:        {model_info.get('hidden_size', '?')}")
        lines.append(f"  intermediate:  {model_info.get('intermediate_size', '?')}")
        lines.append(f"  experts:       {model_info.get('num_experts', '?')}")
        lines.append(f"  slots:         {model_info.get('num_slots', '?')}")
        lines.append(f"  moe layers:    {model_info.get('moe_layer_count', '?')}")

    return "\n".join(lines)


def _load_status(path: Path) -> dict[str, object] | None:
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    json_flag = False
    watch_mode = False
    watch_interval = 2.0
    delta_layers = 0
    delta_interval = 0.0
    pos_args: list[str] = []

    i = 0
    while i < len(argv):
        a = argv[i]
        if a in ("--json", "-j"):
            json_flag = True
        elif a in ("--watch", "-w"):
            watch_mode = True
            if i + 1 < len(argv) and not argv[i + 1].startswith("-"):
                try:
                    watch_interval = float(argv[i + 1])
                    i += 1
                except ValueError:
                    pass
        elif a == "--delta":
            if i + 1 < len(argv):
                try:
                    delta_layers = int(argv[i + 1])
                    i += 1
                except ValueError:
                    pass
        elif a == "--interval":
            if i + 1 < len(argv):
                try:
                    delta_interval = float(argv[i + 1])
                    i += 1
                except ValueError:
                    pass
        else:
            pos_args.append(a)
        i += 1

    path = _status_path(pos_args)

    if watch_mode:
        signal.signal(signal.SIGINT, lambda *_: sys.exit(0))
        prev_done = -1
        try:
            while True:
                st = _load_status(path) if path.exists() else None
                if st is not None:
                    if json_flag:
                        print(json.dumps(st, indent=2))
                    else:
                        done = _coerce_int(st.get("layers_completed", 0) or 0)
                        delta = done - prev_done if prev_done >= 0 and done >= prev_done else 0
                        prev_done = done
                        print("\033[H\033[J", end="")
                        print(_render(st, delta=delta, interval=watch_interval))
                else:
                    print(f"waiting for {path}...")
                time.sleep(watch_interval)
        except KeyboardInterrupt:
            pass
        return 0

    if not path.exists():
        print(f"status file not found: {path}", file=sys.stderr)
        return 1

    st = _load_status(path)
    if st is None:
        print(f"status file not found or unreadable: {path}", file=sys.stderr)
        return 1
    if json_flag:
        print(json.dumps(st, indent=2))
    else:
        print(_render(st, delta=delta_layers, interval=delta_interval))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
