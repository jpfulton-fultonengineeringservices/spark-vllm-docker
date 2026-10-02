#!/usr/bin/env python3
"""Read and display the pack-build progress status file.

Usage::

    pack-status [--json] [--watch [INTERVAL]] [<status-file>]

Default status file: ``/work/.pack-status.json`` (configurable via env
``PACK_STATUS_FILE`` or as positional argument).

--watch mode refreshes every INTERVAL seconds (default 2). Press Ctrl-C to exit.
"""

from __future__ import annotations

import json
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def _status_path(args: list[str]) -> Path:
    for a in args:
        if not a.startswith("-"):
            return Path(a)
    env = os.environ.get("PACK_STATUS_FILE", "/work/.pack-status.json")
    return Path(env)


def _human_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        m, s = divmod(int(seconds), 60)
        return f"{m}m{s:02d}s"
    h, r = divmod(int(seconds), 3600)
    m, s = divmod(r, 60)
    return f"{h}h{m:02d}m{s:02d}s"


def _render(st: dict) -> str:
    lines: list[str] = []
    stage = st.get("stage", "unknown")
    phase = st.get("phase", "")
    started = st.get("started_at", "")
    last = st.get("last_update", "")

    lines.append(f"stage:      {stage}" + (f" / {phase}" if phase else ""))
    if started:
        lines.append(f"started:    {started}")
    if last:
        lines.append(f"updated:    {last}")

    elapsed = st.get("elapsed_seconds")
    eta = st.get("eta_seconds")
    if elapsed is not None:
        lines.append(f"elapsed:    {_human_duration(elapsed)}")
    if eta is not None and eta > 0:
        lines.append(f"ETA:        {_human_duration(eta)} remaining")

    model = st.get("model")
    if model:
        lines.append(f"model:      {model}")
    cb = st.get("codebook")
    bits = st.get("bits")
    if cb or bits is not None:
        lines.append(f"codebook:   {cb} {f'K{bits}' if bits is not None else ''}")

    total = st.get("layers_total")
    done = st.get("layers_completed")
    cur = st.get("current_layer")
    if total is not None:
        pct = f"{(done or 0) / total * 100:.0f}%" if done is not None else ""
        lines.append(f"layers:     {done or '-'}/{total} {pct}" + (f" (current: {cur})" if cur is not None else ""))
    ld = st.get("layer_detail")
    if ld:
        lines.append(f"detail:     {ld}")

    gpu = st.get("gpu_memory_used_mb")
    gpu_tot = st.get("gpu_memory_total_mb")
    if gpu is not None:
        pct_g = f"{(gpu / gpu_tot * 100):.0f}%" if gpu_tot else ""
        lines.append(f"GPU mem:    {gpu} MB / {gpu_tot or '?'} MB {pct_g}")
    disk = st.get("disk_free_gb")
    if disk is not None:
        lines.append(f"disk free:  {disk} GB")

    errs = st.get("errors", [])
    if errs:
        lines.append("errors:")
        for e in errs:
            lines.append(f"  - {e}")

    model_info = st.get("model_info")
    if model_info:
        lines.append("")
        lines.append("detected model geometry:")
        lines.append(f"  architecture:  {model_info.get('architecture', '?')}")
        lines.append(f"  hidden:        {model_info.get('hidden_size', '?')}")
        lines.append(f"  intermediate:  {model_info.get('intermediate_size', '?')}")
        lines.append(f"  experts:       {model_info.get('num_experts', '?')}")
        lines.append(f"  slots:         {model_info.get('num_slots', '?')}")
        lines.append(f"  moe layers:    {model_info.get('moe_layer_count', '?')}")

    return "\n".join(lines)


def _load_status(path):
    """Parse the status JSON, or return None if absent/unreadable/mid-write."""
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    json_flag = False
    watch_mode = False
    watch_interval = 2.0
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
        else:
            pos_args.append(a)
        i += 1

    path = _status_path(pos_args)

    if watch_mode:
        signal.signal(signal.SIGINT, lambda *_: sys.exit(0))
        try:
            while True:
                st = _load_status(path) if path.exists() else None
                if st is not None:
                    if json_flag:
                        print(json.dumps(st, indent=2))
                    else:
                        print("\033[2J\033[H", end="")  # clear screen
                        print(_render(st))
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
        print(_render(st))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())