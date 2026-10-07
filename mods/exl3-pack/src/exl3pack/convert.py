"""Convert stage: drive ``exllamav3.conversion.convert_model``.

Torch-dependent (runs INSIDE the b12x image). Port of ``pipeline_convert`` from
``pack-pipeline.sh``: resume detection, live progress parsing, and atomic status
updates. Streams ``convert_model`` stdout, updating the status file per layer.
"""

from __future__ import annotations

import re
import subprocess
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from . import monitor

# exllamav3 emits layer/expert/proxy_err/g_sc on various lines.
_LAYER_RE = re.compile(r"(?:layer|quantizing|encoding|calibrat)\D*(\d+)", re.I)
_EXPERT_RE = re.compile(r"expert\D*(\d+)", re.I)
_PERR_RE = re.compile(r"proxy_err\D*([\d.]+)")
_GSC_RE = re.compile(r"g_sc\D*([\d.]+)")


def guess_total_layers(src: Path) -> int:
    """``num_hidden_layers`` from config.json, else the detected MoE count."""
    cfg_path = src / "config.json"
    if cfg_path.exists():
        import json

        cfg = json.loads(cfg_path.read_text())
        tc = cfg.get("text_config", cfg)
        n = int(tc.get("num_hidden_layers", 0) or 0)
        if n > 0:
            return n
    from .geometry import detect

    return detect(src).moe_layer_count


def run(
    src: Path,
    out: Path,
    work: Path,
    *,
    bits: int,
    codebook: str,
    extra_args: list[str] | None = None,
    status_file: Path | None = None,
    progress: Callable[[str], None] = lambda s: print(s, flush=True),
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    sf = status_file or monitor.status_path()

    resume_flag: list[str] = []
    qtensors = work / "qtensors"
    if qtensors.is_dir() and any(qtensors.iterdir()) and not any(out.iterdir()):
        resume_flag = ["-r"]
        progress(f"[convert] resuming: found quantized state in {qtensors}")

    total_layers = guess_total_layers(src)
    model_name = src.name
    progress(f"[convert] {src} -> {out} ({codebook} K{bits}); {total_layers} layers")

    started_at = monitor.timestamp()
    monitor.write_status(
        sf,
        started_at,
        stage="convert",
        phase="starting",
        model=model_name,
        codebook=codebook,
        bits=bits,
        layers_total=total_layers,
    )

    cmd = [
        sys.executable,
        "-u",
        "-m",
        "exllamav3.conversion.convert_model",
        "-i",
        str(src),
        "-o",
        str(out),
        "-w",
        str(work),
        "-b",
        str(bits),
        "-cb",
        codebook,
        *resume_flag,
        *(extra_args or []),
    ]
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        progress(line.rstrip("\n"))
        m = _LAYER_RE.search(line)
        if m:
            layer_num = int(m.group(1))
            t0 = datetime.strptime(started_at, "%Y-%m-%dT%H:%M:%SZ").replace(
                tzinfo=UTC
            )
            elapsed = monitor.now_epoch() - int(t0.timestamp())
            eta = monitor.eta_seconds(elapsed, layer_num, total_layers)
            em = _EXPERT_RE.search(line)
            perr = _PERR_RE.search(line)
            gsc = _GSC_RE.search(line)
            monitor.write_status(
                sf,
                started_at,
                stage="convert",
                phase="quantizing",
                model=model_name,
                codebook=codebook,
                bits=bits,
                layers_total=total_layers,
                current_layer=layer_num,
                layers_completed=layer_num,
                layer_detail=f"experts_{em.group(1)}" if em else "",
                proxy_err=perr.group(1) if perr else "",
                g_sc=gsc.group(1) if gsc else "",
                eta_seconds=eta,
                elapsed_seconds=elapsed,
            )
    code = proc.wait()
    if code != 0:
        progress(f"[convert] FAILED (exit {code})")
        monitor.write_status(
            sf, started_at, stage="convert", phase="error",
            errors=[f"convert_model exited with code {code}"],
        )
        raise SystemExit(code)
    monitor.write_status(sf, started_at, stage="convert", phase="done")
    progress(f"[convert] complete: {out}")
