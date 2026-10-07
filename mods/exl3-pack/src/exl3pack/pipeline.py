"""Pack pipeline orchestration: convert → repack (→ assemble), with status.

Port of ``pack-pipeline.sh``'s ``pipeline_run``. Torch-dependent stages are
imported lazily so this module's planning surface (``plan``) is importable
without the GPU stack.

Disk discipline (per the "output on the NAS, tensor-by-tensor, clean up node
disk" requirement):
- ``work`` and intermediate packs live on node-local NVMe.
- the final artifact is written to the NAS ``v1_out``/``serve_out``;
- ``cleanup`` stages the intermediate ``work`` and ``exl3_out`` off the node
  once the final artifact exists.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from . import monitor
from .spec import PackSpec


@dataclass
class PipelinePlan:
    """Resolved paths + mount plan for a dry-run."""

    spec: PackSpec
    source: Path
    work: Path
    exl3_out: Path
    v1_out: Path
    node: str | None = None
    mounts: list[str] = field(default_factory=list)

    def describe(self) -> str:
        lines = [
            f"model:     {self.spec.slug} (codebook={self.spec.codebook} K{self.spec.bits})",
            f"node:      {self.node or '(unresolved)'}",
            f"source:    {self.source}",
            f"work:      {self.work}   (node-local)",
            f"exl3-out:  {self.exl3_out}",
            f"v1-out:    {self.v1_out}  (NAS)",
        ]
        if self.mounts:
            lines.append("mounts:    " + " ".join(self.mounts))
        return "\n".join(lines)


@dataclass
class CleanupPlan:
    """What to remove from the node after the final artifact is secured."""

    delete_work: bool = False
    delete_exl3_out: bool = False
    delete_source: bool = False

    def describe(self) -> str:
        items = []
        if self.delete_work:
            items.append("work")
        if self.delete_exl3_out:
            items.append("exl3_out")
        if self.delete_source:
            items.append("source")
        return ", ".join(items) if items else "none"


def _rmtree(path: Path, progress: Callable[[str], None]) -> None:
    if path.exists():
        progress(f"[cleanup] removing {path}")
        shutil.rmtree(path, ignore_errors=True)


def run(
    spec: PackSpec,
    source: Path,
    work: Path,
    exl3_out: Path,
    v1_out: Path,
    *,
    cleanup: CleanupPlan | None = None,
    status_file: Path | None = None,
    progress: Callable[[str], None] = lambda s: print(s, flush=True),
) -> None:
    cleanup = cleanup or CleanupPlan()
    sf = status_file or monitor.status_path()
    started_total = monitor.timestamp()

    # Lazy imports: only reached when actually executing.
    from . import convert, repack

    progress("=" * 60)
    progress("pack-build pipeline start")
    progress(f"  source:   {source}")
    progress(f"  work:     {work}")
    progress(f"  exl3:     {exl3_out}")
    progress(f"  v1:       {v1_out}")
    progress(f"  codebook: {spec.codebook}")
    progress(f"  bits:     {spec.bits}")
    progress("=" * 60)

    monitor.write_status(
        sf, started_total, stage="pipeline", phase="preflight", model=spec.slug,
        codebook=spec.codebook, bits=spec.bits,
    )
    _preflight(source, work, progress)

    monitor.write_status(sf, started_total, stage="convert", phase="starting")
    t0 = monitor.now_epoch()
    convert.run(
        source, exl3_out, work,
        bits=spec.bits, codebook=spec.codebook, status_file=sf, progress=progress,
    )
    convert_dur = monitor.now_epoch() - t0
    progress(f"[pipeline] convert duration: {convert_dur // 60}m {convert_dur % 60}s")

    monitor.write_status(sf, started_total, stage="repack", phase="starting")
    t1 = monitor.now_epoch()
    repack.run(exl3_out, v1_out, bits=spec.bits, self_check_on=True, progress=progress)
    repack_dur = monitor.now_epoch() - t1
    progress(f"[pipeline] repack duration: {repack_dur // 60}m {repack_dur % 60}s")

    if cleanup.delete_work:
        _rmtree(work, progress)
    if cleanup.delete_exl3_out:
        _rmtree(exl3_out, progress)

    total_dur = monitor.now_epoch() - _parse_epoch(started_total)
    v1_size = _human_size(v1_out)
    progress("=" * 60)
    progress("pack-build pipeline complete")
    progress(f"  convert:  {convert_dur // 60}m {convert_dur % 60}s")
    progress(f"  repack:   {repack_dur // 60}m {repack_dur % 60}s")
    progress(f"  total:    {total_dur // 60}m {total_dur % 60}s")
    progress(f"  output:   {v1_out} ({v1_size})")
    progress("=" * 60)
    monitor.write_status(
        sf, started_total, stage="pipeline", phase="done",
        elapsed_seconds=total_dur, output_size=v1_size,
        convert_duration=convert_dur, repack_duration=repack_dur,
    )


def _parse_epoch(ts: str) -> int:
    from datetime import UTC, datetime

    try:
        return int(
            datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC).timestamp()
        )
    except ValueError:
        return 0


def _human_size(path: Path) -> str:
    total = 0
    if path.exists():
        for p in path.rglob("*"):
            if p.is_file():
                total += p.stat().st_size
    for unit in ("B", "K", "M", "G", "T"):
        if total < 1024:
            return f"{total:.0f}{unit}"
        total //= 1024
    return f"{total}T"


def _preflight(source: Path, work: Path, progress: Callable[[str], None]) -> None:
    """Mirror pipeline_preflight: source present, work writable, libs importable."""
    if not source.is_dir():
        raise SystemExit(f"[pipeline] ERROR: source not found: {source}")
    work.mkdir(parents=True, exist_ok=True)
    for mod in ("exllamav3", "b12x"):
        try:
            __import__(mod)
        except ImportError:
            raise SystemExit(f"[pipeline] ERROR: {mod} not importable") from None
    progress("=== pack-build pre-flight OK ===")
