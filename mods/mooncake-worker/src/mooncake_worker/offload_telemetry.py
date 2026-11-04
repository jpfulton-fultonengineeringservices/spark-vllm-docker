"""Host-side Mooncake offload-tier metrics exporter (stdlib only).

Serves Prometheus text exposition on :9004 with:

- ``mooncake_offload_dir_bytes`` / ``mooncake_offload_dir_files``: apparent size
  and file count of the PRIMARY offload tree (first ``--root``, tier="staging")
  -- the only lines in these families, so the dashboard's label-stripping
  parser stays unambiguous.
- ``mooncake_offload_tier_dir_bytes`` / ``mooncake_offload_tier_dir_files``:
  per-additional-root series labeled ``tier="drain",root="..."`` (NAS drain).
- ``mooncake_offload_staging_max_bytes``: the staging tier's byte cap
  (``--staging-max-bytes``), for staging-utilization panels -- NOT a statvfs
  reading; the staging fs is usually the node root, whose 84% fill would
  otherwise sit permanently past dashboard red thresholds.
- ``mooncake_offload_mount_total_bytes`` / ``mooncake_offload_mount_free_bytes``:
  volume capacity via statvfs on the matched mountpoint of EACH root that
  resolves to an NFS mount (deduped by mountpoint). The NVMe staging parent
  matches no NFS mount and contributes nothing here.
- ``mooncake_offload_nfs_read_bytes_total`` / ``mooncake_offload_nfs_write_bytes_total``:
  /proc/self/mountstats server wire counters for the /nas-1 NFS mount (mount-wide:
  all consumers, not mooncake-only).
- ``mooncake_offload_scrape_ok``: 1 when the last directory refresh succeeded.

Designed to run as a plain ``python3`` script on each GX10 host from the shared
checkout (no pip installs, no wheel). Provisioned by
infra/cluster-config scripts/setup-mooncake-offload-metrics.sh.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import shutil
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

DEFAULT_PORT = 9004
DEFAULT_INTERVAL = 60.0
DEFAULT_STAGING_MAX_BYTES = 274_877_906_944  # 256 GiB; compose default

# Index of server-read / server-written byte counters within the mountstats
# "bytes" line (NFSIOS_SERVERREADBYTES / NFSIOS_SERVERWRITTENBYTES).
SERVER_READ_IDX = 4
SERVER_WRITE_IDX = 5


@dataclass(frozen=True)
class NfsMountStats:
    """Per-mount NFS client byte counters from /proc/self/mountstats."""

    mountpoint: str
    server: str
    fstype: str
    server_read_bytes: int
    server_written_bytes: int


@dataclass(frozen=True)
class RootSample:
    """One offload-tree root's cached dir gauges plus its tier label."""

    root: str
    tier: str
    dir_bytes: int
    dir_files: int


@dataclass(frozen=True)
class NfsMountSample:
    """One NFS mount's rendered gauges: statvfs capacity + wire counters."""

    mountpoint: str
    server: str
    total_bytes: int | None
    free_bytes: int | None
    server_read_bytes: int | None
    server_written_bytes: int | None


def parse_mountstats(text: str) -> dict[str, NfsMountStats]:
    """Parse /proc/self/mountstats content into per-mount NFS byte counters.

    Two header shapes are accepted:

    - kernel format: ``device <server> mounted on <mountpoint> with fstype
      <fstype> ...`` (modern kernels; the bytes line is tab-separated as
      ``bytes:\\t<int> <int> ...``).
    - legacy format: ``<mountpoint> from <server> fstype <fstype> ...``.

    Only sections whose fstype token contains "nfs" (case-insensitive) and
    that carry a "bytes" line with the full counter tuple are kept. Within a
    kept section the "bytes" line's integers map to NFSIOS_* counters;
    SERVER_READ_IDX/SERVER_WRITE_IDX select server-read / server-written.
    """
    mounts: dict[str, NfsMountStats] = {}
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        header = lines[i]
        is_kernel = header.startswith("device ") and " mounted on " in header and (
            " with fstype " in header
        )
        is_legacy = header.startswith("/") and " from " in header and " fstype " in header
        if not (is_kernel or is_legacy):
            i += 1
            continue
        if is_kernel:
            tokens = header.split()
            mountpoint = tokens[tokens.index("on") + 1]
            server = tokens[1]
            fstype_idx = tokens.index("fstype") + 1
            fstype = tokens[fstype_idx] if fstype_idx < len(tokens) else ""
            section_first_token = "device"
        else:
            tokens = header.split()
            mountpoint = tokens[0]
            server = tokens[tokens.index("from") + 1]
            fstype_idx = tokens.index("fstype") + 1
            fstype = tokens[fstype_idx] if fstype_idx < len(tokens) else ""
            section_first_token = "/"
        i += 1
        if "nfs" not in fstype.lower():
            continue
        byte_vals: list[int] | None = None
        while i < len(lines):
            section_line = lines[i]
            starts_next = section_line.startswith(section_first_token) and (
                section_line != header
            )
            if starts_next:
                break  # next mount section header; bytes line never appeared
            first = section_line.split()[0].rstrip(":") if section_line.split() else ""
            if first == "bytes":
                ints: list[int] = []
                for tok in section_line.replace(":", " ").split()[1:]:
                    with contextlib.suppress(ValueError):
                        ints.append(int(tok))
                if len(ints) >= 6:
                    byte_vals = ints
                break
            i += 1
        if byte_vals is None:
            continue
        mounts[mountpoint] = NfsMountStats(
            mountpoint=mountpoint,
            server=server,
            fstype=fstype,
            server_read_bytes=byte_vals[SERVER_READ_IDX],
            server_written_bytes=byte_vals[SERVER_WRITE_IDX],
        )
    return mounts


def find_mount_for(root: str, mounts: dict[str, NfsMountStats]) -> NfsMountStats | None:
    """Return the mount whose mountpoint is the longest prefix of realpath(root)."""
    real = os.path.realpath(root)
    best: NfsMountStats | None = None
    for mountpoint in sorted(mounts, key=len, reverse=True):
        if real == mountpoint or real.startswith(mountpoint.rstrip("/") + "/"):
            best = mounts[mountpoint]
            break
    return best


def collect_dir(root: str) -> tuple[int, int]:
    """Walk root iteratively; return (apparent size bytes, regular-file count)."""
    total = 0
    files = 0
    stack = [root]
    while stack:
        current = stack.pop()
        with os.scandir(current) as it:
            for entry in it:
                st = entry.stat(follow_symlinks=False)
                if entry.is_file(follow_symlinks=False):
                    total += st.st_size
                    files += 1
                elif entry.is_dir(follow_symlinks=False):
                    stack.append(entry.path)
    return total, files


def _fmt_labels(extra: dict[str, str]) -> str:
    parts = ",".join(f'{k}="{v}"' for k, v in sorted(extra.items()))
    return "{" + parts + "}"


def render_series(
    primary: RootSample,
    extras: Sequence[RootSample],
    nfs_mounts: Sequence[NfsMountSample],
    scrape_ok: bool,
    staging_max_bytes: int,
) -> str:
    """Assemble the Prometheus text exposition body (pure, no I/O)."""
    out = [
        "# HELP mooncake_offload_dir_bytes Apparent size of the Mooncake offload tree.",
        "# TYPE mooncake_offload_dir_bytes gauge",
        f'mooncake_offload_dir_bytes{{tier="{primary.tier}"}} {primary.dir_bytes}',
        "# HELP mooncake_offload_dir_files File count in the Mooncake offload tree.",
        "# TYPE mooncake_offload_dir_files gauge",
        f'mooncake_offload_dir_files{{tier="{primary.tier}"}} {primary.dir_files}',
    ]
    if staging_max_bytes > 0:
        out.extend(
            [
                "# HELP mooncake_offload_staging_max_bytes Staging tier byte cap"
                " (MOONCAKE_OFFLOAD_STAGING_MAX_BYTES compose contract).",
                "# TYPE mooncake_offload_staging_max_bytes gauge",
                f'mooncake_offload_staging_max_bytes{{tier="{primary.tier}"}} {staging_max_bytes}',
            ]
        )
    # Per-extra-root dir series under a SEPARATE family name. The panel parser
    # matches `mooncake_offload_dir_bytes` by exact name with last-match-wins,
    # so extra roots must NOT reuse that family; their own family keeps the
    # primary series unambiguous while both tiers stay visible in one scrape.
    for e in extras:
        out.extend(
            [
                "# HELP mooncake_offload_tier_dir_bytes Apparent size of an"
                " additional offload tree root (e.g. the NAS drain tier).",
                "# TYPE mooncake_offload_tier_dir_bytes gauge",
                f'mooncake_offload_tier_dir_bytes{{tier="{e.tier}",root="{e.root}"}} {e.dir_bytes}',
                "# HELP mooncake_offload_tier_dir_files File count in an"
                " additional offload tree root.",
                "# TYPE mooncake_offload_tier_dir_files gauge",
                f'mooncake_offload_tier_dir_files{{tier="{e.tier}",root="{e.root}"}} {e.dir_files}',
            ]
        )
    for m in nfs_mounts:
        mount_label = _fmt_labels({"mount": m.mountpoint})
        out.extend(
            [
                "# HELP mooncake_offload_mount_total_bytes Total size of the offload"
                " NFS volume.",
                "# TYPE mooncake_offload_mount_total_bytes gauge",
            ]
        )
        if m.total_bytes is not None:
            out.append(f'mooncake_offload_mount_total_bytes{mount_label} {m.total_bytes}')
        out.extend(
            [
                "# HELP mooncake_offload_mount_free_bytes Free size of the offload"
                " NFS volume.",
                "# TYPE mooncake_offload_mount_free_bytes gauge",
            ]
        )
        if m.free_bytes is not None:
            out.append(f'mooncake_offload_mount_free_bytes{mount_label} {m.free_bytes}')
        if m.server_read_bytes is not None:
            server_label = _fmt_labels({"mount": m.mountpoint, "server": m.server})
            out.extend(
                [
                    "# HELP mooncake_offload_nfs_read_bytes_total NFS server read"
                    " bytes for the offload mount (mount-wide, all consumers).",
                    "# TYPE mooncake_offload_nfs_read_bytes_total counter",
                    f'mooncake_offload_nfs_read_bytes_total{server_label} {m.server_read_bytes}',
                ]
            )
        if m.server_written_bytes is not None:
            out.extend(
                [
                    "# HELP mooncake_offload_nfs_write_bytes_total NFS server"
                    " written bytes for the offload mount (mount-wide, all"
                    " consumers).",
                    "# TYPE mooncake_offload_nfs_write_bytes_total counter",
                    f'mooncake_offload_nfs_write_bytes_total{server_label} {m.server_written_bytes}',
                ]
            )
    out.extend(
        [
            "# HELP mooncake_offload_scrape_ok 1 when the last offload-dir refresh succeeded.",
            "# TYPE mooncake_offload_scrape_ok gauge",
            f"mooncake_offload_scrape_ok {1 if scrape_ok else 0}",
        ]
    )
    return "\n".join(out) + "\n"


class OffloadCollector:
    """Caches directory gauges on a refresh cadence; parses mountstats per scrape."""

    def __init__(
        self,
        root: str,
        extra_roots: Sequence[str] = (),
        staging_max_bytes: int = DEFAULT_STAGING_MAX_BYTES,
        mountstats_path: str = "/proc/self/mountstats",
        read_mountstats: Callable[[], str] | None = None,
    ) -> None:
        self.root = root
        self.extra_roots = list(extra_roots)
        self.staging_max_bytes = staging_max_bytes
        self._read_mountstats = read_mountstats or (
            lambda: Path(mountstats_path).read_text(encoding="ascii")
        )
        self._lock = threading.Lock()
        self._primary = RootSample(root=root, tier="staging", dir_bytes=0, dir_files=0)
        self._extras: list[RootSample] = []
        self._nfs_mounts: list[NfsMountSample] = []
        self._scrape_ok = False

    def refresh(self) -> None:
        """Re-walk every offload tree; statvfs the NFS mounts the roots resolve to."""
        # Walks and statvfs run OUTSIDE the lock: a missing root or a slow
        # multi-hundred-GB walk must not block a scrape against a stale-but-
        # consistent snapshot, and one root's failure must not abort the rest.
        primary, primary_ok = self._walk_root(self.root, tier="staging")
        extras: list[RootSample] = []
        extras_ok: list[bool] = []
        for r in self.extra_roots:
            sample, ok = self._walk_root(r, tier="drain")
            extras.append(sample)
            extras_ok.append(ok)

        # NFS capacity + wire counters come from whichever roots resolve to an
        # NFS mount, deduped by mountpoint. The staging parent matches no NFS
        # mount; the drain root resolves to /nas-1.
        mounts = parse_mountstats(self._read_mountstats())
        nfs_samples: dict[str, NfsMountSample] = {}
        for sample in [primary, *extras]:
            mount = find_mount_for(sample.root, mounts)
            if mount is None or mount.mountpoint in nfs_samples:
                continue
            total = free = None
            try:
                usage = shutil.disk_usage(mount.mountpoint)
                total, free = usage.total, usage.free
            except OSError:
                pass
            nfs_samples[mount.mountpoint] = NfsMountSample(
                mountpoint=mount.mountpoint,
                server=mount.server,
                total_bytes=total,
                free_bytes=free,
                server_read_bytes=mount.server_read_bytes,
                server_written_bytes=mount.server_written_bytes,
            )

        dir_ok = primary_ok and all(extras_ok)
        with self._lock:
            self._primary = primary
            self._extras = extras
            self._nfs_mounts = list(nfs_samples.values())
            self._scrape_ok = dir_ok

    def _walk_root(self, root: str, tier: str) -> RootSample:
        """Walk one root; degrade to zeroed gauges on any failure."""
        ok = False
        try:
            dir_bytes, dir_files = collect_dir(root)
            ok = True
        except (FileNotFoundError, PermissionError, OSError):
            dir_bytes, dir_files = 0, 0
        return RootSample(root=root, tier=tier, dir_bytes=dir_bytes, dir_files=dir_files), ok

    def start_refresher(self, interval: float) -> None:
        """Run refresh() in a daemon thread every interval seconds; never skip a cycle."""

        def loop() -> None:
            import time

            while True:
                started = time.monotonic()
                try:
                    self.refresh()
                except Exception as exc:
                    print(f"WARN offload refresher failed: {exc}", flush=True)
                elapsed = time.monotonic() - started
                if elapsed > interval:
                    print(
                        f"WARN offload refresh took {elapsed:.1f}s > interval {interval:.1f}s",
                        flush=True,
                    )
                time.sleep(max(interval - elapsed, 0.0))

        threading.Thread(target=loop, daemon=True).start()

    def render(self) -> str:
        """Render the exposition body: live mountstats counters + cached dir gauges."""
        with self._lock:
            primary, extras, nfs_mounts, scrape_ok = (
                self._primary,
                self._extras,
                self._nfs_mounts,
                self._scrape_ok,
            )
        return render_series(
            primary=primary,
            extras=extras,
            nfs_mounts=nfs_mounts,
            scrape_ok=scrape_ok,
            staging_max_bytes=self.staging_max_bytes,
        )


class _MetricsHandler(BaseHTTPRequestHandler):
    collector: OffloadCollector

    def do_GET(self) -> None:  # stdlib handler naming convention
        body = self.collector.render().encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        pass  # keep journald output to the single startup line


def serve(port: int, roots: Sequence[str], interval: float, staging_max_bytes: int) -> None:
    collector = OffloadCollector(
        roots[0], extra_roots=roots[1:], staging_max_bytes=staging_max_bytes
    )
    collector.refresh()
    collector.start_refresher(interval)

    class Handler(_MetricsHandler):
        pass

    Handler.collector = collector
    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    server.daemon_threads = True
    print(f"mooncake-offload-metrics listening on :{port} roots={list(roots)}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Mooncake offload-tier metrics exporter")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--root",
        action="append",
        default=None,
        help=(
            "Offload tree root to walk. Repeatable: the FIRST is the primary"
            " (staging tier, drives the dashboard cards); later roots emit"
            " additional labeled series (e.g. the NAS drain tier)."
            " Default: MOONCAKE_OFFLOAD_DIR or /nas-1/mooncake-store."
        ),
    )
    parser.add_argument("--interval", type=float, default=DEFAULT_INTERVAL)
    parser.add_argument(
        "--staging-max-bytes",
        type=int,
        default=int(os.environ.get("MOONCAKE_OFFLOAD_STAGING_MAX_BYTES", DEFAULT_STAGING_MAX_BYTES)),
        help="Staging tier byte cap (mooncake_offload_staging_max_bytes gauge).",
    )
    args = parser.parse_args(argv)
    roots = args.root or [
        os.environ.get("MOONCAKE_OFFLOAD_DIR", "/nas-1/mooncake-store")
    ]
    serve(args.port, roots, args.interval, args.staging_max_bytes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
