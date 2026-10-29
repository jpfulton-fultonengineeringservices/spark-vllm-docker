"""Host-side Mooncake offload-tier metrics exporter (stdlib only).

Serves Prometheus text exposition on :9004 with:

- ``mooncake_offload_dir_bytes`` / ``mooncake_offload_dir_files``: apparent size and
  file count of the Mooncake offload tree (default /nas-1/mooncake-store).
- ``mooncake_offload_mount_total_bytes`` / ``mooncake_offload_mount_free_bytes``:
  NAS volume capacity via statvfs on the matched NFS mountpoint.
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
from collections.abc import Callable
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

DEFAULT_PORT = 9004
DEFAULT_INTERVAL = 60.0

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
    dir_bytes: int,
    dir_files: int,
    mount_total: int | None,
    mount_free: int | None,
    scrape_ok: bool,
    mount_label: str,
    nfs_read: int | None,
    nfs_write: int | None,
    server_label: str,
) -> str:
    """Assemble the Prometheus text exposition body (pure, no I/O)."""
    out = [
        "# HELP mooncake_offload_dir_bytes Apparent size of the Mooncake offload tree.",
        "# TYPE mooncake_offload_dir_bytes gauge",
        f'mooncake_offload_dir_bytes{mount_label} {dir_bytes}',
        "# HELP mooncake_offload_dir_files File count in the Mooncake offload tree.",
        "# TYPE mooncake_offload_dir_files gauge",
        f'mooncake_offload_dir_files{mount_label} {dir_files}',
        "# HELP mooncake_offload_mount_total_bytes Total size of the offload NAS volume.",
        "# TYPE mooncake_offload_mount_total_bytes gauge",
    ]
    if mount_total is not None:
        out.append(f'mooncake_offload_mount_total_bytes{mount_label} {mount_total}')
    out.extend(
        [
            "# HELP mooncake_offload_mount_free_bytes Free size of the offload NAS volume.",
            "# TYPE mooncake_offload_mount_free_bytes gauge",
        ]
    )
    if mount_free is not None:
        out.append(f'mooncake_offload_mount_free_bytes{mount_label} {mount_free}')
    if nfs_read is not None:
        out.extend(
            [
                "# HELP mooncake_offload_nfs_read_bytes_total NFS server read bytes for the"
                " offload mount (mount-wide, all consumers).",
                "# TYPE mooncake_offload_nfs_read_bytes_total counter",
                f'mooncake_offload_nfs_read_bytes_total{server_label} {nfs_read}',
            ]
        )
    if nfs_write is not None:
        out.extend(
            [
                "# HELP mooncake_offload_nfs_write_bytes_total NFS server written bytes for"
                " the offload mount (mount-wide, all consumers).",
                "# TYPE mooncake_offload_nfs_write_bytes_total counter",
                f'mooncake_offload_nfs_write_bytes_total{server_label} {nfs_write}',
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
        mountstats_path: str = "/proc/self/mountstats",
        read_mountstats: Callable[[], str] | None = None,
    ) -> None:
        self.root = root
        self._read_mountstats = read_mountstats or (
            lambda: Path(mountstats_path).read_text(encoding="ascii")
        )
        self._lock = threading.Lock()
        self._dir_bytes = 0
        self._dir_files = 0
        self._mount_total: int | None = None
        self._mount_free: int | None = None
        self._mount_label = '{"mount":"none"}'
        self._scrape_ok = False

    def refresh(self) -> None:
        """Re-walk the offload tree and statvfs the matched mountpoint."""
        mounts = parse_mountstats(self._read_mountstats())
        mount = find_mount_for(self.root, mounts)
        dir_ok = False
        try:
            dir_bytes, dir_files = collect_dir(self.root)
            dir_ok = True
        except (FileNotFoundError, PermissionError, OSError):
            dir_bytes, dir_files = 0, 0
        mount_total: int | None = None
        mount_free: int | None = None
        try:
            stat_target = mount.mountpoint if mount is not None else self.root
            usage = shutil.disk_usage(stat_target)
            mount_total, mount_free = usage.total, usage.free
        except OSError:
            pass
        with self._lock:
            self._dir_bytes = dir_bytes
            self._dir_files = dir_files
            self._mount_total = mount_total
            self._mount_free = mount_free
            self._mount_label = _fmt_labels({"mount": mount.mountpoint if mount else "none"})
            self._scrape_ok = dir_ok

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
        mounts = parse_mountstats(self._read_mountstats())
        mount = find_mount_for(self.root, mounts)
        with self._lock:
            dir_bytes, dir_files = self._dir_bytes, self._dir_files
            mount_total, mount_free = self._mount_total, self._mount_free
            mount_label, scrape_ok = self._mount_label, self._scrape_ok
        if mount is not None:
            return render_series(
                dir_bytes=dir_bytes,
                dir_files=dir_files,
                mount_total=mount_total,
                mount_free=mount_free,
                scrape_ok=scrape_ok,
                mount_label=mount_label,
                nfs_read=mount.server_read_bytes,
                nfs_write=mount.server_written_bytes,
                server_label=_fmt_labels(
                    {"mount": mount.mountpoint, "server": mount.server}
                ),
            )
        return render_series(
            dir_bytes=dir_bytes,
            dir_files=dir_files,
            mount_total=mount_total,
            mount_free=mount_free,
            scrape_ok=scrape_ok,
            mount_label=mount_label,
            nfs_read=None,
            nfs_write=None,
            server_label="",
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


def serve(port: int, root: str, interval: float) -> None:
    collector = OffloadCollector(root)
    collector.refresh()
    collector.start_refresher(interval)

    class Handler(_MetricsHandler):
        pass

    Handler.collector = collector
    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    server.daemon_threads = True
    print(f"mooncake-offload-metrics listening on :{port} root={root}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Mooncake offload-tier metrics exporter")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--root",
        default=os.environ.get("MOONCAKE_OFFLOAD_DIR", "/nas-1/mooncake-store"),
    )
    parser.add_argument("--interval", type=float, default=DEFAULT_INTERVAL)
    args = parser.parse_args(argv)
    serve(args.port, args.root, args.interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
