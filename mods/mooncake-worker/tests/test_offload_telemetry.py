"""Tests for the Mooncake offload-tier metrics exporter."""

from __future__ import annotations

import threading
import urllib.request

from mooncake_worker.offload_telemetry import (
    OffloadCollector,
    collect_dir,
    find_mount_for,
    parse_mountstats,
)

MOUNTSTATS = """\
device sysfs mounted on /sys with fstype sysfs
device nfsd mounted on /proc/fs/nfsd with fstype nfsd
device systemd-1 mounted on /nas-1 with fstype autofs
device /dev/nvme0n1p2 mounted on /cluster-shared with fstype ext4
device 10.100.171.1:/export/cluster-shared mounted on /cluster-shared with fstype nfs4 statvers=1.1
\topts:\trw,vers=4.2,proto=tcp
\tage:\t100
\tcaps:\tcaps=0xc003c0b7
\tnfsv4:\tbm0=0xfdffafff,sessions
\tsec:\tflavor=1,pseudoflavor=1
\tevents:\t1 2 3 4 5 6 7 8 9 10
\tbytes:\t1000 2000 0 0 500 600 7 8
\tRPC iostats version: 1.1  p/v: 100003/4 (nfs)
device 10.10.10.202:/volume1/fulton-infra mounted on /nas-1 with fstype nfs4 statvers=1.1
\topts:\trw,vers=4.1,proto=tcp,nconnect=8
\tage:\t200
\tcaps:\tcaps=0xc003c0b7
\tnfsv4:\tbm0=0xfdffafff,sessions
\tsec:\tflavor=1,pseudoflavor=1
\tevents:\t1 2 3 4 5 6 7 8 9 10
\tbytes:\t3000 4000 0 0 700 800 9 10
\tRPC iostats version: 1.1  p/v: 100003/4 (nfs)
/legacy-nfs from 10.10.10.202:/volume1/fulton-infra fstype nfs4 (NFSv4.1) vers 4.1
\tint realtime: normal read 3000
\tbytes 3000 4000 0 0 700 800
/tmp type tmpfs (rw,relatime)
/lost-nfs from 10.0.0.9:/export/nope fstype nfs4 (NFSv4.0) vers 4.0
\tint realtime: normal read 1
"""


def test_parse_mountstats_selects_nfs_and_indices() -> None:
    mounts = parse_mountstats(MOUNTSTATS)
    assert "/tmp" not in mounts
    assert "/legacy-nfs" in mounts  # legacy header form also parsed
    nas = mounts["/nas-1"]
    assert nas.server == "10.10.10.202:/volume1/fulton-infra"
    assert nas.fstype == "nfs4"
    assert nas.server_read_bytes == 700
    assert nas.server_written_bytes == 800
    shared = mounts["/cluster-shared"]
    assert shared.server_read_bytes == 500
    assert shared.server_written_bytes == 600


def test_parse_mountstats_skips_bytesless() -> None:
    assert "/lost-nfs" not in parse_mountstats(MOUNTSTATS)


def test_find_mount_for_longest_prefix() -> None:
    mounts = parse_mountstats(MOUNTSTATS)
    assert find_mount_for("/nas-1/mooncake-store", mounts) is mounts["/nas-1"]
    assert find_mount_for("/opt/foo", mounts) is None
    assert find_mount_for("/nas-1", mounts) is mounts["/nas-1"]


def test_collect_dir_sums_apparent_size(tmp_path: object) -> None:
    from pathlib import Path

    root = Path(str(tmp_path))
    (root / "a").write_bytes(b"x" * 10)
    (root / "b").write_bytes(b"y" * 20)
    nested = root / "nested"
    nested.mkdir()
    (nested / "c").write_bytes(b"z" * 5)
    total, files = collect_dir(str(root))
    assert (total, files) == (35, 3)


def _read_fixture() -> str:
    return MOUNTSTATS


def test_render_emits_series_and_degrades() -> None:
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "data.bin").write_bytes(b"q" * 128)
        collector = OffloadCollector(str(root), read_mountstats=_read_fixture)
        collector.refresh()
        body = collector.render()
        assert 'mooncake_offload_dir_bytes{tier="staging"} 128' in body
        # A non-NFS primary resolves no NFS mount -> no mount gauges/counters.
        assert "mooncake_offload_mount_free_bytes" not in body
        assert "mooncake_offload_nfs_read_bytes_total" not in body
        assert "mooncake_offload_staging_max_bytes" in body
        assert "mooncake_offload_scrape_ok 1" in body

    # A single NFS root keeps the drain-tier shape: missing root -> zeroed
    # gauges + scrape_ok 0, but the /nas-1 mount still drives capacity + NFS
    # counters (the pre-multi-root behavior the dashboards rely on).
    collector = OffloadCollector("/nas-1/mooncake-store", read_mountstats=_read_fixture)
    collector.refresh()
    body = collector.render()
    assert "mooncake_offload_scrape_ok 0" in body
    assert 'mooncake_offload_dir_bytes{tier="staging"} 0' in body
    # Capacity gauges/statvfs only render on a host with real /nas-1; the
    # deterministic fixture signals are the mount-derived wire counters.
    assert "# HELP mooncake_offload_mount_total_bytes" in body
    assert "# HELP mooncake_offload_mount_free_bytes" in body
    assert 'mooncake_offload_nfs_read_bytes_total{mount="/nas-1",server=' in body
    assert 'mooncake_offload_nfs_write_bytes_total{mount="/nas-1",server=' in body


def test_two_root_tiers_keep_primary_unambiguous() -> None:
    """The deployed two-tier shape: NVMe staging primary + NAS drain extra.

    Pins the two contracts that beat the 2026-11-04 regression: the
    `mooncake_offload_dir_bytes` family carries ONLY the staging sample (the
    panel parser strips labels, last match wins, so an extra root reusing
    that family would blank the card), and the drain root -- not the staging
    parent -- drives the /nas-1 mount gauges + NFS wire counters.
    """
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as td:
        staging = Path(td) / "staging"
        drain = Path(td) / "drain"
        (staging / "bucket-a").mkdir(parents=True)
        (staging / "bucket-a" / "x.bin").write_bytes(b"q" * 100)
        (drain / "old").mkdir(parents=True)
        (drain / "old" / "y.bin").write_bytes(b"z" * 50)

        # Scenario 1: neither root is NFS-backed -> no mount gauges/counters
        # at all (statvfs of the node fs must never leak into these panels).
        collector = OffloadCollector(
            str(staging),
            extra_roots=[str(drain)],
            read_mountstats=_read_fixture,
            staging_max_bytes=274_877_906_944,
        )
        collector.refresh()
        body = collector.render()
        # Primary family: exactly ONE dir_bytes line, the staging sample.
        assert 'mooncake_offload_dir_bytes{tier="staging"} 100' in body
        assert body.count("mooncake_offload_dir_bytes{") == 1
        assert 'mooncake_offload_dir_files{tier="staging"} 1' in body
        # Extra tier under its own family, keyed by root.
        assert 'mooncake_offload_tier_dir_bytes{tier="drain",root=' in body
        assert " 50" in body.split('mooncake_offload_tier_dir_bytes{tier="drain",root=')[1].split("\n")[0]
        # Staging cap gauge present with the compose-contract value.
        assert "mooncake_offload_staging_max_bytes" in body
        assert "274877906944" in body
        assert "mooncake_offload_mount_total_bytes" not in body
        assert "mooncake_offload_mount_free_bytes" not in body
        assert "mooncake_offload_nfs_read_bytes_total" not in body
        assert "mooncake_offload_nfs_write_bytes_total" not in body
        assert "mooncake_offload_scrape_ok 1" in body

        # Scenario 2: production shape -- drain root under /nas-1 resolves the
        # fixture's /nas-1 NFS mount, so the drain root (not the staging
        # parent) drives the mount gauges' HELP lines and the NFS wire
        # counters. The staging parent contributes nothing NFS-shaped. The
        # drain walk degrades to zeroed gauges where /nas-1 is absent (dev
        # hosts); wire counters come from mountstats either way.
        collector = OffloadCollector(
            str(staging),
            extra_roots=["/nas-1/mooncake-store"],
            read_mountstats=_read_fixture,
            staging_max_bytes=274_877_906_944,
        )
        collector.refresh()
        body = collector.render()
        assert "# HELP mooncake_offload_mount_total_bytes" in body
        assert "# HELP mooncake_offload_mount_free_bytes" in body
        assert 'mooncake_offload_nfs_read_bytes_total{mount="/nas-1",server=' in body
        assert 'mooncake_offload_nfs_write_bytes_total{mount="/nas-1",server=' in body
        # Primary family STILL only the staging sample.
        assert body.count("mooncake_offload_dir_bytes{") == 1


def test_http_endpoint_serves_metrics() -> None:
    import tempfile
    from http.server import ThreadingHTTPServer

    from mooncake_worker.offload_telemetry import _MetricsHandler

    with tempfile.TemporaryDirectory() as td:
        collector = OffloadCollector(td, read_mountstats=_read_fixture)
        collector.refresh()

        class Handler(_MetricsHandler):
            pass

        Handler.collector = collector
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{server.server_address[1]}/metrics", timeout=5
            ) as resp:
                body = resp.read().decode()
            assert 'mooncake_offload_dir_bytes{tier="staging"}' in body
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
