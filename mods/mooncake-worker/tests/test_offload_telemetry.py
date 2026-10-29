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
/nas-1 from 10.10.10.202:/volume1/fulton-infra fstype nfs4 (NFSv4.1) vers 4.1
\tint realtime: normal read 3000
\tbytes 3000 4000 0 0 700 800
/tmp type tmpfs (rw,relatime)
/lost-nfs from 10.0.0.9:/export/nope fstype nfs4 (NFSv4.0) vers 4.0
\tint realtime: normal read 1
"""


def test_parse_mountstats_selects_nfs_and_indices() -> None:
    mounts = parse_mountstats(MOUNTSTATS)
    assert "/tmp" not in mounts
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
        assert 'mooncake_offload_dir_bytes{mount="none"} 128' in body
        assert "mooncake_offload_scrape_ok 1" in body
        assert "mooncake_offload_nfs_read_bytes_total" not in body

    collector = OffloadCollector("/nas-1/mooncake-store", read_mountstats=_read_fixture)
    collector.refresh()
    body = collector.render()
    assert "mooncake_offload_scrape_ok 0" in body
    assert 'mooncake_offload_dir_bytes{mount="/nas-1"} 0' in body
    assert 'mooncake_offload_nfs_read_bytes_total{mount="/nas-1",server=' in body
    assert 'mooncake_offload_nfs_write_bytes_total{mount="/nas-1",server=' in body


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
            assert "mooncake_offload_mount_free_bytes" in body
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
