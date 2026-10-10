"""Offline tests for the matched-set installer (torch-free, vendored-only)."""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from mooncake_worker import gates, installer


def _fake_site_with_stock_worker() -> tuple[Path, Path]:
    """Build a fake dist-packages tree whose store dir holds a stub worker."""
    site = Path(tempfile.mkdtemp())
    store = site / installer.STORE_SUBPATH
    store.mkdir(parents=True)
    worker = store / "worker.py"
    worker.write_text("# stock stub\n")
    return site, worker


class InstallerTests(unittest.TestCase):
    def test_unknown_stock_worker_fails_closed(self):
        site, worker = _fake_site_with_stock_worker()
        try:
            with self.assertRaises(installer.InstallError):
                installer.install(site, apply_envs_shim=False)
            self.assertEqual(gates.classify_installed(worker), "unknown")
        finally:
            shutil.rmtree(site, ignore_errors=True)

    def test_missing_store_dir_raises(self):
        site = Path(tempfile.mkdtemp())
        try:
            with self.assertRaises(installer.InstallError):
                installer.install(site, apply_envs_shim=False)
        finally:
            shutil.rmtree(site, ignore_errors=True)

    def test_verify_rejects_non_ported(self):
        site, _worker = _fake_site_with_stock_worker()
        try:
            with self.assertRaises(installer.InstallError):
                installer.verify(site)
        finally:
            shutil.rmtree(site, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
