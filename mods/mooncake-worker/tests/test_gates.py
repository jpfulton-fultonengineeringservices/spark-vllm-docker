"""Offline tests for the mooncake-worker installer gates (torch-free)."""

from __future__ import annotations

import tempfile
import unittest
import unittest.mock
from pathlib import Path

from mooncake_worker import gates

MOD_DIR = Path(__file__).resolve().parents[1]


class VendorIntegrityTests(unittest.TestCase):
    def test_vendor_set_complete_and_gated(self):
        vendor = gates.verify_vendor()
        self.assertEqual(sorted(p.name for p in vendor.iterdir()), sorted(gates.PACKAGE_FILES))

    def test_classify_on_vendored_worker_is_ported(self):
        self.assertEqual(
            gates.classify_installed(gates.vendor_dir() / "worker.py"), "already_ported"
        )

    def test_md5_constants_match_vendored_and_documented_stock(self):
        self.assertEqual(gates.md5_of(gates.vendor_dir() / "worker.py"), gates.MD5_MOD_WORKER)
        # Recorded from the pristine image:
        #   docker run --rm --entrypoint sh vllm-node-b12x:latest \
        #     md5sum .../mooncake/store/worker.py  ->  b180493c... (2633 lines)
        self.assertEqual(gates.MD5_STOCK_WORKER, "b180493c964225f6a9282b30a47fb967")
        # Derived coder390 image (sha256:e9f6e32941ec…, eugr nightly-20261016
        # base, vLLM 0.32.0rc1.dev301+g1ac942555), re-probed 2026-10-25:
        #   docker run --rm --entrypoint bash vllm-node-b12x:latest \
        #     md5sum .../mooncake/store/worker.py  ->  8cc1f01a... (2494 lines)
        self.assertEqual(
            gates.MD5_STOCK_WORKER_CODER390_DERIVED, "8cc1f01a49aebec1ccda89d1545e445c"
        )
        self.assertEqual(
            gates.MD5_STOCK_WORKERS,
            (gates.MD5_STOCK_WORKER, gates.MD5_STOCK_WORKER_CODER390_DERIVED),
        )

    def test_derived_image_stock_worker_is_classified_stock(self):
        """The pinned coder390 image's stock worker admits the overlay."""
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as fh:
            fh.write(b"x = 1\n" * 100)
            path = fh.name
        try:
            with unittest.mock.patch.object(
                gates, "md5_of", return_value=gates.MD5_STOCK_WORKER_CODER390_DERIVED
            ):
                self.assertEqual(gates.classify_installed(path), "stock")
        finally:
            Path(path).unlink(missing_ok=True)

    def test_unknown_worker_is_classified_unknown(self):
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as fh:
            fh.write(b"x = 1\n")
            path = fh.name
        try:
            self.assertEqual(gates.classify_installed(path), "unknown")
        finally:
            Path(path).unlink(missing_ok=True)

    def test_incomplete_vendor_set_raises(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "vendor" / "store").mkdir(parents=True)
            with self.assertRaises(gates.GateError):
                gates.verify_vendor(root)


if __name__ == "__main__":
    unittest.main()
