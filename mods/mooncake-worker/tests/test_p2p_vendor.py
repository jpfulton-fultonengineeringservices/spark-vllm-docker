"""Offline gates for the vendored P2P MooncakeConnector set.

Torch-free, per the house rule (gates.py stays importable on host + container).
"""

from __future__ import annotations

import re
import unittest.mock
from pathlib import Path

import pytest

from mooncake_worker import gates


def test_p2p_vendor_set_complete_and_parses() -> None:
    vendor = gates.verify_p2p_vendor()
    assert vendor.is_dir()
    for name in gates.P2P_PACKAGE_FILES:
        assert (vendor / name).is_file()


def test_p2p_facade_md5_matches_tip() -> None:
    vendor = gates.p2p_vendor_dir()
    assert gates.md5_of(vendor / "mooncake_connector.py") == gates.MD5_MOD_P2P_CONNECTOR


def test_p2p_port_shim_present_and_exhaustive() -> None:
    """The single get_kv_cache_layout drift site is shimmed; no other refs."""
    vendor = gates.p2p_vendor_dir()
    worker_src = (vendor / "worker.py").read_text()
    assert "def _kv_cache_layout(" in worker_src
    assert "self.kv_cache_layout = _kv_cache_layout(self.vllm_config)" in worker_src
    for name in gates.P2P_PACKAGE_FILES:
        src = (vendor / name).read_text()
        for match in re.finditer(r"\bget_kv_cache_layout\b", src):
            context = src[max(0, match.start() - 160) : match.start() + 40]
            # Only the shim's own docstring may mention the symbol.
            assert "lacks" in context, f"raw get_kv_cache_layout ref in {name}"


def test_p2p_no_rdma_utils_shipped() -> None:
    """rdma_utils.py is orphaned upstream (no importers) — must not ship."""
    vendor = gates.p2p_vendor_dir()
    assert not (vendor / "rdma_utils.py").exists()


def test_classify_p2p_fails_closed_on_uncaptured_stock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stock md5 captured 2026-10-12; simulate the pre-capture sentinel state."""
    monkeypatch.setattr(gates, "MD5_STOCK_P2P_CONNECTOR", "UNKNOWN-CAPTURE-ON-NODE")
    with pytest.raises(gates.GateError, match="not captured"):
        gates.classify_p2p_installed("whatever")


def test_classify_p2p_fails_closed_on_missing_facade() -> None:
    """A nonexistent facade path gates, never crashes with FileNotFoundError."""
    with pytest.raises(gates.GateError, match="missing"):
        gates.classify_p2p_installed("whatever")


def test_derived_image_stock_p2p_facade_is_classified_stock(
    tmp_path: Path,
) -> None:
    """The pinned coder390 image's monolith facade admits the overlay.

    Derived image sha256:e9f6e32941ec… (eugr nightly-20261016 base, vLLM
    0.32.0rc1.dev301+g1ac942555), probed 2026-10-25: 3082 lines,
    host_staging count 0.
    """
    facade = tmp_path / "mooncake_connector.py"
    facade.write_text("# derived image stock facade\n")
    with unittest.mock.patch.object(
        gates, "md5_of", return_value=gates.MD5_STOCK_P2P_CONNECTOR_CODER390_DERIVED
    ):
        assert gates.classify_p2p_installed(facade) == "stock_p2p"


def test_p2p_stock_base_tuple_covers_both_lineages() -> None:
    assert gates.MD5_STOCK_P2P_CONNECTORS == (
        "5697ebd6979f7fdcc92106a8f70103d5",  # vllm-node-b12x 2026-10-12
        "c626af08807a0b78350850e6370ecc90",  # coder390-derived e9f6e329
    )


def test_envs_shim_p2p_roundtrip(tmp_path: Path) -> None:
    """inject_p2p_envs: adds entries, idempotent, AST-valid, verify passes."""
    from mooncake_worker import envs_shim

    envs_py = tmp_path / "envs.py"
    envs_py.write_text(
        "import os\n"
        "class Envs:\n"
        "    environment_variables = {\n" + envs_shim.DICT_ANCHOR + "    }\n"
    )
    added = envs_shim.inject_p2p_envs(envs_py)
    assert added == 2
    assert envs_shim.inject_p2p_envs(envs_py) == 0  # idempotent
    envs_shim.verify_p2p_envs(envs_py)
    import ast

    ast.parse(envs_py.read_text())


def test_envs_shim_p2p_fail_closed_on_unknown_anchor(tmp_path: Path) -> None:
    from mooncake_worker import envs_shim

    envs_py = tmp_path / "envs.py"
    envs_py.write_text("import os\nX = {}\n")
    with pytest.raises(envs_shim.EnvShimError, match="anchor"):
        envs_shim.inject_p2p_envs(envs_py)
