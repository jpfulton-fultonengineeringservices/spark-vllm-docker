"""Integrity gates for the vendored Mooncake store package.

Fail-closed md5 gates against known image states. No vLLM imports — this
module must stay torch-free so it runs on the host and in the container
alike.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

# The stock connector shipped in vllm-node-b12x:latest
# (vLLM 0.1.dev21546+g502d6cb5a, 2633 lines, host_staging count 0).
MD5_STOCK_WORKER = "b180493c964225f6a9282b30a47fb967"

# The vendored branch tip
# (Fulton-Engineering-Services/vllm @ cuda13.3-aarch64-gb10-glm5next-modular,
#  4e79d6be6f, 2574 lines — carries the completed host-staging series).
# 2026-10-11: + LookupKeyClient SND/RCVTIMEO + rebuild-on-Again (2574→2619
# lines, fix-prefill-lookup-livelock).
MD5_MOD_WORKER = "09388dea01fd935d7cb0335fb0cf736e"

PACKAGE_FILES: tuple[str, ...] = (
    "__init__.py",
    "connector.py",
    "coordinator.py",
    "data.py",
    "metrics.py",
    "protocol.py",
    "scheduler.py",
    "worker.py",
)

# --- P2P connector vendored set -------------------------------------------
# The image ships the P2P MooncakeConnector as a 2,666-line monolith
# (mooncake_connector.py, pre-decomposition lineage) plus the welded
# siblings (base.py, scheduler.py, worker.py). The Fulton fork
# (mooncake-host-staging-decompose @ 4a2519311) replaces the whole
# directory with the 13-file decomposed package carrying host-staging
# (_PinnedStagingArena: D2H -> pinned RDMA -> H2D; GB10 has no GPUDirect).
# One port edit vs the branch tip: the single get_kv_cache_layout() call
# in worker.py is shimmed (_kv_cache_layout) because the b12x image tree
# lacks that helper; every file's import of it was dropped with
# NULL_BLOCK_ID kept. rdma_utils.py is NOT shipped (orphaned, no importers).
P2P_PACKAGE_FILES: tuple[str, ...] = (
    "__init__.py",
    "base.py",
    "mooncake_connector.py",
    "scheduler.py",
    "worker.py",
    "_worker_send.py",
    "_worker_receive.py",
    "_worker_setup.py",
    "_transfer_planning.py",
    "_protocol.py",
    "staging.py",
    "stats.py",
    "mooncake_utils.py",
)

# Vendored branch tip (mooncake-host-staging-decompose @ 4a2519311) after
# the port shim. Facade md5 doubles as the install-verify sentinel (the
# file the connector factory imports by class name).
MD5_MOD_P2P_CONNECTOR = "8bd9c7fc214e032271c843dbce51ce5d"

# The stock image's monolith connector. Captured live 2026-10-12 on
# gx10-node1 vllm_node (vLLM 0.1.dev21546+g502d6cb5a.d20261003):
# image tree is monolith + mooncake_utils/rdma_utils/stats + store/.
MD5_STOCK_P2P_CONNECTOR = "5697ebd6979f7fdcc92106a8f70103d5"


class GateError(RuntimeError):
    """A fail-closed gate rejected the operation."""


def md5_of(path: str | Path) -> str:
    digest = hashlib.md5()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def vendor_dir(package_root: Path | None = None) -> Path:
    root = package_root if package_root is not None else Path(__file__).resolve().parent
    vendor = root / "vendor" / "store"
    if not vendor.is_dir():
        raise GateError(f"vendored store package missing: {vendor}")
    return vendor


def p2p_vendor_dir(package_root: Path | None = None) -> Path:
    root = package_root if package_root is not None else Path(__file__).resolve().parent
    vendor = root / "vendor" / "p2p"
    if not vendor.is_dir():
        raise GateError(f"vendored p2p package missing: {vendor}")
    return vendor


def verify_p2p_vendor(package_root: Path | None = None) -> Path:
    """Ensure the vendored P2P set is complete, parses, and facade md5 matches."""
    vendor = p2p_vendor_dir(package_root)
    for name in P2P_PACKAGE_FILES:
        path = vendor / name
        if not path.is_file():
            raise GateError(f"p2p vendor set incomplete: missing {name}")
        try:
            compile(path.read_text(), str(path), "exec")
        except SyntaxError as exc:
            raise GateError(f"vendored p2p {name} does not parse: {exc}") from exc
    actual = md5_of(vendor / "mooncake_connector.py")
    if actual != MD5_MOD_P2P_CONNECTOR:
        raise GateError(
            f"vendored p2p mooncake_connector.py md5 {actual} != expected"
            f" {MD5_MOD_P2P_CONNECTOR} (vendor set corrupted; re-extract from"
            " branch tip 4a2519311 and re-apply the port shim)"
        )
    # The port shim must be present: no get_kv_cache_layout code refs outside
    # the shim docstring, and the shim function exists in worker.py.
    worker_src = (vendor / "worker.py").read_text()
    if "def _kv_cache_layout(" not in worker_src:
        raise GateError("p2p port shim (_kv_cache_layout) missing from worker.py")
    return vendor


def classify_p2p_installed(installed_connector: str | Path) -> str:
    """Classify the installed P2P mooncake_connector.py state.

    Returns one of:
      - "p2p_ported"  (facade == vendored tip)
      - "stock_p2p"   (facade == pristine image monolith)
      - "unknown"     (image drift; installer must fail closed)
    """
    if MD5_STOCK_P2P_CONNECTOR.startswith("UNKNOWN"):
        raise GateError(
            "MD5_STOCK_P2P_CONNECTOR not captured yet — probe the image's"
            " vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py"
            " md5 on the cluster node before installing"
        )
    actual = md5_of(installed_connector)
    if actual == MD5_MOD_P2P_CONNECTOR:
        return "p2p_ported"
    if actual == MD5_STOCK_P2P_CONNECTOR:
        return "stock_p2p"
    return "unknown"


def verify_vendor(package_root: Path | None = None) -> Path:
    """Ensure the vendored set is complete, parses, and worker md5 matches."""
    vendor = vendor_dir(package_root)
    for name in PACKAGE_FILES:
        path = vendor / name
        if not path.is_file():
            raise GateError(f"vendor set incomplete: missing {name}")
        try:
            compile(path.read_text(), str(path), "exec")
        except SyntaxError as exc:
            raise GateError(f"vendored {name} does not parse: {exc}") from exc
    actual = md5_of(vendor / "worker.py")
    if actual != MD5_MOD_WORKER:
        raise GateError(
            f"vendored worker.py md5 {actual} != expected {MD5_MOD_WORKER}"
            " (vendor set corrupted; re-extract from the branch tip)"
        )
    return vendor


def classify_installed(installed_worker: str | Path) -> str:
    """Classify the installed worker.py state for the installer.

    Returns one of:
      - "already_ported"  (worker == vendored branch tip)
      - "stock"           (worker == pristine vllm-node-b12x)
      - "unknown"         (image drift; installer must fail closed)
    """
    actual = md5_of(installed_worker)
    if actual == MD5_MOD_WORKER:
        return "already_ported"
    if actual == MD5_STOCK_WORKER:
        return "stock"
    return "unknown"


def _selftest() -> int:
    try:
        vendor = verify_vendor()
    except GateError as exc:
        print(f"SELFTEST FAIL: {exc}", file=sys.stderr)
        return 1
    worker = vendor / "worker.py"
    state = classify_installed(worker)
    if state != "already_ported":
        print(
            f"SELFTEST FAIL: classify_installed on the vendored worker returned {state!r}",
            file=sys.stderr,
        )
        return 1
    print(f"SELFTEST OK: vendor={vendor} worker_md5={MD5_MOD_WORKER}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
