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
MD5_MOD_WORKER = "b11bd659baf724298b2f347dad503b4e"

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
