"""Install the vendored host-staged Mooncake store package over the image's copy.

Matched-set atomic install: all eight store/ files move together because the
FES worker reads ``ReqMeta.partial_tail_offloads`` while the stock image's
``data.py`` defines ``boundary_state_offloads`` — a lone-worker swap would
``AttributeError``.
"""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from . import envs_shim, gates

VLLM_SITE_DEFAULT = "/usr/local/lib/python3.12/dist-packages"

STORE_SUBPATH = Path("vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store")


class InstallError(RuntimeError):
    """Installer refused a state it cannot safely handle."""


@dataclass(frozen=True)
class SitePaths:
    """Resolved absolute paths for one vLLM install."""

    site: Path
    store_dir: Path
    envs_py: Path

    @classmethod
    def resolve(cls, site: str | Path = VLLM_SITE_DEFAULT) -> SitePaths:
        site_path = Path(site)
        return cls(
            site=site_path,
            store_dir=site_path / STORE_SUBPATH,
            envs_py=site_path / "vllm" / "envs.py",
        )


def _atomic_copy(src: Path, dst: Path) -> None:
    _, tmp_name = tempfile.mkstemp(dir=dst.parent, prefix=f".{dst.name}.", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with src.open("rb") as fh_in, tmp.open("wb") as fh_out:
            shutil.copyfileobj(fh_in, fh_out)
        tmp.replace(dst)
    finally:
        tmp.unlink(missing_ok=True)


def install(
    site: str | Path = VLLM_SITE_DEFAULT,
    *,
    package_root: Path | None = None,
    apply_envs_shim: bool = True,
) -> str:
    """Install the vendored package. Returns a human-readable status line.

    Raises InstallError (fail-closed) on unknown/partial states.
    """
    paths = SitePaths.resolve(site)
    if not paths.store_dir.is_dir():
        raise InstallError(f"vLLM store dir missing: {paths.store_dir} (vLLM not installed?)")

    vendor = gates.verify_vendor(package_root)

    state = gates.classify_installed(paths.store_dir / "worker.py")
    if state == "already_ported":
        return f"already installed (worker md5={gates.MD5_MOD_WORKER})"
    if state == "unknown":
        actual = gates.md5_of(paths.store_dir / "worker.py")
        raise InstallError(
            f"installed worker md5 {actual} is not a known base"
            f" (stock={gates.MD5_STOCK_WORKER}, ported={gates.MD5_MOD_WORKER});"
            " re-port the vendored set against the new image — do NOT blind-overwrite"
        )

    for name in gates.PACKAGE_FILES:
        _atomic_copy(vendor / name, paths.store_dir / name)

    new_md5 = gates.md5_of(paths.store_dir / "worker.py")
    if new_md5 != gates.MD5_MOD_WORKER:
        raise InstallError(f"post-install worker md5 {new_md5} mismatch")

    if apply_envs_shim and paths.envs_py.is_file():
        envs_shim.inject(paths.envs_py)

    return (
        f"installed host-staged store package (8 files;"
        f" worker {gates.MD5_STOCK_WORKER} -> {gates.MD5_MOD_WORKER})"
    )


def verify(site: str | Path = VLLM_SITE_DEFAULT) -> str:
    """Confirm the installed state is the vendored one (and envs resolves)."""
    paths = SitePaths.resolve(site)
    state = gates.classify_installed(paths.store_dir / "worker.py")
    if state != "already_ported":
        raise InstallError(f"installed worker state is {state!r}, expected 'already_ported'")
    if paths.envs_py.is_file() and envs_shim.needs_shim(paths.envs_py):
        raise InstallError(
            f"{envs_shim.TARGET_ENV} absent from {paths.envs_py} (envs shim not applied)"
        )
    return f"verified host-staged store package (worker md5={gates.MD5_MOD_WORKER})"
