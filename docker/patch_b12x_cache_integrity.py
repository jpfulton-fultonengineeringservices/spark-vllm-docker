#!/usr/bin/env python3
"""Apply the temporary upstream CuTe cache-integrity fix to installed B12X."""

import argparse
import importlib.machinery
import importlib.util
from pathlib import Path
import subprocess
import sys


PATCH = Path(__file__).with_name("b12x-cache-integrity.patch")


def candidate_site_dirs(site_dirs: list[Path]) -> list[Path]:
    """Import roots that may hold flashinfer, found without executing it."""
    candidates = list(site_dirs)
    try:
        spec = importlib.machinery.PathFinder.find_spec("flashinfer")
    except Exception:  # An unusable sys.path entry must not fail the build.
        spec = None
    for location in (spec.submodule_search_locations or []) if spec else []:
        candidates.append(Path(location).parent)
    candidates.extend(Path(entry) for entry in sys.path if entry)
    return candidates


def vendored_package(site_dirs: list[Path]) -> Path | None:
    """Locate flashinfer's bundled b12x package without importing GPU code.

    Every filesystem probe runs first: importing ``flashinfer`` to resolve a
    spec executes its package initializer, which pulls in Torch. The spec probe
    remains only as a fallback for layouts the directory scan cannot see.
    """
    for site in candidate_site_dirs(site_dirs):
        package = site / "flashinfer/experimental/b12x"
        if (package / "_lib/compiler.py").is_file():
            return package
    try:
        spec = importlib.util.find_spec("flashinfer.experimental.b12x")
    except Exception:  # A broken flashinfer install is reported, not raised.
        return None
    locations = list(spec.submodule_search_locations or []) if spec else []
    for location in [Path(item) for item in locations]:
        if (location / "_lib/compiler.py").is_file():
            return location
    return None


def installed_package() -> tuple[Path | None, str]:
    """Return the installed package and whether it already carries the fix.

    Newer flashinfer-python wheels vendor b12x as
    ``flashinfer/experimental/b12x`` and leave only a ``site-packages/b12x``
    import shim behind. The vendored tree either still needs the reviewed
    upstream diff (``unpatched-vendored``) or already contains it (``bundled``).
    """
    spec = importlib.util.find_spec("b12x")
    if spec is None:
        package = vendored_package([])
        if package is not None:
            return package, ("bundled" if (package / "_lib/cache_integrity.py").is_file()
                             else "unpatched-vendored")
        return None, "absent"
    locations = [Path(item) for item in spec.submodule_search_locations or []]
    standalone = len(locations) == 1 and locations[0].name == "b12x"
    if standalone:
        root = locations[0].parent
        if (root / "b12x/_lib/compiler.py").is_file():
            return root, "standalone"
    package = vendored_package([location.parent for location in locations])
    if package is not None:
        return package, ("bundled" if (package / "_lib/cache_integrity.py").is_file()
                         else "unpatched-vendored")
    if not standalone:
        raise SystemExit("Unable to locate the installed B12X package")
    return locations[0].parent, "unknown"


def apply_patch(root: Path) -> bool:
    """Check every package hunk before writing; accept an identical reapply."""
    command = ["git", "apply"]
    if not (root / "b12x/_lib/compile_plan.py").exists():
        # PyPI 1.3.0 has the same object cache but no preparation planner.
        # Its normal compile/load path still receives integrity and fsync fixes.
        command.append("--exclude=b12x/_lib/compile_plan.py")
    command.append("--include=b12x/**")
    result = subprocess.run(
        command + ["--check", str(PATCH)], cwd=root, capture_output=True, text=True,
    )
    if result.returncode:
        repeated = subprocess.run(
            command + ["--reverse", "--check", str(PATCH)],
            cwd=root, capture_output=True, text=True,
        )
        if repeated.returncode == 0:
            return False
        raise RuntimeError(
            "B12X source differs from the reviewed cache implementation; "
            "refresh the upstream patch before building.\n" + result.stderr.strip()
        )
    subprocess.run(command + [str(PATCH)], cwd=root, check=True)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_root", nargs="?", type=Path,
                        help="Checkout or site-packages directory containing b12x/")
    parser.add_argument("--installed", action="store_true")
    args = parser.parse_args()
    if args.installed:
        if args.source_root is not None:
            parser.error("Use either source_root or --installed")
        # Locate the package without importing B12X, Torch, or initializing CUDA.
        root, layout = installed_package()
        if layout == "absent":
            print("B12X is not installed; cache-integrity patch is not applicable")
            return
        if layout == "bundled":
            print("Flashinfer bundles the B12X cache-integrity fix; nothing to patch")
            return
        if layout == "unpatched-vendored":
            raise SystemExit(
                f"B12X at {root} lacks _lib/cache_integrity.py; the reviewed patch "
                "targets a standalone b12x package, refresh it for this layout"
            )
    elif args.source_root is not None:
        root = args.source_root.resolve()
    else:
        parser.error("Provide source_root or --installed")
    if root is None:
        raise SystemExit("Unable to locate the installed B12X package")
    if not (root / "b12x/_lib/compiler.py").is_file():
        raise SystemExit(f"B12X CuTe compiler not found under {root}")
    try:
        changed = apply_patch(root)
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"Unable to patch B12X cache integrity: {error}") from error
    print("Patched B12X CuTe cache validation and durable writes" if changed else
          "B12X cache integrity is already patched; skipping")


if __name__ == "__main__":
    main()
