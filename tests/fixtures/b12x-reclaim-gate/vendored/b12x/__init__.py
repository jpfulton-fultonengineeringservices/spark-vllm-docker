"""Shim-only b12x package mirroring the current image layout.

The real implementation lives at flashinfer.experimental.b12x; this package
contains nothing but a sys.meta_path aliaser, exactly like
/usr/local/lib/python3.12/dist-packages/b12x/__init__.py in vllm-node-b12x.
"""

import importlib.util
import sys

_REAL = "b12x"
_TARGET = "flashinfer.experimental.b12x"


class _VendoredAliaser:
    def find_spec(self, fullname, path=None, target=None):
        if fullname != _REAL and not fullname.startswith(_REAL + "."):
            return None
        mapped = _TARGET + fullname[len(_REAL):]
        try:
            upstream = importlib.util.find_spec(mapped)
        except (ImportError, ModuleNotFoundError):
            return None
        if upstream is None or upstream.origin is None:
            return None
        return importlib.util.spec_from_file_location(
            fullname,
            upstream.origin,
            submodule_search_locations=upstream.submodule_search_locations,
        )


if not any(isinstance(f, _VendoredAliaser) for f in sys.meta_path):
    sys.meta_path.insert(0, _VendoredAliaser())
