"""exl3pack — shared EXL3 pack builder.

One library, many models: the pack pipeline logic lives here; per-model
directories supply only a :class:`~exl3pack.spec.PackSpec`. The torch/b12x/
exllamav3-dependent stages (``convert``, ``repack``, ``assemble``) import those
at call time so that the geometry/spec/paths/status/monitor surface stays
importable on a host without the GPU stack.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
