"""Per-model pack specification.

A spec is the *only* thing a model directory must provide. It carries the
auto-detected geometry (for cross-checking the real checkpoint), the codebook /
bitrate, the dense-format label, and the node-map key used to resolve a
node-local source.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Geometry:
    """Expected model geometry — must match what ``detect()`` reads."""

    architecture: str
    hidden_size: int
    intermediate_size: int
    num_experts: int
    num_slots: int
    moe_layer_count: int
    num_hidden_layers: int
    dense_layers: list[int] = field(default_factory=list)

    def as_comparable(self) -> dict[str, object]:
        """Fields compared in the detect-parity check (no per-layer lists)."""
        return {
            "architecture": self.architecture,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_experts": self.num_experts,
            "num_slots": self.num_slots,
            "moe_layer_count": self.moe_layer_count,
            "num_hidden_layers": self.num_hidden_layers,
            "dense_layers": self.dense_layers,
        }


@dataclass(frozen=True)
class PackSpec:
    """Everything the shared pipeline needs to know about one model."""

    slug: str
    geometry: Geometry
    codebook: str = "mcg"
    bits: int = 3
    dense_format: str = "bf16"
    # Module basenames the exl3 method leaves unquantized (bf16 dense).
    ignored_layers: list[str] = field(default_factory=list)
    # Key into node-model-map.json for resolving a node-local source.
    node_map_key: str | None = None

    def __post_init__(self) -> None:
        if self.codebook not in ("mcg", "lut_e4m3", "lut_fp16"):
            raise ValueError(
                f"spec {self.slug}: codebook {self.codebook!r} is not a b12x "
                "exl3-v1 codebook (mcg | lut_e4m3 | lut_fp16)"
            )
        if self.geometry.num_slots == 0:
            raise ValueError(
                f"spec {self.slug}: num_slots is 0 — moe_intermediate_size must "
                "be a multiple of 32 for the exl3-v1 container"
            )


def load_spec_from_file(path: Path) -> PackSpec:
    """Import a ``spec.py`` and return its ``SPEC`` attribute.

    Raises ``FileNotFoundError`` / ``AttributeError`` / ``TypeError`` with a
    clear message when the module does not expose a ``PackSpec``.
    """
    import importlib.util

    if not path.is_file():
        raise FileNotFoundError(f"spec file not found: {path}")
    spec_mod = importlib.util.spec_from_file_location(f"_exl3spec_{path.stem}", path)
    if spec_mod is None or spec_mod.loader is None:
        raise ImportError(f"cannot load spec module from {path}")
    module = importlib.util.module_from_spec(spec_mod)
    spec_mod.loader.exec_module(module)
    spec = getattr(module, "SPEC", None)
    if not isinstance(spec, PackSpec):
        raise TypeError(f"{path} must define SPEC: PackSpec (got {type(spec).__name__})")
    return spec


def load_spec(slug: str, specs_root: Path) -> PackSpec:
    """Resolve ``<specs_root>/<slug>/pack-build/spec.py`` (in-image layout)."""
    return load_spec_from_file(specs_root / slug / "pack-build" / "spec.py")
