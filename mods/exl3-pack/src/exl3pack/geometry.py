"""Autodetect model geometry from a HuggingFace model directory.

Torch-free port of the original ``pack-model-info.py``: reads ``config.json``
and the safetensors weight map (index, else shard headers) and emits the
geometry record the pack pipeline keys off. Works for any MoE transformer
(MiMo-V2, DeepSeek-V4, Kimi-K3, Qwen-MoE); no per-model config needed.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

MOE_EXPERT_MARKER = ".mlp.experts."


@dataclass
class DetectedGeometry:
    source: str
    architecture: str | None
    hidden_size: int
    intermediate_size: int
    num_experts: int
    num_slots: int
    moe_layer_count: int
    num_hidden_layers: int = 0
    moe_layers: list[int] = field(default_factory=list)
    dense_layers: list[int] = field(default_factory=list)
    text_config: dict[str, object] = field(default_factory=dict)
    warning: str | None = None

    def to_dict(self) -> dict[str, object]:
        d = asdict(self)
        if self.warning is not None:
            d["_warning"] = self.warning
        del d["warning"]
        return d


def _load_config(source: Path) -> dict[str, object]:
    cfg_path = source / "config.json"
    if not cfg_path.exists():
        raise SystemExit(f"config.json not found in {source}")
    return json.loads(cfg_path.read_text())  # type: ignore[no-any-return]


def _text_config(cfg: dict[str, object]) -> dict[str, object]:
    tc = cfg.get("text_config", cfg)
    return tc if isinstance(tc, dict) else cfg


def _as_int(value: object, default: int = 0) -> int:
    """Coerce a config value to ``int`` (config JSON may hold int/float/str)."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str) and value.strip():
        try:
            return int(value)
        except ValueError:
            return default
    return default


def _weight_map(source: Path) -> dict[str, str] | None:
    idx = source / "model.safetensors.index.json"
    if idx.exists():
        wm = json.loads(idx.read_text()).get("weight_map")
        return wm if isinstance(wm, dict) else None
    return None


def _list_keys_from_safetensors(source: Path) -> list[str]:
    try:
        from safetensors import safe_open
    except ImportError:
        print(
            "safetensors not available; install safetensors or use an indexed model",
            file=__import__("sys").stderr,
        )
        raise SystemExit(1) from None
    shards = sorted(source.glob("*.safetensors"))
    if not shards:
        return []
    keys: list[str] = []
    for shard in shards:
        with safe_open(str(shard), framework="pt") as f:
            keys.extend(f.keys())
    return keys


def _all_keys(source: Path) -> list[str]:
    wm = _weight_map(source)
    if wm is not None:
        return list(wm.keys())
    return _list_keys_from_safetensors(source)


def _layer_index(key: str) -> int | None:
    """Return the ``model.layers.<N>.`` index in ``key`` or ``None``."""
    parts = key.split(".")
    for i, part in enumerate(parts):
        if part == "layers" and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                return None
    return None


def _find_moe_layers(keys: list[str]) -> list[int]:
    layers: set[int] = set()
    for k in keys:
        if MOE_EXPERT_MARKER in k:
            idx = _layer_index(k)
            if idx is not None:
                layers.add(idx)
    return sorted(layers)


def _find_dense_layers(keys: list[str], moe_indices: set[int]) -> list[int]:
    all_layers: set[int] = set()
    for k in keys:
        if k.startswith("model.layers."):
            idx = _layer_index(k)
            if idx is not None:
                all_layers.add(idx)
    return sorted(all_layers - moe_indices)


def detect(source: Path) -> DetectedGeometry:
    cfg = _load_config(source)
    tc = _text_config(cfg)

    architectures = cfg.get("architectures", [])
    arch = architectures[0] if isinstance(architectures, list) and architectures else None
    hidden_size = _as_int(tc.get("hidden_size", 0))
    inter = _as_int(tc.get("moe_intermediate_size", tc.get("intermediate_size", 0)))
    num_experts = _as_int(
        tc.get("n_routed_experts", tc.get("num_experts", tc.get("num_local_experts", 0)))
    )
    num_slots = inter // 32 if inter % 32 == 0 else 0
    num_hidden_layers = _as_int(tc.get("num_hidden_layers", 0))

    keys = _all_keys(source)
    moe_layers = _find_moe_layers(keys)
    dense_layers = _find_dense_layers(keys, set(moe_layers))

    warning: str | None = None
    if num_slots == 0 and inter > 0:
        warning = (
            f"intermediate_size {inter} not cleanly divisible by 32; "
            "b12x exl3-v1 requires slot_channels=32"
        )
    if not moe_layers:
        warning = (
            "no MoE expert layers detected in weight map; "
            "model may be dense or use non-standard key naming"
        )

    scalar_text_config: dict[str, object] = {
        k: v
        for k, v in tc.items()
        if isinstance(v, (str, int, float, bool, type(None)))
    }

    return DetectedGeometry(
        source=str(source),
        architecture=arch,
        hidden_size=hidden_size,
        intermediate_size=inter,
        num_experts=num_experts,
        num_slots=num_slots,
        moe_layer_count=len(moe_layers),
        num_hidden_layers=num_hidden_layers,
        moe_layers=moe_layers,
        dense_layers=dense_layers,
        text_config=scalar_text_config,
        warning=warning,
    )
