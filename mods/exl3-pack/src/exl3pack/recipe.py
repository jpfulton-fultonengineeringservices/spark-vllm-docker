"""Pregenerate a model-global EXL3 bitrate recipe for sharded convert.

``exllamav3.conversion.allocation.create_q_strategy`` is **model-global**: it
walks the full module tree, sums ``weights_numel()``, and spends one bit budget
across layer groups by priority. A per-shard re-allocation would therefore
diverge, so the parallel pack pipeline pregenerates ONE recipe here and passes
``--recipe``/``-rcp`` to every shard. ``convert_model.prepare()`` loads that file
into ``in_args["recipe_strategy"]`` and ``main()`` feeds it to
``create_q_strategy_from_recipe``, which requires the recipe to cover the
budgeted (``qbits_key == "bits"``) tensors **exactly**.

Recipe YAML consumed by ``convert_model -rcp``::

    tensors:            # module key -> bpw (budgeted + head/mtp/vision aux keys)
      "model.layers.0.self_attn.q_proj": 3
    achieved_bpw: 3.0012
    head_bits: 6

Reuse note: the pinned exllamav3 wheel does **not** ship a recipe emitter.
``exllamav3/conversion/optimize_model.py`` is a different tool (a
``measurement.json`` -> output-dir sizing pass, ``main(args, job_state)``); it
writes no recipe YAML, so this is a fresh implementation rather than a wrapper.

Torch / exllamav3 are heavyweight and image-only, so every heavy import is lazy:
``--dry-run``, :func:`resolve_recipe_path` and :func:`write_recipe` stay
stdlib-only (mirroring ``exl3pack.convert`` and ``exl3pack.assemble``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

RECIPE_FILENAME = "recipe.yaml"

# Defaults mirror ``convert_model``'s ``-hb``/``-mb``/``-vb`` when unset.
DEFAULT_HEAD_BITS = 6
DEFAULT_MTP_BITS = 4
DEFAULT_VISION_BITS = 0


@dataclass(frozen=True)
class Recipe:
    """A model-global bitrate allocation, serialized for ``convert_model -rcp``."""

    tensors: dict[str, float]
    achieved_bpw: float
    head_bits: float = DEFAULT_HEAD_BITS
    mtp_bits: float | None = None

    def to_dict(self) -> dict[str, object]:
        """The exact mapping ``convert_model`` reads.

        ``tensors`` / ``achieved_bpw`` / ``head_bits`` are what ``convert_model``
        reads. ``mtp_bits`` is emitted as a convenience for the shard driver
        (``convert_model``'s ``--mtp_bits`` is *not* read from the recipe, so the
        driver must pass ``-mb`` identically on every shard; extra top-level keys
        are ignored by ``prepare()``).
        """
        out: dict[str, object] = {
            "tensors": {k: self.tensors[k] for k in sorted(self.tensors)},
            "achieved_bpw": self.achieved_bpw,
            "head_bits": self.head_bits,
        }
        if self.mtp_bits is not None:
            out["mtp_bits"] = self.mtp_bits
        return out


def resolve_recipe_path(work: Path, out: Path | None = None) -> Path:
    """``out`` when given, else ``<work>/recipe.yaml`` (the shard-invariant default)."""
    return _resolve_out(out) if out is not None else Path(work) / RECIPE_FILENAME


def _resolve_out(out: Path) -> Path:
    """Accept either a recipe file path or a directory (``out/recipe.yaml``)."""
    out = Path(out)
    if out.suffix == "" or out.is_dir():
        return out / RECIPE_FILENAME
    return out


def render_recipe(recipe: Recipe) -> str:
    """Serialize a recipe to YAML (stdlib-only; keys are JSON-quoted).

    JSON double-quoted scalars are valid YAML, and every value is an int/float,
    so ``yaml.safe_load`` (what ``convert_model`` uses) round-trips this exactly.
    """
    data = recipe.to_dict()
    lines = [
        f"achieved_bpw: {json.dumps(data['achieved_bpw'])}",
        f"head_bits: {json.dumps(data['head_bits'])}",
    ]
    if "mtp_bits" in data:
        lines.append(f"mtp_bits: {json.dumps(data['mtp_bits'])}")
    lines.append("tensors:")
    tensors = data["tensors"]
    assert isinstance(tensors, dict)
    for key, bpw in tensors.items():
        lines.append(f"  {json.dumps(key)}: {json.dumps(bpw)}")
    return "\n".join(lines) + "\n"


def _as_rate(value: str | float) -> float:
    """Parse a bitrate, keeping whole numbers as ints (matches YAML round-trip)."""
    r = float(value)
    return int(r) if r.is_integer() else r


def parse_recipe(text: str) -> Recipe:
    """Minimal stdlib parser for :func:`render_recipe`'s output (host tests/dry-run).

    ``convert_model`` itself parses with PyYAML; this reader exists only so the
    torch-free host path can round-trip without a YAML dependency.
    """
    achieved: float | None = None
    head_bits: float = DEFAULT_HEAD_BITS
    mtp_bits: float | None = None
    tensors: dict[str, float] = {}
    in_tensors = False
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if not line.startswith((" ", "\t")):
            in_tensors = False
            if line.startswith("achieved_bpw:"):
                achieved = float(line.split(":", 1)[1].strip())
            elif line.startswith("head_bits:"):
                head_bits = _as_rate(line.split(":", 1)[1].strip())
            elif line.startswith("mtp_bits:"):
                mtp_bits = _as_rate(line.split(":", 1)[1].strip())
            elif line.startswith("tensors:"):
                in_tensors = True
            continue
        if in_tensors:
            key_part, _, value_part = line.partition(":")
            key = key_part.strip()
            if key.startswith('"') and key.endswith('"'):
                key = json.loads(key)
            value = value_part.strip()
            tensors[key] = _as_rate(value)
    if achieved is None:
        raise ValueError("recipe is missing 'achieved_bpw'")
    if not tensors:
        raise ValueError("recipe must contain a non-empty 'tensors' mapping")
    return Recipe(tensors=tensors, achieved_bpw=achieved, head_bits=head_bits, mtp_bits=mtp_bits)


def write_recipe(out: Path, recipe: Recipe) -> Path:
    """Atomically write ``recipe.yaml`` (or a directory receiving one)."""
    path = _resolve_out(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(render_recipe(recipe), encoding="utf8")
    tmp.replace(path)
    return path


def read_recipe(path: Path) -> Recipe:
    """Read a recipe file back (see :func:`parse_recipe` for the parser)."""
    return parse_recipe(Path(path).read_text(encoding="utf8"))


def _build_in_args(
    source: Path,
    *,
    head_bits: float,
    mtp_bits: float,
    vision_bits: int,
    hq: bool,
    extra_args: list[str] | None,
) -> dict[str, Any]:
    """The ``in_args`` subset ``convert_model.get_base_model`` + ``create_q_strategy`` read.

    Mirrors ``convert_model.prepare()``: ``--head_bits``/``--mtp_bits``/
    ``--vision_bits`` default to 6/4/0 (auto) when unset. ``extra_args`` may carry the
    budget-relevant ``convert_model`` flags (``-mb/--mtp_bits``, ``-vb/
    --vision_bits``, ``-hq/--hq``) so the recipe matches that convert invocation.
    """
    in_args: dict[str, Any] = {
        "in_dir": str(source),
        "head_bits": head_bits,
        "mtp_bits": mtp_bits,
        "vision_bits": vision_bits,
        "hq": hq,
    }
    flags = list(extra_args or [])
    i = 0
    while i < len(flags):
        flag = flags[i]
        if flag in ("-mb", "--mtp_bits") and i + 1 < len(flags):
            in_args["mtp_bits"] = _as_rate(flags[i + 1])
            i += 2
            continue
        if flag in ("-vb", "--vision_bits") and i + 1 < len(flags):
            in_args["vision_bits"] = int(float(flags[i + 1]))
            i += 2
            continue
        if flag in ("-hb", "--head_bits") and i + 1 < len(flags):
            in_args["head_bits"] = _as_rate(flags[i + 1])
            i += 2
            continue
        if flag in ("-hq", "--hq"):
            in_args["hq"] = True
            i += 1
            continue
        i += 1
    return in_args


def _budgeted_keys(model: object, mtp_model: object) -> set[str]:
    """Keys of every budgeted (``qbits_key == "bits"``) Linear in the full tree.

    ``create_q_strategy`` merges head/mtp/vision aux targets into its returned
    map, but ``create_q_strategy_from_recipe`` reads *only* the budgeted keys
    from the recipe (head/mtp/vision come from separate bpw args). So the recipe
    must carry exactly these keys — no more (aux values would be silently
    overwritten) and no less (the loader raises on any missing budgeted tensor).
    """
    keys: set[str] = set()

    def walk(module: object) -> None:
        has_qmap = getattr(module, "qmap", None) is not None
        if has_qmap and getattr(module, "qbits_key", None) == "bits":
            keys.add(str(getattr(module, "key", None)))
        for child in getattr(module, "modules", []) or []:
            walk(child)

    modules = list(getattr(model, "modules", []) or [])
    if mtp_model is not None:
        modules += list(getattr(mtp_model, "modules", []) or [])
    for top in modules:
        walk(top)
    return keys


def build(
    source: Path,
    *,
    bits: int,
    codebook: str,
    head_bits: float | None = None,
    hq: bool = False,
    extra_args: list[str] | None = None,
) -> Recipe:
    """Build the model-global allocation (torch + exllamav3; lazy imports).

    Reuses ``convert_model.get_base_model`` so the module tree (and therefore the
    module keys) is constructed exactly as ``convert`` does, then calls
    ``create_q_strategy`` once over the full tree.
    """
    import importlib

    convert_model = importlib.import_module("exllamav3.conversion.convert_model")
    allocation = importlib.import_module("exllamav3.conversion.allocation")

    in_args = _build_in_args(
        Path(source),
        head_bits=DEFAULT_HEAD_BITS if head_bits is None else head_bits,
        mtp_bits=DEFAULT_MTP_BITS,
        vision_bits=DEFAULT_VISION_BITS,
        hq=hq,
        extra_args=extra_args,
    )
    config, model, mtp_model, vision_model, _tokenizer, _ref = convert_model.get_base_model(
        in_args
    )

    strategy: dict[str, float]
    achieved: float
    strategy, achieved = allocation.create_q_strategy(
        model,
        mtp_model,
        config,
        float(bits),
        float(in_args["head_bits"]),
        float(in_args["mtp_bits"]),
        bool(in_args["hq"]),
        vision_model=vision_model,
        vision_bpw=in_args["vision_bits"],
        # 1.5 / 2.5 / 3.5 bpw exist for the mul1 codebook only.
        half_steps=codebook == "mul1",
    )
    budgeted = _budgeted_keys(model, mtp_model)
    return Recipe(
        tensors={k: _as_rate(v) for k, v in strategy.items() if k in budgeted},
        achieved_bpw=float(achieved),
        head_bits=_as_rate(in_args["head_bits"]),
        mtp_bits=_as_rate(in_args["mtp_bits"]),
    )


def emit(
    source: Path,
    out: Path,
    *,
    bits: int,
    codebook: str,
    head_bits: float | None = None,
    hq: bool = False,
    extra_args: list[str] | None = None,
) -> Path:
    """Build the model-global allocation and serialize it to ``out`` (``recipe.yaml``)."""
    recipe = build(
        Path(source),
        bits=bits,
        codebook=codebook,
        head_bits=head_bits,
        hq=hq,
        extra_args=extra_args,
    )
    return write_recipe(out, recipe)


def dry_run_plan(
    source: Path | None,
    work: Path,
    *,
    slug: str,
    codebook: str,
    bits: int,
    out: Path | None = None,
) -> dict[str, object]:
    """Resolve the paths a real ``recipe`` run would use — no tree walk, stdlib only."""
    return {
        "model": slug,
        "codebook": codebook,
        "bits": bits,
        "source": str(source) if source else None,
        "work": str(work),
        "recipe": str(resolve_recipe_path(work, out)),
        "head_bits": DEFAULT_HEAD_BITS,
        "note": "builds the full module tree and calls create_q_strategy once (in-image only)",
    }


__all__ = [
    "DEFAULT_HEAD_BITS",
    "DEFAULT_MTP_BITS",
    "DEFAULT_VISION_BITS",
    "RECIPE_FILENAME",
    "Recipe",
    "build",
    "dry_run_plan",
    "emit",
    "parse_recipe",
    "read_recipe",
    "render_recipe",
    "resolve_recipe_path",
    "write_recipe",
]
