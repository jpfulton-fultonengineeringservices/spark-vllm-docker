"""Torch-free tests for the model-global EXL3 recipe pregenerator.

Covers the CLI wiring (``recipe`` dispatch + ``--dry-run``) and the recipe
serialization, both without importing torch — the recipe module keeps its
torch/exllamav3 imports lazy so the host path stays stdlib-only.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from exl3pack import paths
from exl3pack.cli import _cmd_recipe, build_parser
from exl3pack.recipe import (
    DEFAULT_HEAD_BITS,
    RECIPE_FILENAME,
    Recipe,
    _budgeted_keys,
    parse_recipe,
    render_recipe,
    resolve_recipe_path,
)
from exl3pack.spec import load_spec_from_file


def _specs_root() -> Path:
    """Same resolution as ``tests/test_core.py``: env override, else repo ``mods/``."""
    env = os.environ.get("EXL3_SPECS_ROOT")
    if env:
        return Path(env)
    return Path(__file__).parent.parent.parent  # mods/


SPECS = _specs_root()
SLUG = "mimo-v2.6-flash-rl-uncensored"


def _spec():
    return load_spec_from_file(SPECS / SLUG / "pack-build" / "spec.py")


def test_recipe_subcommand_is_wired() -> None:
    ap = build_parser()
    args = ap.parse_args(["recipe", "--source", "/opt/llm/staging/m", "--dry-run"])
    assert args.func is _cmd_recipe


def test_recipe_dry_run_resolves_work_recipe_path(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    spec = _spec()
    ap = build_parser()
    args = ap.parse_args(
        [
            "--spec", str(SPECS / SLUG / "pack-build" / "spec.py"),
            "--local-root", str(tmp_path),
            "recipe",
            "--source", "/opt/llm/staging/m",
            "--dry-run",
        ]
    )
    assert args.func is _cmd_recipe
    rc = args.func(args)
    assert rc == 0
    out = capsys.readouterr().out
    plan = json.loads(out)
    work = tmp_path / "fes-projects" / "exl3-mimo-build" / f"{spec.slug}-work-k{spec.bits}"
    assert plan["recipe"] == str(work / RECIPE_FILENAME)
    assert plan["work"] == str(work)
    assert plan["bits"] == spec.bits
    assert plan["codebook"] == spec.codebook


def test_recipe_dry_run_is_torch_free(tmp_path: Path) -> None:
    """`--dry-run` must work on a host with no torch/exllamav3 installed."""
    spec_path = SPECS / SLUG / "pack-build" / "spec.py"
    script = (
        "import sys; "
        "from exl3pack.cli import build_parser; "
        "args = build_parser().parse_args(["
        f"'--spec', '{spec_path}', '--local-root', '{tmp_path}', "
        "'recipe', '--source', '/x', '--dry-run']); "
        "rc = args.func(args); "
        "assert 'torch' not in sys.modules, 'torch was imported'; "
        "assert 'exllamav3' not in sys.modules, 'exllamav3 was imported'; "
        "raise SystemExit(rc)"
    )
    env = dict(os.environ)
    src = str(Path(__file__).parent.parent / "src")
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert RECIPE_FILENAME in proc.stdout


def test_resolve_recipe_path_defaults_to_work() -> None:
    assert resolve_recipe_path(Path("/w")) == Path("/w") / RECIPE_FILENAME
    assert resolve_recipe_path(Path("/w"), Path("/r/x.yaml")) == Path("/r/x.yaml")


def test_recipe_round_trip() -> None:
    r = Recipe(
        tensors={
            "model.layers.0.self_attn.q_proj": 3,
            "model.layers.0.mlp.experts.0.gate_proj": 2,
        },
        achieved_bpw=3.0012,
        head_bits=6,
    )
    back = parse_recipe(render_recipe(r))
    assert back.achieved_bpw == r.achieved_bpw
    assert back.head_bits == r.head_bits
    assert back.tensors == r.tensors


def test_recipe_default_head_bits() -> None:
    assert Recipe(tensors={"a": 3}, achieved_bpw=3.0).head_bits == DEFAULT_HEAD_BITS


def test_recipe_half_rate_head_and_tensors() -> None:
    """Half-integer rates (mul1 codebook) must survive the round-trip as floats."""
    r = Recipe(
        tensors={"a": 1.5, "b": 3.5},
        achieved_bpw=2.5,
        head_bits=2.5,
        mtp_bits=4.5,
        codebook="mul1",
    )
    back = parse_recipe(render_recipe(r))
    assert back.tensors == {"a": 1.5, "b": 3.5}
    assert back.head_bits == 2.5
    assert back.mtp_bits == 4.5
    assert back.codebook == "mul1"


class _Mod:
    def __init__(self, key: str, qmap: object, qbits_key: str, children: list | None = None):
        self.key = key
        self.qmap = qmap
        self.qbits_key = qbits_key
        self.modules = children or []


def test_budgeted_keys_excludes_aux_targets() -> None:
    """The recipe must carry ONLY budgeted 'bits' tensors, not head/mtp/vision."""
    root = [
        _Mod("blk.q_proj", "q", "bits"),
        _Mod("blk.head", "q", "head_bits"),
        _Mod("blk.mtp", "q", "mtp_bits"),
        _Mod("blk.vision", "q", "vision"),
        _Mod("blk.noparam", None, "bits"),  # qmap None -> not a budgeted Linear
    ]
    assert _budgeted_keys(_Mod("r", None, "", root), None) == {"blk.q_proj"}


def test_parse_recipe_rejects_empty_tensors() -> None:
    with pytest.raises(ValueError):
        parse_recipe("achieved_bpw: 3\nhead_bits: 6\ntensors:\n")


def test_parse_recipe_rejects_missing_achieved() -> None:
    with pytest.raises(ValueError):
        parse_recipe('head_bits: 6\ntensors:\n  "a": 3\n')


def test_recipe_map_defaults_to_local_source() -> None:
    nm = paths.load_node_map(paths.default_map_path())
    spec = _spec()
    p = paths.resolve_source(nm, "gx10-becc", spec.node_map_key or spec.slug)
    assert p is not None
    assert str(p).endswith(SLUG)


def test_work_root_matches_cli_naming() -> None:
    """The dry-run work dir must equal what the stage commands resolve."""
    nm = paths.load_node_map(paths.default_map_path())
    spec = _spec()
    root = paths.resolve_work_root(nm, "gx10-becc")
    assert root / f"{spec.slug}-work-k{spec.bits}" == (
        root / f"{SLUG}-work-k{spec.bits}"
    )
