"""Torch-free contract tests for the dist-coordinator argument path.

Guards the failure seen on gx10 nodes: the coordinator built an args dict
without ``bits``, so ``convert_model.prepare`` raised
``AttributeError: ... no attribute 'bits'`` after the worker containers had
already launched.

These assert the contract ``Coordinator.run`` depends on: every attribute
``prepare`` reads must be present with a usable value. Torch and exllamav3 are
faked in ``sys.modules`` exactly as ``test_distributed.py`` does, so this runs
on a torch-free host.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from unittest import mock

import pytest

from exl3pack import cli, paths
from exl3pack.cli import _EXL3_CODEBOOKS, build_parser
from exl3pack.spec import load_spec

MODEL = "mimo-v2.6-flash-rl-uncensored"
# Every args.X attribute prepare() reads, generated from the pinned wheel
# (see gen_prepare_attrs.py). Not hand-maintained.
REQUIRED_BY_PREPARE: tuple[str, ...] = tuple(
    json.loads(Path(__file__).with_name("prepare_attrs.json").read_text())
)


def _specs_root() -> Path:
    """Same resolution as ``test_recipe.py``: env override, else repo ``mods/``."""
    env = os.environ.get("EXL3_SPECS_ROOT")
    if env:
        return Path(env)
    return Path(__file__).parent.parent.parent  # mods/


_DIST_TESTS = Path(__file__).with_name("test_distributed.py")
_spec = importlib.util.spec_from_file_location("_dist_fakes", _DIST_TESTS)
assert _spec is not None and _spec.loader is not None
_dist_fakes = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_dist_fakes)
_fake_gpu_modules = _dist_fakes._fake_gpu_modules


def _dist_argv(extra: list[str]) -> list[str]:
    return [
        "--model", MODEL,
        "--specs-root", str(_specs_root()),
        "dist-coordinator",
        "--source", "/src",
        "--work", "/work",
        "--exl3-out", "/out",
        "--recipe", "/recipe.yaml",
        "--nodes", "gx10-cb11,gx10-f1d8",
        *extra,
    ]


def _built_coordinator_args(extra: list[str]) -> dict[str, object]:
    """Run the real arg-building path and capture what Coordinator receives."""
    captured: dict[str, object] = {}

    class _StopBeforeWorkers(Exception):
        pass

    class _FakeCoordinator:
        def __init__(self, args, endpoints, work, log_dir=None):  # noqa: ANN001
            captured["args"] = args
            captured["log_dir"] = log_dir
            raise _StopBeforeWorkers

    args_ns = build_parser().parse_args(_dist_argv(extra))
    with _fake_gpu_modules():
        sys.modules.pop("exl3pack.distributed", None)
        distributed = importlib.import_module("exl3pack.distributed")
        with mock.patch.dict(os.environ, {"EXL3_SPECS_ROOT": str(_specs_root())}), \
                mock.patch.object(paths, "load_node_map", lambda _p: {}), \
                mock.patch.object(distributed, "Coordinator", _FakeCoordinator), \
                pytest.raises(_StopBeforeWorkers):
            cli._cmd_dist_coordinator(args_ns)
    return dict(captured["args"])  # type: ignore[arg-type]


@pytest.mark.parametrize("extra", [[], ["--bits", "3", "--codebook", "mcg"]])
def test_coordinator_args_carry_every_attribute_prepare_reads(extra: list[str]) -> None:
    args = _built_coordinator_args(extra)
    for key in REQUIRED_BY_PREPARE:
        assert key in args, (
            f"coordinator args missing {key!r}; prepare() will raise AttributeError. "
            "Upstream defaults live in gen_prepare_attrs.py."
        )


def test_bits_default_comes_from_model_spec_when_flag_omitted() -> None:
    spec = load_spec(MODEL, _specs_root())
    assert _built_coordinator_args([])["bits"] == spec.bits


def test_explicit_bits_flag_overrides_spec_default() -> None:
    assert _built_coordinator_args(["--bits", "4"])["bits"] == 4


def test_codebook_is_always_one_of_the_accepted_names() -> None:
    assert _built_coordinator_args([])["codebook"] in _EXL3_CODEBOOKS


def test_bogus_codebook_is_rejected_at_parse_time() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(_dist_argv(["--codebook", "bogus"]))


@pytest.mark.image
def test_required_attrs_match_live_prepare_signature() -> None:
    """In-image drift guard: the fixture must match what the pinned
    exllamav3's prepare() actually reads. Runs only where exllamav3 is
    importable (the exl3-pack image); host runs skip on the image marker."""
    import inspect  # noqa: PLC0415

    import exllamav3.conversion.convert_model as cm  # noqa: PLC0415

    src = inspect.getsource(cm.prepare)
    tree = ast.parse(src)
    live = {
        n.attr
        for n in ast.walk(tree)
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "args"
    }
    assert live == set(REQUIRED_BY_PREPARE), (
        f"prepare() reads {sorted(live - set(REQUIRED_BY_PREPARE))} not in fixture; "
        f"fixture has {sorted(set(REQUIRED_BY_PREPARE) - live)} prepare() no longer reads. "
        "Regenerate with: uv run --extra hosttest python tests/gen_prepare_attrs.py"
    )
