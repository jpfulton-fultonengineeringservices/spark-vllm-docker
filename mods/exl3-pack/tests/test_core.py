"""Host unit tests (torch-free) for the shared exl3pack library."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from exl3pack import paths
from exl3pack.geometry import detect
from exl3pack.monitor import eta_seconds, write_status
from exl3pack.spec import Geometry, PackSpec, load_spec_from_file
from exl3pack.status import _progress_bar, _render

FIXTURES = Path(__file__).parent / "fixtures"


def _specs_root() -> Path:
    """Resolve the per-model spec root.

    Prefer ``EXL3_SPECS_ROOT`` (set in the image to ``/opt/exl3-specs``); else
    walk up from this file to the repo ``mods/`` tree.
    """
    env = os.environ.get("EXL3_SPECS_ROOT")
    if env:
        return Path(env)
    return Path(__file__).parent.parent.parent  # mods/


def _map_path() -> Path:
    return paths.default_map_path()


SPECS = _specs_root()


def test_detect_flash_uncensored_geometry() -> None:
    g = detect(FIXTURES / "flash-unc").to_dict()
    assert g["architecture"] == "MiMoV2ForCausalLM"
    assert g["hidden_size"] == 4096
    assert g["intermediate_size"] == 2048
    assert g["num_experts"] == 256
    assert g["num_slots"] == 64
    assert g["moe_layer_count"] == 47
    assert g["num_hidden_layers"] == 48
    assert g["dense_layers"] == [0]
    assert "_warning" not in g


def test_detect_pro_uncensored_geometry() -> None:
    g = detect(FIXTURES / "pro-unc").to_dict()
    assert g["hidden_size"] == 6144
    assert g["num_experts"] == 384
    assert g["moe_layer_count"] == 69
    assert g["num_hidden_layers"] == 70
    assert g["num_slots"] == 64
    assert "_warning" not in g


@pytest.mark.parametrize(
    ("slug", "fixture"),
    [
        ("mimo-v2.6-flash-rl-uncensored", "flash-unc"),
        ("mimo-v2.6-pro-rl-uncensored", "pro-unc"),
    ],
)
def test_spec_matches_detected_geometry(slug: str, fixture: str) -> None:
    spec = load_spec_from_file(SPECS / slug / "pack-build" / "spec.py")
    detected = detect(FIXTURES / fixture).to_dict()
    for key, expected in spec.geometry.as_comparable().items():
        assert detected[key] == expected, (slug, key)


def test_spec_rejects_unknown_codebook() -> None:
    with pytest.raises(ValueError):
        PackSpec(
            slug="x",
            geometry=Geometry("A", 1, 32, 1, 1, 1, 1),
            codebook="mul1",
        )


def test_spec_rejects_zero_slots() -> None:
    with pytest.raises(ValueError):
        PackSpec(slug="x", geometry=Geometry("A", 1, 32, 1, 0, 1, 1))


def test_node_map_resolves_local_source() -> None:
    nm = paths.load_node_map(_map_path())
    p = paths.resolve_source(nm, "home-gx10-node4", "mimo-v2.6-pro-rl-uncensored")
    assert p == Path("/opt/llm/staging/mimo-v2.6-pro-rl-uncensored")
    p3 = paths.resolve_source(nm, "gx10-becc", "mimo-v2.6-flash-rl-uncensored")
    assert p3 == Path("/opt/llm/staging/mimo-v2.6-flash-rl-uncensored")


def test_node_map_falls_back_to_nas() -> None:
    nm = paths.load_node_map(_map_path())
    p = paths.resolve_source(nm, "home-gx10-node2", "mimo-v2.6-pro-rl-uncensored")
    assert p == Path("/nas-1/models/mimo/mimo-v2.6-pro-rl-uncensored")


def test_eta_formula() -> None:
    assert eta_seconds(0, 0, 10) == 0
    assert eta_seconds(100, 5, 10) == 100  # 100 * (10-5)/5
    assert eta_seconds(60, 3, 9) == 120  # 60 * 6/3


def test_status_render_and_bar() -> None:
    assert "5/10" in _progress_bar(5, 10)
    st = {
        "stage": "convert",
        "phase": "quantizing",
        "layers_total": 47,
        "layers_completed": 10,
        "codebook": "mcg",
        "bits": 3,
        "model": "m",
    }
    assert "convert" in _render(st)
    # The delta/rate line the driver's poll() depends on.
    assert "Δ poll" in _render(st, delta=3, interval=5.0)


def test_status_driver_poll_invocation(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The driver's poll() calls `status.py --delta N --interval S <file>`.

    Regression guard: those flags must be parsed (not treated as the path), and
    the delta line must render — otherwise the driver aborts under `set -e`.
    """
    from exl3pack import status as status_mod

    sf = tmp_path / ".pack-status.json"
    write_status(
        sf, "2026-01-01T00:00:00Z",
        stage="convert", phase="quantizing", layers_total=47, layers_completed=10,
        codebook="mcg", bits=3, model="m",
    )
    rc = status_mod.main(["--delta", "3", "--interval", "5", str(sf)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Δ poll: +3 layers" in out
    assert "convert" in out


def test_status_json_and_env_fallback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from exl3pack import status as status_mod

    sf = tmp_path / ".pack-status.json"
    write_status(sf, "2026-01-01T00:00:00Z", stage="repack", phase="assembling")

    # --json prints the raw record.
    assert status_mod.main(["--json", str(sf)]) == 0
    assert json.loads(capsys.readouterr().out)["stage"] == "repack"

    # No positional arg: fall back to PACK_STATUS_FILE.
    monkeypatch.setenv("PACK_STATUS_FILE", str(sf))
    assert status_mod.main([]) == 0
    assert "repack" in capsys.readouterr().out


def test_write_status_atomic_merge(tmp_path: Path) -> None:
    sf = tmp_path / ".pack-status.json"
    write_status(sf, "2026-01-01T00:00:00Z", stage="convert", phase="starting")
    write_status(sf, "2026-01-01T00:00:00Z", phase="quantizing", layers_total=47)
    data = json.loads(sf.read_text())
    assert data["stage"] == "convert"
    assert data["phase"] == "quantizing"
    assert data["layers_total"] == 47
    assert "last_update" in data
    assert data["started_at"] == "2026-01-01T00:00:00Z"
