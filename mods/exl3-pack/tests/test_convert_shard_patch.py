"""Host unit tests (torch-free) for the convert_model range-seam patch.

Exercises ``tools/patch_convert_shard.py`` against a transcription of the
installed ``exllamav3/conversion/convert_model.py`` source shape (the three
anchor blocks: CLI parser, ``prepare`` job args, quantization loop) plus
fail-loud and idempotence cases.  No torch, no GPU, no exllamav3 import.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

TOOLS = Path(__file__).parents[1] / "tools"
SCRIPT = TOOLS / "patch_convert_shard.py"
FIXTURES = Path(__file__).parent / "fixtures"

# Verbatim transcription of the installed convert_model.py shapes the patch
# anchors on. Kept independent of the tool's own constants on purpose: if the
# patch's anchors drift from this, the apply assertions below go red.
MAX_MODULE_ARG = (
    'parser.add_argument("--max_module", type = int, help = "End quantization '
    'after this many modules, includes embedding and norm layers (for debug '
    'purposes)", default = None)'
)
PREPARE_ARGS = (
    "    # Momentary args\n"
    '    in_args["image_dump"] = args.image_dump\n'
    '    in_args["max_module"] = args.max_module\n'
)
LOOP_BLOCK = (
    "        # If resuming, skip along to checkpoint index\n"
    '        if idx < job_state["next_module_idx"]:\n'
    "            continue\n"
    "\n"
    '        if args["max_module"] is not None and idx > args["max_module"]:\n'
    '            print(" !! --max_module reached, stopping job")\n'
    "            sys.exit()\n"
)


def make_convert_model() -> str:
    return (
        "import argparse\n"
        "import sys\n"
        "\n"
        "parser = argparse.ArgumentParser(allow_abbrev = False)\n"
        'parser.add_argument("-i", "--in_dir", type = str, default = None)\n'
        f"{MAX_MODULE_ARG}\n"
        "\n"
        "def prepare(args):\n"
        "    in_args = {}\n"
        f"{PREPARE_ARGS}"
        "    return in_args\n"
        "\n"
        "def main(args, job_state):\n"
        "    model = None\n"
        "    for idx, module in enumerate(model.modules):\n"
        "        free_mem()\n"
        "\n"
        f"{LOOP_BLOCK}"
        "        quantize(module)\n"
    )


def load_tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location("patch_convert_shard", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def write_target(root: Path, text: str) -> Path:
    target = root / "exllamav3" / "conversion" / "convert_model.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text)
    return target


def test_apply_adds_module_start_lower_bound() -> None:
    mod = load_tool()
    patched = mod.apply_patch(make_convert_model())

    assert '"--module-start"' in patched
    assert 'in_args["min_module"] = args.module_start' in patched
    assert 'args.get("min_module") is not None and idx < args["min_module"]' in patched

    # The bound is inclusive of --module-start and exclusive past --max_module.
    guard = '        if args.get("min_module") is not None and idx < args["min_module"]:'
    assert guard in patched
    assert patched.index(guard) < patched.index('idx > args["max_module"]')
    compile(patched, "convert_model.py", "exec")


def test_apply_is_idempotent() -> None:
    mod = load_tool()
    once = mod.apply_patch(make_convert_model())
    twice = mod.apply_patch(once)
    assert once == twice


def test_apply_preserves_resume_skip_order() -> None:
    mod = load_tool()
    patched = mod.apply_patch(make_convert_model())
    resume = '        if idx < job_state["next_module_idx"]:'
    lower = '        if args.get("min_module") is not None and idx < args["min_module"]:'
    assert patched.index(resume) < patched.index(lower)


def test_apply_is_a_noop_when_already_patched() -> None:
    mod = load_tool()
    patched = mod.apply_patch(make_convert_model())
    assert mod.apply_patch(patched) == patched


def test_apply_fails_loud_on_unexpected_source_shape() -> None:
    mod = load_tool()
    broken = make_convert_model().replace('in_args["max_module"] = args.max_module', "")
    with pytest.raises(mod.PatchError):
        mod.apply_patch(broken)


def test_apply_fails_loud_on_missing_target(tmp_path: Path) -> None:
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(tmp_path)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert "is missing under" in proc.stderr


def test_cli_applies_and_then_reports_already_patched(tmp_path: Path) -> None:
    write_target(tmp_path, make_convert_model())
    first = subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(tmp_path)],
        capture_output=True,
        text=True,
    )
    assert first.returncode == 0
    assert "--module-start" in (tmp_path / "exllamav3/conversion/convert_model.py").read_text()

    second = subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(tmp_path)],
        capture_output=True,
        text=True,
    )
    assert second.returncode == 0
    assert "already carries" in second.stdout


def test_check_is_zero_when_applicable_and_nonzero_when_missing(tmp_path: Path) -> None:
    missing = subprocess.run(
        [sys.executable, str(SCRIPT), "--check", "--root", str(tmp_path)],
        capture_output=True,
        text=True,
    )
    assert missing.returncode != 0

    write_target(tmp_path, make_convert_model())
    applicable = subprocess.run(
        [sys.executable, str(SCRIPT), "--check", "--root", str(tmp_path)],
        capture_output=True,
        text=True,
    )
    assert applicable.returncode == 0


def test_convert_run_passes_range_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import exl3pack.convert as convert

    captured: list[str] = []

    class _Proc:
        stdout: list[str] = []

        def wait(self) -> int:
            return 0

    monkeypatch.setattr(
        convert.subprocess,
        "Popen",
        lambda cmd, **kw: captured.append(cmd) or _Proc(),
    )
    convert.run(
        FIXTURES / "flash-unc",
        tmp_path / "out",
        tmp_path / "work",
        bits=4,
        codebook="mul1",
        module_start=12,
        max_module=40,
        status_file=tmp_path / "status.json",
    )
    cmd = captured[0]
    assert cmd[cmd.index("--module-start") + 1] == "12"
    assert cmd[cmd.index("--max_module") + 1] == "40"


def test_convert_run_omits_range_flags_when_unset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import exl3pack.convert as convert

    captured: list[str] = []

    class _Proc:
        stdout: list[str] = []

        def wait(self) -> int:
            return 0

    monkeypatch.setattr(
        convert.subprocess,
        "Popen",
        lambda cmd, **kw: captured.append(cmd) or _Proc(),
    )
    convert.run(
        FIXTURES / "flash-unc",
        tmp_path / "out",
        tmp_path / "work",
        bits=4,
        codebook="mul1",
        status_file=tmp_path / "status.json",
    )
    cmd = captured[0]
    assert "--module-start" not in cmd
    assert "--max_module" not in cmd
