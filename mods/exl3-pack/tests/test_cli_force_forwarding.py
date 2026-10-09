"""CLI `assemble --force --status-file --log-dir` kwargs reach `assemble.run`.

Argparse-level: monkeypatch ``assemble.run`` and drive ``_cmd_run_stage`` via the
real parser so the forwarding path is exercised without torch or a NAS.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from exl3pack import cli


def test_assemble_flags_forwarded(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    spec_dir = tmp_path / "specs"
    (spec_dir / "mymodel" / "pack-build").mkdir(parents=True)
    (spec_dir / "mymodel" / "pack-build" / "spec.py").write_text(
        "from exl3pack.spec import Geometry, PackSpec\n"
        "SPEC = PackSpec(\n"
        '    slug="mymodel",\n'
        '    geometry=Geometry(\n'
        '        architecture="MiMoV2ForCausalLM", hidden_size=8,\n'
        '        intermediate_size=32, num_experts=1, num_slots=32,\n'
        '        moe_layer_count=1, num_hidden_layers=2,\n'
        '    ),\n'
        '    bits=3, codebook="mcg",\n'
        ")\n"
    )
    node_map = tmp_path / "node-map.json"
    node_map.write_text(
        json.dumps(
            {
                "schema": "node-model-map/v1",
                "local_root": str(tmp_path / "nvme"),
                "nodes": {
                    "node1": {
                        "alias": "n1",
                        "cluster_node": "gx10-node1",
                        "checkpoints": {},
                    }
                },
            }
        )
    )
    status = tmp_path / "custom-status.json"
    log_dir = tmp_path / "custom-logs"
    source = tmp_path / "source"
    source.mkdir()
    exl3_out = tmp_path / "exl3-out"
    v1_out = tmp_path / "v1-out"
    for d in (exl3_out, v1_out):
        d.mkdir()

    argv = [
        "--spec", str(spec_dir / "mymodel" / "pack-build" / "spec.py"),
        "--node-map", str(node_map),
        "assemble",
        "--source", str(source),
        "--exl3-out", str(exl3_out),
        "--v1-out", str(v1_out),
        "--force",
        "--status-file", str(status),
        "--log-dir", str(log_dir),
    ]
    args = cli.build_parser().parse_args(argv)

    captured: dict[str, object] = {}

    def fake_run(*fargs, **fkwargs):  # type: ignore[no-untyped-def]
        captured["args"] = fargs
        captured["kwargs"] = fkwargs

    with patch.object(cli, "_rmtree"):
        import exl3pack.assemble as assemble_mod

        with patch.object(assemble_mod, "run", side_effect=fake_run):
            assert cli._cmd_run_stage(args) == 0

    kwargs = captured["kwargs"]
    assert kwargs["force"] is True
    assert kwargs["status_file"] == status
    assert kwargs["log_dir"] == log_dir
    # positional source/pack/out are the resolved pipeline paths
    fargs = captured["args"]
    assert Path(fargs[0]) == source
    assert Path(fargs[1]) == exl3_out
    assert Path(fargs[2]) == v1_out


def test_assemble_status_defaults_to_v1_out(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    spec_dir = tmp_path / "specs"
    (spec_dir / "mymodel" / "pack-build").mkdir(parents=True)
    (spec_dir / "mymodel" / "pack-build" / "spec.py").write_text(
        "from exl3pack.spec import Geometry, PackSpec\n"
        "SPEC = PackSpec(\n"
        '    slug="mymodel",\n'
        '    geometry=Geometry(\n'
        '        architecture="MiMoV2ForCausalLM", hidden_size=8,\n'
        '        intermediate_size=32, num_experts=1, num_slots=32,\n'
        '        moe_layer_count=1, num_hidden_layers=2,\n'
        '    ),\n'
        '    bits=3, codebook="mcg",\n'
        ")\n"
    )
    node_map = tmp_path / "node-map.json"
    node_map.write_text(
        json.dumps(
            {
                "schema": "node-model-map/v1",
                "local_root": str(tmp_path / "nvme"),
                "nodes": {
                    "node1": {
                        "alias": "n1",
                        "cluster_node": "gx10-node1",
                        "checkpoints": {},
                    }
                },
            }
        )
    )
    source = tmp_path / "source"
    source.mkdir()
    exl3_out = tmp_path / "exl3-out"
    v1_out = tmp_path / "v1-out"
    for d in (exl3_out, v1_out):
        d.mkdir()

    monkeypatch.delenv("PACK_STATUS_FILE", raising=False)
    argv = [
        "--spec", str(spec_dir / "mymodel" / "pack-build" / "spec.py"),
        "--node-map", str(node_map),
        "assemble",
        "--source", str(source),
        "--exl3-out", str(exl3_out),
        "--v1-out", str(v1_out),
    ]
    args = cli.build_parser().parse_args(argv)

    captured: dict[str, object] = {}

    def fake_run(*fargs, **fkwargs):  # type: ignore[no-untyped-def]
        captured["kwargs"] = fkwargs

    with patch.object(cli, "_rmtree"):
        import exl3pack.assemble as assemble_mod

        with patch.object(assemble_mod, "run", side_effect=fake_run):
            assert cli._cmd_run_stage(args) == 0

    kwargs = captured["kwargs"]
    # No flag: run() applies its own v1-out default (out/.pack-status.json).
    assert kwargs["status_file"] is None
    assert kwargs["force"] is False
