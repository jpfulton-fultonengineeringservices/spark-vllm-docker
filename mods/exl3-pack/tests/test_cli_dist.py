from __future__ import annotations

from exl3pack.cli import build_parser


def test_help_lists_distributed_commands(capsys) -> None:
    build_parser().print_help()
    output = capsys.readouterr().out
    assert "dist-coordinator" in output
    assert "dist-worker" in output


def test_dist_coordinator_help_lists_expected_args(capsys) -> None:
    parser = build_parser()
    try:
        parser.parse_args(["dist-coordinator", "--help"])
    except SystemExit as exc:
        assert exc.code == 0
    output = capsys.readouterr().out
    for arg in (
        "--source", "--work", "--exl3-out", "--recipe", "--bits",
        "--codebook", "--nodes", "--gather-timeout", "--checkpoint-interval",
    ):
        assert arg in output


def test_dist_worker_help_lists_expected_args(capsys) -> None:
    parser = build_parser()
    try:
        parser.parse_args(["dist-worker", "--help"])
    except SystemExit as exc:
        assert exc.code == 0
    output = capsys.readouterr().out
    for arg in ("--inbox", "--shared", "--device", "--stop"):
        assert arg in output
