"""exl3-pack CLI.

Subcommands: ``detect | convert | repack | assemble | pipeline | recipe | status |
selftest | plan``. A model is selected with ``--spec <spec.py>`` or
``--model <slug> --specs-root <dir>``. ``--dry-run`` (or ``plan``) prints the
resolved paths + docker mount plan without executing anything.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import TypedDict

from . import paths
from .geometry import detect
from .logconfig import log_event
from .spec import PackSpec, load_spec, load_spec_from_file

# Codebooks exl3-v1 accepts (see repack.py). Shared by --codebook validation.
_EXL3_CODEBOOKS = ("mcg", "lut_e4m3", "lut_fp16")


class CoordinatorArgs(TypedDict):
    """Exactly the kwargs convert_model.prepare() reads on the Namespace built
    from dist_args; a dropped key is a compile error (types must match the
    upstream argparse defaults baked in below)."""

    in_dir: str
    out_dir: str
    recipe: str
    work_dir: str
    gather_timeout: float
    checkpoint_interval: int
    bits: int
    codebook: str
    resume: bool
    head_bits: float | None
    mtp_bits: float | None
    hq: bool
    override_anyway: bool
    image_dump: bool
    verbose: bool
    out_scales: str
    max_module: int | None
    shard_size: int
    vision_bits: int
    ngram_bits: int
    ngram_file: str
    cal_data: str
    cal_rows: int
    cal_cols: int
    last_checkpoint_index: int
    devices: str
    device_ratios: str
    hessians: str
    hessians_reg: float
    # PackSpec geometry (M): MoE validate range + expert count for the
    # commit-time expert-coverage assert (dist_commit). Flash default keeps
    # minimal host-test args working: MoE layers 1..47, 256 experts.
    moe_layer_lo: int
    moe_layer_hi: int
    moe_expert_count: int


def _resolve_spec(args: argparse.Namespace) -> PackSpec | None:
    if args.spec:
        return load_spec_from_file(Path(args.spec))
    if args.model:
        root = Path(args.specs_root or "/opt/exl3-specs")
        return load_spec(args.model, root)
    return None


def _mounts() -> list[str]:
    return ["-v /nas-1:/nas-1", "-v /opt/llm:/opt/llm"]


def _cmd_detect(args: argparse.Namespace) -> int:
    src = Path(args.source)
    info = detect(src)
    out = info.to_dict()
    spec = _resolve_spec(args)
    if spec is not None:
        mismatches = {
            k: {"expected": v, "detected": out.get(k)}
            for k, v in spec.geometry.as_comparable().items()
            if out.get(k) != v
        }
        out["_spec_parity_ok"] = not mismatches
        if mismatches:
            out["_spec_mismatches"] = mismatches
    print(json.dumps(out, indent=2))
    if args.assert_parity and not out.get("_spec_parity_ok", True):
        return 1
    return 0


def _cmd_plan(args: argparse.Namespace) -> int:
    spec = _resolve_spec(args)
    if spec is None:
        print("plan requires --spec or --model", file=sys.stderr)
        return 2
    node_map = paths.load_node_map(Path(args.node_map))
    source = Path(args.source) if args.source else paths.resolve_source(
        node_map, args.node or "", spec.node_map_key or spec.slug
    )
    work_root = paths.resolve_work_root(node_map, args.node or "", local_root=(
        Path(args.local_root) if args.local_root else None
    ))
    v1_out = Path(args.v1_out) if args.v1_out else paths.resolve_destination(
        spec.slug, Path(args.out_root or "/nas-1/models/mimo")
    )
    plan = {
        "model": spec.slug,
        "codebook": spec.codebook,
        "bits": spec.bits,
        "node": args.node,
        "source": str(source) if source else None,
        "work": str(work_root / f"{spec.slug}-work-k{spec.bits}"),
        "exl3_out": str(work_root / f"{spec.slug}-mcg-k{spec.bits}"),
        "v1_out": str(v1_out),
        "mounts": _mounts(),
        "cleanup": {"work": True, "exl3_out": True},
        "note": "output written to NAS; work/intermediate on node-local NVMe and removed after",
    }
    print(json.dumps(plan, indent=2))
    return 0


def _cmd_recipe(args: argparse.Namespace) -> int:
    spec = _resolve_spec(args)
    if spec is None:
        print("recipe requires --spec or --model", file=sys.stderr)
        return 2

    node_map = paths.load_node_map(Path(args.node_map))
    source = Path(args.source) if args.source else paths.resolve_source(
        node_map, args.node or "", spec.node_map_key or spec.slug
    )
    work_root = paths.resolve_work_root(
        node_map,
        args.node or "",
        local_root=Path(args.local_root) if args.local_root else None,
    )
    work = Path(args.work) if args.work else work_root / f"{spec.slug}-work-k{spec.bits}"
    out = Path(args.out) if args.out else work / "recipe.yaml"

    if args.dry_run:
        from . import recipe

        print(json.dumps(
            recipe.dry_run_plan(
                source,
                work,
                slug=spec.slug,
                codebook=spec.codebook,
                bits=spec.bits,
                out=out,
            ),
            indent=2,
        ))
        return 0
    if source is None:
        print("could not resolve a source checkpoint", file=sys.stderr)
        return 2

    from . import recipe

    result = recipe.emit(
        source,
        out,
        bits=spec.bits,
        codebook=spec.codebook,
        head_bits=args.head_bits,
        hq=args.hq,
    )
    print(f"recipe written: {result}")
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    from . import status as status_mod

    argv = []
    if args.watch:
        argv.append("--watch")
        if args.interval:
            argv.append(str(args.interval))
    if args.delta:
        argv += ["--delta", str(args.delta)]
    if args.delta_interval:
        argv += ["--interval", str(args.delta_interval)]
    if args.json:
        argv.append("--json")
    if args.status_file:
        argv.append(args.status_file)
    return status_mod.main(argv)


def _cmd_selftest(args: argparse.Namespace) -> int:
    """In-image integration check: torch family imports + FP8 dequant dispatch.

    Exercises the riskiest ported logic — the per-group-vs-flat FP8 scale-grid
    dequant (``_dequant_fp8_block_dispatch``) — with a synthetic interleaved grid
    (must pick per-group) and a flat grid (must pick flat). This is the coverage
    a torch-free host cannot run. Mirrors ``tests/test_dequant_dispatch.py``.
    """
    checks: list[dict[str, object]] = []
    for mod in ("torch", "safetensors", "b12x", "exllamav3"):
        try:
            __import__(mod)
            checks.append({"import": mod, "ok": True})
        except Exception as e:  # pragma: no cover - image-only path
            checks.append({"import": mod, "ok": False, "reason": str(e)})

    dispatch = _selftest_dispatch()
    checks.append(dispatch)
    report = {"checks": checks, "ok": all(c["ok"] for c in checks)}
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


def _cmd_typecheck(args: argparse.Namespace) -> int:
    """In-image type safety: mypy strict + stubtest for exllamav3 signatures.

    Verifies that stubs/exllamav3/*.pyi match the runtime API of the installed
    exllamav3 package. Requires the image to include stubs/ at /opt/exl3-pack/stubs
    and mypy installed (see Dockerfile.exl3-pack).
    """
    import subprocess
    import sys

    checks: list[dict[str, object]] = []

    root = Path("/opt/exl3-pack")
    if not (root / "stubs").is_dir():
        report = {"checks": [{"check": "stubs_present", "ok": False,
                              "reason": f"{root / 'stubs'} missing from image"}],
                  "ok": False}
        print(json.dumps(report, indent=2))
        return 1

    # stubtest resolves stubs via MYPYPATH (PEP 561 stubs dir importable);
    # mypy resolves src/ + stubs/ from the config's own paths when run with
    # cwd=/opt/exl3-pack (where pyproject.toml, src/, stubs/ all sit).
    env_path = f"{root / 'src'}{os.pathsep}{root / 'stubs'}"
    env = {**os.environ, "MYPYPATH": env_path}

    def _run(name: str, cmd: list[str]) -> None:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True,
                                    timeout=300, cwd=str(root), env=env)
        except FileNotFoundError as e:
            checks.append({"check": name, "ok": False, "reason": str(e)})
            return
        checks.append({
            "check": name,
            "ok": result.returncode == 0,
            "stdout": result.stdout[-2000:] if result.stdout else None,
            "stderr": result.stderr[-2000:] if result.stderr else None,
        })

    # 1) stubtest: validate stubs against the installed runtime package
    _run("stubtest_exllamav3", [sys.executable, "-m", "mypy.stubtest",
                                 "--mypy-config-file", str(root / "pyproject.toml"),
                                 "exllamav3.conversion.convert_model"])

    # 2) mypy: validate src/exl3pack against the stubs using repo config
    #    (pyproject.toml baked alongside src/ in the image)
    _run("mypy_src", [sys.executable, "-m", "mypy",
                        "--config-file", str(root / "pyproject.toml"),
                        "--no-error-summary"])

    report = {"checks": checks, "ok": all(c["ok"] for c in checks)}
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


def _selftest_dispatch() -> dict[str, object]:
    """Exercise `_dequant_fp8_block_dispatch` on synthetic flat + interleaved grids."""
    try:
        import torch

        from .assemble import (
            _dequant_fp8_block,
            _dequant_fp8_block_dispatch,
            _dequant_fp8_block_per_group,
        )
    except Exception as e:  # pragma: no cover - image-only path
        return {"check": "dequant_dispatch", "ok": False, "reason": str(e)}
    bs, groups, m, n = 4, 4, 12, 8
    gen = torch.Generator().manual_seed(7)
    weight = torch.randn(m, n, generator=gen).to(torch.float8_e4m3fn)
    # Interleaved grid: rows_per_group=3 -> ceil(3/4)*4 = 4 grid rows (vs flat 3).
    inter = torch.rand(4, n // bs, generator=gen) + 0.5
    out_i, pg_i = _dequant_fp8_block_dispatch("selftest.inter", weight, inter, (bs, bs), groups)
    ok_i = pg_i and torch.equal(
        out_i, _dequant_fp8_block_per_group(weight, inter, (bs, bs), groups)
    )
    # Flat grid: exact multiple -> flat path.
    flat = torch.rand(-(-m // bs), n // bs, generator=gen) + 0.5
    out_f, pg_f = _dequant_fp8_block_dispatch("selftest.flat", weight, flat, (bs, bs), groups)
    ok_f = (not pg_f) and torch.equal(out_f, _dequant_fp8_block(weight, flat, (bs, bs)))
    return {"check": "dequant_dispatch", "ok": bool(ok_i and ok_f),
            "per_group_path": bool(pg_i), "flat_path": bool(not pg_f)}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="exl3-pack", description=__doc__)
    ap.add_argument("--spec", help="path to a per-model spec.py")
    ap.add_argument("--model", help="model slug (resolved under --specs-root)")
    ap.add_argument("--specs-root", default=None, help="root of per-model spec dirs")
    ap.add_argument("--node-map", default=str(paths.default_map_path()))
    ap.add_argument("--node", default=None, help="node alias/id or cluster node")
    ap.add_argument("--local-root", default=None)
    ap.add_argument("--out-root", default="/nas-1/models/mimo")
    ap.add_argument("--dry-run", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("detect")
    p.add_argument("source")
    p.add_argument("--assert-parity", action="store_true")
    p.set_defaults(func=_cmd_detect)

    p = sub.add_parser("plan")
    p.add_argument("--source", default=None)
    p.add_argument("--v1-out", default=None)
    p.set_defaults(func=_cmd_plan)

    p = sub.add_parser("recipe")
    p.add_argument("--source", default=None)
    p.add_argument("--work", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--head-bits", type=float, default=None)
    p.add_argument("--hq", action="store_true")
    p.add_argument("--dry-run", action="store_true", default=argparse.SUPPRESS)
    p.set_defaults(func=_cmd_recipe)

    for name in ("convert", "repack", "assemble", "pipeline"):
        p = sub.add_parser(name)
        p.add_argument("--source", default=None)
        p.add_argument("--work", default=None)
        p.add_argument("--exl3-out", default=None)
        p.add_argument("--v1-out", default=None)
        # Shared recipe (model-global quantization plan).
        p.add_argument("--recipe", default=None)
        p.add_argument("--cleanup", choices=("none", "work", "all"), default="all")
        p.set_defaults(func=_cmd_run_stage, stage=name)

    p = sub.add_parser("status")
    p.add_argument("status_file", nargs="?", default=None)
    p.add_argument("--watch", action="store_true")
    p.add_argument("--interval", type=float, default=5.0)
    p.add_argument("--delta", type=int, default=0, help="layers advanced since last poll")
    p.add_argument("--delta-interval", type=float, default=0.0,
                   help="seconds between the two polls the delta covers")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=_cmd_status)

    p = sub.add_parser("selftest")
    p.set_defaults(func=_cmd_selftest)

    p = sub.add_parser("typecheck")
    p.set_defaults(func=_cmd_typecheck)

    p = sub.add_parser("dist-coordinator")
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--work", required=True, type=Path)
    p.add_argument("--exl3-out", required=True, type=Path)
    p.add_argument("--recipe", required=True, type=Path)
    p.add_argument("--bits", type=int, default=None)
    p.add_argument(
        "--codebook", default=None, choices=_EXL3_CODEBOOKS,
        help="EXL3 codebook name (default: coordinator uses mcg)",
    )
    p.add_argument("--nodes", required=True, help="comma-separated node slugs/aliases")
    p.add_argument(
        "--gather-timeout", type=float, default=600.0,
        help="per-shard gather timeout in seconds (default 600.0)",
    )
    p.add_argument(
        "--checkpoint-interval", type=int, default=120,
        help="checkpoint cadence in seconds (default 120)",
    )
    p.add_argument("--node-map", default=str(paths.default_map_path()))
    p.add_argument(
        "--log-dir", type=Path, default=None,
        help="write structured JSONL lifecycle logs under this directory",
    )
    p.set_defaults(func=_cmd_dist_coordinator)

    p = sub.add_parser("dist-worker")
    p.add_argument("--inbox", required=True, type=Path)
    p.add_argument("--shared", required=True, type=Path)
    p.add_argument("--device", type=int, default=0, help="GPU device index (default 0)")
    p.add_argument("--stop", required=True, type=Path)
    p.add_argument(
        "--log-dir", type=Path, default=None,
        help="write structured JSONL lifecycle logs under this directory",
    )
    p.set_defaults(func=_cmd_dist_worker)

    p = sub.add_parser("help")
    p.set_defaults(func=_cmd_help)
    return ap


def _cmd_help(args: argparse.Namespace) -> int:
    build_parser().print_help()
    return 0


def _cmd_run_stage(args: argparse.Namespace) -> int:
    spec = _resolve_spec(args)
    if spec is None:
        print(f"{args.stage} requires --spec or --model", file=sys.stderr)
        return 2
    node_map = paths.load_node_map(Path(args.node_map))
    source = Path(args.source) if args.source else paths.resolve_source(
        node_map, args.node or "", spec.node_map_key or spec.slug
    )
    if source is None:
        print("could not resolve a source checkpoint", file=sys.stderr)
        return 2
    work_root = paths.resolve_work_root(node_map, args.node or "")
    work = Path(args.work) if args.work else work_root / f"{spec.slug}-work-k{spec.bits}"
    exl3_out = Path(args.exl3_out) if args.exl3_out else work_root / f"{spec.slug}-mcg-k{spec.bits}"
    v1_out = Path(args.v1_out) if args.v1_out else paths.resolve_destination(
        spec.slug, Path(args.out_root)
    )
    if args.dry_run:
        from .pipeline import PipelinePlan

        print(PipelinePlan(
            spec=spec, source=source, work=work, exl3_out=exl3_out,
            v1_out=v1_out, node=args.node, mounts=_mounts(),
        ).describe())
        return 0

    from .pipeline import CleanupPlan, run

    cleanup = CleanupPlan(
        delete_work=args.cleanup in ("work", "all"),
        delete_exl3_out=args.cleanup == "all",
    )
    if args.stage == "pipeline":
        run(spec, source, work, exl3_out, v1_out, cleanup=cleanup)
        return 0
    if args.stage == "convert":
        from . import convert

        extra_args = ["--recipe", args.recipe] if getattr(args, "recipe", None) else None
        convert.run(
            source,
            exl3_out,
            work,
            bits=spec.bits,
            codebook=spec.codebook,
            extra_args=extra_args,
        )
        return 0
    if args.stage == "repack":
        from . import repack

        repack.run(exl3_out, v1_out, bits=spec.bits, self_check_on=True)
        if cleanup.delete_exl3_out:
            _rmtree(exl3_out)
        return 0
    if args.stage == "assemble":
        from . import assemble

        assemble.run(
            source,
            exl3_out,
            v1_out,
            cleanup=assemble.CleanupPolicy(delete_pack_after_copy=cleanup.delete_exl3_out),
        )
        return 0
    print(f"unknown stage {args.stage}", file=sys.stderr)
    return 2


def _rmtree(path: Path) -> None:
    import shutil

    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


def _cmd_dist_coordinator(args: argparse.Namespace) -> int:
    """Run the distributed quantization coordinator in-process.

    Resolves the per-node work roots from the node map, builds
    ``WorkerEndpoint`` records under ``<work>/dist/inbox/<node>``,
    and hands the loop to :class:`exl3pack.distributed.Coordinator`.
    """
    from .dist_types import WorkerEndpoint
    from .distributed import Coordinator


    paths.load_node_map(Path(args.node_map))
    nodes = [n.strip() for n in args.nodes.split(",") if n.strip()]
    if not nodes:
        print("error: --nodes must list at least one node", file=sys.stderr)
        return 2

    work = Path(args.work)
    # PackSpec geometry -> commit-time MoE validation (dist_commit): the MoE
    # validate range and expert count come from the spec the run was launched
    # with, never from baked flash constants. No spec (raw --bits path) keeps
    # the upstream-compatible flash default.
    _spec_geo = _resolve_spec(args)
    if _spec_geo is None and args.bits is None:
        print("dist-coordinator: --bits or --model required", file=sys.stderr)
        return 2
    dist_args: CoordinatorArgs = {
        "in_dir": str(args.source),
        "out_dir": str(args.exl3_out),
        # bits/codebook resolved below from --bits or PackSpec; placeholders
        # replaced before any consumer runs.
        "bits": 0,
        "codebook": "mcg",
        "recipe": str(args.recipe),
        "work_dir": str(work),
        "gather_timeout": float(args.gather_timeout),
        "checkpoint_interval": int(args.checkpoint_interval),
        # prepare() also reads these; supply upstream argparse defaults
        "resume": False,
        "head_bits": None,
        "mtp_bits": None,
        "hq": False,
        "override_anyway": False,
        "image_dump": False,
        "verbose": False,
        "out_scales": "always",
        "max_module": None,
        # override() table entries with upstream argparse defaults
        # NB: override() treats falsy values as absent (table default applies),
        # so ""/0/False here behave identically to missing keys on fresh runs;
        # on resume, a falsy CLI value cannot override the saved job.json value.
        "shard_size": 8192,
        "vision_bits": 0,
        "ngram_bits": 0,
        "ngram_file": "",
        "cal_data": "",
        "cal_rows": 250,
        "cal_cols": 2048,
        "last_checkpoint_index": -1,
        "devices": "0",
        "device_ratios": "",
        "hessians": "",
        "hessians_reg": 0.025,
        "moe_layer_lo": 1,
        "moe_layer_hi": (
            47 if _spec_geo is None else _spec_geo.geometry.moe_layer_count
        ),
        "moe_expert_count": (
            256 if _spec_geo is None else _spec_geo.geometry.num_experts
        ),
    }
    if args.bits is not None:
        dist_args["bits"] = int(args.bits)
    else:
        if _spec_geo is None:
            print("dist-coordinator: --bits or --model required", file=sys.stderr)
            return 2
        dist_args["bits"] = int(_spec_geo.bits)
    dist_args["codebook"] = args.codebook or "mcg"

    endpoints: list[WorkerEndpoint] = []
    for node in nodes:
        inbox = work / "dist" / "inbox" / node
        endpoints.append(WorkerEndpoint(node=node, inbox=str(inbox), device=0))

    coord = Coordinator(dist_args, endpoints, work, log_dir=getattr(args, "log_dir", None))
    try:
        coord.run()
    except Exception as exc:
        # Evidence: the exception must land in the coordinator JSONL with a
        # full traceback before the process dies (--rm destroys the container
        # and stdout with it). Narrow excepts here previously let crashes
        # escape with zero trace in the log (rc=1, blind rerun).
        log_event(
            coord.log,
            logging.ERROR,
            "coordinator.error",
            fields={"error": str(exc), "traceback": traceback.format_exc()},
        )
        print(f"error: coordinator failed: {exc}", file=sys.stderr)
        raise
    return 0


def _cmd_dist_worker(args: argparse.Namespace) -> int:
    """Run a per-node quantization worker polling its inbox."""
    from .worker import serve

    try:
        serve(
            Path(args.inbox),
            Path(args.shared),
            int(args.device),
            Path(args.stop),
            log_dir=getattr(args, "log_dir", None),
        )
    except (RuntimeError, TimeoutError, OSError) as exc:
        print(f"error: worker failed: {exc}", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
