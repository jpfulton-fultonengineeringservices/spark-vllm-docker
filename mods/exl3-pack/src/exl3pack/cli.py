"""exl3-pack CLI.

Subcommands: ``detect | convert | repack | assemble | pipeline | recipe | status |
selftest | plan``. A model is selected with ``--spec <spec.py>`` or
``--model <slug> --specs-root <dir>``. ``--dry-run`` (or ``plan``) prints the
resolved paths + docker mount plan without executing anything.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import paths
from .geometry import detect
from .spec import PackSpec, load_spec, load_spec_from_file


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
        # Shard range + shared recipe (parallel packing; see exl3-pack-shard.sh).
        p.add_argument("--module-start", type=int, default=None)
        p.add_argument("--max_module", type=int, default=None)
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
            module_start=getattr(args, "module_start", None),
            max_module=getattr(args, "max_module", None),
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


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
