#!/usr/bin/env python3
"""spark-vllm-docker eval rig CLI.

Usage:
  cli.py run <config.toml> [--dry-run] [extra lm-eval args...]
  cli.py viz <results-dir> [--model M] [--run TS|latest] [--out PATH]
             [--local] [--html] [--zeno] [--zeno-project NAME]
  cli.py setup

`run` parses the TOML, bootstraps eval/.venv (via install.sh) if the harness
is not installed, translates the config to an `lm-eval run` argv, and execs it.
`viz` targets a single run (results_<ts>.json + samples_<task>_<ts>.jsonl),
selected by --model/--run or an interactive model -> run wizard, and renders a
markdown summary plus a self-contained HTML report; it optionally uploads
per-task projects to a Zeno hub (from the config's [zeno] table).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
VENV = SCRIPT_DIR / ".venv"
INSTALL_SH = SCRIPT_DIR / "install.sh"
LMEVAL = VENV / "bin" / "lm-eval"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from report import (
    RunRef,
    discover_runs,
    load_run,
    metric_rows,
    read_asset,
    render_html,
    render_markdown,
    resolve_metric_key,
)

DEFAULT_MODEL_ARGS_KEYS = (
    "model",
    "base_url",
    "tokenizer_backend",
    "num_concurrent",
    "max_images",
    "max_retries",
    "timeout",
)

# Task-name prefixes whose tasks are `unsafe_code: true` (they execute the
# model's generated Python). The harness refuses these unless
# `--confirm_run_unsafe_code` is passed, so the rig adds it automatically.
UNSAFE_CODE_TASK_PREFIXES = ("humaneval", "mbpp", "cruxeval", "longproc")

# ---------------------------------------------------------------- run

def build_argv(cfg: dict, dry_run: bool) -> list[str]:
    run = cfg["run"]
    model = cfg["model"]
    zeno = cfg.get("zeno", {})

    name = run["name"]
    tasks = run["tasks"]
    backend = model["backend"]
    base_url = model["base_url"]
    model_id = model["model"]

    out_dir = SCRIPT_DIR / run.get("output_dir", "results") / name
    out_dir.mkdir(parents=True, exist_ok=True)

    margs: dict[str, str | int] = {
        "model": model_id,
        "base_url": base_url,
        "tokenizer_backend": model.get("tokenizer_backend", "huggingface"),
    }
    if (n := model.get("num_concurrent")) is not None:
        margs["num_concurrent"] = n
    if (tok_repo := model.get("tokenizer")):
        margs["tokenizer"] = tok_repo
    if backend == "local-chat-completions" and (mi := model.get("max_images")) is not None:
        margs["max_images"] = mi
    for key in ("max_retries", "timeout"):
        if (v := model.get(key)) is not None:
            margs[key] = v
    if (env_name := model.get("auth_token_env")) and (tok := os.environ.get(env_name)):
        # local-completions/local-chat-completions build the request header from
        # `api_key` (TemplateAPI.header -> OPENAI_API_KEY), NOT from the
        # `auth_token` model arg. Propagate the named env var to OPENAI_API_KEY
        # so the `Authorization: Bearer <tok>` header is populated. Direct
        # assignment (not setdefault) so the config's named var wins for this
        # run. Kept out of `model_args` because the harness prints them and
        # stores them in results_*.json.
        os.environ["OPENAI_API_KEY"] = tok
        # `auth_token` is independently consumed by the remote tokenizer
        # backend (RemoteTokenizer) when tokenizer_backend="remote"; forward it
        # only then, so the secret is not written into the printed model_args of
        # ordinary runs.
        if model.get("tokenizer_backend") == "remote":
            margs["auth_token"] = tok

    model_args = ",".join(f"{k}={v}" for k, v in margs.items())

    argv = [str(LMEVAL), "run"]
    argv += ["--model", backend]
    argv += ["--model_args", model_args]
    argv += ["--tasks", ",".join(tasks)]
    if (nf := run.get("num_fewshot")) is not None:
        argv += ["--num_fewshot", str(nf)]
    if (bs := run.get("batch_size")) is not None:
        argv += ["--batch_size", str(bs)]
    if (lim := run.get("limit", 0)):
        argv += ["--limit", str(lim)]
    argv += ["--output_path", str(out_dir)]
    if run.get("log_samples", True):
        argv += ["--log_samples"]
    if (seed := run.get("seed")):
        argv += ["--seed", str(seed)]
    if run.get("apply_chat_template"):
        argv += ["--apply_chat_template"]
    if (gk := run.get("gen_kwargs")):
        argv += ["--gen_kwargs", str(gk)]
    if any(t.startswith(UNSAFE_CODE_TASK_PREFIXES) for t in tasks):
        argv += ["--confirm_run_unsafe_code"]
        # The evaluate `code_eval` metric refuses to run without this opt-in.
        os.environ.setdefault("HF_ALLOW_CODE_EVAL", "1")

    if zeno.get("enabled"):
        # Record the zeno project base so viz can find it later.
        (out_dir / "zeno.json").write_text(
            json.dumps({"project": zeno.get("project", name),
                        "endpoint": zeno.get("endpoint", "")}, indent=2)
        )
    return argv


def cmd_run(cfg_path: str, dry_run: bool, extra: list[str]) -> int:
    cfg = tomllib.loads(Path(cfg_path).read_text())

    if not LMEVAL.exists():
        print(f"[cli] no venv at {VENV}; running install.sh")
        r = subprocess.run(["bash", str(INSTALL_SH)], check=False)
        if r.returncode != 0:
            return r.returncode

    argv = build_argv(cfg, dry_run) + list(extra)
    print("[cli] exec:", " ".join(argv))
    if dry_run:
        return 0
    os.execv(str(argv[0]), argv)


# ---------------------------------------------------------------- viz

def build_task_table(data: list[dict], config: dict) -> tuple[list[tuple], dict]:
    """Mirror harness scripts/zeno_visualize.py generate_dataset/generate_system_df."""
    if config.get("filter_list"):
        ids = [f"{x['doc_id']}.{x['filter']}" for x in data]
    else:
        ids = [x["doc_id"] for x in data]
    labels = [x["target"] for x in data]
    if config["output_type"] == "loglikelihood":
        instance = [x["arguments"]["gen_args_0"]["arg_0"] for x in data]
    elif config["output_type"] == "multiple_choice":
        instance = [
            x["arguments"]["gen_args_0"]["arg_0"]
            + "\n\n"
            + "\n".join([f"- {y[1]}" for y in x["arguments"]])
            for x in data
        ]
    else:  # generate_until
        instance = [x["arguments"]["gen_args_0"]["arg_0"] for x in data]
    return list(zip(ids, instance, labels)), config


def build_system_df(data: list[dict], config: dict) -> dict:
    if config.get("filter_list"):
        ids = [f"{x['doc_id']}.{x['filter']}" for x in data]
        filt = [x["filter"] for x in data]
    else:
        ids = [x["doc_id"] for x in data]
        filt = None
    system: dict = {"id": ids, "doc_id": ids}
    if filt is not None:
        system["filter"] = filt
    if config["output_type"] == "loglikelihood":
        system["output"] = ["correct" if x["filtered_resps"][0][1] else "incorrect" for x in data]
    elif config["output_type"] == "multiple_choice":
        system["output"] = [", ".join([str(y[0]) for y in x["filtered_resps"]]) for x in data]
        system["num_answers"] = [len(x["filtered_resps"]) for x in data]
    else:
        system["output"] = [str(x["filtered_resps"][0]) for x in data]
        system["output_length"] = [len(str(x["filtered_resps"][0])) for x in data]
    metrics = {}
    for m in config["metric_list"]:
        key = resolve_metric_key(m["metric"], data)
        if key is not None:
            metrics[key] = [x[key] for x in data if key in x]
    system.update(metrics)
    return system


def headline(ref: RunRef) -> str:
    """Short `metric=value` headline for a run, used by the wizard."""
    try:
        data = json.loads(ref.results_file.read_text())
    except (OSError, json.JSONDecodeError):
        return ""
    results = data.get("results") or {}
    for task in ref.tasks:
        higher = (data.get("higher_is_better") or {}).get(task, {})
        rows = metric_rows(results.get(task, {}), higher)
        if rows:
            return f"{rows[0].metric}={rows[0].value:.3f}"
    return ""


def prompt_choice(count: int, attempts: int = 3) -> int | None:
    """Read a 1-based index from stdin; None after `attempts` bad tries or EOF."""
    for _ in range(attempts):
        try:
            raw = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print(file=sys.stderr)
            return None
        if raw.isdigit() and 1 <= int(raw) <= count:
            return int(raw) - 1
        print(f"enter a number 1-{count}", file=sys.stderr)
    return None


def wizard(candidates: list[RunRef]) -> RunRef | None:
    """Two-level picker: model, then run (newest-first order preserved)."""
    by_model: dict[str, list[RunRef]] = {}
    for r in candidates:
        by_model.setdefault(r.model, []).append(r)
    models = list(by_model)
    if len(models) == 1:
        chosen = models[0]
    else:
        print("Select model:")
        for i, m in enumerate(models, 1):
            runs = by_model[m]
            print(
                f"  {i}) {m}  [{runs[0].run_name}]  runs={len(runs)}  "
                f"newest={runs[0].timestamp}  {headline(runs[0])}"
            )
        idx = prompt_choice(len(models))
        if idx is None:
            return None
        chosen = models[idx]
    runs = by_model[chosen]
    if len(runs) == 1:
        return runs[0]
    print("Select run:")
    for i, r in enumerate(runs, 1):
        print(f"  {i}) {r.timestamp}  tasks={len(r.tasks)}  {headline(r)}")
    idx = prompt_choice(len(runs))
    return None if idx is None else runs[idx]


def select_run(
    runs: list[RunRef],
    model: str | None,
    run_ts: str | None,
    interactive: bool,
) -> RunRef | None:
    """Pick one run from `runs` (newest-first) by flags or interactively."""
    candidates = runs
    if model:
        needle = model.lower()
        candidates = [
            r for r in runs
            if needle in r.model.lower()
            or needle in r.model_name.lower()
            or needle in r.run_name.lower()
        ]
        if not candidates:
            print(f"[viz] no run matches --model {model}", file=sys.stderr)
            for r in runs:
                print(f"  {r.model} ({r.run_name}) @ {r.timestamp}", file=sys.stderr)
            return None
    if run_ts:
        want = run_ts.replace("_", "-")  # accept 07_58_54 for 07-58-54
        if want == "latest":
            return candidates[0]
        for r in candidates:
            if r.timestamp == want:
                return r
        print(f"[viz] no run with timestamp {want}", file=sys.stderr)
        for r in candidates:
            print(f"  {r.timestamp}  {r.run_name}/{r.model}", file=sys.stderr)
        return None
    if len(candidates) == 1:
        return candidates[0]
    if interactive:
        return wizard(candidates)
    print(
        "[viz] multiple runs found; pass --model NAME [--run TS|latest] or run interactively",
        file=sys.stderr,
    )
    for r in candidates:
        print(f"  {r.model} ({r.run_name}) @ {r.timestamp}  tasks={len(r.tasks)}", file=sys.stderr)
    return None


def cmd_viz(results_dir: str, model: str | None, run_ts: str | None, out: str | None,
            local: bool, html: bool, zeno: bool, zeno_project: str | None) -> int:
    # Relative paths resolve against eval/ (SCRIPT_DIR), not the caller's
    # CWD — the documented usage is `eval.sh viz results` from the repo root.
    rdir = Path(results_dir)
    if not rdir.is_absolute():
        rdir = SCRIPT_DIR / rdir
    if not rdir.is_dir():
        print(f"[viz] no such dir: {rdir}", file=sys.stderr)
        return 1
    runs = discover_runs(rdir)
    if not runs:
        print(f"[viz] no runs found under {rdir}", file=sys.stderr)
        return 1

    ref = select_run(runs, model, run_ts, interactive=sys.stdin.isatty())
    if ref is None:
        return 2
    run = load_run(ref)
    banner = f"[viz] run: {ref.run_name} / {ref.model} @ {ref.timestamp}"
    print(banner)
    print(banner, file=sys.stderr)

    md = render_markdown(run)
    if local or not zeno:
        print(md)
    if html:
        try:
            css = read_asset("report.css")
            js = read_asset("report.js")
        except FileNotFoundError as exc:
            print(f"[viz] {exc}", file=sys.stderr)
            return 1
        doc = render_html(run, css, js)
        hp = Path(out) if out else rdir / "report.html"
        if hp.is_dir():
            hp = hp / "report.html"
        hp.parent.mkdir(parents=True, exist_ok=True)
        hp.write_text(doc)
        print(f"[viz] wrote {hp}")
        size_mb = len(doc) / (1024 * 1024)
        if size_mb > 25:
            print(f"[viz] warning: report is {size_mb:.1f} MB", file=sys.stderr)

    if not zeno:
        return 0

    # Zeno upload
    # build_argv writes <output_dir>/<run-name>/zeno.json, so it sits beside
    # the run directory (rdir/<run_name>/), not beside the results root.
    zcfg_path = rdir / ref.run_name / "zeno.json"
    project_base = zeno_project or (json.loads(zcfg_path.read_text())["project"]
                                    if zcfg_path.exists() else "eval")
    endpoint = json.loads(zcfg_path.read_text())["endpoint"] if zcfg_path.exists() else ""
    key_env = os.environ.get("ZENO_API_KEY_ENV", "ZENO_API_KEY")
    api_key = os.environ.get(key_env)
    if not api_key:
        print(f"[viz] Zeno upload: ${key_env} not set; set it and rerun with --zeno",
              file=sys.stderr)
        return 1
    try:
        from zeno_client import ZenoClient, ZenoMetric  # type: ignore
    except ModuleNotFoundError:
        print("[viz] zeno-client not installed (re-run install.sh with zeno extra)",
              file=sys.stderr)
        return 1
    kwargs = {}
    if endpoint:
        kwargs["endpoint"] = endpoint
    client = ZenoClient(api_key, **kwargs)
    import pandas as pd

    for task_data in run.tasks:
        config, samples = task_data.config, task_data.samples
        dataset_rows, _ = build_task_table(samples, config)
        df = pd.DataFrame(dataset_rows, columns=["id", "data", "labels"])
        metrics = [ZenoMetric(name=k, type="mean", columns=[k])
                   for m in config.get("metric_list", [])
                   if m.get("aggregation") == "mean"
                   for k in [resolve_metric_key(m["metric"], samples)]
                   if k]
        project = client.create_project(
            name=f"{project_base}_{task_data.task}",
            view="text-classification",
            metrics=metrics,
        )
        project.upload_dataset(df, id_column="id", data_column="data",
                               label_column="labels")
        sys_df = pd.DataFrame(build_system_df(samples, config))
        project.upload_system(sys_df, name=task_data.task, id_column="id",
                              output_column="output")
        print(f"[viz] zeno: uploaded {task_data.task} -> {project_base}_{task_data.task}")
    return 0


# ---------------------------------------------------------------- main

def main() -> int:
    p = argparse.ArgumentParser(description="spark-vllm-docker eval rig")
    sub = p.add_subparsers(dest="cmd", required=True)
    rp = sub.add_parser("run", help="run an eval from a TOML config")
    rp.add_argument("config")
    rp.add_argument("--dry-run", action="store_true", help="print argv, do not exec")
    rp.add_argument("extra", nargs="*", help="extra lm-eval args")
    vp = sub.add_parser("viz", help="render report / upload to Zeno")
    vp.add_argument("results_dir")
    vp.add_argument("--model", default=None, help="model slug/name substring filter")
    vp.add_argument("--run", dest="run_ts", default=None, help="run timestamp, or 'latest'")
    vp.add_argument("--out", default=None,
                    help="HTML output path (default <results_dir>/report.html)")
    vp.add_argument("--local", action="store_true", help="print markdown table")
    vp.add_argument("--html", action="store_true", help="write report.html")
    vp.add_argument("--zeno", action="store_true", help="upload to Zeno hub")
    vp.add_argument("--zeno-project", default=None)
    sub.add_parser("setup", help="force venv install")
    args = p.parse_args()

    if args.cmd == "run":
        return cmd_run(args.config, args.dry_run, args.extra)
    if args.cmd == "viz":
        return cmd_viz(args.results_dir, args.model, args.run_ts, args.out,
                       args.local, args.html, args.zeno, args.zeno_project)
    r = subprocess.run(["bash", str(INSTALL_SH), "--refresh"], check=False)
    return r.returncode


if __name__ == "__main__":
    sys.exit(main())
