#!/usr/bin/env python3
"""spark-vllm-docker eval rig CLI.

Usage:
  cli.py run <config.toml> [--dry-run] [extra lm-eval args...]
  cli.py viz <results-dir> [--local] [--html] [--zeno] [--zeno-project NAME]
  cli.py setup

`run` parses the TOML, bootstraps eval/.venv (via install.sh) if the harness
is not installed, translates the config to an `lm-eval run` argv, and execs it.
`viz` renders a local markdown/HTML report from results_<timestamp>.json /
samples_<timestamp>.jsonl files and optionally uploads per-task projects to a
Zeno hub (cloud or self-hosted endpoint from the config's [zeno] table).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
VENV = SCRIPT_DIR / ".venv"
INSTALL_SH = SCRIPT_DIR / "install.sh"
LMEVAL = VENV / "bin" / "lm-eval"

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
        r = subprocess.run(["bash", str(INSTALL_SH)])
        if r.returncode != 0:
            return r.returncode

    argv = build_argv(cfg, dry_run) + list(extra)
    print("[cli] exec:", " ".join(argv))
    if dry_run:
        return 0
    os.execv(str(argv[0]), argv)


# ---------------------------------------------------------------- viz

def find_task_files(results_dir: Path, task: str) -> tuple[Path, Path] | None:
    """Newest results_*.json + matching samples_*.jsonl for a task."""
    results = sorted(results_dir.glob("results_*.json"))
    for rf in reversed(results):
        data = json.loads(rf.read_text())
        if task in data.get("configs", {}):
            m = re.match(r"results_(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}(?:\.\d+)?)\.json$", rf.name)
            ts = m.group(1) if m else None
            samples = results_dir / f"samples_{task}_{ts}.jsonl"
            if samples.exists():
                return rf, samples
    return None


def iter_samples(path: Path):
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def build_task_table(data: list[dict], config: dict) -> tuple[list[dict], dict]:
    """Mirror harness scripts/zeno_visualize.py generate_dataset/generate_system_df."""
    config.get("filter_list")
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
    metrics = {
        m["metric"]: [x[m["metric"]] for x in data] for m in config["metric_list"]
    }
    system.update(metrics)
    return system


def collect_tasks(results_dir: Path) -> list[tuple[str, dict, list[dict], Path]]:
    """(task, results-config, samples, samples_path) for every task present."""
    out: list[tuple[str, dict, list[dict], Path]] = []
    seen: set[str] = set()
    # The harness nests outputs as <output_path>/<sanitized_model>/results_*.json.
    for rf in sorted(results_dir.rglob("results_*.json")):
        data = json.loads(rf.read_text())
        for task, config in data.get("configs", {}).items():
            if task in seen:
                continue
            ts_m = re.match(r"results_(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}(?:\.\d+)?)\.json$", rf.name)
            ts = ts_m.group(1) if ts_m else "*"
            sp = rf.parent / f"samples_{task}_{ts}.jsonl"
            if not sp.exists():
                continue
            out.append((task, config, list(iter_samples(sp)), sp))
            seen.add(task)
    return out


def report_markdown(tasks: list[tuple[str, dict, list[dict], Path]]) -> str:
    lines = ["| task | metric | value | stderr |", "|---|---|---|---|"]
    for task, config, samples, _ in tasks:
        for m in config.get("metric_list", []):
            metric = m["metric"]
            vals = [x[metric] for x in samples if metric in x and x[metric] is not None]
            if not vals:
                lines.append(f"| {task} | {metric} | (missing) | |")
                continue
            mean = sum(vals) / len(vals)
            stderr = (sum((v - mean) ** 2 for v in vals) / max(len(vals) - 1, 1)) ** 0.5
            lines.append(f"| {task} | {metric} | {mean:.4f} | ±{stderr:.4f} |")
    return "\n".join(lines)


def report_html(tasks: list[tuple[str, dict, list[dict], Path]], md: str) -> str:
    rows = md.splitlines()[2:]
    trs = "\n".join(
        "<tr><td>" + "</td><td>".join(c.strip() for c in r.strip().strip("|").split("|"))
        + "</td></tr>"
        for r in rows
    )
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>eval report</title>
<style>
body{{font-family:-apple-system,sans-serif;margin:2em;}}
table{{border-collapse:collapse;}} td,th{{border:1px solid #ccc;padding:6px 12px;}}
th{{background:#f4f4f4;}}
</style></head><body>
<h1>lm-eval report</h1>
<table><tr><td>task</td><td>metric</td><td>value</td><td>stderr</td></tr>{trs}</table>
</body></html>"""


def cmd_viz(results_dir: str, local: bool, html: bool, zeno: bool,
            zeno_project: str | None) -> int:
    rdir = Path(results_dir)
    if not rdir.is_dir():
        print(f"[viz] no such dir: {rdir}", file=sys.stderr)
        return 1
    tasks = collect_tasks(rdir)
    if not tasks:
        print(f"[viz] no samples/results found under {rdir}", file=sys.stderr)
        return 1

    md = report_markdown(tasks)
    if local or not zeno:
        print(md)
        if html:
            hp = rdir / "report.html"
            hp.write_text(report_html(tasks, md))
            print(f"[viz] wrote {hp}")

    if not zeno:
        return 0

    # Zeno upload
    zcfg_path = rdir.parent / "zeno.json"
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

    for task, config, samples, _ in tasks:
        dataset_rows, _ = build_task_table(samples, config)
        df = pd.DataFrame(dataset_rows, columns=["id", "data", "labels"])
        metrics = [ZenoMetric(name=m["metric"], type="mean", columns=[m["metric"]])
                   for m in config.get("metric_list", [])
                   if m.get("aggregation") == "mean"]
        project = client.create_project(
            name=f"{project_base}_{task}",
            view="text-classification",
            metrics=metrics,
        )
        project.upload_dataset(df, id_column="id", data_column="data",
                               label_column="labels")
        sys_df = pd.DataFrame(build_system_df(samples, config))
        project.upload_system(sys_df, name=task, id_column="id", output_column="output")
        print(f"[viz] zeno: uploaded {task} -> {project_base}_{task}")
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
    vp.add_argument("--local", action="store_true", help="print markdown table")
    vp.add_argument("--html", action="store_true", help="write report.html")
    vp.add_argument("--zeno", action="store_true", help="upload to Zeno hub")
    vp.add_argument("--zeno-project", default=None)
    sub.add_parser("setup", help="force venv install")
    args = p.parse_args()

    if args.cmd == "run":
        return cmd_run(args.config, args.dry_run, args.extra)
    if args.cmd == "viz":
        return cmd_viz(args.results_dir, args.local, args.html, args.zeno,
                       args.zeno_project)
    r = subprocess.run(["bash", str(INSTALL_SH), "--refresh"])
    return r.returncode


if __name__ == "__main__":
    sys.exit(main())
