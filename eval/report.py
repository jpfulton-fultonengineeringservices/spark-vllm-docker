#!/usr/bin/env python3
"""Run-targeted report builder for the eval rig.

Discovers lm-eval runs (one `results_<ts>.json` plus its sibling
`samples_<task>_<ts>.jsonl` files), extracts harness aggregates as the
authoritative metric source, and renders a markdown summary plus a
self-contained HTML report.

Rendering is stdlib-only: `eval.sh` runs this under
`uv run --python 3.12 --no-project`, an interpreter without jinja2,
matplotlib or pandas.
"""

from __future__ import annotations

import html
import json
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

RESULTS_RE = re.compile(r"results_(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}(?:\.\d+)?)\.json$")
AGG_RE = re.compile(r"(?P<name>.+?)(?P<stderr>_stderr)?,(?P<group>.+)")
ASSET_DIR = Path(__file__).resolve().parent / "assets"

MAX_STR = 2048
SAMPLE_PAGE = 25


# ---------------------------------------------------------------- discovery


@dataclass
class RunRef:
    """One harness run: a results JSON plus its sibling samples files."""

    results_file: Path
    run_name: str
    model: str
    timestamp: str
    date: float
    model_name: str
    tasks: list[str]


@dataclass
class MetricRow:
    metric: str
    group: str
    value: float
    stderr: float | None
    higher_is_better: bool


@dataclass
class TaskData:
    task: str
    config: dict
    agg: dict
    rows: list[MetricRow]
    samples: list[dict]
    samples_path: Path | None
    coverage: dict
    unsafe_code: bool = False
    extra: dict = field(default_factory=dict)


@dataclass
class RunData:
    ref: RunRef
    meta: dict
    tasks: list[TaskData]

    @property
    def sample_count(self) -> int:
        return sum(len(t.samples) for t in self.tasks)


def discover_runs(results_dir: Path) -> list[RunRef]:
    """All well-formed runs under `results_dir`, newest first."""
    refs: list[RunRef] = []
    for p in results_dir.rglob("results_*.json"):
        m = RESULTS_RE.search(p.name)
        if not m:
            continue
        try:
            data = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        tasks = sorted(data.get("configs") or {})
        if not tasks:
            continue
        rel = p.relative_to(results_dir)
        refs.append(
            RunRef(
                results_file=p,
                run_name=rel.parts[0],
                model=p.parent.name,
                timestamp=m.group(1),
                date=float(data.get("date") or 0.0),
                model_name=str(data.get("model_name") or p.parent.name),
                tasks=tasks,
            )
        )
    refs.sort(key=lambda r: r.timestamp, reverse=True)
    return refs


def iter_samples(path: Path) -> Iterator[dict]:
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def resolve_metric_key(metric: str, samples: list[dict]) -> str | None:
    """Map a metric_list entry to its per-sample key.

    Code-eval metrics (pass@k) store the metric *function source* in
    configs[task].metric_list[].metric; the real per-sample key lives in
    each sample's `metrics` list (e.g. `pass@1`, `pass_at_1`).
    """
    if any(metric in x for x in samples):
        return metric
    for x in samples:
        for name in x.get("metrics") or []:
            if any(name in s for s in samples):
                return name
    return None


def metric_rows(agg: dict, per_task_higher: dict) -> list[MetricRow]:
    """Aggregate metric values + bootstrap stderr from a results JSON."""
    rows: list[MetricRow] = []
    for key, value in agg.items():
        if key in ("name", "alias", "sample_len"):
            continue
        m = AGG_RE.fullmatch(key)
        if not m:
            if isinstance(value, (int, float)):
                rows.append(MetricRow(key, "none", float(value), None,
                                      bool(per_task_higher.get(key, True))))
            continue
        if m.group("stderr"):
            continue
        name, group = m.group("name"), m.group("group")
        if not isinstance(value, (int, float)):
            continue
        stderr = agg.get(f"{name}_stderr,{group}")
        rows.append(
            MetricRow(
                metric=name,
                group=group,
                value=float(value),
                stderr=float(stderr) if isinstance(stderr, (int, float)) else None,
                higher_is_better=bool(per_task_higher.get(name, True)),
            )
        )
    return rows


def load_run(ref: RunRef) -> RunData:
    data = json.loads(ref.results_file.read_text())
    higher = data.get("higher_is_better") or {}
    n_samples = data.get("n-samples") or {}
    tasks: list[TaskData] = []
    for task in ref.tasks:
        config = (data.get("configs") or {}).get(task, {})
        agg = (data.get("results") or {}).get(task, {})
        sp = ref.results_file.parent / f"samples_{task}_{ref.timestamp}.jsonl"
        samples = list(iter_samples(sp)) if sp.exists() else []
        tasks.append(
            TaskData(
                task=task,
                config=config,
                agg=agg,
                rows=metric_rows(agg, higher.get(task, {})),
                samples=samples,
                samples_path=sp if sp.exists() else None,
                coverage=n_samples.get(task, {}),
                unsafe_code=bool(config.get("unsafe_code")),
            )
        )
    meta = {
        "run_name": ref.run_name,
        "model": ref.model,
        "model_name": ref.model_name,
        "timestamp": ref.timestamp,
        "date_iso": (
            datetime.fromtimestamp(ref.date).isoformat(timespec="seconds") if ref.date else ""
        ),
        "backend": str(data.get("model_source") or (data.get("config") or {}).get("model") or ""),
        "base_url": str(((data.get("config") or {}).get("model_args") or {}).get("base_url") or ""),
        "harness_git": str(data.get("upper_git_hash") or data.get("git_hash") or ""),
        "lm_eval_version": str(data.get("lm_eval_version") or ""),
        "transformers_version": str(data.get("transformers_version") or ""),
        "total_seconds": float(data.get("total_evaluation_time_seconds") or 0.0),
        "task_count": len(tasks),
        "sample_count": sum(len(t.samples) for t in tasks),
    }
    return RunData(ref=ref, meta=meta, tasks=tasks)


# ---------------------------------------------------------------- samples


def sanitize(obj: Any, *, max_str: int = MAX_STR) -> Any:
    """Copy data while excluding embedded image payloads and opaque base64 blobs.

    Long prompts and model outputs are valuable report content, not binary
    blobs; preserve them. Replace image data URLs even when embedded inside a
    prompt, and replace very long strings only when the entire string is
    base64-like.
    """
    if isinstance(obj, str):
        image_url = re.compile(r"data:image/[^;,\s]+;base64,[A-Za-z0-9+/=\r\n]+")
        if "data:image" in obj:
            obj = image_url.sub(
                lambda match: f"[base64-image ~{len(match.group(0)) // 1024} KiB]", obj
            )
        if len(obj) > max_str and re.fullmatch(r"[A-Za-z0-9+/=\r\n]+", obj):
            return f"[binary-like ~{len(obj) // 1024} KiB]"
        return obj
    if isinstance(obj, dict):
        return {str(k): sanitize(v, max_str=max_str) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize(v, max_str=max_str) for v in obj]
    return obj


def _flatten_response(value: Any) -> str:
    """Unwrap resps/filtered_resps nesting down to a displayable string."""
    while isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    if isinstance(value, (list, tuple)):
        value = " ".join(str(v) for v in value)
    return "" if value is None else str(value)


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def sample_score_keys(sample: dict, config: dict) -> list[str]:
    """Per-sample metric keys: metrics listed by the sample exist on it."""
    keys = [k for k in (sample.get("metrics") or []) if k in sample]
    if keys:
        return keys
    for m in config.get("metric_list") or []:
        k = resolve_metric_key(str(m.get("metric", "")), [sample])
        if k:
            keys.append(k)
    return keys


def sample_record(sample: dict, config: dict) -> dict:
    """Explorer record for one harness sample."""
    output_type = str(config.get("output_type") or "generate_until")
    filt = sample.get("filter")
    doc_id = sample.get("doc_id")
    sid = str(doc_id) if filt in (None, "", "none") else f"{doc_id}.{filt}"

    scores: dict[str, float] = {}
    for k in sample_score_keys(sample, config):
        v = _to_float(sample.get(k))
        if v is not None:
            scores[k] = v

    if not scores:
        outcome = "fail"
    elif all(v >= 1.0 for v in scores.values()):
        outcome = "pass"
    elif all(v == 0.0 for v in scores.values()):
        outcome = "fail"
    else:
        outcome = "partial"

    arguments = sample.get("arguments") or {}
    args = [v for _, v in sorted(arguments.items())]
    prompt = ""
    if args and isinstance(args[0], dict):
        prompt = str(args[0].get("arg_0") or "")

    filtered = sample.get("filtered_resps")
    if isinstance(filtered, list) and filtered:
        output = _flatten_response(filtered[0])
    else:
        output = _flatten_response(sample.get("resps"))

    choices: list[list[Any]] = []
    picked: int | None = None
    correct: int | None = None
    if output_type in ("multiple_choice", "loglikelihood"):
        scores_mc: list[float] = []
        for i, arg in enumerate(args):
            if not isinstance(arg, dict):
                continue
            text = str(arg.get("arg_1") or "")
            score = None
            if isinstance(filtered, list) and i < len(filtered):
                inner = filtered[i]
                if isinstance(inner, (list, tuple)) and inner:
                    score = _to_float(inner[0])
                else:
                    score = _to_float(inner)
            if score is not None:
                scores_mc.append(score)
            choices.append([text, score])
        if scores_mc:
            picked = max(
                range(len(choices)),
                key=lambda i: choices[i][1] if choices[i][1] is not None else float("-inf"),
            )
        target = _to_float(sample.get("target"))
        correct = int(target) if target is not None else None
        if picked is not None and correct is not None:
            scores.setdefault("acc", 1.0 if picked == correct else 0.0)
        if output and output.startswith(("-", "[")):
            output = "(loglikelihood)"

    return {
        "id": sid,
        "outcome": outcome,
        "scores": scores,
        "prompt": sanitize(prompt),
        "output": sanitize(output) if output else "(no output)",
        "reference": sanitize(str(sample.get("target") or "")),
        "choices": [[str(t), s] for t, s in choices],
        "picked": picked,
        "correct": correct,
    }


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "task"


# ---------------------------------------------------------------- payload


def build_payload(run: RunData) -> dict:
    tasks = []
    for t in run.tasks:
        records = [sample_record(s, t.config) for s in t.samples]
        counts = {"pass": 0, "fail": 0, "partial": 0}
        for r in records:
            counts[r["outcome"]] += 1
        filters = [
            str(f.get("name"))
            for f in (t.config.get("filter_list") or [])
            if isinstance(f, dict)
        ]
        tasks.append(
            {
                "task": t.task,
                "slug": slugify(t.task),
                "output_type": str(t.config.get("output_type") or ""),
                "num_fewshot": int(t.config.get("num_fewshot") or 0),
                "repeats": int(t.config.get("repeats") or 1),
                "dataset_path": str(t.config.get("dataset_path") or ""),
                "test_split": str(t.config.get("test_split") or ""),
                "generation_kwargs": sanitize(t.config.get("generation_kwargs") or {}),
                "filter_names": filters,
                "description": str(t.config.get("description") or ""),
                "unsafe_code": t.unsafe_code,
                "metrics": [
                    {
                        "metric": r.metric,
                        "group": r.group,
                        "value": r.value,
                        "stderr": r.stderr,
                        "higher_is_better": r.higher_is_better,
                    }
                    for r in t.rows
                ],
                "coverage": {
                    "original": int(t.coverage.get("original") or 0),
                    "effective": int(t.coverage.get("effective") or 0),
                },
                "counts": counts,
                "sample_note": (
                    ""
                    if t.samples_path
                    else f"samples file missing: samples_{t.task}_{run.ref.timestamp}.jsonl"
                ),
                "samples": records,
            }
        )
    return {"run": run.meta, "tasks": tasks}


# ---------------------------------------------------------------- markdown


def render_markdown(run: RunData) -> str:
    lines = [
        f"## {run.meta['run_name']} / {run.meta['model']} @ {run.meta['timestamp']}",
        "",
        (f"backend `{run.meta['backend']}`  server `{run.meta['base_url']}`  "
        f"lm_eval {run.meta['lm_eval_version']}  git {run.meta['harness_git'][:8]}"),
        "",
        "| task | metric | group | value | stderr | n |",
        "|---|---|---|---|---|---|",
    ]
    for t in run.tasks:
        n = t.coverage.get("effective") or len(t.samples)
        if not t.rows:
            lines.append(f"| {t.task} | (no aggregates) | | | | {n} |")
            continue
        for r in t.rows:
            stderr = f"±{r.stderr:.4f}" if r.stderr is not None else ""
            lines.append(f"| {t.task} | {r.metric} | {r.group} | {r.value:.4f} | {stderr} | {n} |")
    return "\n".join(lines)


# ---------------------------------------------------------------- svg


def _esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def svg_bars(rows: list[MetricRow], width: int = 760) -> str:
    """Horizontal bar chart of metric values with stderr whiskers."""
    if not rows:
        return '<p class="muted">no metrics to chart</p>'
    label_w = 240
    bar_h, gap = 22, 10
    height = len(rows) * (bar_h + gap) + 20
    plot_w = width - label_w - 80
    values = [r.value for r in rows]
    lo = 0.0 if min(values) >= 0.0 and max(values) <= 1.0 else min(values)
    hi = 1.0 if lo == 0.0 else max(values)
    if hi <= lo:
        hi = lo + 1.0
    span = hi - lo

    def x(v: float) -> float:
        return label_w + (v - lo) / span * plot_w

    parts = [
        (
            f'<svg class="chart" viewBox="0 0 {width} {height}" '
            f'width="{width}" height="{height}" role="img">'
        )
    ]
    for i, r in enumerate(rows):
        y = 10 + i * (bar_h + gap)
        label = f"{r.metric}"
        if r.group and r.group != "none":
            label += f" ({r.group})"
        parts.append(
            f'<text x="0" y="{y + bar_h * 0.72:.0f}" class="chart-label">{_esc(label)}</text>'
        )
        w = max(x(r.value) - label_w, 0.0)
        cls = "bar-pos" if r.value >= 0 else "bar-neg"
        parts.append(
            f'<rect x="{label_w}" y="{y}" width="{w:.2f}" height="{bar_h}" '
            f'class="{cls}" rx="3"></rect>'
        )
        if r.stderr:
            x1, x2 = x(max(r.value - r.stderr, lo)), x(min(r.value + r.stderr, hi))
            yc = y + bar_h / 2
            parts.append(
                f'<line x1="{x1:.2f}" y1="{yc:.1f}" x2="{x2:.2f}" y2="{yc:.1f}" '
                'class="whisker"></line>'
            )
            parts.append(
                f'<line x1="{x1:.2f}" y1="{yc - 4:.1f}" x2="{x1:.2f}" y2="{yc + 4:.1f}" '
                'class="whisker"></line>'
            )
            parts.append(
                f'<line x1="{x2:.2f}" y1="{yc - 4:.1f}" x2="{x2:.2f}" y2="{yc + 4:.1f}" '
                'class="whisker"></line>'
            )
        vt = f"{r.value:.3f}" + (f" ±{r.stderr:.3f}" if r.stderr else "")
        parts.append(
            f'<text x="{width - 4}" y="{y + bar_h * 0.72:.0f}" class="chart-value" '
            f'text-anchor="end">{_esc(vt)}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def svg_coverage(tasks: list[TaskData], width: int = 760) -> str:
    """Stacked coverage bars: effective vs original sample counts per task."""
    if not tasks:
        return ""
    label_w = 240
    bar_h, gap = 22, 10
    height = len(tasks) * (bar_h + gap) + 20
    plot_w = width - label_w - 80
    hi = max([int(t.coverage.get("original") or 0) for t in tasks] + [1])
    parts = [
        (
            f'<svg class="chart" viewBox="0 0 {width} {height}" '
            f'width="{width}" height="{height}" role="img">'
        )
    ]
    for i, t in enumerate(tasks):
        y = 10 + i * (bar_h + gap)
        eff = int(t.coverage.get("effective") or len(t.samples))
        orig = int(t.coverage.get("original") or eff)
        parts.append(
            f'<text x="0" y="{y + bar_h * 0.72:.0f}" class="chart-label">{_esc(t.task)}</text>'
        )
        parts.append(
            f'<rect x="{label_w}" y="{y}" width="{orig / hi * plot_w:.2f}" '
            f'height="{bar_h}" class="bar-orig" rx="3"></rect>'
        )
        parts.append(
            f'<rect x="{label_w}" y="{y}" width="{eff / hi * plot_w:.2f}" '
            f'height="{bar_h}" class="bar-eff" rx="3"></rect>'
        )
        parts.append(
            f'<text x="{width - 4}" y="{y + bar_h * 0.72:.0f}" class="chart-value" '
            f'text-anchor="end">{eff}/{orig}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


# ---------------------------------------------------------------- html


def _metric_cards(task: dict) -> str:
    out = []
    for m in task["metrics"]:
        arrow = "▲" if m["higher_is_better"] else "▼"
        stderr = f" ±{m['stderr']:.4f}" if m["stderr"] is not None else ""
        group = "" if m["group"] == "none" else f'<span class="group">{_esc(m["group"])}</span>'
        out.append(
            f'<div class="metric-card"><div class="metric-name">{_esc(m["metric"])}{group}</div>'
            f'<div class="metric-value">{m["value"]:.4f}'
            f'<span class="metric-stderr">{stderr}</span></div>'
            f'<div class="metric-dir">{arrow} is better</div></div>'
        )
    return "".join(out) or '<p class="muted">no aggregates for this task</p>'


def _histogram(task: dict) -> str:
    values = [
        s["scores"][k]
        for s in task["samples"]
        for k in (list(s["scores"])[:1])
        if k in s["scores"]
    ]
    if not values:
        return ""
    lo, hi = 0.0, 1.0
    bins = [0] * 10
    for v in values:
        idx = min(max(int(v * 10), 0), 9)
        bins[idx] += 1
    top = max(bins) or 1
    bars = "".join(
        f'<div class="hist-bar" style="height:{max(b / top * 100, 2):.0f}%" '
        f'title="{b} samples"></div>'
        for b in bins
    )
    return (
        f'<div class="hist"><div class="hist-bars">{bars}</div>'
        f'<div class="hist-scale"><span>{lo:.2f}</span><span>{hi:.2f}</span></div></div>'
    )


def _mis_selections(task: dict) -> str:
    bad = [
        s
        for s in task["samples"]
        if s["picked"] is not None and s["correct"] is not None and s["picked"] != s["correct"]
    ]
    if not bad:
        return ""
    rows = []
    for s in bad[:5]:
        picked_txt = s["choices"][s["picked"]][0] if s["picked"] < len(s["choices"]) else "?"
        correct_txt = s["choices"][s["correct"]][0] if s["correct"] < len(s["choices"]) else "?"
        rows.append(
            f"<tr><td>{_esc(s['id'])}</td><td>{_esc(picked_txt)}</td><td>{_esc(correct_txt)}</td>"
            f"<td>{_esc(s['prompt'][-200:])}</td></tr>"
        )
    return (
        f"<h4>Mis-selections (first 5 of {len(bad)})</h4>"
        '<table class="data"><thead><tr><th>id</th><th>picked</th>'
        f"<th>correct</th><th>prompt tail</th></tr></thead><tbody>{''.join(rows)}</tbody></table>"
    )


def _config_summary(task: dict) -> str:
    items = {
        "dataset": task["dataset_path"],
        "split": task["test_split"],
        "output_type": task["output_type"],
        "num_fewshot": task["num_fewshot"],
        "repeats": task["repeats"],
        "filters": ", ".join(task["filter_names"]) or "-",
        "unsafe_code": task["unsafe_code"],
        "generation_kwargs": json.dumps(task["generation_kwargs"], ensure_ascii=True),
    }
    chips = "".join(
        f'<span class="chip"><b>{_esc(k)}</b> {_esc(v)}</span>' for k, v in items.items()
    )
    desc = f'<p class="muted">{_esc(task["description"][:400])}</p>' if task["description"] else ""
    return f'<div class="chips">{chips}</div>{desc}'


def _sample_card(s: dict) -> str:
    scores = " ".join(f"{k}={v:g}" for k, v in s["scores"].items())
    body = [
        (f'<div class="sample-head"><span class="badge badge-{s["outcome"]}">{s["outcome"]}</span>'
        f'<span class="sample-id">{_esc(s["id"])}</span>'
        f'<span class="sample-scores">{_esc(scores)}</span></div>')
    ]
    if s["choices"]:
        rows = "".join(
            f'<tr{" class=\"picked\"" if i == s["picked"] else ""}><td>{i}</td>'
            f"<td>{_esc(t)}</td><td>{'' if v is None else f'{v:.3f}'}</td></tr>"
            for i, (t, v) in enumerate(s["choices"])
        )
        body.append(
            '<table class="choices"><thead><tr>'
            "<th>i</th><th>choice</th><th>score</th></tr></thead>"
            f"<tbody>{rows}</tbody></table>"
        )
    body.append(f'<details><summary>prompt</summary><pre>{_esc(s["prompt"])}</pre></details>')
    body.append(f'<details open><summary>output</summary><pre>{_esc(s["output"])}</pre></details>')
    body.append(f'<details><summary>reference</summary><pre>{_esc(s["reference"])}</pre></details>')
    return f'<article class="sample-card outcome-{s["outcome"]}">{"".join(body)}</article>'


def _static_samples(task: dict) -> str:
    """Server-rendered first page so the report is useful without JS."""
    if not task["samples"]:
        note = task["sample_note"] or "no samples recorded for this task"
        return f'<p class="muted">{_esc(note)}</p>'
    cards = "".join(_sample_card(s) for s in task["samples"][:SAMPLE_PAGE])
    return f'<div class="static-samples">{cards}</div>'


def _summary_table(tasks: list[dict]) -> str:
    rows = []
    for t in tasks:
        n = t["coverage"]["effective"] or len(t["samples"])
        if not t["metrics"]:
            rows.append(
                f'<tr><td>{_esc(t["task"])}</td>'
                f'<td colspan="4" class="muted">no aggregates</td><td>{n}</td></tr>'
            )
            continue
        for m in t["metrics"]:
            stderr = f"±{m['stderr']:.4f}" if m["stderr"] is not None else ""
            rows.append(
                f"<tr><td>{_esc(t['task'])}</td><td>{_esc(m['metric'])}</td>"
                f"<td>{_esc(m['group'])}</td><td>{m['value']:.4f}</td>"
                f"<td>{_esc(stderr)}</td><td>{n}</td></tr>"
            )
    return (
        '<table class="data"><thead><tr><th>task</th><th>metric</th><th>group</th>'
        "<th>value</th><th>stderr</th><th>n</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def render_html(run: RunData, css: str, js: str) -> str:
    payload = build_payload(run)
    payload_json = json.dumps(payload, ensure_ascii=True).replace("</", "<\\/")
    meta = run.meta
    secs = meta["total_seconds"]
    duration = f"{int(secs // 60)}m {int(secs % 60)}s" if secs else "-"
    nav = (
        '<button class="task-tab active" type="button" data-tab="overview">Overview</button>'
        + "".join(
            f'<button class="task-tab" type="button" data-tab="{_esc(t["slug"])}">'
            f'{_esc(t["task"])}</button>'
            for t in payload["tasks"]
        )
    )
    overview_metrics = "".join(
        f'<section class="overview-task"><h3>{_esc(t["task"])}</h3>'
        f'<div class="metric-grid">{_metric_cards(t)}</div></section>'
        for t in payload["tasks"]
    )

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>eval report — {_esc(meta['run_name'])} / {_esc(meta['model'])}</title>
<style>{css}</style>
</head><body>
<header class="topbar">
  <div>
    <h1>{_esc(meta['run_name'])} <span class="muted">/ {_esc(meta['model'])}</span></h1>
    <div class="muted">{_esc(meta['timestamp'])} · {_esc(meta['date_iso'])}</div>
  </div>
  <button id="theme-toggle" type="button">theme</button>
</header>
<div class="report-layout">
  <nav class="tasknav" aria-label="Report tabs">{nav}</nav>
  <main>
  <div id="overview-tab" class="tab-panel active">
  <section id="run-meta" class="panel">
    <h2>Run</h2>
    <div class="chips">
      <span class="chip"><b>model</b> {_esc(meta['model_name'])}</span>
      <span class="chip"><b>backend</b> {_esc(meta['backend'])}</span>
      <span class="chip"><b>server</b> {_esc(meta['base_url'])}</span>
      <span class="chip"><b>harness git</b> {_esc(meta['harness_git'][:12])}</span>
      <span class="chip"><b>lm_eval</b> {_esc(meta['lm_eval_version'])}</span>
      <span class="chip"><b>transformers</b> {_esc(meta['transformers_version'])}</span>
      <span class="chip"><b>tasks</b> {meta['task_count']}</span>
      <span class="chip"><b>samples</b> {meta['sample_count']}</span>
      <span class="chip"><b>wall</b> {_esc(duration)}</span>
    </div>
  </section>
  <section id="summary" class="panel">
    <h2>Task metrics</h2>{overview_metrics}
    <h2>Aggregate metrics</h2>{_summary_table(payload['tasks'])}
  </section>
  <section id="charts" class="panel">
    <h2>Charts</h2>
    <h3>Metric values (with stderr)</h3>
    {svg_bars(
        [MetricRow(m["metric"], m["group"], m["value"], m["stderr"], m["higher_is_better"])
         for t in payload["tasks"] for m in t["metrics"]]
    )}
    <h3>Sample coverage (evaluated / available)</h3>
    {svg_coverage(run.tasks)}
  </section>
  </div>
  <div id="tasks">{''.join(
      f'<section class="tab-panel task-tab-panel" id="tab-{_esc(t["slug"])}">'
      f'<section class="task-card" id="task-{_esc(t["slug"])}"><h2>{_esc(t["task"])}</h2>'
      f'{_config_summary(t)}<div class="metric-grid">{_metric_cards(t)}</div>'
      f'<div class="counts"><span class="badge badge-pass">pass {t["counts"]["pass"]}</span>'
      f'<span class="badge badge-fail">fail {t["counts"]["fail"]}</span>'
      f'<span class="badge badge-partial">partial {t["counts"]["partial"]}</span></div>'
      f'{_histogram(t)}{_mis_selections(t)}<h3>Sample explorer</h3>'
      f'<div class="explorer-tools task-explorer-tools" data-task="{_esc(t["task"])}">'
      f'<input type="search" placeholder="search this task (press /)">'
      f'<select class="outcome-filter"><option value="all">all outcomes</option>'
      f'<option value="pass">pass only</option><option value="fail">fail only</option>'
      f'<option value="partial">partial only</option></select>'
      f'<span class="count muted"></span><button class="prev" type="button">prev</button>'
      f'<button class="next" type="button">next</button></div>'
      f'<noscript>{_static_samples(t)}</noscript>'
      f'<div class="explorer-list task-explorer-list" data-task="{_esc(t["task"])}"></div>'
      f'</section></section>' for t in payload["tasks"]
  )}</div>
  <section id="explorer" class="panel overview-only">
    <h2>Run-wide explorer</h2>
    <div class="explorer-tools" data-task="__global__">
      <input id="search" type="search"
             placeholder="search prompt / output / reference / id  (press /)">
      <select id="outcome-filter"><option value="all">all outcomes</option>
        <option value="pass">pass only</option><option value="fail">fail only</option>
        <option value="partial">partial only</option>
      </select>
      <select id="task-filter"><option value="all">all tasks</option></select>
      <span id="count" class="muted"></span>
      <button id="prev" type="button">prev</button>
      <button id="next" type="button">next</button>
    </div>
    <div id="explorer-list" class="explorer-list"></div>
  </section>
  </main>
</div>
<script id="eval-data" type="application/json">{payload_json}</script>
<script>{js}</script>
</body></html>
"""


def read_asset(name: str) -> str:
    path = ASSET_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"missing asset: {path}")
    return path.read_text()
