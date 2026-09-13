#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reproduce the Kimi-K3 / 16 H200 / 1000-in / 1000-out / concurrency-1 search.

Run with the Python interpreter from the installed AIConfigurator environment::

    python tools/kimi_k3_estimates.py
    python tools/kimi_k3_estimates.py --suite estimates
    python tools/kimi_k3_estimates.py --suite parallelism
    python tools/kimi_k3_estimates.py --suite refinement
    python tools/kimi_k3_estimates.py --suite report --ground-truth-repo /path/to/kimi-k3
    python tools/kimi_k3_estimates.py --dry-run

The sweeps are loops over cli_estimate(), the same API used in the original
search, not the CLI's SLA optimizer. Defaults pin the performance DB versions
resolved in that search: vLLM 0.24.0 and SGLang 0.5.14. Quantization is inferred
from the checkpoint; prefix caching and speculative decoding stay disabled.

Outputs go into a new results/kimi-k3-<timestamp>/ directory in this repository:
metadata.json, results.jsonl (including errors and per-op sources), summary.csv,
best.json, commands.txt (equivalent CLI calls), and run.log. The report suite
also writes comparison.json and a Quarto report.qmd. Missing profiles
and OOM candidates are recorded without stopping the sweep. Only results with
16 GPUs, effective concurrency 1, and no KV-cache warning enter the ranking.
Pipeline cases can report higher concurrency even with batch_size=1.

The ranking covers evaluable configurations, not every possible deployment.
Even SILICON mode can use empirical, theoretical, and low-fidelity fallback
data; inspect per_ops_source and run.log before interpreting the predictions.
The installed package supplies the code/data; this script does not modify
sys.path to load the repository's source tree.
"""

import argparse
import csv
import importlib.metadata
import json
import logging
import math
import os
import shlex
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

VERSIONS = {"vllm": "0.24.0", "sglang": "0.5.14"}
COMMON = dict(model_path="moonshotai/Kimi-K3", system_name="h200_sxm", isl=1000, osl=1000, batch_size=1)
CLI_NAMES = {"system_name": "system", "backend_name": "backend", "mode": "estimate-mode"}
REPORT_DIRS = {
    "TP16": "conc-sweep-real-1000-1000",
    "TEP16": "conc-sweep-tep-1000-1000",
    "TP8xPP2": "conc-sweep-pp2-1000-1000",
    "TP8xDP2": "conc-sweep-dp2-1000-1000",
}
REPORT_CONFIGS = {
    "TP16": dict(tp_size=16, pp_size=1, attention_dp_size=1, moe_tp_size=16, moe_ep_size=1),
    "TEP16": dict(tp_size=16, pp_size=1, attention_dp_size=1, moe_tp_size=1, moe_ep_size=16),
    "TP8xPP2": dict(tp_size=8, pp_size=2, attention_dp_size=1, moe_tp_size=8, moe_ep_size=1),
    # With attention TP8 × DP2, the installed Kimi-K3 database represents the
    # recipe's expert-parallel global group as MoE TP1 × EP16. The intuitive
    # TP8 × EP2 tuple has no profile row in vLLM 0.24.0.
    "TP8xDP2": dict(tp_size=8, pp_size=1, attention_dp_size=2, moe_tp_size=1, moe_ep_size=16),
}
FIELDS = [
    "case",
    "suite",
    "status",
    "eligible",
    "database_mode",
    "backend_name",
    "backend_version",
    "mode",
    "tp_size",
    "pp_size",
    "attention_dp_size",
    "moe_tp_size",
    "moe_ep_size",
    "ctx_tokens",
    "ttft_ms",
    "tpot_ms",
    "request_latency_ms",
    "tokens_s",
    "memory_gb",
    "gpus",
    "concurrency",
    "kv_cache_warning",
    "error",
]


def load_ground_truth(repo):
    """Load the report's four 16-GPU artifact sweeps."""
    root = Path(repo) / "bench-results"
    ground_truth = {}
    for strategy, dirname in REPORT_DIRS.items():
        directory = root / dirname
        if not directory.is_dir():
            raise FileNotFoundError(f"Missing ground-truth artifact directory: {directory}")
        for path in directory.glob("conc-*.json"):
            try:
                concurrency = int(path.stem.split("-", 1)[1])
            except ValueError:
                continue
            data = json.loads(path.read_text())
            ground_truth[(strategy, concurrency)] = {
                "output_tok_s": data["output_throughput"],
                "ttft_ms": data["mean_ttft_ms"],
                "tpot_ms": data["mean_tpot_ms"],
                "itl_ms": data["mean_itl_ms"],
                "e2el_ms": data["mean_e2el_ms"],
                "artifact": (
                    "https://github.com/mnmehta/kimi-k3/blob/main/bench-results/"
                    f"{dirname}/{path.name}"
                ),
            }
    return ground_truth


def cases(ground_truth=None):
    """The three comparison points, original 30-point sweep, and 27 refinements."""
    for name, backend, ep in [
        ("baseline", "vllm", 16),
        ("vllm-tp16", "vllm", 1),
        ("sglang-tp16", "sglang", 1),
    ]:
        yield "estimates", name, dict(backend_name=backend, moe_ep_size=ep, moe_tp_size=16 // ep)

    for backend in VERSIONS:
        for tp in [16, 8, 4, 2, 1]:
            for ep in [1, 2, 4, 8, 16]:
                if ep <= tp:
                    yield "parallelism", f"{backend}-tp{tp}-pp{16 // tp}-ep{ep}", dict(
                        backend_name=backend, tp_size=tp, pp_size=16 // tp, moe_tp_size=tp // ep, moe_ep_size=ep
                    )

    for backend, ep in [("vllm", 1), ("vllm", 16), ("sglang", 1)]:
        for ctx in [128, 256, 512, 1000, 2048]:
            yield "refinement", f"{backend}-ep{ep}-ctx{ctx}", dict(
                backend_name=backend, moe_tp_size=16 // ep, moe_ep_size=ep, ctx_tokens=ctx
            )
    for backend in VERSIONS:
        for ep in [1, 2, 4, 8, 16]:
            yield "refinement", f"{backend}-ep{ep}-hybrid", dict(
                backend_name=backend, moe_tp_size=16 // ep, moe_ep_size=ep, database_mode="HYBRID"
            )
        yield "refinement", f"{backend}-disagg-8-plus-8", dict(
            backend_name=backend,
            mode="disagg",
            tp_size=8,
            moe_tp_size=8,
            moe_ep_size=1,
            prefill_batch_size=1,
            decode_batch_size=1,
            prefill_num_workers=1,
            decode_num_workers=1,
        )

    if ground_truth is not None:
        for strategy, config in REPORT_CONFIGS.items():
            for concurrency in sorted(c for s, c in ground_truth if s == strategy):
                # AIC reports global concurrency as batch_size * PP * attention-DP.
                # For PP2/DP2, odd target concurrency values cannot be represented
                # by an integer per-worker batch; the recorded effective value is
                # retained in the comparison report.
                scale = config["pp_size"] * config["attention_dp_size"]
                batch_size = max(1, math.ceil(concurrency / scale))
                yield "report", f"{strategy}-c{concurrency}", {
                    "backend_name": "vllm",
                    "backend_version": VERSIONS["vllm"],
                    "tp_size": config["tp_size"],
                    "pp_size": config["pp_size"],
                    "attention_dp_size": config["attention_dp_size"],
                    "moe_tp_size": config["moe_tp_size"],
                    "moe_ep_size": config["moe_ep_size"],
                    "batch_size": batch_size,
                    "ground_truth_strategy": strategy,
                    "ground_truth_concurrency": concurrency,
                }


def make_calls(args, ground_truth=None):
    versions = {"vllm": args.vllm_version, "sglang": args.sglang_version}
    for suite, name, overrides in cases(ground_truth):
        if args.suite not in ("all", suite):
            continue
        config = dict(
            **COMMON,
            mode="agg",
            tp_size=16,
            pp_size=1,
            attention_dp_size=1,
            ctx_tokens=1000,
            database_mode="SILICON",
        )
        metadata = {key: overrides.pop(key) for key in list(overrides) if key.startswith("ground_truth_")}
        config.update(overrides)
        config["backend_version"] = versions[config["backend_name"]]
        yield dict(case=f"{suite}/{name}", suite=suite, config=config, **metadata)


def cli_command(config):
    # Use the entry point next to the interpreter used to run this script.
    words = [str(Path(sys.executable).with_name("aiconfigurator")), "cli", "estimate"]
    for key, value in config.items():
        words.extend(["--" + CLI_NAMES.get(key, key.replace("_", "-")), str(value)])
    words.extend(["--detail", "source"])
    return shlex.join(words)


def estimate(call, cli_estimate):
    row = dict(call, status="error", eligible=False)
    start = time.monotonic()
    try:
        result = cli_estimate(**call["config"])
        row.update(
            status="ok",
            backend_version=result.backend_version,
            ttft_ms=result.ttft,
            tpot_ms=result.tpot,
            request_latency_ms=result.request_latency,
            tokens_s=result.tokens_per_second,
            memory_gb=result.memory,
            gpus=result.num_total_gpus,
            concurrency=result.concurrency,
            kv_cache_warning=result.kv_cache_warning,
            raw=result.raw,
            per_ops_data=result.per_ops_data,
            per_ops_source=result.per_ops_source,
            moe_comm_fallbacks=[str(item) for item in result.moe_comm_fallbacks],
        )
        row["eligible"] = (
            result.num_total_gpus == 16
            and result.concurrency == 1
            and not result.kv_cache_warning
            and math.isfinite(result.request_latency)
            and result.request_latency > 0
        )
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
        logging.exception("Estimate failed: %s", call["case"])
    row["elapsed_s"] = time.monotonic() - start
    return row


def save_summary(output, rows):
    with (output / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({**row["config"], **row})
    eligible = sorted((row for row in rows if row["eligible"]), key=lambda row: row["request_latency_ms"])
    best = eligible[0] if eligible else None
    (output / "best.json").write_text(json.dumps(best, indent=2, default=str) + "\n")
    if best:
        print(f"\nBest evaluable result: {best['case']} ({best['config']['database_mode']})")
        print(
            f"Latency {best['request_latency_ms'] / 1000:.3f} s; TTFT {best['ttft_ms']:.3f} ms; "
            f"TPOT {best['tpot_ms']:.3f} ms; memory {best['memory_gb']:.2f} GB/GPU"
        )
        print(cli_command(best["config"]))
    else:
        print("\nNo result met the 16-GPU, concurrency-1 constraints. See results.jsonl and run.log.")
    errors = sum(row["status"] == "error" for row in rows)
    excluded = len(rows) - len(eligible) - errors
    print(f"{len(rows)} calls: {len(eligible)} eligible, {errors} errors, {excluded} excluded.")
    print(f"Results: {output}")
    return 0 if best else 1


def save_report(output, rows, ground_truth, repo):
    """Write a Quarto AIC-versus-measured report for the four 16-GPU runs."""
    report_rows = []
    for row in rows:
        strategy = row.get("ground_truth_strategy")
        concurrency = row.get("ground_truth_concurrency")
        truth = ground_truth[(strategy, concurrency)]
        report_rows.append(
            {
                "strategy": strategy,
                "target_c": concurrency,
                "aic_c": row.get("concurrency"),
                "aic_batch": row["config"].get("batch_size"),
                "gt_ttft": truth["ttft_ms"],
                "aic_ttft": row.get("ttft_ms"),
                "gt_tpot": truth["tpot_ms"],
                "aic_tpot": row.get("tpot_ms"),
                "gt_itl": truth["itl_ms"],
                "gt_tok_s": truth["output_tok_s"],
                "aic_tok_s": row.get("tokens_s"),
                "artifact": truth["artifact"],
                "status": row["status"],
                "error": row.get("error"),
            }
        )
    (output / "comparison.json").write_text(json.dumps(report_rows, indent=2, default=str) + "\n")

    qmd = f'''---
title: "Kimi-K3 AIConfigurator estimates versus H200 ground truth"
subtitle: "16-GPU strategy comparison at 1000 input / 1000 output tokens"
date: "{datetime.now(UTC).date().isoformat()}"
format:
  html:
    theme: darkly
    page-layout: full
    grid:
      sidebar-width: 280px
      body-width: 1800px
      margin-width: 220px
      gutter-width: 1.5rem
    toc: true
    toc-depth: 2
    code-fold: true
    code-tools: false
    embed-resources: true
    html-math-method: plain
    fig-width: 14
    fig-height: 5
execute:
  echo: false
  warning: false
jupyter: kimi-k3-reports
---

This report compares AIConfigurator estimates with the published vLLM-bench artifacts for the four 16-GPU strategies. The 32-GPU P/D strategy is intentionally omitted. Ground truth is the artifact's `mean_ttft_ms`, `mean_tpot_ms`, `mean_itl_ms`, and `output_throughput`; AIC uses the matching 1000-token/1000-token `agg` estimate.

Ground-truth repository: `{Path(repo).resolve()}`. AIC backend: vLLM 0.24.0; system: H200 SXM; model: `moonshotai/Kimi-K3`.

For PP2 and DP2, AIC's effective concurrency is `batch_size × PP × attention-DP`; odd target concurrencies therefore use the next representable batch and are shown explicitly.

```{{python}}
#| label: setup
import json
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from IPython.display import HTML, display

pio.templates.default = "plotly_dark"
pio.renderers.default = "notebook"
records = json.loads(Path("comparison.json").read_text())
df = pd.DataFrame(records)
ok = df[df["status"] == "ok"].copy()
for col in ["target_c", "aic_c", "gt_ttft", "aic_ttft", "gt_tpot", "aic_tpot", "gt_itl", "gt_tok_s", "aic_tok_s"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
ok = df[df["status"] == "ok"].copy()
df["c_match"] = df["target_c"] == df["aic_c"]
df["ttft_error"] = (df["aic_ttft"] / df["gt_ttft"] - 1) * 100
df["tpot_error"] = (df["aic_tpot"] / df["gt_tpot"] - 1) * 100
df["throughput_error"] = (df["aic_tok_s"] / df["gt_tok_s"] - 1) * 100
ok["c_match"] = ok["target_c"] == ok["aic_c"]
ok["ttft_error"] = (ok["aic_ttft"] / ok["gt_ttft"] - 1) * 100
ok["tpot_error"] = (ok["aic_tpot"] / ok["gt_tpot"] - 1) * 100
ok["throughput_error"] = (ok["aic_tok_s"] / ok["gt_tok_s"] - 1) * 100
strategy_order = ["TP16", "TEP16", "TP8xPP2", "TP8xDP2"]
strategy_colors = dict(zip(strategy_order, ["#1f77b4", "#17becf", "#ff7f0e", "#2ca02c"]))
```

## Comparison chart

The chart shows signed AIC error relative to the measured value. Positive values mean AIC is higher than ground truth; negative values mean it is lower. (Values are clamped at ±100%; the shaded band marks ±20%.)

```{{python}}
#| label: comparison-chart
fig = make_subplots(
    rows=len(strategy_order),
    cols=3,
    subplot_titles=[
        f"{{strategy}} · {{title}}"
        for strategy in strategy_order
        for title in ("Output throughput Δ (%)", "TTFT Δ (%)", "TPOT Δ (%)")
    ],
    shared_xaxes=False,
)
metrics = [("throughput_error", "Output throughput Δ (%)"), ("ttft_error", "TTFT Δ (%)"), ("tpot_error", "TPOT Δ (%)")]
error_limits = {{
    error_col: max(1.0, float(ok[error_col].abs().max()) * 1.05)
    for error_col, _ in metrics
}}
for row_index, strategy in enumerate(strategy_order, 1):
    subset = ok[ok["strategy"] == strategy].sort_values("target_c")
    for col, (error_col, title) in enumerate(metrics, 1):
        plotted_error = subset[error_col].clip(-100, 100)
        fig.add_shape(
            type="rect",
            x0=1,
            x1=max(1, float(subset["target_c"].max())),
            y0=-20,
            y1=20,
            fillcolor="rgba(40, 167, 69, 0.28)",
            line_color="rgba(40, 167, 69, 0.85)",
            line_width=1,
            layer="below",
            row=row_index,
            col=col,
        )
        fig.add_trace(go.Scatter(x=subset["target_c"], y=plotted_error, mode="lines+markers",
            name=f"{{strategy}} · AIC error", legendgroup=strategy, showlegend=(col == 1),
            line=dict(color=strategy_colors[strategy]), marker=dict(size=6),
            customdata=subset[[error_col]],
            hovertemplate="C=%{{x}}<br>Δ=%{{customdata[0]:.1f}}%<extra>" + f"{{strategy}} · AIC error</extra>"),
            row=row_index, col=col)
        fig.add_hline(y=0, line_color="#888888", line_width=1, row=row_index, col=col)
        fig.update_xaxes(type="log", dtick=1, title_text="Concurrency (C)", row=row_index, col=col)
        fig.update_yaxes(
            range=[-100, 100],
            title_text=title,
            ticksuffix="%",
            row=row_index,
            col=col,
        )
fig.update_layout(
    height=1250,
    margin=dict(t=90, b=110),
    legend=dict(orientation="h", y=-0.04),
    hovermode="x unified",
)
fig.show()
```

## Detailed comparison

```{{python}}
#| label: detail-table
detail = df[["strategy", "target_c", "aic_c", "c_match", "gt_ttft", "aic_ttft", "ttft_error", "gt_tpot", "aic_tpot", "tpot_error", "gt_itl", "gt_tok_s", "aic_tok_s", "throughput_error", "status"]].rename(columns={{
    "target_c": "Target C", "aic_c": "AIC C", "c_match": "C match", "gt_ttft": "TTFT GT (ms)", "aic_ttft": "TTFT AIC (ms)", "ttft_error": "TTFT error (%)", "gt_tpot": "TPOT GT (ms)", "aic_tpot": "TPOT AIC (ms)", "tpot_error": "TPOT error (%)", "gt_itl": "ITL GT (ms)", "gt_tok_s": "Output GT (tok/s)", "aic_tok_s": "Output AIC (tok/s)", "throughput_error": "Throughput error (%)"
}})
numeric_columns = detail.select_dtypes(include="number").columns
detail[numeric_columns] = detail[numeric_columns].round(1)

percentage_columns = ["TTFT error (%)", "TPOT error (%)", "Throughput error (%)"]

def error_cell_style(value):
    if pd.isna(value):
        return ""
    absolute_error = abs(float(value))
    if absolute_error <= 20:
        color = "#d4edda"
    elif absolute_error <= 100:
        color = "#fff3cd"
    else:
        color = "#f8d7da"
    return f"background-color: {color};"

styled_detail = detail.style.map(error_cell_style, subset=percentage_columns)
display(HTML(styled_detail.to_html(index=False, na_rep="—", classes="table table-striped table-hover", border=0)))
```

## Mean absolute percentage error

```{{python}}
#| label: mape-table
mape = ok.groupby("strategy", sort=False).agg(
    TTFT=("ttft_error", lambda s: s.abs().mean()),
    TPOT=("tpot_error", lambda s: s.abs().mean()),
    **{{"Output throughput": ("throughput_error", lambda s: s.abs().mean()), "Points compared": ("strategy", "size")}}
).reindex(strategy_order).reset_index()
for col in ["TTFT", "TPOT", "Output throughput"]:
    mape[col] = mape[col].map(lambda value: f"{{value:.1f}}%" if pd.notna(value) else "—")
display(HTML(mape.to_html(index=False, classes="table table-striped table-hover", border=0)))
```

## Interpretation

AIC's single-point `agg` model represents steady-state scheduler behavior differently from the benchmark's finite burst sweep. TTFT is therefore most comparable at low concurrency; at high concurrency, benchmark queueing and KV-cache limits can dominate. TPOT/ITL is the cleaner decode comparison, while ITL is reported as the ground-truth companion to TPOT.

The complete machine-readable comparison is in `comparison.json`; raw AIC rows and per-operation source tags are in `results.jsonl`.
'''
    (output / "report.qmd").write_text(qmd)
    print(f"Report: {output / 'report.qmd'}")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", choices=["all", "estimates", "parallelism", "refinement", "report"], default="all")
    parser.add_argument("--ground-truth-repo", type=Path, help="Cloned mnmehta/kimi-k3 repository for --suite report.")
    parser.add_argument(
        "--output-dir", type=Path, help="New or empty directory; defaults to results/kimi-k3-<timestamp>."
    )
    parser.add_argument("--vllm-version", default=VERSIONS["vllm"])
    parser.add_argument("--sglang-version", default=VERSIONS["sglang"])
    parser.add_argument("--dry-run", action="store_true", help="Print equivalent CLI calls without importing the SDK.")
    args = parser.parse_args()
    ground_truth = None
    if args.suite == "report":
        if args.ground_truth_repo is None:
            parser.error("--suite report requires --ground-truth-repo")
        ground_truth = load_ground_truth(args.ground_truth_repo)
    calls = list(make_calls(args, ground_truth))
    if args.dry_run:
        for call in calls:
            print(f"# {call['case']}\n{cli_command(call['config'])}")
        return 0

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    output = (args.output_dir or Path(__file__).resolve().parents[1] / "results" / f"kimi-k3-{timestamp}").resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error(f"Output directory must be new or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output / "matplotlib"))
    logging.basicConfig(filename=output / "run.log", level=logging.WARNING, force=True)

    from aiconfigurator.cli import api

    metadata = dict(
        started_utc=timestamp,
        python=sys.executable,
        api_file=api.__file__,
        suite=args.suite,
        packages={name: importlib.metadata.version(name) for name in ["aiconfigurator", "aiconfigurator-core"]},
        target=dict(gpus=16, concurrency=1, **COMMON),
        calls=calls,
    )
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (output / "commands.txt").write_text(
        "# Equivalent CLI calls, using the currently selected Python environment.\n"
        f"export MPLCONFIGDIR={shlex.quote(os.environ['MPLCONFIGDIR'])}\n\n"
        + "\n\n".join(f"# {call['case']}\n{cli_command(call['config'])}" for call in calls)
        + "\n"
    )
    rows = []
    print(f"Running {len(calls)} calls using {api.__file__}\nResults: {output}", flush=True)
    with (output / "results.jsonl").open("w") as handle:
        for index, call in enumerate(calls, 1):
            row = estimate(call, api.cli_estimate)
            rows.append(row)
            handle.write(json.dumps(row, default=str) + "\n")
            handle.flush()
            if row["status"] == "error":
                detail = row["error"]
            else:
                detail = (
                    f"{row['request_latency_ms'] / 1000:.3f} s, concurrency={row['concurrency']}, "
                    f"GPUs={row['gpus']}, eligible={row['eligible']}"
                )
            print(f"[{index}/{len(calls)}] {call['case']}: {detail}", flush=True)
    if args.suite == "report":
        return save_report(output, rows, ground_truth, args.ground_truth_repo)
    return save_summary(output, rows)


if __name__ == "__main__":
    raise SystemExit(main())
