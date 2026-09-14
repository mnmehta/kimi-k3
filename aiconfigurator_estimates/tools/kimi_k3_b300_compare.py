#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run manifest-driven B300 AIConfigurator comparisons for Kimi K3.

This tool is intentionally separate from the existing H200 script so we can
iterate on B300-specific manifests without disturbing the validated H200 flow.
The manifest is JSON to avoid adding a YAML dependency during validation.
"""

from __future__ import annotations

import argparse
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


DEFAULT_MANIFEST = {
    "name": "kimi-k3-b300-vllm-blog-3row-comparison",
    "description": (
        "Three-row B300 comparison built from the vLLM Kimi K3 optimization blog's "
        "v0.27.1 throughput and TTFT table."
    ),
    "defaults": {
        "model_path": "moonshotai/Kimi-K3",
        "system_name": "b300_sxm",
        "database_mode": "SILICON",
        "mode": "agg",
    },
    "aic_candidates": [
        {
            "id": "vllm-next-tp8-moe-tp8",
            "label": "vLLM TP8 MoE TP8",
            "backend_family": "vllm",
            "config": {
                "backend_name": "vllm",
                "backend_version": "next",
                "tp_size": 8,
                "pp_size": 1,
                "attention_dp_size": 1,
                "moe_tp_size": 8,
                "moe_ep_size": 1,
            },
        }
    ],
    "points": [
        {
            "id": "blog-vllm-0271-c1",
            "label": "Blog vLLM 0.27.1 C=1",
            "source_kind": "web",
            "evidence_status": "measured",
            "scenario": "recipe-benchmark",
            "hardware": "B300 single node",
            "system_name": "b300_sxm",
            "node_count": 1,
            "gpu_count": 8,
            "backend_family": "vllm",
            "backend_label": "vLLM 0.27.1",
            "runtime_config": "TP8, DSpark 8 speculative tokens, prefix caching disabled, 8k/1k",
            "isl": 8000,
            "osl": 1000,
            "concurrency_target": 1,
            "metrics": {"output_tok_s": 83.3, "ttft_ms": 2262.9},
            "provenance_url": "https://16ca0411.vllm-blog-source.pages.dev/2026/09/13/kimi-k3-performance-optimization",
            "notes": "Blog performance table row for v0.27.1.",
        },
        {
            "id": "blog-vllm-0271-c4",
            "label": "Blog vLLM 0.27.1 C=4",
            "source_kind": "web",
            "evidence_status": "measured",
            "scenario": "recipe-benchmark",
            "hardware": "B300 single node",
            "system_name": "b300_sxm",
            "node_count": 1,
            "gpu_count": 8,
            "backend_family": "vllm",
            "backend_label": "vLLM 0.27.1",
            "runtime_config": "TP8, DSpark 8 speculative tokens, prefix caching disabled, 8k/1k",
            "isl": 8000,
            "osl": 1000,
            "concurrency_target": 4,
            "metrics": {"output_tok_s": 166.7, "ttft_ms": 2314.9},
            "provenance_url": "https://16ca0411.vllm-blog-source.pages.dev/2026/09/13/kimi-k3-performance-optimization",
            "notes": "Blog performance table row for v0.27.1.",
        },
        {
            "id": "blog-vllm-0271-c16",
            "label": "Blog vLLM 0.27.1 C=16",
            "source_kind": "web",
            "evidence_status": "measured",
            "scenario": "recipe-benchmark",
            "hardware": "B300 single node",
            "system_name": "b300_sxm",
            "node_count": 1,
            "gpu_count": 8,
            "backend_family": "vllm",
            "backend_label": "vLLM 0.27.1",
            "runtime_config": "TP8, DSpark 8 speculative tokens, prefix caching disabled, 8k/1k",
            "isl": 8000,
            "osl": 1000,
            "concurrency_target": 16,
            "metrics": {"output_tok_s": 258.3, "ttft_ms": 7601.1},
            "provenance_url": "https://16ca0411.vllm-blog-source.pages.dev/2026/09/13/kimi-k3-performance-optimization",
            "notes": "Blog performance table row for v0.27.1.",
        },
    ],
}


def load_manifest(path: Path | None) -> dict:
    if path is None:
        return json.loads(json.dumps(DEFAULT_MANIFEST))
    return json.loads(path.read_text())


def make_output_dir(path: Path | None) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    return (path or Path(__file__).resolve().parents[1] / "results" / f"b300-kimi-k3-{timestamp}").resolve()


def aic_command(config: dict) -> str:
    cli_names = {"system_name": "system", "backend_name": "backend", "mode": "estimate-mode"}
    words = [str(Path(sys.executable).with_name("aiconfigurator")), "cli", "estimate"]
    for key, value in config.items():
        words.extend(["--" + cli_names.get(key, key.replace("_", "-")), str(value)])
    words.extend(["--detail", "source"])
    return shlex.join(words)


def build_calls(manifest: dict) -> list[dict]:
    defaults = dict(manifest.get("defaults", {}))
    points = manifest.get("points", [])
    candidates = manifest.get("aic_candidates", [])
    calls = []
    for point in points:
        if point.get("evidence_status") != "measured":
            continue
        for candidate in candidates:
            if candidate.get("backend_family") != point.get("backend_family"):
                continue
            config = dict(defaults)
            config.update(candidate["config"])
            config["system_name"] = point.get("system_name", config.get("system_name"))
            config["isl"] = int(point["isl"])
            config["osl"] = int(point["osl"])
            config["ctx_tokens"] = int(point.get("ctx_tokens", point["isl"]))
            scale = int(config.get("pp_size", 1)) * int(config.get("attention_dp_size", 1))
            config["batch_size"] = max(1, math.ceil(int(point["concurrency_target"]) / scale))
            calls.append(
                {
                    "point_id": point["id"],
                    "point_label": point["label"],
                    "candidate_id": candidate["id"],
                    "candidate_label": candidate["label"],
                    "backend_family": point["backend_family"],
                    "config": config,
                }
            )
    return calls


def run_call(call: dict, cli_estimate) -> dict:
    row = dict(call, status="error")
    started = time.monotonic()
    try:
        result = cli_estimate(**call["config"])
        row.update(
            status="ok",
            aic_concurrency=result.concurrency,
            aic_gpus=result.num_total_gpus,
            aic_ttft_ms=result.ttft,
            aic_tpot_ms=result.tpot,
            aic_request_latency_ms=result.request_latency,
            aic_output_tok_s=result.tokens_per_second,
            aic_memory_gb=result.memory,
            aic_kv_cache_warning=result.kv_cache_warning,
            per_ops_source=result.per_ops_source,
            moe_comm_fallbacks=[str(item) for item in result.moe_comm_fallbacks],
        )
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
        logging.exception("Estimate failed for %s vs %s", call["point_id"], call["candidate_id"])
    row["elapsed_s"] = time.monotonic() - started
    return row


def with_errors(point: dict, row: dict) -> dict:
    out = {
        "point_id": point["id"],
        "point_label": point["label"],
        "backend_label": point.get("backend_label"),
        "candidate_id": row["candidate_id"],
        "candidate_label": row["candidate_label"],
        "target_concurrency": point["concurrency_target"],
        "measured_output_tok_s": point["metrics"].get("output_tok_s"),
        "measured_request_throughput": point["metrics"].get("request_throughput"),
        "measured_ttft_ms": point["metrics"].get("ttft_ms"),
        "measured_tpot_ms": point["metrics"].get("tpot_ms"),
        "aic_output_tok_s": row.get("aic_output_tok_s"),
        "aic_request_latency_ms": row.get("aic_request_latency_ms"),
        "aic_ttft_ms": row.get("aic_ttft_ms"),
        "aic_tpot_ms": row.get("aic_tpot_ms"),
        "aic_concurrency": row.get("aic_concurrency"),
        "aic_gpus": row.get("aic_gpus"),
        "status": row["status"],
        "error": row.get("error"),
        "provenance_url": point.get("provenance_url"),
    }

    def pct(aic: float | None, measured: float | None) -> float | None:
        if aic is None or measured in (None, 0):
            return None
        return 100.0 * (aic - measured) / measured

    out["output_tok_s_error_pct"] = pct(out["aic_output_tok_s"], out["measured_output_tok_s"])
    out["ttft_error_pct"] = pct(out["aic_ttft_ms"], out["measured_ttft_ms"])
    out["tpot_error_pct"] = pct(out["aic_tpot_ms"], out["measured_tpot_ms"])
    out["concurrency_match"] = out["aic_concurrency"] == out["target_concurrency"]
    return out


def mean_abs_pct(rows: list[dict], key: str) -> float | None:
    vals = [abs(row[key]) for row in rows if row.get(key) is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def save_report(output_dir: Path, manifest: dict, manifest_label: str, raw_rows: list[dict]) -> int:
    points_by_id = {point["id"]: point for point in manifest["points"]}
    comparison_rows = [with_errors(points_by_id[row["point_id"]], row) for row in raw_rows]
    (output_dir / "comparison.json").write_text(json.dumps(comparison_rows, indent=2) + "\n")

    by_candidate = {}
    for row in comparison_rows:
        by_candidate.setdefault(row["candidate_id"], []).append(row)

    summary_rows = []
    for candidate_id, rows in sorted(by_candidate.items()):
        ok_rows = [row for row in rows if row["status"] == "ok"]
        summary_rows.append(
            {
                "candidate_id": candidate_id,
                "candidate_label": rows[0]["candidate_label"],
                "points_total": len(rows),
                "points_ok": len(ok_rows),
                "throughput_mape_pct": mean_abs_pct(ok_rows, "output_tok_s_error_pct"),
                "ttft_mape_pct": mean_abs_pct(ok_rows, "ttft_error_pct"),
                "tpot_mape_pct": mean_abs_pct(ok_rows, "tpot_error_pct"),
            }
        )
    (output_dir / "summary.json").write_text(json.dumps(summary_rows, indent=2) + "\n")

    measured_has_ttft = any(row.get("measured_ttft_ms") is not None for row in comparison_rows)
    measured_has_tpot = any(row.get("measured_tpot_ms") is not None for row in comparison_rows)
    non_measured = [point for point in manifest["points"] if point.get("evidence_status") != "measured"]
    blog_url = "https://16ca0411.vllm-blog-source.pages.dev/2026/09/13/kimi-k3-performance-optimization"
    repo_root = "https://github.com/mnmehta/kimi-k3"
    script_github = f"{repo_root}/blob/main/aiconfigurator_estimates/tools/kimi_k3_b300_compare.py"
    results_github = f"{repo_root}/tree/main/aiconfigurator_estimates/results"
    commands_github = (
        f"{repo_root}/blob/main/"
        f"{output_dir.relative_to(Path.cwd()).as_posix()}/commands.txt"
    )

    def fmt_pct(value: float | None) -> str:
        return "—" if value is None else f"{value:+.1f}%"

    lines = [
        "# Kimi-K3 B300 comparison",
        "",
        f"This report compares AIC Kimi K3 B300 estimates to data in the blog at [Kimi K3 Performance Optimizations in vLLM, 2.2–2.8× Higher Throughput]({blog_url}).",
        "",
        "## Artifacts",
        "",
        f"- Script: [aiconfigurator_estimates/tools/kimi_k3_b300_compare.py]({script_github})",
        f"- Commands source: [commands.txt]({commands_github})",
        f"- Results root: [aiconfigurator_estimates/results]({results_github})",
        "",
        "## Candidate summary",
        "",
        "MAPE = Mean Absolute Percentage Error.",
        "",
        "| Candidate | Points ok | Throughput MAPE | TTFT MAPE | TPOT MAPE |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['candidate_label']} | {row['points_ok']}/{row['points_total']} | "
            f"{'—' if row['throughput_mape_pct'] is None else f'{row['throughput_mape_pct']:.1f}%'} | "
            f"{'—' if row['ttft_mape_pct'] is None else f'{row['ttft_mape_pct']:.1f}%'} | "
            f"{'—' if row['tpot_mape_pct'] is None else f'{row['tpot_mape_pct']:.1f}%'} |"
        )
    header = [
        "",
        "## Point comparisons",
        "",
        "| Point | Candidate | Target C | AIC C | Measured tok/s | AIC tok/s | Tok/s error |",
    ]
    divider = ["|---|---|---:|---:|---:|---:|---:|"]
    if measured_has_ttft:
        header[3] += " Measured TTFT | AIC TTFT | TTFT error |"
        divider[0] += "---:|---:|---:|"
    if measured_has_tpot:
        header[3] += " Measured TPOT | AIC TPOT | TPOT error |"
        divider[0] += "---:|---:|---:|"
    lines.extend(header + divider)
    for row in comparison_rows:
        if row["status"] != "ok":
            line = (
                f"| {row['point_label']} | {row['candidate_label']} | {row['target_concurrency']} | — | "
                f"{row['measured_output_tok_s']:.1f} | — | — |"
            )
            if measured_has_ttft:
                line += f" {'—' if row['measured_ttft_ms'] is None else f'{row['measured_ttft_ms']:.1f}'} | — | — |"
            if measured_has_tpot:
                line += f" {'—' if row['measured_tpot_ms'] is None else f'{row['measured_tpot_ms']:.1f}'} | — | — |"
            lines.append(line)
            continue
        line = (
            f"| {row['point_label']} | {row['candidate_label']} | {row['target_concurrency']} | {row['aic_concurrency']} | "
            f"{row['measured_output_tok_s']:.1f} | {row['aic_output_tok_s']:.1f} | {fmt_pct(row['output_tok_s_error_pct'])} |"
        )
        if measured_has_ttft:
            line += (
                f" {'—' if row['measured_ttft_ms'] is None else f'{row['measured_ttft_ms']:.1f}'} | "
                f"{row['aic_ttft_ms']:.1f} | {fmt_pct(row['ttft_error_pct'])} |"
            )
        if measured_has_tpot:
            line += (
                f" {'—' if row['measured_tpot_ms'] is None else f'{row['measured_tpot_ms']:.1f}'} | "
                f"{row['aic_tpot_ms']:.1f} | {fmt_pct(row['tpot_error_pct'])} |"
            )
        lines.append(line)
    if non_measured:
        lines.extend(
            [
                "",
                "## Context-only external points",
                "",
                "| Label | Status | Scenario | Notes |",
                "|---|---|---|---|",
            ]
        )
        for point in non_measured:
            lines.append(
                f"| {point['label']} | {point['evidence_status']} | {point.get('scenario', '')} | {point.get('notes', '')} |"
            )

    lines.extend(
        [
            "",
            "## Notes",
            "",
        "- The AIC side uses the local queryable B300 vLLM slot `next`, which resolves to `0.27.0`; the measured rows are the blog's `0.27.1` numbers.",
        "- This manifest assumes `tp_size=8`, `pp_size=1`, `attention_dp_size=1`, `moe_tp_size=8`, `moe_ep_size=1` for every comparison row.",
        "- Only the measured fields present in the manifest are rendered in the point table.",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Optional path to a JSON manifest. Defaults to the built-in three-row B300 blog comparison.",
    )
    parser.add_argument("--output-dir", type=Path, help="Optional new or empty output directory.")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved AIC commands without executing them.")
    parsed = parser.parse_args()

    manifest = load_manifest(parsed.manifest)
    calls = build_calls(manifest)
    output_dir = make_output_dir(parsed.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / "matplotlib"))
    logging.basicConfig(filename=output_dir / "run.log", level=logging.WARNING, force=True)
    manifest_output = output_dir / "manifest.json"
    manifest_output.write_text(json.dumps(manifest, indent=2) + "\n")

    metadata = {
        "started_utc": datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ"),
        "python": sys.executable,
        "manifest": str(manifest_output),
        "manifest_source": "built-in default" if parsed.manifest is None else str(parsed.manifest.resolve()),
        "packages": {
            name: importlib.metadata.version(name) for name in ["aiconfigurator", "aiconfigurator-core"]
        },
        "calls": calls,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (output_dir / "commands.txt").write_text(
        "# Equivalent AIConfigurator commands for this comparison run.\n"
        f"export MPLCONFIGDIR={shlex.quote(os.environ['MPLCONFIGDIR'])}\n\n"
        + "\n\n".join(
            f"# {call['point_id']} vs {call['candidate_id']}\n{aic_command(call['config'])}" for call in calls
        )
        + "\n"
    )

    if parsed.dry_run:
        print((output_dir / "commands.txt").read_text())
        return 0

    from aiconfigurator.cli import api

    raw_rows = []
    with (output_dir / "results.jsonl").open("w") as handle:
        for index, call in enumerate(calls, 1):
            row = run_call(call, api.cli_estimate)
            raw_rows.append(row)
            handle.write(json.dumps(row, default=str) + "\n")
            handle.flush()
            if row["status"] == "ok":
                print(
                    f"[{index}/{len(calls)}] {call['point_id']} vs {call['candidate_id']}: "
                    f"tok/s={row['aic_output_tok_s']:.1f} c={row['aic_concurrency']} gpus={row['aic_gpus']}",
                    flush=True,
                )
            else:
                print(
                    f"[{index}/{len(calls)}] {call['point_id']} vs {call['candidate_id']}: {row['error']}",
                    flush=True,
                )

    manifest_label = str(manifest_output) if parsed.manifest is None else str(parsed.manifest.resolve())
    return save_report(output_dir, manifest, manifest_label, raw_rows)


if __name__ == "__main__":
    raise SystemExit(main())
