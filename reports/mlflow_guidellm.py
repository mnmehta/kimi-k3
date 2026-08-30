"""Fetch GuideLLM `benchmark_output.json` from MLflow and slim it for Quarto reports.

Credentials (never commit these):

- `MLFLOW_TRACKING_URI` (default: IBM BenchFlow tracking server)
- `MLFLOW_TRACKING_USERNAME` / `MLFLOW_TRACKING_PASSWORD`
- `MLFLOW_WORKSPACE` (default: `benchflow`)
- `MLFLOW_TRACKING_INSECURE_TLS=true` if the server uses a private CA

Local render can skip the env vars when `KUBECONFIG` can read secret
`mlflow-ui-auth` in namespace `benchflow`. GitHub Actions must set the
username/password secrets; Pages itself only serves the rendered HTML.

Slim aggregates are cached under `reports/.mlflow-cache/` (gitignored).
Set `MLFLOW_REFRESH=1` to re-download.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlencode

import pandas as pd
import requests
import urllib3

DEFAULT_TRACKING_URI = "https://mlflow.apps.psap-automation.ibm.rhperfscale.org"
DEFAULT_WORKSPACE = "benchflow"
DEFAULT_ARTIFACT = "results/benchmark_output.json"
DEFAULT_K8S_NAMESPACE = "benchflow"
DEFAULT_K8S_SECRET = "mlflow-ui-auth"

_CACHE_ENV = "MLFLOW_CACHE_DIR"
_REFRESH_ENV = "MLFLOW_REFRESH"


def mlflow_run_url(experiment_id: str, run_id: str, workspace: str | None = None) -> str:
    ws = workspace or os.environ.get("MLFLOW_WORKSPACE") or DEFAULT_WORKSPACE
    uri = _tracking_uri()
    return f"{uri}/#/experiments/{experiment_id}/runs/{run_id}?workspace={ws}"


def load_guidellm_run(
    run_id: str,
    *,
    experiment_id: str,
    label: str,
    artifact: str = DEFAULT_ARTIFACT,
    extra: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any], str]:
    """Download (or cache) a GuideLLM MLflow artifact and return a plot frame."""
    payload = fetch_guidellm_metrics(
        run_id, experiment_id=experiment_id, artifact=artifact
    )
    frame = payload_to_frame(payload, label=label, extra=extra)
    return frame, payload, mlflow_run_url(experiment_id, run_id)


def fetch_guidellm_metrics(
    run_id: str,
    *,
    experiment_id: str | None = None,
    artifact: str = DEFAULT_ARTIFACT,
) -> dict[str, Any]:
    cache_path = _cache_dir() / run_id / "metrics.json"
    if cache_path.exists() and not _refresh():
        return json.loads(cache_path.read_text())

    raw = _download_artifact_json(run_id, artifact)
    payload = slim_guidellm(raw, run_id=run_id, experiment_id=experiment_id, artifact=artifact)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def slim_guidellm(
    data: dict[str, Any],
    *,
    run_id: str,
    experiment_id: str | None = None,
    artifact: str = DEFAULT_ARTIFACT,
) -> dict[str, Any]:
    rows = [_slim_benchmark(b) for b in data.get("benchmarks") or []]
    spec = (data.get("config") or {}).get("spec") or {}
    backend = spec.get("backend") or {}
    constraints = spec.get("constraints") or []
    max_duration = None
    for item in constraints:
        if isinstance(item, dict) and item.get("kind") == "max_duration":
            max_duration = item.get("seconds")
            break
    source = {
        "mlflow_experiment_id": experiment_id,
        "mlflow_run_id": run_id,
        "mlflow_url": mlflow_run_url(experiment_id or "", run_id) if experiment_id else None,
        "artifact": artifact,
        "harness": "guidellm",
        "max_duration_s": max_duration,
        "endpoint": backend.get("request_format"),
        "model": backend.get("model"),
    }
    return {"source": source, "rows": rows}


def payload_to_frame(
    payload: dict[str, Any],
    *,
    label: str,
    extra: dict[str, Any] | None = None,
) -> pd.DataFrame:
    extra = extra or {}
    rows = []
    for d in payload["rows"]:
        row = {
            "run": label,
            "concurrency": int(d["concurrency"]),
            "num_prompts": d["successful_requests"],
            "duration_s": d["duration_s"],
            "completed": d["successful_requests"],
            "failed": d["errored_requests"] or 0,
            "incomplete": d["incomplete_requests"] or 0,
            "request_throughput": d["req_s_total_mean"],
            "output_throughput": d["output_tok_s_total_mean"],
            "output_throughput_successful": d["output_tok_s_successful_mean"],
            "mean_ttft_ms": d["mean_ttft_ms"],
            "p90_ttft_ms": d.get("p95_ttft_ms"),
            "p99_ttft_ms": d["p99_ttft_ms"],
            "mean_tpot_ms": d["mean_tpot_ms"],
            "mean_itl_ms": d["mean_itl_ms"],
            "mean_e2el_ms": d["mean_e2el_ms"],
        }
        row.update(extra)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("concurrency").reset_index(drop=True)


def _slim_benchmark(benchmark: dict[str, Any]) -> dict[str, Any]:
    metrics = benchmark.get("metrics") or {}
    strategy = (benchmark.get("config") or {}).get("strategy") or {}
    totals = metrics.get("request_totals") or {}

    def stat(metric: str, status: str, field: str, default=None):
        block = (metrics.get(metric) or {}).get(status) or {}
        if field in ("p95", "p99"):
            return (block.get("percentiles") or {}).get(field, default)
        return block.get(field, default)

    e2e_s = stat("request_latency", "successful", "mean") or 0.0
    return {
        "concurrency": int(strategy.get("streams") or strategy.get("max_concurrency")),
        "measured_concurrency_total_mean": stat("request_concurrency", "total", "mean"),
        "measured_concurrency_successful_mean": stat("request_concurrency", "successful", "mean"),
        "duration_s": benchmark.get("duration"),
        "successful_requests": totals.get("successful") or 0,
        "errored_requests": totals.get("errored") or 0,
        "incomplete_requests": totals.get("incomplete") or 0,
        "output_tok_s_total_mean": stat("output_tokens_per_second", "total", "mean"),
        "output_tok_s_successful_mean": stat("output_tokens_per_second", "successful", "mean"),
        "req_s_total_mean": stat("requests_per_second", "total", "mean"),
        "req_s_successful_mean": stat("requests_per_second", "successful", "mean"),
        "mean_ttft_ms": stat("time_to_first_token_ms", "successful", "mean"),
        "median_ttft_ms": stat("time_to_first_token_ms", "successful", "median"),
        "p95_ttft_ms": stat("time_to_first_token_ms", "successful", "p95"),
        "p99_ttft_ms": stat("time_to_first_token_ms", "successful", "p99"),
        "mean_tpot_ms": stat("time_per_output_token_ms", "successful", "mean"),
        "median_tpot_ms": stat("time_per_output_token_ms", "successful", "median"),
        "mean_itl_ms": stat("inter_token_latency_ms", "successful", "mean"),
        "median_itl_ms": stat("inter_token_latency_ms", "successful", "median"),
        "mean_e2el_ms": e2e_s * 1000.0,
        "median_e2el_ms": (stat("request_latency", "successful", "median") or 0.0) * 1000.0,
        "total_output_tokens_successful": stat("output_token_count", "successful", "total_sum"),
        "total_output_tokens_total": stat("output_token_count", "total", "total_sum"),
    }


def _cache_dir() -> Path:
    override = os.environ.get(_CACHE_ENV, "").strip()
    if override:
        return Path(override)
    return Path(__file__).resolve().parent / ".mlflow-cache"


def _refresh() -> bool:
    return os.environ.get(_REFRESH_ENV, "").strip().lower() in {"1", "true", "yes"}


def _tracking_uri() -> str:
    return (
        os.environ.get("MLFLOW_TRACKING_URI", "").strip()
        or DEFAULT_TRACKING_URI
    ).rstrip("/")


def _tls_verify() -> bool:
    return os.environ.get("MLFLOW_TRACKING_INSECURE_TLS", "").strip().lower() not in {
        "1",
        "true",
        "yes",
    }


def _ensure_credentials() -> tuple[str, str]:
    username = os.environ.get("MLFLOW_TRACKING_USERNAME", "").strip()
    password = os.environ.get("MLFLOW_TRACKING_PASSWORD", "").strip()
    if username and password:
        return username, password
    loaded = _credentials_from_kube_secret()
    if loaded:
        username, password, extra = loaded
        os.environ.setdefault("MLFLOW_TRACKING_USERNAME", username)
        os.environ.setdefault("MLFLOW_TRACKING_PASSWORD", password)
        for key, value in extra.items():
            os.environ.setdefault(key, value)
        return username, password
    raise RuntimeError(
        "MLflow credentials missing. Set MLFLOW_TRACKING_USERNAME and "
        "MLFLOW_TRACKING_PASSWORD, or export KUBECONFIG so secret "
        f"{DEFAULT_K8S_SECRET} in namespace {DEFAULT_K8S_NAMESPACE} can be read."
    )


def _credentials_from_kube_secret() -> tuple[str, str, dict[str, str]] | None:
    namespace = os.environ.get("MLFLOW_K8S_NAMESPACE", DEFAULT_K8S_NAMESPACE)
    secret = os.environ.get("MLFLOW_K8S_SECRET", DEFAULT_K8S_SECRET)
    try:
        raw = subprocess.check_output(
            ["kubectl", "get", "secret", secret, "-n", namespace, "-o", "json"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    import base64

    data = {
        k: base64.b64decode(v).decode()
        for k, v in (json.loads(raw).get("data") or {}).items()
    }
    username = (data.get("username") or data.get("user") or "").strip()
    password = (data.get("password") or "").strip()
    if not username or not password:
        return None
    extra: dict[str, str] = {}
    uri = (data.get("tracking-uri") or data.get("tracking_uri") or "").strip()
    if uri:
        extra["MLFLOW_TRACKING_URI"] = uri
    workspace = (data.get("workspace") or "").strip()
    if workspace:
        extra["MLFLOW_WORKSPACE"] = workspace
    extra.setdefault("MLFLOW_TRACKING_INSECURE_TLS", "true")
    return username, password, extra


def _headers() -> dict[str, str]:
    workspace = os.environ.get("MLFLOW_WORKSPACE", "").strip() or DEFAULT_WORKSPACE
    # IBM/MLflow 3 workspaces honor X-MLFLOW-WORKSPACE; keep the unprefixed
    # name for older proxies.
    return {
        "X-MLFLOW-WORKSPACE": workspace,
        "MLFLOW-WORKSPACE": workspace,
    }


def _download_artifact_json(run_id: str, artifact: str) -> dict[str, Any]:
    username, password = _ensure_credentials()
    verify = _tls_verify()
    if not verify:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    auth = (username, password)
    uri = _tracking_uri()
    encoded_path = quote(artifact, safe="/")
    candidates = [
        f"{uri}/get-artifact?{urlencode({'path': artifact, 'run_uuid': run_id})}",
        f"{uri}/ajax-api/2.0/mlflow/get-artifact?{urlencode({'run_uuid': run_id, 'path': artifact})}",
        f"{uri}/api/2.0/mlflow-artifacts/artifacts/{run_id}/{encoded_path}",
    ]
    last_error = None
    for url in candidates:
        try:
            with requests.get(
                url,
                auth=auth,
                headers=_headers(),
                verify=verify,
                stream=True,
                timeout=300,
            ) as response:
                if response.status_code == 404:
                    last_error = f"{response.status_code} {url}"
                    continue
                response.raise_for_status()
                with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
                    tmp_path = Path(tmp.name)
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            tmp.write(chunk)
                try:
                    payload = json.loads(tmp_path.read_text())
                finally:
                    tmp_path.unlink(missing_ok=True)
                if not isinstance(payload, dict) or "benchmarks" not in payload:
                    last_error = f"unexpected artifact JSON from {url}"
                    continue
                return payload
        except requests.RequestException as exc:
            last_error = str(exc)
            continue
    raise RuntimeError(
        f"failed to download MLflow artifact {artifact!r} for run {run_id}: {last_error}"
    )
