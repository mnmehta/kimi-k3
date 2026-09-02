#!/usr/bin/env bash
# Run AgentX MVP from the in-cluster aiperf-agentx pod against an SGLang endpoint.
#
# Mirrors run-agentx-mvp-inpod-c1234.sh but targets SGLang pods/ports.
#
# Prerequisites:
#   - SGLang recipe pods Ready (deploy-sglang-pp2.sh or deploy-sglang-tp16-ep16.sh)
#   - kubectl apply -f manifests/aiperf-agentx.yaml
#
# Usage:
#   KUBECONFIG=../kubeconfig-kimi-k3.yaml \
#   CONCURRENCIES="1 2" \
#   LOCAL_OUT_PREFIX=bench-results/sglang/agentx-mvp-pp2 \
#   ./scripts/run-agentx-mvp-sglang-inpod.sh
#
# Optional env:
#   URL=sglang-recipe-0.sglang-recipe:30000
#   CONCURRENCIES="1 2 3 4"
#   RESTART_BETWEEN=0       # SGLang restart not automated; default off
#   AIPERF_VERSION=0.12.0.dev20260803
#   LOCAL_OUT_PREFIX=bench-results/sglang/agentx-mvp-pp2
#   STS_NAME=sglang-recipe  # StatefulSet/pod name prefix
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NS="${NS:-kimi-k3}"
POD="${POD:-aiperf-agentx}"
STS_NAME="${STS_NAME:-sglang-recipe}"
PORT="${PORT:-30000}"
URL="${URL:-${STS_NAME}-0.${STS_NAME}:${PORT}}"
MODEL="${MODEL:-/models/Kimi-K3}"
TOKENIZER="${TOKENIZER:-moonshotai/Kimi-K3}"
CONCURRENCIES="${CONCURRENCIES:-1 2 3 4}"
BENCHMARK_DURATION="${BENCHMARK_DURATION:-900}"
RANDOM_SEED="${RANDOM_SEED:-20260817}"
MAX_CONTEXT_LENGTH="${MAX_CONTEXT_LENGTH:-262144}"
PUBLIC_DATASET="${PUBLIC_DATASET:-semianalysis_cc_traces_weka_with_subagents_256k}"
AIPERF_VERSION="${AIPERF_VERSION:-0.12.0.dev20260803}"
LOCAL_OUT_PREFIX="${LOCAL_OUT_PREFIX:-$ROOT/bench-results/sglang/agentx-mvp-pp2}"
NNODES="${NNODES:-2}"
RESTART_BETWEEN="${RESTART_BETWEEN:-0}"
SKIP_FIRST_RESTART="${SKIP_FIRST_RESTART:-1}"
MASTER_LOG="${MASTER_LOG:-${LOCAL_OUT_PREFIX}.master.log}"
_FIRST_C=1

mkdir -p "$(dirname "$MASTER_LOG")"
: >"$MASTER_LOG"

log() { echo "$*" | tee -a "$MASTER_LOG"; }

ensure_pod() {
  if ! kubectl -n "$NS" get pod "$POD" >/dev/null 2>&1; then
    kubectl apply -f "$ROOT/manifests/aiperf-agentx.yaml"
  fi
  kubectl -n "$NS" wait --for=condition=Ready "pod/$POD" --timeout=10m
}

ensure_aiperf() {
  kubectl -n "$NS" exec "$POD" -- bash -lc "
    set -euo pipefail
    if ! python3 -c 'import aiperf,sys; print(aiperf.__version__)' 2>/dev/null | grep -q '$AIPERF_VERSION'; then
      echo '==> installing aiperf==$AIPERF_VERSION'
      python3 -m pip install -q --upgrade 'pip' 'setuptools' 'wheel'
      python3 -m pip install -q 'aiperf==$AIPERF_VERSION'
    fi
    python3 -c 'import aiperf; print(\"aiperf\", aiperf.__version__)'
  "
}

ensure_endpoint() {
  log "==> Checking SGLang health at $URL"
  kubectl -n "$NS" exec "$POD" -- bash -lc "
    set -euo pipefail
    curl -sf --max-time 10 'http://$URL/health' >/dev/null && echo 'HEALTHY'
    curl -sf --max-time 10 'http://$URL/v1/models' | head -c 200
    echo
  "
}

archive_sglang_logs() {
  local local_out="$1"
  mkdir -p "$local_out"
  log "==> Archiving SGLang serve logs into $local_out"
  for i in $(seq 0 $((NNODES - 1))); do
    local out="$local_out/sglang-recipe-$i.log"
    if kubectl -n "$NS" exec "${STS_NAME}-$i" -- test -f /tmp/sglang-recipe.log 2>/dev/null; then
      kubectl -n "$NS" exec "${STS_NAME}-$i" -- cat /tmp/sglang-recipe.log >"$out"
      log "    rank-$i -> $out ($(wc -c <"$out") bytes)"
    else
      log "    rank-$i: /tmp/sglang-recipe.log missing"
    fi
  done
}

copy_results() {
  local c="$1"
  local remote="/results/sglang-agentx-c${c}"
  local local_out="${LOCAL_OUT_PREFIX}-c${c}"
  mkdir -p "$local_out"
  kubectl -n "$NS" exec "$POD" -- tar -C "$remote" -czf - . \
    | tar -C "$local_out" -xzf -
  archive_sglang_logs "$local_out"
  log "==> results -> $local_out"
  ls -la "$local_out" | tee -a "$MASTER_LOG"
}

run_one() {
  local c="$1"
  local remote="/results/sglang-agentx-c${c}"

  ensure_endpoint
  log "===== START C=$c $(date -u +%Y-%m-%dT%H:%M:%SZ) ====="

  kubectl -n "$NS" exec "$POD" -- bash -lc "rm -rf '$remote' && mkdir -p '$remote'"
  set +e
  kubectl -n "$NS" exec "$POD" -- bash -lc "
    set -euo pipefail
    export AIPERF_DATASET_CONFIGURATION_TIMEOUT=1800
    export AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT=1800
    aiperf profile \
      --scenario inferencex-agentx-mvp \
      --url '$URL' \
      --model '$MODEL' \
      --tokenizer '$TOKENIZER' \
      --tokenizer-trust-remote-code \
      --max-context-length '$MAX_CONTEXT_LENGTH' \
      --endpoint-type chat \
      --public-dataset '$PUBLIC_DATASET' \
      --concurrency '$c' \
      --use-server-token-count \
      --streaming \
      --extra-inputs ignore_eos:true \
      --cache-bust first_turn_prefix \
      --system-idle-gap-cap-seconds 10 \
      --trajectory-start-min-ratio 0.0 \
      --trajectory-start-max-ratio 1.0 \
      --benchmark-duration '$BENCHMARK_DURATION' \
      --random-seed '$RANDOM_SEED' \
      --artifact-dir '$remote' \
      --ui simple
  " 2>&1 | tee -a "${LOCAL_OUT_PREFIX}-c${c}.run.log"
  local rc=${PIPESTATUS[0]}
  set -e

  log "===== DONE C=$c exit=$rc $(date -u +%Y-%m-%dT%H:%M:%SZ) ====="

  if [[ $rc -ne 0 ]]; then
    log "ERROR: C=$c failed — archiving crash logs"
    mkdir -p "${LOCAL_OUT_PREFIX}-c${c}-crash"
    archive_sglang_logs "${LOCAL_OUT_PREFIX}-c${c}-crash"
    return "$rc"
  fi

  copy_results "$c"

  # Validate error rate
  python3 - <<PY
import json
from pathlib import Path
p = Path("${LOCAL_OUT_PREFIX}-c${c}") / "profile_export_aiperf.json"
if not p.exists():
    print(f"C=${c} WARNING: no profile_export_aiperf.json found")
else:
    d = json.loads(p.read_text())
    ok = (d.get("request_count") or {}).get("avg") or 0
    err = (d.get("error_request_count") or {}).get("avg") or 0
    rate = (d.get("request_error_rate") or {}).get("avg") or 0
    print(f"C=${c} ok={ok} err={err} error_rate={rate:.2f}%")
    if rate > 10:
        raise SystemExit(f"ERROR: C=${c} error_rate {rate:.1f}% > 10%")
PY
}

{
  log "===== MASTER START $(date -u +%Y-%m-%dT%H:%M:%SZ) ====="
  log "URL=$URL STS_NAME=$STS_NAME NNODES=$NNODES"
  log "CONCURRENCIES=$CONCURRENCIES RESTART_BETWEEN=$RESTART_BETWEEN"
  log "LOCAL_OUT_PREFIX=$LOCAL_OUT_PREFIX"
  ensure_pod
  ensure_aiperf
  for c in $CONCURRENCIES; do
    run_one "$c" || {
      log "===== STOPPED after C=$c failure $(date -u +%Y-%m-%dT%H:%M:%SZ) ====="
      exit 1
    }
  done
  log "===== ALL DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) ====="
} 2>&1 | tee -a "$MASTER_LOG"
