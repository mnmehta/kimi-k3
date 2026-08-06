#!/usr/bin/env bash
# Run AgentX MVP from the in-cluster aiperf-agentx pod (no local port-forward).
#
# By default restarts vLLM serve before each concurrency so GPU/CPU KV and
# prefix cache are not inherited from the prior run.
#
# Prerequisites:
#   - recipe pods Ready (first bring-up: ./scripts/deploy.sh <recipe>)
#   - kubectl apply -f manifests/aiperf-agentx.yaml
#
# Optional env:
#   URL=vllm-recipe-0.vllm-recipe:8000
#   CONCURRENCIES="2 3 4 5 6 7 8"
#   RECIPE=pp2-humming-agentx-kv-offload
#   RESTART_BETWEEN=1   # set 0 to skip serve restart between C
#   AIPERF_VERSION=0.12.0.dev20260803
#   LOCAL_OUT_PREFIX=bench-results/agentx-mvp-pp2-humming-kv-offload
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export KUBECONFIG="${KUBECONFIG:-/Users/mimehta/kubeconfigs/kubeconfig.fozzie}"
NS="${NS:-kimi-k3}"
POD="${POD:-aiperf-agentx}"
URL="${URL:-vllm-recipe-0.vllm-recipe:8000}"
MODEL="${MODEL:-/models/Kimi-K3}"
TOKENIZER="${TOKENIZER:-moonshotai/Kimi-K3}"
CONCURRENCIES="${CONCURRENCIES:-1 2 3 4}"
BENCHMARK_DURATION="${BENCHMARK_DURATION:-900}"
RANDOM_SEED="${RANDOM_SEED:-20260804}"
MAX_CONTEXT_LENGTH="${MAX_CONTEXT_LENGTH:-262144}"
PUBLIC_DATASET="${PUBLIC_DATASET:-semianalysis_cc_traces_weka_with_subagents_256k}"
AIPERF_VERSION="${AIPERF_VERSION:-0.12.0.dev20260803}"
LOCAL_OUT_PREFIX="${LOCAL_OUT_PREFIX:-$ROOT/bench-results/agentx-mvp-pp2-humming-kv-offload}"
NNODES="${NNODES:-2}"
RECIPE="${RECIPE:-pp2-humming-agentx-kv-offload}"
RESTART_BETWEEN="${RESTART_BETWEEN:-1}"
# Set 1 when serve was already restarted before launching this script.
SKIP_FIRST_RESTART="${SKIP_FIRST_RESTART:-0}"
MASTER_LOG="${MASTER_LOG:-$ROOT/bench-results/agentx-mvp-kv-offload-c1234.master.log}"
_FIRST_C=1

mkdir -p "$ROOT/bench-results"
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
  kubectl -n "$NS" exec "$POD" -- bash -lc "
    set -euo pipefail
    curl -sf --max-time 5 'http://$URL/v1/models' | head -c 200
    echo
  "
}

restart_serve() {
  log "==> Restarting serve (recipe=$RECIPE) before next concurrency"
  bash "$ROOT/scripts/restart-recipe-serve.sh" "$RECIPE" 2>&1 | tee -a "$MASTER_LOG"
  ensure_endpoint
}

archive_vllm_logs() {
  local local_out="$1"
  mkdir -p "$local_out"
  for i in $(seq 0 $((NNODES - 1))); do
    local out="$local_out/vllm-recipe-$i.log"
    if kubectl -n "$NS" exec "vllm-recipe-$i" -- test -f /tmp/vllm-recipe.log; then
      kubectl -n "$NS" exec "vllm-recipe-$i" -- cat /tmp/vllm-recipe.log >"$out"
      log "    archived $out ($(wc -c <"$out") bytes)"
    else
      log "    rank-$i: /tmp/vllm-recipe.log missing"
    fi
  done
}

copy_results() {
  local c="$1"
  local remote="/results/agentx-c${c}"
  local local_out="${LOCAL_OUT_PREFIX}-c${c}"
  mkdir -p "$local_out"
  kubectl -n "$NS" exec "$POD" -- tar -C "$remote" -czf - . \
    | tar -C "$local_out" -xzf -
  archive_vllm_logs "$local_out"
  log "==> results -> $local_out"
  ls -la "$local_out" | tee -a "$MASTER_LOG"
}

run_one() {
  local c="$1"
  local remote="/results/agentx-c${c}"
  local do_restart=0
  if [[ "$RESTART_BETWEEN" == "1" ]]; then
    if [[ "$_FIRST_C" == "1" && "$SKIP_FIRST_RESTART" == "1" ]]; then
      do_restart=0
    else
      do_restart=1
    fi
  fi
  _FIRST_C=0
  if [[ "$do_restart" == "1" ]]; then
    restart_serve
  else
    ensure_endpoint
  fi
  log "===== START C=$c $(date -u +%Y-%m-%dT%H:%M:%SZ) ====="
  : >"$ROOT/bench-results/agentx-mvp-kv-offload-c${c}.run.log"
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
  " 2>&1 | tee -a "$ROOT/bench-results/agentx-mvp-kv-offload-c${c}.run.log"
  local rc=${PIPESTATUS[0]}
  set -e
  log "===== DONE C=$c exit=$rc $(date -u +%Y-%m-%dT%H:%M:%SZ) ====="
  if [[ $rc -ne 0 ]]; then
    log "ERROR: C=$c failed — archiving crash logs"
    mkdir -p "${LOCAL_OUT_PREFIX}-c${c}-crash"
    archive_vllm_logs "${LOCAL_OUT_PREFIX}-c${c}-crash"
    return "$rc"
  fi
  copy_results "$c"
  python3 - <<PY
import json
from pathlib import Path
p = Path("${LOCAL_OUT_PREFIX}-c${c}") / "profile_export_aiperf.json"
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
  log "CONCURRENCIES=$CONCURRENCIES RESTART_BETWEEN=$RESTART_BETWEEN SKIP_FIRST_RESTART=$SKIP_FIRST_RESTART RECIPE=$RECIPE"
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
