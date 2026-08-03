#!/usr/bin/env bash
# Sequential end-to-end repro sweeps for the four 16-GPU configs (no P/D).
# Writes into NEW bench-results dirs so existing archives are untouched.
#
# Usage:
#   export KUBECONFIG=...
#   ./scripts/run-repro-sweeps.sh
#
# Optional:
#   TAG=repro-20260803          # directory / in-pod RESULT suffix
#   CONFIGS="real tep pp2 dp2"  # subset
#   MAX_POLLS=300               # wait-sweep polls (× POLL_SECS)
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NS="${NS:-kimi-k3}"
TAG="${TAG:-repro-20260803}"
CONFIGS="${CONFIGS:-real tep pp2 dp2}"
POLL_SECS="${POLL_SECS:-60}"
MAX_POLLS="${MAX_POLLS:-300}"  # 5h @ 60s — covers long TP16/TEP tiers
export NNODES=2

HOST_LOG="$ROOT/bench-results/repro-sweep-${TAG}.log"
mkdir -p "$ROOT/bench-results"

echo "=== repro sweeps TAG=$TAG CONFIGS=$CONFIGS $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "host log: $HOST_LOG"
: "${KUBECONFIG:?export KUBECONFIG=... first}"

wait_healthy() {
  local label="$1"
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) waiting for $label /health"
  for n in $(seq 1 360); do
    if kubectl -n "$NS" exec vllm-recipe-0 -- curl -sf -m 2 http://127.0.0.1:8000/health >/dev/null 2>&1; then
      echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $label healthy poll=$n"
      kubectl -n "$NS" exec vllm-recipe-0 -- curl -sS http://127.0.0.1:8000/v1/models || true
      return 0
    fi
    if (( n % 6 == 0 )); then
      echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $label still loading poll=$n"
      kubectl -n "$NS" exec vllm-recipe-0 -- bash -c \
        'grep -E "Model loading|Available KV|Application startup|CUDA out of memory|Engine core|ValueError" /tmp/vllm-recipe.log | tail -8' \
        2>/dev/null || true
    fi
    if kubectl -n "$NS" exec vllm-recipe-0 -- grep -qE \
      'CUDA out of memory|Engine core initialization failed|ValueError: To serve at least one request' \
      /tmp/vllm-recipe.log 2>/dev/null; then
      if ! kubectl -n "$NS" exec vllm-recipe-0 -- pgrep -f 'vllm serve' >/dev/null 2>&1; then
        echo "$label serve died" >&2
        kubectl -n "$NS" exec vllm-recipe-0 -- tail -80 /tmp/vllm-recipe.log >&2 || true
        return 1
      fi
    fi
    sleep 10
  done
  echo "$label health timeout" >&2
  kubectl -n "$NS" exec vllm-recipe-0 -- tail -100 /tmp/vllm-recipe.log >&2 || true
  return 1
}

launch_sweep() {
  local start_wrapper="$1"  # e.g. start-real-sweep.sh
  local result_dir="$2"
  local concs="$3"

  kubectl -n "$NS" cp "$ROOT/scripts/bench-concurrency-sweep.sh" vllm-recipe-0:/tmp/bench-concurrency-sweep.sh
  kubectl -n "$NS" cp "$ROOT/scripts/start-sweep.sh" vllm-recipe-0:/tmp/start-sweep.sh
  kubectl -n "$NS" cp "$ROOT/scripts/$start_wrapper" "vllm-recipe-0:/tmp/$start_wrapper"
  kubectl -n "$NS" exec vllm-recipe-0 -- env \
    RESULT_DIR="$result_dir" \
    CONCURRENCIES="$concs" \
    bash "/tmp/$start_wrapper"
}

run_config() {
  local key="$1"          # real|tep|pp2|dp2
  local deploy="$2"
  local start_wrapper="$3"
  local concs="$4"

  local result="/tmp/vllm-bench-sweep-${key}-${TAG}"
  local local_out="$ROOT/bench-results/conc-sweep-${key}-1000-1000-${TAG}"

  if [[ -f "$local_out/summary.tsv" ]]; then
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) SKIP $key — $local_out/summary.tsv already exists"
    return 0
  fi

  echo ""
  echo "######################################################################"
  echo "# CONFIG=$key  RESULT=$result"
  echo "# LOCAL_OUT=$local_out"
  echo "# CONCURRENCIES=$concs"
  echo "######################################################################"

  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) deploy $key"
  # shellcheck disable=SC2086
  bash $deploy

  wait_healthy "$key"

  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) launch sweep $key"
  launch_sweep "$start_wrapper" "$result" "$concs"

  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) wait+copy $key"
  RESULT="$result" \
    LOCAL_OUT="$local_out" \
    NNODES="$NNODES" \
    POLL_SECS="$POLL_SECS" \
    MAX_POLLS="$MAX_POLLS" \
    "$ROOT/scripts/wait-sweep.sh"

  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) DONE $key"
  ls -la "$local_out" | head -25
}

FULL_C="1 2 4 8 16 32 64 128 256 512"
DP2_C="1 2 4 8 16"  # match archived DP2 (KV-limited; higher C not informative)

for cfg in $CONFIGS; do
  case "$cfg" in
    real)
      run_config real \
        "$ROOT/scripts/deploy-recipe.sh" \
        start-real-sweep.sh \
        "$FULL_C"
      ;;
    tep)
      run_config tep \
        "$ROOT/scripts/deploy-recipe-tep.sh" \
        start-tep-sweep.sh \
        "$FULL_C"
      ;;
    pp2)
      run_config pp2 \
        "$ROOT/scripts/deploy-recipe-pp.sh" \
        start-pp2-sweep.sh \
        "$FULL_C"
      ;;
    dp2)
      run_config dp2 \
        "$ROOT/scripts/deploy-recipe-dp.sh" \
        start-dp2-sweep.sh \
        "$DP2_C"
      ;;
    *)
      echo "unknown CONFIGS entry: $cfg" >&2
      exit 1
      ;;
  esac
done

echo ""
echo "=== all requested repro sweeps finished $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "Dirs under bench-results/*-${TAG}/"
ls -d "$ROOT"/bench-results/conc-sweep-*-1000-1000-"$TAG" 2>/dev/null || true
