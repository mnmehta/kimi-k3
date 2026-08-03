#!/usr/bin/env bash
# Wait for DP2 /health, run C=1..512 sweep, copy results locally.
set -euo pipefail
NS="${NS:-kimi-k3}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) waiting for DP2 /health"
for n in $(seq 1 360); do
  if kubectl -n "$NS" exec vllm-recipe-0 -- curl -sf -m 2 http://127.0.0.1:8000/health >/dev/null 2>&1; then
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) healthy poll=$n"
    kubectl -n "$NS" exec vllm-recipe-0 -- curl -sS http://127.0.0.1:8000/v1/models || true
    kubectl -n "$NS" exec vllm-recipe-0 -- bash -c \
      'grep -E "Model loading took|Available KV|GPU KV cache|Maximum concurrency|Skipping model_runner|Application startup|kv_cache_memory" /tmp/vllm-recipe.log | tail -25' || true
    break
  fi
  if (( n % 6 == 0 )); then
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) still loading poll=$n"
    kubectl -n "$NS" exec vllm-recipe-0 -- bash -c \
      'grep -E "Model loading took|Available KV|Skipping model_runner|Application startup|CUDA out of memory|Engine core initialization|ValueError|patched" /tmp/vllm-recipe.log | tail -8' \
      2>/dev/null || true
  fi
  if kubectl -n "$NS" exec vllm-recipe-0 -- grep -qE 'CUDA out of memory|Engine core initialization failed|ValueError: To serve at least one request' /tmp/vllm-recipe.log 2>/dev/null; then
    if ! kubectl -n "$NS" exec vllm-recipe-0 -- pgrep -f 'vllm serve' >/dev/null 2>&1; then
      echo "serve died; fatal error in log" >&2
      kubectl -n "$NS" exec vllm-recipe-0 -- grep -E 'Model loading took|CUDA out of memory|Skipping|ValueError|RuntimeError' /tmp/vllm-recipe.log | tail -25 >&2
      exit 1
    fi
  fi
  sleep 10
done
if ! kubectl -n "$NS" exec vllm-recipe-0 -- curl -sf -m 2 http://127.0.0.1:8000/health >/dev/null 2>&1; then
  echo "health timeout" >&2
  kubectl -n "$NS" exec vllm-recipe-0 -- tail -100 /tmp/vllm-recipe.log >&2 || true
  exit 1
fi

echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) launching DP2 sweep C=1..512"
kubectl -n "$NS" cp "$ROOT/scripts/bench-concurrency-sweep.sh" vllm-recipe-0:/tmp/bench-concurrency-sweep.sh
kubectl -n "$NS" cp "$ROOT/scripts/start-sweep.sh" vllm-recipe-0:/tmp/start-sweep.sh
kubectl -n "$NS" cp "$ROOT/scripts/start-dp2-sweep.sh" vllm-recipe-0:/tmp/start-dp2-sweep.sh
kubectl -n "$NS" exec vllm-recipe-0 -- bash /tmp/start-dp2-sweep.sh

echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) monitoring sweep"
RESULT=/tmp/vllm-bench-sweep-dp2-512 INTERVAL_S=60 "$ROOT/scripts/monitor-dp2-sweep.sh"

echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) copying results"
RESULT=/tmp/vllm-bench-sweep-dp2-512 LOCAL_OUT=bench-results/conc-sweep-dp2-1000-1000 \
  "$ROOT/scripts/copy-sweep-results.sh"
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) sweep artifacts ready"
ls -la "$ROOT/bench-results/conc-sweep-dp2-1000-1000" | head -30
