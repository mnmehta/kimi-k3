#!/usr/bin/env bash
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NS=kimi-k3
RESULT=/tmp/vllm-bench-sweep-full-256
LOCAL_OUT="$ROOT/bench-results/conc-sweep-full-1000-1000"
echo "=== full-dummy sweep monitor $(date -u) ==="
for j in $(seq 1 120); do
  sleep 30
  if ! kubectl -n "$NS" exec vllm-recipe-0 -- curl -sf -m 2 http://127.0.0.1:8000/health >/dev/null 2>&1; then
    echo "SERVER_DIED at $(date -u)"
    kubectl -n "$NS" exec vllm-recipe-0 -- tail -40 /tmp/vllm-recipe.log || true
    exit 1
  fi
  kubectl -n "$NS" exec vllm-recipe-0 -- bash -c "
    echo \"--- \$(date -u +%H:%M:%S) ---\"
    ls $RESULT/conc-*.json 2>/dev/null | xargs -n1 basename 2>/dev/null | tr '\n' ' '; echo
    ps -eo etime,cmd | awk '/vllm-rs bench serve/{print \"RUNNING\",\$1,\$NF; f=1} END{if(!f) print \"NO_BENCH\"}'
    grep -E '^== concurrency|seed_latency|sweep complete' $RESULT/nohup.out | tail -5
  " || echo kubectl_fail
  if kubectl -n "$NS" exec vllm-recipe-0 -- grep -q 'sweep complete' "$RESULT/nohup.out" 2>/dev/null; then
    echo SWEEP_DONE
    kubectl -n "$NS" exec vllm-recipe-0 -- cat "$RESULT/summary.tsv"
    RESULT="$RESULT" LOCAL_OUT="$LOCAL_OUT" "$ROOT/scripts/copy-sweep-results.sh"
    exit 0
  fi
done
echo monitor_timeout
exit 1
