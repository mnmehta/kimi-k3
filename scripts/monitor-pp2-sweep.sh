#!/usr/bin/env bash
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NS=kimi-k3
RESULT=/tmp/vllm-bench-sweep-pp2-512
LOCAL_OUT="$ROOT/bench-results/conc-sweep-pp2-1000-1000"
echo "=== pp2 sweep monitor $(date -u) ==="
for j in $(seq 1 180); do
  sleep 30
  if ! kubectl -n "$NS" exec vllm-recipe-0 -- curl -sf -m 2 http://127.0.0.1:8000/health >/dev/null 2>&1; then
    echo "SERVER_DIED at $(date -u)"
    kubectl -n "$NS" exec vllm-recipe-0 -- tail -40 /tmp/vllm-recipe.log || true
    exit 1
  fi
  kubectl -n "$NS" exec vllm-recipe-0 -- bash -c "
    echo \"--- \$(date -u +%H:%M:%S) ---\"
    ls $RESULT/conc-*.json 2>/dev/null | xargs -n1 basename 2>/dev/null | tr '\n' ' '; echo
    if [[ -f $RESULT/sweep.pid ]] && kill -0 \$(cat $RESULT/sweep.pid) 2>/dev/null; then
      echo SWEEP_ALIVE pid=\$(cat $RESULT/sweep.pid)
    else
      echo SWEEP_PID_DEAD
    fi
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
