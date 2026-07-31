#!/usr/bin/env bash
# Host helper: wait for in-pod sweep to finish, then copy results + vLLM logs.
set -euo pipefail

export KUBECONFIG="${KUBECONFIG:-/Users/mimehta/kubeconfigs/kubeconfig.fozzie}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NS="${NS:-kimi-k3}"
RESULT="${RESULT:?RESULT required}"
LOCAL_OUT="${LOCAL_OUT:?LOCAL_OUT required}"
POLL_SECS="${POLL_SECS:-60}"
MAX_POLLS="${MAX_POLLS:-180}"

echo "=== wait-sweep $(date -u) RESULT=$RESULT LOCAL_OUT=$LOCAL_OUT ==="
for n in $(seq 1 "$MAX_POLLS"); do
  sleep "$POLL_SECS"
  st=$(kubectl -n "$NS" exec vllm-recipe-0 -- bash -c "
    if grep -q 'sweep complete' $RESULT/nohup.out 2>/dev/null; then echo DONE; exit 0; fi
    if [[ -f $RESULT/sweep.pid ]] && ! kill -0 \$(cat $RESULT/sweep.pid) 2>/dev/null; then echo DEAD; exit 0; fi
    last=\$(grep -E '^== concurrency' $RESULT/nohup.out 2>/dev/null | tail -1)
    jsons=\$(ls $RESULT/conc-*.json 2>/dev/null | sed 's|.*/||' | tr '\n' ' ')
    echo PROGRESS:\$last :: \$jsons
  " 2>/dev/null || echo KUBECTL_FAIL)
  echo "$(date -u +%H:%M:%S) $st"
  case "$st" in
    DONE*)
      RESULT="$RESULT" LOCAL_OUT="$LOCAL_OUT" "$ROOT/scripts/copy-sweep-results.sh"
      exit 0
      ;;
    DEAD*)
      echo "sweep process died before completion" >&2
      RESULT="$RESULT" LOCAL_OUT="$LOCAL_OUT" "$ROOT/scripts/copy-sweep-results.sh" || true
      exit 1
      ;;
  esac
  if ! kubectl -n "$NS" exec vllm-recipe-0 -- curl -sf -m 2 http://127.0.0.1:8000/health >/dev/null 2>&1; then
    echo "SERVER_DIED" >&2
    exit 1
  fi
done
echo "TIMEOUT waiting for sweep" >&2
exit 1
