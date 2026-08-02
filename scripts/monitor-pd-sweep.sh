#!/usr/bin/env bash
set -euo pipefail
export KUBECONFIG="${KUBECONFIG:-/Users/mimehta/kubeconfigs/kubeconfig.fozzie}"
NS="${NS:-kimi-k3}"
RESULT="${RESULT:-/tmp/vllm-bench-sweep-pd-512}"
INTERVAL_S="${INTERVAL_S:-60}"

while true; do
  echo "==== $(date -u +%Y-%m-%dT%H:%M:%SZ) ===="
  kubectl -n "$NS" exec vllm-recipe-0 -- bash -c "
    if [[ -f $RESULT/sweep.pid ]]; then
      pid=\$(cat $RESULT/sweep.pid)
      if kill -0 \$pid 2>/dev/null; then
        if ps -o stat= -p \$pid 2>/dev/null | grep -q Z; then echo SWEEP_ZOMBIE pid=\$pid
        else echo SWEEP_ALIVE pid=\$pid; fi
      else echo SWEEP_DEAD pid=\$pid; fi
    else
      echo NO_PID_FILE
    fi
    ls -la $RESULT/conc-*.json 2>/dev/null | awk '{print \$NF}' | xargs -n1 basename 2>/dev/null || echo '(no conc json yet)'
    tail -n 6 $RESULT/nohup.out 2>/dev/null || true
    if [[ -f $RESULT/summary.tsv ]]; then echo '--- summary ---'; cat $RESULT/summary.tsv; fi
  " 2>/dev/null || echo "(exec failed)"
  if kubectl -n "$NS" exec vllm-recipe-0 -- test -f "$RESULT/summary.tsv" 2>/dev/null; then
    if kubectl -n "$NS" exec vllm-recipe-0 -- bash -c "
      pid=\$(cat $RESULT/sweep.pid 2>/dev/null || true)
      if [[ -z \$pid ]]; then exit 0; fi
      if ! kill -0 \$pid 2>/dev/null; then exit 0; fi
      if ps -o stat= -p \$pid 2>/dev/null | grep -q Z; then exit 0; fi
      if grep -q 'sweep complete' $RESULT/nohup.out 2>/dev/null; then exit 0; fi
      exit 1
    " 2>/dev/null; then
      echo "Sweep complete."
      exit 0
    fi
  fi
  sleep "$INTERVAL_S"
done
