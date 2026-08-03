#!/usr/bin/env bash
set -euo pipefail
NS="${NS:-kimi-k3}"
RESULT="${RESULT:-/tmp/vllm-bench-sweep-tep-512}"
INTERVAL_S="${INTERVAL_S:-30}"

while true; do
  echo "==== $(date -u +%Y-%m-%dT%H:%M:%SZ) ===="
  kubectl -n "$NS" exec vllm-recipe-0 -- bash -c "
    if [[ -f $RESULT/sweep.pid ]]; then
      pid=\$(cat $RESULT/sweep.pid)
      if kill -0 \$pid 2>/dev/null; then echo SWEEP_ALIVE pid=\$pid; else echo SWEEP_DEAD pid=\$pid; fi
    else
      echo NO_PID_FILE
    fi
    ls -la $RESULT/conc-*.json 2>/dev/null | awk '{print \$NF}' | xargs -n1 basename 2>/dev/null || echo '(no conc json yet)'
    tail -n 8 $RESULT/nohup.out 2>/dev/null || true
    if [[ -f $RESULT/summary.tsv ]]; then echo '--- summary ---'; cat $RESULT/summary.tsv; fi
  " 2>/dev/null || echo "(exec failed)"
  if kubectl -n "$NS" exec vllm-recipe-0 -- test -f "$RESULT/summary.tsv" 2>/dev/null; then
    # Done if pid gone/zombie, or nohup printed "sweep complete" (zombie parents fool kill -0).
    if kubectl -n "$NS" exec vllm-recipe-0 -- bash -c "
      pid=\$(cat $RESULT/sweep.pid 2>/dev/null || true)
      if [[ -z \$pid ]]; then exit 0; fi
      if ! kill -0 \$pid 2>/dev/null; then exit 0; fi
      # Zombie: still kill -0 ok, but state is Z
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
