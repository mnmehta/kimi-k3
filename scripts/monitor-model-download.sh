#!/usr/bin/env bash
# Poll per-rank model download progress on vllm-recipe pods.
set -euo pipefail

export KUBECONFIG="${KUBECONFIG:-/Users/mimehta/kubeconfigs/kubeconfig.fozzie}"
NS="${NS:-kimi-k3}"
NNODES="${NNODES:-2}"
INTERVAL_S="${INTERVAL_S:-60}"

while true; do
  echo "==== $(date -u +%Y-%m-%dT%H:%M:%SZ) ===="
  done_n=0
  for i in $(seq 0 $((NNODES - 1))); do
    if kubectl -n "$NS" exec "vllm-recipe-$i" -- test -f /models/Kimi-K3/.download-complete 2>/dev/null; then
      echo "rank-$i COMPLETE"
      kubectl -n "$NS" exec "vllm-recipe-$i" -- bash -c \
        'cat /models/Kimi-K3/.download-complete; du -sh /models/Kimi-K3 2>/dev/null; ls /models/Kimi-K3/*.safetensors 2>/dev/null | wc -l'
      done_n=$((done_n + 1))
    else
      echo "rank-$i in progress"
      kubectl -n "$NS" exec "vllm-recipe-$i" -- bash -c \
        'df -h /models 2>/dev/null | tail -1; \
         du -sh /models/Kimi-K3 /models/hf 2>/dev/null || true; \
         ls /models/Kimi-K3/*.safetensors 2>/dev/null | wc -l; \
         tail -n 5 /tmp/model-download.log 2>/dev/null || echo "(no log)"' || echo "  (pod not ready)"
    fi
    echo
  done
  if [[ "$done_n" -eq "$NNODES" ]]; then
    echo "All ranks complete."
    exit 0
  fi
  sleep "$INTERVAL_S"
done
