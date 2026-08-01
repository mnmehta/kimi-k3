#!/usr/bin/env bash
# Download moonshotai/Kimi-K3 onto each recipe pod's /models mount
# (real hostPath /mnt/local/kimi-k3/models when using manifests/vllm-recipe.yaml).
#
# Usage:
#   export KUBECONFIG=/Users/mimehta/kubeconfigs/kubeconfig.fozzie
#   ./scripts/download-model.sh
#   FOLLOW=0 ./scripts/download-model.sh     # start downloads and return
#   ./scripts/monitor-model-download.sh      # watch progress

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export KUBECONFIG="${KUBECONFIG:-/Users/mimehta/kubeconfigs/kubeconfig.fozzie}"
NS="${NS:-kimi-k3}"
FOLLOW="${FOLLOW:-1}"
NNODES="${NNODES:-2}"

echo "==> Ensuring recipe StatefulSet (hostPath /models)"
kubectl apply -f "$ROOT/manifests/namespace.yaml"
kubectl apply -f "$ROOT/manifests/vllm-recipe.yaml"
kubectl -n "$NS" scale statefulset/vllm-recipe --replicas="$NNODES"
kubectl -n "$NS" rollout status statefulset/vllm-recipe --timeout=30m
for i in $(seq 0 $((NNODES - 1))); do
  kubectl -n "$NS" wait --for=condition=Ready "pod/vllm-recipe-$i" --timeout=30m
done
kubectl -n "$NS" get pods -l app=vllm-recipe -o wide

echo "==> Storage check"
for i in $(seq 0 $((NNODES - 1))); do
  kubectl -n "$NS" exec "vllm-recipe-$i" -- bash -c \
    'df -h /models; findmnt -T /models | head -2 || true'
done

echo "==> Copying in-pod download script to each rank"
for i in $(seq 0 $((NNODES - 1))); do
  kubectl -n "$NS" cp \
    "$ROOT/scripts/download-model-inpod.sh" \
    "vllm-recipe-$i:/tmp/download-model-inpod.sh"
done

echo "==> Starting downloads on each recipe pod (parallel)"
for i in $(seq 0 $((NNODES - 1))); do
  # NOTE: do not `pkill -f download-model-inpod.sh` here — it matches this
  # kubectl exec argv and SIGTERMs the launcher (exit 143).
  kubectl -n "$NS" exec "vllm-recipe-$i" -- bash -c \
    'if [[ -f /tmp/model-download.pid ]] && kill -0 "$(cat /tmp/model-download.pid)" 2>/dev/null; then
       echo "download already running pid=$(cat /tmp/model-download.pid)"
       exit 0
     fi
     rm -f /tmp/model-download.log /tmp/model-download.pid
     MODEL_ROOT=/models LOCAL_DIR=/models/Kimi-K3 HF_HOME=/models/hf \
       nohup bash /tmp/download-model-inpod.sh > /tmp/model-download.log 2>&1 &
     echo $! > /tmp/model-download.pid
     echo started download on rank-'"$i"' pid=$(cat /tmp/model-download.pid)'
done

if [[ "$FOLLOW" == "1" ]]; then
  echo "==> Tailing rank-0 download log (Ctrl-C stops follow only)"
  kubectl -n "$NS" exec vllm-recipe-0 -- tail -f /tmp/model-download.log || true
fi

echo "==> Waiting for .download-complete on every rank"
pending=1
while [[ "$pending" == "1" ]]; do
  pending=0
  for i in $(seq 0 $((NNODES - 1))); do
    if ! kubectl -n "$NS" exec "vllm-recipe-$i" -- \
        test -f /models/Kimi-K3/.download-complete 2>/dev/null; then
      pending=1
      lines="$(kubectl -n "$NS" exec "vllm-recipe-$i" -- \
        bash -c 'tail -n 3 /tmp/model-download.log 2>/dev/null || echo "(no log yet)"' 2>/dev/null || true)"
      echo "  rank-$i: still downloading... $lines"
    else
      echo "  rank-$i: complete"
      kubectl -n "$NS" exec "vllm-recipe-$i" -- cat /models/Kimi-K3/.download-complete || true
    fi
  done
  if [[ "$pending" == "1" ]]; then
    sleep 60
  fi
done

echo
echo "Weights ready on hostPath. Next: ./scripts/deploy-recipe.sh"
echo "Storage notes: manifests/STORAGE.md"
