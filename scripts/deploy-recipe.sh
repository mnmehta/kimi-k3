#!/usr/bin/env bash
# Deploy multi-node TP recipe with REAL weights from PVC kimi-k3-model.
# Recipe: https://recipes.vllm.ai/moonshotai/Kimi-K3
#
# Prerequisites:
#   ./scripts/download-model.sh   # PVC Bound + Job complete
#
# Usage:
#   export KUBECONFIG=/Users/mimehta/kubeconfigs/kubeconfig.fozzie
#   ./scripts/deploy-recipe.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export KUBECONFIG="${KUBECONFIG:-/Users/mimehta/kubeconfigs/kubeconfig.fozzie}"
NS="${NS:-kimi-k3}"
NNODES="${NNODES:-2}"
PVC="${PVC:-kimi-k3-model}"
MODEL_PATH="${MODEL_PATH:-/models/Kimi-K3}"
LOAD_FORMAT="${LOAD_FORMAT:-}"

echo "==> Applying namespace + recipe StatefulSet (container-local /models; see manifests/STORAGE.md)"
kubectl apply -f "$ROOT/manifests/namespace.yaml"
kubectl apply -f "$ROOT/manifests/vllm-recipe.yaml"
kubectl -n "$NS" scale statefulset/vllm-recipe --replicas="$NNODES"
kubectl -n "$NS" rollout status statefulset/vllm-recipe --timeout=30m

echo "==> Waiting for pods Ready"
for i in $(seq 0 $((NNODES - 1))); do
  kubectl -n "$NS" wait --for=condition=Ready "pod/vllm-recipe-$i" --timeout=30m
done
kubectl -n "$NS" get pods -l app=vllm-recipe -o wide

if kubectl -n "$NS" get pvc "$PVC" >/dev/null 2>&1; then
  echo "==> PVC $PVC status (optional; hostPath fallback may be active)"
  kubectl -n "$NS" get pvc "$PVC" || true
fi

echo "==> Verifying weights on each rank"
for i in $(seq 0 $((NNODES - 1))); do
  kubectl -n "$NS" exec "vllm-recipe-$i" -- bash -c \
    "test -f ${MODEL_PATH}/.download-complete && test -f ${MODEL_PATH}/config.json && \
     echo rank-$i ok && cat ${MODEL_PATH}/.download-complete && du -sh ${MODEL_PATH}"
done

MASTER_ADDR="$(kubectl -n "$NS" get pod vllm-recipe-0 -o jsonpath='{.status.podIP}')"
if [[ -z "$MASTER_ADDR" ]]; then
  echo "failed to resolve MASTER_ADDR from vllm-recipe-0" >&2
  exit 1
fi
echo "==> MASTER_ADDR=$MASTER_ADDR"

echo "==> Copying in-pod serve script"
for i in $(seq 0 $((NNODES - 1))); do
  kubectl -n "$NS" cp \
    "$ROOT/scripts/run-vllm-kimi-k3-recipe.sh" \
    "vllm-recipe-$i:/tmp/run-recipe.sh"
done

echo "==> Stopping any previous serve processes"
for i in $(seq 0 $((NNODES - 1))); do
  kubectl -n "$NS" exec "vllm-recipe-$i" -- \
    bash -c 'pkill -f "vllm serve" || true' || true
done
sleep 2

LOAD_ENV=""
if [[ -n "$LOAD_FORMAT" ]]; then
  LOAD_ENV="LOAD_FORMAT=$LOAD_FORMAT"
fi

echo "==> Starting workers (headless) then head"
for i in $(seq 1 $((NNODES - 1))); do
  kubectl -n "$NS" exec "vllm-recipe-$i" -- bash -c \
    "rm -f /tmp/vllm-recipe.log; \
     NODE_RANK=$i MASTER_ADDR=$MASTER_ADDR NNODES=$NNODES MODEL=$MODEL_PATH $LOAD_ENV \
     nohup bash /tmp/run-recipe.sh > /tmp/vllm-recipe.log 2>&1 & \
     echo started rank $i pid=\$!"
done

kubectl -n "$NS" exec vllm-recipe-0 -- bash -c \
  "rm -f /tmp/vllm-recipe.log; \
   NODE_RANK=0 MASTER_ADDR=$MASTER_ADDR NNODES=$NNODES MODEL=$MODEL_PATH $LOAD_ENV \
   nohup bash /tmp/run-recipe.sh > /tmp/vllm-recipe.log 2>&1 & \
   echo started rank 0 pid=\$!"

echo "==> Waiting for /health (weight load can take a long time)"
echo "    kubectl -n $NS exec vllm-recipe-0 -- tail -f /tmp/vllm-recipe.log"
for _ in $(seq 1 360); do
  if kubectl -n "$NS" exec vllm-recipe-0 -- \
      curl -sf -m 2 http://127.0.0.1:8000/health >/dev/null 2>&1; then
    echo "Healthy: http://$MASTER_ADDR:8000  (pod vllm-recipe-0)"
    kubectl -n "$NS" exec vllm-recipe-0 -- curl -sS http://127.0.0.1:8000/v1/models
    echo
    exit 0
  fi
  sleep 10
done

echo "Timed out waiting for health. Recent rank-0 log:" >&2
kubectl -n "$NS" exec vllm-recipe-0 -- tail -80 /tmp/vllm-recipe.log >&2 || true
echo "Rank-1 log:" >&2
kubectl -n "$NS" exec vllm-recipe-1 -- tail -40 /tmp/vllm-recipe.log >&2 || true
exit 1
