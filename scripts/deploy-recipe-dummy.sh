#!/usr/bin/env bash
# Deploy the multi-node TP recipe (dummy weights) onto the kimi-k3 namespace.
# Recipe: https://recipes.vllm.ai/moonshotai/Kimi-K3
#
# Usage:
#   export KUBECONFIG=/Users/mimehta/kubeconfigs/kubeconfig.fozzie
#   ./scripts/deploy-recipe-dummy.sh
#
# Tears down previous vllm-recipe pods, applies manifests, then starts
# rank-0 (API server) and rank-1 (--headless) with the recipe flags.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export KUBECONFIG="${KUBECONFIG:-/Users/mimehta/kubeconfigs/kubeconfig.fozzie}"
NS="${NS:-kimi-k3}"
NNODES="${NNODES:-2}"

echo "==> Applying namespace + StatefulSet ($NNODES nodes × 8 GPUs)"
kubectl apply -f "$ROOT/manifests/namespace.yaml"
# Scale before apply so first create uses desired replica count
kubectl apply -f "$ROOT/manifests/vllm-recipe.yaml"
kubectl -n "$NS" scale statefulset/vllm-recipe --replicas="$NNODES"
kubectl -n "$NS" rollout status statefulset/vllm-recipe --timeout=30m

echo "==> Waiting for pods Ready"
for i in $(seq 0 $((NNODES - 1))); do
  kubectl -n "$NS" wait --for=condition=Ready "pod/vllm-recipe-$i" --timeout=30m
done

kubectl -n "$NS" get pods -l app=vllm-recipe -o wide

# With hostNetwork, podIP is the node IP — fine as NCCL master addr.
MASTER_ADDR="$(kubectl -n "$NS" get pod vllm-recipe-0 -o jsonpath='{.status.podIP}')"
if [[ -z "$MASTER_ADDR" ]]; then
  echo "failed to resolve MASTER_ADDR from vllm-recipe-0" >&2
  exit 1
fi
echo "==> MASTER_ADDR=$MASTER_ADDR"

echo "==> Copying in-pod serve script"
for i in $(seq 0 $((NNODES - 1))); do
  kubectl -n "$NS" cp \
    "$ROOT/scripts/run-vllm-kimi-k3-recipe-dummy.sh" \
    "vllm-recipe-$i:/tmp/run-recipe-dummy.sh"
done

echo "==> Stopping any previous serve processes"
for i in $(seq 0 $((NNODES - 1))); do
  kubectl -n "$NS" exec "vllm-recipe-$i" -- \
    bash -c 'pkill -f "vllm serve" || true' || true
done
sleep 2

echo "==> Starting workers (headless) then head"
# Start non-zero ranks first so they wait on the rendezvous.
for i in $(seq 1 $((NNODES - 1))); do
  kubectl -n "$NS" exec "vllm-recipe-$i" -- bash -c \
    "rm -f /tmp/vllm-recipe.log; \
     NODE_RANK=$i MASTER_ADDR=$MASTER_ADDR NNODES=$NNODES \
     nohup bash /tmp/run-recipe-dummy.sh > /tmp/vllm-recipe.log 2>&1 & \
     echo started rank $i pid=\$!"
done

kubectl -n "$NS" exec vllm-recipe-0 -- bash -c \
  "rm -f /tmp/vllm-recipe.log; \
   NODE_RANK=0 MASTER_ADDR=$MASTER_ADDR NNODES=$NNODES \
   nohup bash /tmp/run-recipe-dummy.sh > /tmp/vllm-recipe.log 2>&1 & \
   echo started rank 0 pid=\$!"

echo "==> Tail rank-0 until /health is up (Ctrl-C stops tail only)"
echo "    kubectl -n $NS exec vllm-recipe-0 -- tail -f /tmp/vllm-recipe.log"
echo
for _ in $(seq 1 120); do
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
