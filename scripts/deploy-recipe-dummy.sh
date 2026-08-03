#!/usr/bin/env bash
# Deploy the multi-node recipe (dummy weights) onto the kimi-k3 namespace.
# Recipe: https://recipes.vllm.ai/moonshotai/Kimi-K3
#
# Default: TP16. Also:
#   TP8×PP2: ./scripts/deploy-recipe-dummy-pp.sh
#   TP8×DP2: ./scripts/deploy-recipe-dummy-dp.sh
#
# Usage:
#   export KUBECONFIG=/Users/mimehta/kubeconfigs/kubeconfig.fozzie
#   ./scripts/deploy-recipe-dummy.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export KUBECONFIG="${KUBECONFIG:-/Users/mimehta/kubeconfigs/kubeconfig.fozzie}"
NS="${NS:-kimi-k3}"
NNODES="${NNODES:-2}"
PP_SIZE="${PP_SIZE:-1}"
DP_SIZE="${DP_SIZE:-1}"
DP_SIZE_LOCAL="${DP_SIZE_LOCAL:-1}"
DP_RPC_PORT="${DP_RPC_PORT:-13345}"
if [[ "$DP_SIZE" -gt 1 ]]; then
  TP_SIZE="${TP_SIZE:-8}"
else
  TP_SIZE="${TP_SIZE:-$((NNODES * 8 / PP_SIZE))}"
fi

echo "==> Parallelism TP=${TP_SIZE} PP=${PP_SIZE} DP=${DP_SIZE} DP_LOCAL=${DP_SIZE_LOCAL} NNODES=${NNODES} (dummy)"
echo "==> Applying namespace + StatefulSet ($NNODES nodes × 8 GPUs)"
kubectl apply -f "$ROOT/manifests/namespace.yaml"
kubectl apply -f "$ROOT/manifests/vllm-recipe.yaml"
kubectl -n "$NS" scale statefulset/vllm-recipe --replicas="$NNODES"
kubectl -n "$NS" rollout status statefulset/vllm-recipe --timeout=30m

echo "==> Waiting for pods Ready"
for i in $(seq 0 $((NNODES - 1))); do
  kubectl -n "$NS" wait --for=condition=Ready "pod/vllm-recipe-$i" --timeout=30m
done

kubectl -n "$NS" get pods -l app=vllm-recipe -o wide

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
  kubectl -n "$NS" cp "$ROOT/scripts/stop-inpod-vllm.sh" "vllm-recipe-$i:/tmp/stop-inpod-vllm.sh"
  kubectl -n "$NS" exec "vllm-recipe-$i" -- bash /tmp/stop-inpod-vllm.sh 2>/dev/null || true
done
sleep 2

OPT_ENV=""
for v in GPU_MEM_UTIL MAX_NUM_SEQS MAX_MODEL_LEN MAX_NUM_BATCHED_TOKENS \
         VLLM_USE_RUST_FRONTEND VLLM_USE_V2_MODEL_RUNNER DP_RPC_PORT \
         ENABLE_EXPERT_PARALLEL NUM_LAYERS NUM_EXPERTS NUM_EXPERTS_PER_TOKEN \
         NUM_SHARED_EXPERTS ENABLE_TORCH_PROFILER PROFILE_DIR VLLM_RPC_TIMEOUT \
         VLLM_ENABLE_K3_LATENT_MOE_TAIL_FUSION NCCL_DMABUF_ENABLE \
         NCCL_IB_HCA NCCL_IB_GID_INDEX NCCL_IB_DISABLE \
         PYTORCH_CUDA_ALLOC_CONF MODEL; do
  if [[ -n "${!v+x}" ]]; then
    OPT_ENV+=" $v=${!v}"
  fi
done
PAR_ENV="TP_SIZE=$TP_SIZE PP_SIZE=$PP_SIZE DP_SIZE=$DP_SIZE DP_SIZE_LOCAL=$DP_SIZE_LOCAL NNODES=$NNODES$OPT_ENV"

if [[ "$DP_SIZE" -gt 1 ]]; then
  echo "==> Starting DP workers (headless) then DP head"
  for i in $(seq 1 $((NNODES - 1))); do
    kubectl -n "$NS" exec "vllm-recipe-$i" -- bash -c \
      "rm -f /tmp/vllm-recipe.log; \
       DP_START_RANK=$i NODE_RANK=$i MASTER_ADDR=$MASTER_ADDR DATA_PARALLEL_ADDRESS=$MASTER_ADDR $PAR_ENV \
       nohup bash /tmp/run-recipe-dummy.sh > /tmp/vllm-recipe.log 2>&1 & \
       echo started dp_start_rank=$i pid=\$!"
  done
  kubectl -n "$NS" exec vllm-recipe-0 -- bash -c \
    "rm -f /tmp/vllm-recipe.log; \
     DP_START_RANK=0 NODE_RANK=0 MASTER_ADDR=$MASTER_ADDR DATA_PARALLEL_ADDRESS=$MASTER_ADDR $PAR_ENV \
     nohup bash /tmp/run-recipe-dummy.sh > /tmp/vllm-recipe.log 2>&1 & \
     echo started dp_start_rank=0 pid=\$!"
else
  echo "==> Starting workers (headless) then head"
  for i in $(seq 1 $((NNODES - 1))); do
    kubectl -n "$NS" exec "vllm-recipe-$i" -- bash -c \
      "rm -f /tmp/vllm-recipe.log; \
       NODE_RANK=$i MASTER_ADDR=$MASTER_ADDR $PAR_ENV \
       nohup bash /tmp/run-recipe-dummy.sh > /tmp/vllm-recipe.log 2>&1 & \
       echo started rank $i pid=\$!"
  done
  kubectl -n "$NS" exec vllm-recipe-0 -- bash -c \
    "rm -f /tmp/vllm-recipe.log; \
     NODE_RANK=0 MASTER_ADDR=$MASTER_ADDR $PAR_ENV \
     nohup bash /tmp/run-recipe-dummy.sh > /tmp/vllm-recipe.log 2>&1 & \
     echo started rank 0 pid=\$!"
fi

echo "==> Waiting for /health"
echo "    kubectl -n $NS exec vllm-recipe-0 -- tail -f /tmp/vllm-recipe.log"
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
