#!/usr/bin/env bash
# Deploy multi-node recipe with REAL weights on hostPath /models.
# Recipe: https://recipes.vllm.ai/moonshotai/Kimi-K3
#
# Default parallelism: TP16 (PP=1, DP=1).
#   TP8×PP2: ./scripts/deploy-recipe-pp.sh
#   TP8×DP2: ./scripts/deploy-recipe-dp.sh
#     https://recipes.vllm.ai/moonshotai/Kimi-K3?strategy=multi_node_tp_dp
#
# Prerequisites:
#   ./scripts/download-model.sh   # or weights already on /mnt/local/kimi-k3/models
#
# Usage:
#   export KUBECONFIG=/Users/mimehta/kubeconfigs/kubeconfig.fozzie
#   ./scripts/deploy-recipe.sh

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
MODEL_PATH="${MODEL_PATH:-/models/Kimi-K3}"
LOAD_FORMAT="${LOAD_FORMAT:-}"
# hostpath = CoreWeave /mnt/local/kimi-k3/models (default)
# overlay  = ephemeral container-local /models (legacy)
STORAGE_BACKEND="${STORAGE_BACKEND:-hostpath}"
# Engine seq budget (client sweep may oversubscribe up to C=512).
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-128}"

echo "==> Parallelism TP=${TP_SIZE} PP=${PP_SIZE} DP=${DP_SIZE} DP_LOCAL=${DP_SIZE_LOCAL} NNODES=${NNODES} STORAGE_BACKEND=${STORAGE_BACKEND}"

echo "==> Applying namespace + recipe StatefulSet"
kubectl apply -f "$ROOT/manifests/namespace.yaml"
# Stop serve before volume-template changes recreate pods.
for i in $(seq 0 $((NNODES - 1))); do
  kubectl -n "$NS" exec "vllm-recipe-$i" -- bash /tmp/stop-inpod-vllm.sh >/dev/null 2>&1 || true
done || true
case "$STORAGE_BACKEND" in
  hostpath)
    kubectl apply -f "$ROOT/manifests/vllm-recipe.yaml"
    ;;
  overlay)
    kubectl apply -f "$ROOT/manifests/vllm-recipe-overlay.yaml"
    ;;
  *)
    echo "Unknown STORAGE_BACKEND=$STORAGE_BACKEND (use hostpath|overlay)" >&2
    exit 1
    ;;
esac
kubectl -n "$NS" scale statefulset/vllm-recipe --replicas="$NNODES"
# Force rollout when switching storage backend (template may already match).
kubectl -n "$NS" delete pod -l app=vllm-recipe --wait=false 2>/dev/null || true
kubectl -n "$NS" rollout status statefulset/vllm-recipe --timeout=30m

echo "==> Waiting for pods Ready"
for i in $(seq 0 $((NNODES - 1))); do
  kubectl -n "$NS" wait --for=condition=Ready "pod/vllm-recipe-$i" --timeout=30m
done
kubectl -n "$NS" get pods -l app=vllm-recipe -o wide

echo "==> Storage mount check"
for i in $(seq 0 $((NNODES - 1))); do
  kubectl -n "$NS" exec "vllm-recipe-$i" -- bash -c \
    'echo rank-'"$i"'; df -h /models | tail -1; findmnt -T /models | head -2 || true'
done

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

echo "==> Copying in-pod serve + stop scripts"
for i in $(seq 0 $((NNODES - 1))); do
  kubectl -n "$NS" cp \
    "$ROOT/scripts/run-vllm-kimi-k3-recipe.sh" \
    "vllm-recipe-$i:/tmp/run-recipe.sh"
  kubectl -n "$NS" cp \
    "$ROOT/scripts/stop-inpod-vllm.sh" \
    "vllm-recipe-$i:/tmp/stop-inpod-vllm.sh"
done

echo "==> Stopping any previous serve processes"
for i in $(seq 0 $((NNODES - 1))); do
  kubectl -n "$NS" exec "vllm-recipe-$i" -- bash /tmp/stop-inpod-vllm.sh >/dev/null || true
done
sleep 2

LOAD_ENV=""
if [[ -n "$LOAD_FORMAT" ]]; then
  LOAD_ENV="LOAD_FORMAT=$LOAD_FORMAT"
fi

# Optional knobs (set by deploy-recipe-pp.sh / deploy-recipe-dp.sh or caller)
OPT_ENV=""
for v in GPU_MEM_UTIL MAX_NUM_SEQS MAX_MODEL_LEN MAX_NUM_BATCHED_TOKENS \
         NCCL_IB_DISABLE NCCL_IB_GID_INDEX NCCL_IB_HCA NCCL_DMABUF_ENABLE \
         PYTORCH_CUDA_ALLOC_CONF VLLM_USE_RUST_FRONTEND VLLM_USE_V2_MODEL_RUNNER \
         DP_RPC_PORT DATA_PARALLEL_ADDRESS KV_CACHE_MEMORY_BYTES \
         VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE SKIP_MM_PROFILING \
         ENABLE_EXPERT_PARALLEL ENABLE_EP_WEIGHT_FILTER; do
  if [[ -n "${!v+x}" ]]; then
    OPT_ENV+=" $v=${!v}"
  fi
done

PAR_ENV="TP_SIZE=$TP_SIZE PP_SIZE=$PP_SIZE DP_SIZE=$DP_SIZE DP_SIZE_LOCAL=$DP_SIZE_LOCAL NNODES=$NNODES MODEL=$MODEL_PATH $LOAD_ENV$OPT_ENV"

if [[ "$DP_SIZE" -gt 1 ]]; then
  echo "==> Starting DP workers (headless) then DP head (data-parallel-address=$MASTER_ADDR)"
  for i in $(seq 1 $((NNODES - 1))); do
    kubectl -n "$NS" exec "vllm-recipe-$i" -- bash -c \
      "rm -f /tmp/vllm-recipe.log; \
       DP_START_RANK=$i NODE_RANK=$i MASTER_ADDR=$MASTER_ADDR DATA_PARALLEL_ADDRESS=$MASTER_ADDR $PAR_ENV \
       nohup bash /tmp/run-recipe.sh > /tmp/vllm-recipe.log 2>&1 & \
       echo started dp_start_rank=$i pid=\$!"
  done
  kubectl -n "$NS" exec vllm-recipe-0 -- bash -c \
    "rm -f /tmp/vllm-recipe.log; \
     DP_START_RANK=0 NODE_RANK=0 MASTER_ADDR=$MASTER_ADDR DATA_PARALLEL_ADDRESS=$MASTER_ADDR $PAR_ENV \
     nohup bash /tmp/run-recipe.sh > /tmp/vllm-recipe.log 2>&1 & \
     echo started dp_start_rank=0 pid=\$!"
else
  echo "==> Starting workers (headless) then head"
  for i in $(seq 1 $((NNODES - 1))); do
    kubectl -n "$NS" exec "vllm-recipe-$i" -- bash -c \
      "rm -f /tmp/vllm-recipe.log; \
       NODE_RANK=$i MASTER_ADDR=$MASTER_ADDR $PAR_ENV \
       nohup bash /tmp/run-recipe.sh > /tmp/vllm-recipe.log 2>&1 & \
       echo started rank $i pid=\$!"
  done
  kubectl -n "$NS" exec vllm-recipe-0 -- bash -c \
    "rm -f /tmp/vllm-recipe.log; \
     NODE_RANK=0 MASTER_ADDR=$MASTER_ADDR $PAR_ENV \
     nohup bash /tmp/run-recipe.sh > /tmp/vllm-recipe.log 2>&1 & \
     echo started rank 0 pid=\$!"
fi

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
