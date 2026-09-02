#!/usr/bin/env bash
# Deploy SGLang for AgentX workloads with selectable parallelism strategy.
#
# Usage:
#   export KUBECONFIG=...
#   ./scripts/deploy-sglang-agentx.sh tp32-ep32       # flat TP32+EP32 (upstream recipe, DSpark OK)
#   ./scripts/deploy-sglang-agentx.sh pp4              # TP8×PP4 (no DSpark — PP>1 unsupported)
#
# All strategies use 4×8 H200 (32 GPUs), 262K context, mamba-ratio=0.17.
# Optional env overrides:
#   ENABLE_DSPARK=0/1  ENABLE_HICACHE=0/1  HICACHE_SIZE_GB=100  MEM_FRACTION_STATIC=0.90
#   MAMBA_FULL_MEMORY_RATIO=0.17  CONTEXT_LENGTH=262144  MAX_RUNNING_REQUESTS=48

set -euo pipefail

MODE="${1:-}"
if [[ -z "$MODE" ]]; then
  echo "Usage: $0 <tp32-ep32|pp4>" >&2
  echo "  tp32-ep32  Flat TP32+EP32 (upstream H200 high-throughput recipe, DSpark supported)" >&2
  echo "  pp4        TP8×PP4 pipeline parallel (DSpark NOT supported with PP>1)" >&2
  exit 1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NS="${NS:-kimi-k3}"
NNODES="${NNODES:-4}"
STS_NAME="${STS_NAME:-sglang-recipe}"
MODEL_PATH="${MODEL_PATH:-/models/Kimi-K3}"

# Shared AgentX defaults
export MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.90}"
export MAMBA_FULL_MEMORY_RATIO="${MAMBA_FULL_MEMORY_RATIO:-0.17}"
export CONTEXT_LENGTH="${CONTEXT_LENGTH:-262144}"
export MAMBA_SSM_DTYPE="${MAMBA_SSM_DTYPE:-bfloat16}"
export MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-48}"
export PORT="${PORT:-30000}"
export DIST_PORT="${DIST_PORT:-20000}"
export ENABLE_HICACHE="${ENABLE_HICACHE:-0}"
export HICACHE_SIZE_GB="${HICACHE_SIZE_GB:-144}"
export HICACHE_STORAGE_BACKEND="${HICACHE_STORAGE_BACKEND:-}"

# NCCL tuning for cross-node comms
export NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5_5}"
export NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-3}"
unset NCCL_IB_DISABLE 2>/dev/null || true
export NCCL_DMABUF_ENABLE="${NCCL_DMABUF_ENABLE:-0}"

# Mode-specific settings
case "$MODE" in
  tp32-ep32)
    export TP_SIZE="${TP_SIZE:-32}"
    export PP_SIZE=1
    export EP_SIZE="${EP_SIZE:-32}"
    export ENABLE_DSPARK="${ENABLE_DSPARK:-1}"
    RUN_SCRIPT="$ROOT/scripts/run-sglang-kimi-k3-tp32-ep32-agentx.sh"
    MODE_LABEL="TP32+EP32"
    ;;
  pp4)
    export TP_SIZE="${TP_SIZE:-8}"
    export PP_SIZE=4
    export EP_SIZE=1
    # DSpark not supported with PP>1
    if [[ "${ENABLE_DSPARK:-0}" == "1" ]]; then
      echo "WARNING: DSpark not supported with PP>1 — forcing ENABLE_DSPARK=0" >&2
    fi
    export ENABLE_DSPARK=0
    RUN_SCRIPT="$ROOT/scripts/run-sglang-kimi-k3-tp8-pp4-agentx.sh"
    MODE_LABEL="TP8×PP4"
    ;;
  *)
    echo "Unknown mode: $MODE (use tp32-ep32 or pp4)" >&2
    exit 1
    ;;
esac

if [[ ! -f "$RUN_SCRIPT" ]]; then
  echo "Run script not found: $RUN_SCRIPT" >&2
  exit 1
fi

echo "==> SGLang AgentX deployment: ${MODE_LABEL} (mode=$MODE)"
echo "    TP=${TP_SIZE} PP=${PP_SIZE} EP=${EP_SIZE} NNODES=${NNODES}"
echo "    mem-fraction-static=${MEM_FRACTION_STATIC} mamba-full-memory-ratio=${MAMBA_FULL_MEMORY_RATIO}"
echo "    context-length=${CONTEXT_LENGTH} max-running-requests=${MAX_RUNNING_REQUESTS}"
echo "    dspark=${ENABLE_DSPARK} hicache=${ENABLE_HICACHE} hicache-size=${HICACHE_SIZE_GB}GB"
echo "    NCCL_IB_HCA=${NCCL_IB_HCA} NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX}"
echo "    run-script=$(basename "$RUN_SCRIPT")"

# ---------- Manifest ----------
echo "==> Applying namespace + SGLang StatefulSet"
kubectl apply -f "$ROOT/manifests/namespace.yaml"

if [[ -f "$ROOT/manifests/sglang-recipe.yaml" ]]; then
  kubectl apply -f "$ROOT/manifests/sglang-recipe.yaml"
else
  echo "    [INFO] No sglang-recipe.yaml found; reusing existing StatefulSet '$STS_NAME'"
fi

kubectl -n "$NS" scale "statefulset/$STS_NAME" --replicas="$NNODES" 2>/dev/null || true
kubectl -n "$NS" rollout status "statefulset/$STS_NAME" --timeout=30m 2>/dev/null || true

echo "==> Waiting for pods Ready"
for i in $(seq 0 $((NNODES - 1))); do
  kubectl -n "$NS" wait --for=condition=Ready "pod/${STS_NAME}-$i" --timeout=30m
done
kubectl -n "$NS" get pods -l "app=$STS_NAME" -o wide

echo "==> Storage mount check"
for i in $(seq 0 $((NNODES - 1))); do
  kubectl -n "$NS" exec "${STS_NAME}-$i" -- bash -c \
    'echo rank-'"$i"'; df -h /models | tail -1; findmnt -T /models | head -2 || true'
done

echo "==> Verifying weights on each rank"
for i in $(seq 0 $((NNODES - 1))); do
  kubectl -n "$NS" exec "${STS_NAME}-$i" -- bash -c \
    "test -f ${MODEL_PATH}/.download-complete && test -f ${MODEL_PATH}/config.json && \
     echo rank-$i ok && cat ${MODEL_PATH}/.download-complete && du -sh ${MODEL_PATH}"
done

DIST_INIT_ADDR="$(kubectl -n "$NS" get pod "${STS_NAME}-0" -o jsonpath='{.status.podIP}')"
if [[ -z "$DIST_INIT_ADDR" ]]; then
  echo "Failed to resolve DIST_INIT_ADDR from ${STS_NAME}-0" >&2
  exit 1
fi
echo "==> DIST_INIT_ADDR=$DIST_INIT_ADDR"

echo "==> Copying in-pod serve + stop scripts"
for i in $(seq 0 $((NNODES - 1))); do
  kubectl -n "$NS" cp "$RUN_SCRIPT" "${STS_NAME}-$i:/tmp/run-sglang.sh"
  kubectl -n "$NS" cp "$ROOT/scripts/stop-inpod-sglang.sh" "${STS_NAME}-$i:/tmp/stop-inpod-sglang.sh"
done

echo "==> Stopping any previous SGLang processes"
for i in $(seq 0 $((NNODES - 1))); do
  kubectl -n "$NS" exec "${STS_NAME}-$i" -- bash /tmp/stop-inpod-sglang.sh >/dev/null 2>&1 || true
done
sleep 2

# Env passed to the in-pod script
SERVE_ENV="TP_SIZE=$TP_SIZE PP_SIZE=$PP_SIZE EP_SIZE=$EP_SIZE NNODES=$NNODES MODEL=$MODEL_PATH"
SERVE_ENV+=" MEM_FRACTION_STATIC=$MEM_FRACTION_STATIC PORT=$PORT DIST_PORT=$DIST_PORT"
SERVE_ENV+=" CONTEXT_LENGTH=$CONTEXT_LENGTH MAMBA_SSM_DTYPE=$MAMBA_SSM_DTYPE"
SERVE_ENV+=" MAMBA_FULL_MEMORY_RATIO=$MAMBA_FULL_MEMORY_RATIO MAX_RUNNING_REQUESTS=$MAX_RUNNING_REQUESTS"
SERVE_ENV+=" ENABLE_DSPARK=$ENABLE_DSPARK"
SERVE_ENV+=" ENABLE_HICACHE=$ENABLE_HICACHE HICACHE_SIZE_GB=$HICACHE_SIZE_GB HICACHE_STORAGE_BACKEND=$HICACHE_STORAGE_BACKEND"
SERVE_ENV+=" NCCL_IB_HCA=$NCCL_IB_HCA NCCL_IB_GID_INDEX=$NCCL_IB_GID_INDEX"
SERVE_ENV+=" NCCL_DMABUF_ENABLE=$NCCL_DMABUF_ENABLE"

echo "==> Starting SGLang on all ranks (workers first, then rank-0)"
for i in $(seq $((NNODES - 1)) -1 1); do
  kubectl -n "$NS" exec "${STS_NAME}-$i" -- bash -c \
    "rm -f /tmp/sglang-recipe.log; \
     NODE_RANK=$i DIST_INIT_ADDR=$DIST_INIT_ADDR $SERVE_ENV \
     nohup bash /tmp/run-sglang.sh > /tmp/sglang-recipe.log 2>&1 & \
     echo started rank $i pid=\$!"
done

kubectl -n "$NS" exec "${STS_NAME}-0" -- bash -c \
  "rm -f /tmp/sglang-recipe.log; \
   NODE_RANK=0 DIST_INIT_ADDR=$DIST_INIT_ADDR $SERVE_ENV \
   nohup bash /tmp/run-sglang.sh > /tmp/sglang-recipe.log 2>&1 & \
   echo started rank 0 pid=\$!"

echo "==> Waiting for /health (model + draft load can take a long time)"
echo "    kubectl -n $NS exec ${STS_NAME}-0 -- tail -f /tmp/sglang-recipe.log"
for _ in $(seq 1 480); do
  if kubectl -n "$NS" exec "${STS_NAME}-0" -- \
      curl -sf -m 2 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    echo "Healthy: http://${DIST_INIT_ADDR}:${PORT}  (pod ${STS_NAME}-0)"
    kubectl -n "$NS" exec "${STS_NAME}-0" -- curl -sS "http://127.0.0.1:${PORT}/v1/models"
    echo
    echo
    echo "==> SGLang ${MODE_LABEL} AgentX ready. Run AIPerf with:"
    echo "    POD=aiperf-agentx NNODES=4 \\"
    echo "    LOCAL_OUT_PREFIX=bench-results/sglang/agentx-mvp-${MODE}-sglang \\"
    echo "    bash scripts/run-agentx-mvp-sglang-inpod.sh"
    exit 0
  fi
  sleep 10
done

echo "Timed out waiting for health. Recent logs:" >&2
for i in $(seq 0 $((NNODES - 1))); do
  echo "--- rank-$i ---" >&2
  kubectl -n "$NS" exec "${STS_NAME}-$i" -- tail -40 /tmp/sglang-recipe.log >&2 || true
done
exit 1
