#!/usr/bin/env bash
# Deploy multi-node SGLang with TP=8, PP=2, EP=8 (2×8 H200).
# All other flags from the H200 low-latency recipe (tp16+ep16):
#   --moe-runner-backend marlin --decode-attention-backend flashmla
#   --enable-symm-mem --mem-fraction-static 0.85
#   env: NCCL_MNNVL_ENABLE=1, NCCL_CUMEM_ENABLE=1, SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0
#
# Recipe reference:
#   https://docs.sglang.io/cookbook/autoregressive/Moonshotai/Kimi-K3#hw=h200&pdMode=unified&strategy=low-latency
#
# Usage:
#   export KUBECONFIG=...
#   ./scripts/deploy-sglang-pp2.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NS="${NS:-kimi-k3}"
NNODES="${NNODES:-2}"
STS_NAME="${STS_NAME:-sglang-recipe}"
MODEL_PATH="${MODEL_PATH:-/models/Kimi-K3}"

# Parallelism — tp8-pp2 override of H200 recipe
export TP_SIZE="${TP_SIZE:-8}"
export PP_SIZE="${PP_SIZE:-2}"
export MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.90}"
export PORT="${PORT:-30000}"
export DIST_PORT="${DIST_PORT:-20000}"
# Cap context to 8192 (match vLLM benchmark); halve Mamba state for higher concurrency
export CONTEXT_LENGTH="${CONTEXT_LENGTH:-8192}"
export MAMBA_SSM_DTYPE="${MAMBA_SSM_DTYPE:-bfloat16}"
export MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-256}"

# NCCL tuning for cross-node PP P2P (same as vLLM PP2 harness)
export NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5_5}"
export NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-3}"
unset NCCL_IB_DISABLE 2>/dev/null || true
export NCCL_DMABUF_ENABLE="${NCCL_DMABUF_ENABLE:-0}"

echo "==> SGLang PP2 deployment (H200 recipe flags): TP=${TP_SIZE} PP=${PP_SIZE} NNODES=${NNODES}"
echo "    moe-runner-backend=marlin decode-attention-backend=flashmla enable-symm-mem"
echo "    NCCL_IB_HCA=${NCCL_IB_HCA} NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX}"
echo "    MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC} PORT=${PORT}"
echo "    CONTEXT_LENGTH=${CONTEXT_LENGTH} MAMBA_SSM_DTYPE=${MAMBA_SSM_DTYPE} MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS}"

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
  kubectl -n "$NS" cp \
    "$ROOT/scripts/run-sglang-kimi-k3-tp8-pp2.sh" \
    "${STS_NAME}-$i:/tmp/run-sglang.sh"
  kubectl -n "$NS" cp \
    "$ROOT/scripts/stop-inpod-sglang.sh" \
    "${STS_NAME}-$i:/tmp/stop-inpod-sglang.sh"
done

echo "==> Stopping any previous SGLang processes"
for i in $(seq 0 $((NNODES - 1))); do
  kubectl -n "$NS" exec "${STS_NAME}-$i" -- bash /tmp/stop-inpod-sglang.sh >/dev/null 2>&1 || true
done
sleep 2

# Env passed to run-sglang-kimi-k3.sh
SERVE_ENV="TP_SIZE=$TP_SIZE PP_SIZE=$PP_SIZE NNODES=$NNODES MODEL=$MODEL_PATH"
SERVE_ENV+=" MEM_FRACTION_STATIC=$MEM_FRACTION_STATIC PORT=$PORT DIST_PORT=$DIST_PORT"
SERVE_ENV+=" CONTEXT_LENGTH=$CONTEXT_LENGTH MAMBA_SSM_DTYPE=$MAMBA_SSM_DTYPE MAX_RUNNING_REQUESTS=$MAX_RUNNING_REQUESTS"
SERVE_ENV+=" NCCL_IB_HCA=$NCCL_IB_HCA NCCL_IB_GID_INDEX=$NCCL_IB_GID_INDEX"
SERVE_ENV+=" NCCL_DMABUF_ENABLE=$NCCL_DMABUF_ENABLE"

echo "==> Starting SGLang on all ranks (rank-1+ first, then rank-0)"
for i in $(seq 1 $((NNODES - 1))); do
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

echo "==> Waiting for /health (model load can take a long time)"
echo "    kubectl -n $NS exec ${STS_NAME}-0 -- tail -f /tmp/sglang-recipe.log"
for _ in $(seq 1 360); do
  if kubectl -n "$NS" exec "${STS_NAME}-0" -- \
      curl -sf -m 2 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    echo "Healthy: http://${DIST_INIT_ADDR}:${PORT}  (pod ${STS_NAME}-0)"
    kubectl -n "$NS" exec "${STS_NAME}-0" -- curl -sS "http://127.0.0.1:${PORT}/v1/models"
    echo
    echo
    echo "==> SGLang PP2 ready. Run benchmarks with:"
    echo "    BASE_URL=http://127.0.0.1:${PORT} vllm bench serve \\"
    echo "      --backend openai --model moonshotai/Kimi-K3 \\"
    echo "      --endpoint /v1/completions --dataset-name random \\"
    echo "      --random-input-len 1000 --random-output-len 1000 \\"
    echo "      --ignore-eos --request-rate inf --max-concurrency 8 --num-prompts 16"
    exit 0
  fi
  sleep 10
done

echo "Timed out waiting for health. Recent rank-0 log:" >&2
kubectl -n "$NS" exec "${STS_NAME}-0" -- tail -80 /tmp/sglang-recipe.log >&2 || true
echo "Rank-1 log:" >&2
kubectl -n "$NS" exec "${STS_NAME}-1" -- tail -40 /tmp/sglang-recipe.log >&2 || true
exit 1
