#!/usr/bin/env bash
# Multi-node deploy engine (TP / TEP / PP / DP). Expects env already set by deploy.sh.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
NS="${NS:-kimi-k3}"
NNODES="${NNODES:?NNODES required}"
PP_SIZE="${PP_SIZE:-1}"
DP_SIZE="${DP_SIZE:-1}"
DP_SIZE_LOCAL="${DP_SIZE_LOCAL:-1}"
DP_RPC_PORT="${DP_RPC_PORT:-13345}"
if [[ -z "${TP_SIZE:-}" ]]; then
  if [[ "$DP_SIZE" -gt 1 ]]; then
    TP_SIZE=8
  else
    TP_SIZE=$((NNODES * 8 / PP_SIZE))
  fi
fi
MODEL_PATH="${MODEL_PATH:-${MODEL:-/models/Kimi-K3}}"
MODEL="${MODEL:-$MODEL_PATH}"
LOAD_FORMAT="${LOAD_FORMAT:-}"
STORAGE_BACKEND="${DEPLOY_STORAGE_BACKEND:-${STORAGE_BACKEND:-hostpath}}"
VERIFY_WEIGHTS="${DEPLOY_VERIFY_WEIGHTS:-1}"
SERVE_SCRIPT="${DEPLOY_SERVE_SCRIPT:-scripts/run-vllm-kimi-k3-recipe.sh}"
HEALTH_PORT="${DEPLOY_HEALTH_PORT:-${PORT:-8000}}"
HEALTH_POLLS="${DEPLOY_HEALTH_TIMEOUT_POLLS:-360}"
NAME="${DEPLOY_NAME:-multi_node}"

if [[ "$SERVE_SCRIPT" != /* ]]; then
  SERVE_SCRIPT_HOST="$ROOT/$SERVE_SCRIPT"
else
  SERVE_SCRIPT_HOST="$SERVE_SCRIPT"
fi
if [[ ! -f "$SERVE_SCRIPT_HOST" ]]; then
  echo "serve script not found: $SERVE_SCRIPT_HOST" >&2
  exit 1
fi

echo "==> Deploy ${NAME}: TP=${TP_SIZE} PP=${PP_SIZE} DP=${DP_SIZE} DP_LOCAL=${DP_SIZE_LOCAL} NNODES=${NNODES} STORAGE=${STORAGE_BACKEND} VERIFY_WEIGHTS=${VERIFY_WEIGHTS}"

echo "==> Applying namespace + recipe StatefulSet"
kubectl apply -f "$ROOT/manifests/namespace.yaml"
for i in $(seq 0 $((NNODES - 1))); do
  kubectl -n "$NS" exec "vllm-recipe-$i" -- bash /tmp/stop-inpod-vllm.sh >/dev/null 2>&1 || true
done || true
case "$STORAGE_BACKEND" in
  hostpath) kubectl apply -f "$ROOT/manifests/vllm-recipe.yaml" ;;
  overlay)  kubectl apply -f "$ROOT/manifests/vllm-recipe-overlay.yaml" ;;
  *) echo "Unknown STORAGE_BACKEND=$STORAGE_BACKEND" >&2; exit 1 ;;
esac
kubectl -n "$NS" scale statefulset/vllm-recipe --replicas="$NNODES"
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

if [[ "$VERIFY_WEIGHTS" == "1" ]]; then
  echo "==> Verifying weights on each rank"
  for i in $(seq 0 $((NNODES - 1))); do
    kubectl -n "$NS" exec "vllm-recipe-$i" -- bash -c \
      "test -f ${MODEL_PATH}/.download-complete && test -f ${MODEL_PATH}/config.json && \
       echo rank-$i ok && cat ${MODEL_PATH}/.download-complete && du -sh ${MODEL_PATH}"
  done
else
  echo "==> Skipping weight verification (dummy / verify_weights=false)"
fi

MASTER_ADDR="$(kubectl -n "$NS" get pod vllm-recipe-0 -o jsonpath='{.status.podIP}')"
if [[ -z "$MASTER_ADDR" ]]; then
  echo "failed to resolve MASTER_ADDR from vllm-recipe-0" >&2
  exit 1
fi
echo "==> MASTER_ADDR=$MASTER_ADDR"

echo "==> Copying in-pod serve + stop scripts ($SERVE_SCRIPT)"
for i in $(seq 0 $((NNODES - 1))); do
  kubectl -n "$NS" cp "$SERVE_SCRIPT_HOST" "vllm-recipe-$i:/tmp/run-recipe.sh"
  kubectl -n "$NS" cp "$ROOT/scripts/stop-inpod-vllm.sh" "vllm-recipe-$i:/tmp/stop-inpod-vllm.sh"
  kubectl -n "$NS" cp "$ROOT/scripts/lib/patch_humming_situ.sh" "vllm-recipe-$i:/tmp/patch_humming_situ.sh"
done

echo "==> Stopping any previous serve processes"
for i in $(seq 0 $((NNODES - 1))); do
  kubectl -n "$NS" exec "vllm-recipe-$i" -- bash /tmp/stop-inpod-vllm.sh >/dev/null || true
done
sleep 2

# PR #50510 allowlist: Humming rejects MoEActivation.SITU until upstream lands.
# Patch only while MOE_BACKEND=humming; always unpatch first for a clean image state.
echo "==> Humming SiTU allowlist (MOE_BACKEND=${MOE_BACKEND:-})"
for i in $(seq 0 $((NNODES - 1))); do
  kubectl -n "$NS" exec "vllm-recipe-$i" -- bash /tmp/patch_humming_situ.sh unpatch >/dev/null || true
  if [[ "${MOE_BACKEND:-}" == "humming" ]]; then
    kubectl -n "$NS" exec "vllm-recipe-$i" -- bash /tmp/patch_humming_situ.sh patch
  else
    kubectl -n "$NS" exec "vllm-recipe-$i" -- bash /tmp/patch_humming_situ.sh status || true
  fi
done

# Forward optional knobs if set in the environment.
OPT_ENV=""
for v in GPU_MEM_UTIL MAX_NUM_SEQS MAX_MODEL_LEN MAX_NUM_BATCHED_TOKENS \
         NCCL_IB_DISABLE NCCL_IB_GID_INDEX NCCL_IB_HCA NCCL_DMABUF_ENABLE \
         PYTORCH_CUDA_ALLOC_CONF VLLM_USE_RUST_FRONTEND VLLM_USE_V2_MODEL_RUNNER \
         DP_RPC_PORT DATA_PARALLEL_ADDRESS KV_CACHE_MEMORY_BYTES \
         VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE SKIP_MM_PROFILING \
         ENABLE_EXPERT_PARALLEL ENABLE_EP_WEIGHT_FILTER \
         NUM_LAYERS NUM_EXPERTS NUM_EXPERTS_PER_TOKEN NUM_SHARED_EXPERTS \
         ENABLE_TORCH_PROFILER PROFILE_DIR VLLM_RPC_TIMEOUT \
         VLLM_ENABLE_K3_LATENT_MOE_TAIL_FUSION MOE_BACKEND ATTENTION_BACKEND PORT; do
  if [[ -n "${!v+x}" ]]; then
    OPT_ENV+=" $v=${!v}"
  fi
done

LOAD_ENV=""
if [[ -n "${LOAD_FORMAT:-}" ]]; then
  LOAD_ENV="LOAD_FORMAT=$LOAD_FORMAT"
fi

PAR_ENV="TP_SIZE=$TP_SIZE PP_SIZE=$PP_SIZE DP_SIZE=$DP_SIZE DP_SIZE_LOCAL=$DP_SIZE_LOCAL NNODES=$NNODES MODEL=$MODEL $LOAD_ENV$OPT_ENV"

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

echo "==> Waiting for /health on :${HEALTH_PORT} (up to $((HEALTH_POLLS * 10))s)"
echo "    kubectl -n $NS exec vllm-recipe-0 -- tail -f /tmp/vllm-recipe.log"
for _ in $(seq 1 "$HEALTH_POLLS"); do
  if kubectl -n "$NS" exec vllm-recipe-0 -- \
      curl -sf -m 2 "http://127.0.0.1:${HEALTH_PORT}/health" >/dev/null 2>&1; then
    echo "Healthy: http://$MASTER_ADDR:${HEALTH_PORT}  (pod vllm-recipe-0)"
    kubectl -n "$NS" exec vllm-recipe-0 -- curl -sS "http://127.0.0.1:${HEALTH_PORT}/v1/models" || true
    echo
    exit 0
  fi
  sleep 10
done

echo "Timed out waiting for health. Recent rank-0 log:" >&2
kubectl -n "$NS" exec vllm-recipe-0 -- tail -80 /tmp/vllm-recipe.log >&2 || true
if [[ "$NNODES" -gt 1 ]]; then
  echo "Rank-1 log:" >&2
  kubectl -n "$NS" exec vllm-recipe-1 -- tail -40 /tmp/vllm-recipe.log >&2 || true
fi
exit 1
