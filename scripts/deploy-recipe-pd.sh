#!/usr/bin/env bash
# Deploy Prefill/Decode disaggregation (pd_cluster) on 4×8 H200.
# Recipe: https://recipes.vllm.ai/moonshotai/Kimi-K3?strategy=pd_cluster
#
# Layout (recipe default):
#   Prefill: vllm-recipe-0,1  TEP16  :8001  kv_producer  NIXL :5557
#   Decode:  vllm-recipe-2,3  TEP16  :8002  kv_consumer  NIXL :5558
#   Router:  disagg_proxy_demo on prefill head :8000  (wait-pd-and-sweep)
#
# Usage:
#   export KUBECONFIG=/Users/mimehta/kubeconfigs/kubeconfig.fozzie
#   NNODES=4 ./scripts/download-model.sh   # or sync-model-hostpath.sh
#   ./scripts/deploy-recipe-pd.sh
#   ./scripts/wait-pd-and-sweep.sh

set -euo pipefail

export KUBECONFIG="${KUBECONFIG:-/Users/mimehta/kubeconfigs/kubeconfig.fozzie}"
NS="${NS:-kimi-k3}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PREFILL_PORT="${PREFILL_PORT:-8001}"
DECODE_PORT="${DECODE_PORT:-8002}"
NIXL_PREFILL_PORT="${NIXL_PREFILL_PORT:-5557}"
NIXL_DECODE_PORT="${NIXL_DECODE_PORT:-5558}"

# Recipe: TEP16 per role (2 nodes × 8 GPUs).
TP_SIZE="${TP_SIZE:-16}"
NNODES_PER_ROLE="${NNODES_PER_ROLE:-2}"
ENABLE_EXPERT_PARALLEL="${ENABLE_EXPERT_PARALLEL:-1}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.97}"
# 1000/1000 sweep; recipe default max-model-len is 32768.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
PREFILL_MAX_NUM_SEQS="${PREFILL_MAX_NUM_SEQS:-5}"
DECODE_MAX_NUM_SEQS="${DECODE_MAX_NUM_SEQS:-16}"
VLLM_USE_RUST_FRONTEND="${VLLM_USE_RUST_FRONTEND:-0}"
VLLM_USE_V2_MODEL_RUNNER="${VLLM_USE_V2_MODEL_RUNNER:-1}"
NCCL_DMABUF_ENABLE="${NCCL_DMABUF_ENABLE:-0}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# NIXL + expandable_segments: harness patches allow expandable without CuMem
# (CuMem weight pool OOMs on tight H200 paths). Keep cumem off by default.
ENABLE_CUMEM_ALLOCATOR="${ENABLE_CUMEM_ALLOCATOR:-0}"
NO_DISABLE_HYBRID_KV_CACHE_MANAGER="${NO_DISABLE_HYBRID_KV_CACHE_MANAGER:-1}"
PREFILL_ENFORCE_EAGER="${PREFILL_ENFORCE_EAGER:-1}"

# Shrink default 1e9 NIXL buffer; keep on CPU.
KV_BUFFER_SIZE="${KV_BUFFER_SIZE:-33554432}"
KV_BUFFER_DEVICE="${KV_BUFFER_DEVICE:-cpu}"
PREFILL_KV="{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"kv_producer\",\"kv_buffer_size\":${KV_BUFFER_SIZE},\"kv_buffer_device\":\"${KV_BUFFER_DEVICE}\",\"kv_load_failure_policy\":\"fail\"}"
DECODE_KV="{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"kv_consumer\",\"kv_buffer_size\":${KV_BUFFER_SIZE},\"kv_buffer_device\":\"${KV_BUFFER_DEVICE}\",\"kv_load_failure_policy\":\"fail\"}"
DECODE_CC='{"cudagraph_mode":"FULL_DECODE_ONLY"}'

echo "==> Ensuring 4 recipe pods"
kubectl apply -f "$ROOT/manifests/vllm-recipe.yaml"
kubectl -n "$NS" scale statefulset/vllm-recipe --replicas=4
kubectl -n "$NS" rollout status statefulset/vllm-recipe --timeout=30m
for i in 0 1 2 3; do
  kubectl -n "$NS" wait --for=condition=Ready "pod/vllm-recipe-$i" --timeout=30m
done
kubectl -n "$NS" get pods -l app=vllm-recipe -o wide

echo "==> Verifying weights on all ranks"
for i in 0 1 2 3; do
  kubectl -n "$NS" exec "vllm-recipe-$i" -- bash -c \
    'test -f /models/Kimi-K3/.download-complete && du -sh /models/Kimi-K3 && echo rank-'"$i"' ok'
done

echo "==> Stopping prior serve / router"
for i in 0 1 2 3; do
  kubectl -n "$NS" cp "$ROOT/scripts/stop-inpod-vllm.sh" "vllm-recipe-$i:/tmp/stop-inpod-vllm.sh"
  kubectl -n "$NS" exec "vllm-recipe-$i" -- bash /tmp/stop-inpod-vllm.sh 2>/dev/null || true
  kubectl -n "$NS" exec "vllm-recipe-$i" -- bash -c \
    'pkill -f disagg_proxy_demo.py 2>/dev/null || true; pkill -f "vllm serve" 2>/dev/null || true' \
    2>/dev/null || true
done
sleep 4

PREFILL_IP="$(kubectl -n "$NS" get pod vllm-recipe-0 -o jsonpath='{.status.podIP}')"
DECODE_IP="$(kubectl -n "$NS" get pod vllm-recipe-2 -o jsonpath='{.status.podIP}')"
echo "==> P/D 4×8 H200 (recipe layout)"
echo "    Prefill: pods 0-1 TEP TP=${TP_SIZE} EP=${ENABLE_EXPERT_PARALLEL} :${PREFILL_PORT} master=${PREFILL_IP}"
echo "    Decode:  pods 2-3 TEP TP=${TP_SIZE} EP=${ENABLE_EXPERT_PARALLEL} :${DECODE_PORT} master=${DECODE_IP}"
echo "$PREFILL_IP" > /tmp/pd-prefill-ip
echo "$DECODE_IP" > /tmp/pd-decode-ip

for i in 0 1 2 3; do
  kubectl -n "$NS" cp "$ROOT/scripts/run-vllm-kimi-k3-recipe.sh" "vllm-recipe-$i:/tmp/run-recipe.sh"
done
kubectl -n "$NS" exec vllm-recipe-0 -- bash -c \
  'cp /vllm-workspace/examples/disaggregated/disaggregated_serving/disagg_proxy_demo.py /tmp/disagg_proxy_demo.py'

start_role_rank() {
  local pod=$1 node_rank=$2 master_addr=$3 port=$4 nixl_host=$5 nixl_port=$6
  local kv_cfg=$7 max_seqs=$8 enforce_eager=$9 compilation_cfg=${10:-}

  # P/D + hybrid Mamba: 3-read conv transfer needs DS layout.
  local ssm_layout="${VLLM_SSM_CONV_STATE_LAYOUT:-DS}"

  local env_args=(
    TP_SIZE="$TP_SIZE" PP_SIZE=1 DP_SIZE=1 NNODES="$NNODES_PER_ROLE"
    NODE_RANK="$node_rank" MASTER_ADDR="$master_addr"
    MODEL=/models/Kimi-K3
    ENABLE_EXPERT_PARALLEL="$ENABLE_EXPERT_PARALLEL"
    GPU_MEM_UTIL="$GPU_MEM_UTIL" MAX_MODEL_LEN="$MAX_MODEL_LEN"
    MAX_NUM_BATCHED_TOKENS="$MAX_NUM_BATCHED_TOKENS"
    MAX_NUM_SEQS="$max_seqs"
    VLLM_USE_RUST_FRONTEND="$VLLM_USE_RUST_FRONTEND"
    VLLM_USE_V2_MODEL_RUNNER="$VLLM_USE_V2_MODEL_RUNNER"
    VLLM_SSM_CONV_STATE_LAYOUT="$ssm_layout"
    NCCL_DMABUF_ENABLE="$NCCL_DMABUF_ENABLE"
    PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF"
    ENABLE_CUMEM_ALLOCATOR="$ENABLE_CUMEM_ALLOCATOR"
    NO_DISABLE_HYBRID_KV_CACHE_MANAGER="$NO_DISABLE_HYBRID_KV_CACHE_MANAGER"
    UCX_NET_DEVICES=all
    PORT="$port"
    KV_TRANSFER_CONFIG="$kv_cfg"
    ENFORCE_EAGER="$enforce_eager"
    VLLM_NIXL_SIDE_CHANNEL_PORT="$nixl_port"
    VLLM_NIXL_SIDE_CHANNEL_HOST="$nixl_host"
  )
  if [[ -n "$compilation_cfg" ]]; then
    env_args+=(COMPILATION_CONFIG="$compilation_cfg")
  fi

  kubectl -n "$NS" exec "$pod" -- env "${env_args[@]}" \
    bash -c 'rm -f /tmp/vllm-recipe.log; nohup bash /tmp/run-recipe.sh > /tmp/vllm-recipe.log 2>&1 & echo started=$!'
}

echo "==> Starting decode workers then head (pods 3 → 2)"
start_role_rank vllm-recipe-3 1 "$DECODE_IP" "$DECODE_PORT" "$DECODE_IP" "$NIXL_DECODE_PORT" \
  "$DECODE_KV" "$DECODE_MAX_NUM_SEQS" 0 "$DECODE_CC"
sleep 3
start_role_rank vllm-recipe-2 0 "$DECODE_IP" "$DECODE_PORT" "$DECODE_IP" "$NIXL_DECODE_PORT" \
  "$DECODE_KV" "$DECODE_MAX_NUM_SEQS" 0 "$DECODE_CC"

echo "==> Starting prefill workers then head (pods 1 → 0)"
start_role_rank vllm-recipe-1 1 "$PREFILL_IP" "$PREFILL_PORT" "$PREFILL_IP" "$NIXL_PREFILL_PORT" \
  "$PREFILL_KV" "$PREFILL_MAX_NUM_SEQS" "$PREFILL_ENFORCE_EAGER" ""
sleep 3
start_role_rank vllm-recipe-0 0 "$PREFILL_IP" "$PREFILL_PORT" "$PREFILL_IP" "$NIXL_PREFILL_PORT" \
  "$PREFILL_KV" "$PREFILL_MAX_NUM_SEQS" "$PREFILL_ENFORCE_EAGER" ""

sleep 5
echo "==> Launch banners"
for i in 0 2; do
  echo "--- vllm-recipe-$i ---"
  kubectl -n "$NS" exec "vllm-recipe-$i" -- bash -c \
    'grep -E "Launching|kv-transfer|EP=|nnodes|nixl|cumem" /tmp/vllm-recipe.log | head -20' || true
done
echo "==> Next: ./scripts/wait-pd-and-sweep.sh"
