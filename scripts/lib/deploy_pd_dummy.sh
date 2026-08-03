#!/usr/bin/env bash
# Deploy Prefill/Decode disaggregation (pd_cluster) with DUMMY weights on 4×8.
# Prefill: pods 0–1 TEP16 :8001 kv_producer
# Decode:  pods 2–3 TEP16 :8002 kv_consumer
# Router:  disagg_proxy_demo on prefill head :8000
#
# Usage:
#   export KUBECONFIG=...
#   ENABLE_TORCH_PROFILER=1 PROFILE_DIR=/tmp/vllm_profile/pd \
#     NUM_LAYERS=4 NUM_EXPERTS=8 NUM_EXPERTS_PER_TOKEN=2 NUM_SHARED_EXPERTS=1 \
#     MAX_MODEL_LEN=1024 ./scripts/deploy-recipe-dummy-pd.sh

set -euo pipefail

NS="${NS:-kimi-k3}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PREFILL_PORT="${PREFILL_PORT:-8001}"
DECODE_PORT="${DECODE_PORT:-8002}"
NIXL_PREFILL_PORT="${NIXL_PREFILL_PORT:-5557}"
NIXL_DECODE_PORT="${NIXL_DECODE_PORT:-5558}"
ROUTER_PORT="${ROUTER_PORT:-8000}"

TP_SIZE="${TP_SIZE:-16}"
NNODES_PER_ROLE="${NNODES_PER_ROLE:-2}"
ENABLE_EXPERT_PARALLEL="${ENABLE_EXPERT_PARALLEL:-1}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1024}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-1024}"
PREFILL_MAX_NUM_SEQS="${PREFILL_MAX_NUM_SEQS:-5}"
DECODE_MAX_NUM_SEQS="${DECODE_MAX_NUM_SEQS:-16}"
NUM_LAYERS="${NUM_LAYERS:-4}"
# EP16 requires num_experts % 16 == 0 (8 experts from single-pod smoke is too small).
NUM_EXPERTS="${NUM_EXPERTS:-32}"
NUM_EXPERTS_PER_TOKEN="${NUM_EXPERTS_PER_TOKEN:-2}"
NUM_SHARED_EXPERTS="${NUM_SHARED_EXPERTS:-1}"
ENABLE_TORCH_PROFILER="${ENABLE_TORCH_PROFILER:-0}"
PROFILE_DIR="${PROFILE_DIR:-/tmp/vllm_profile/pd}"
VLLM_RPC_TIMEOUT="${VLLM_RPC_TIMEOUT:-1800000}"
VLLM_USE_RUST_FRONTEND="${VLLM_USE_RUST_FRONTEND:-0}"
VLLM_USE_V2_MODEL_RUNNER="${VLLM_USE_V2_MODEL_RUNNER:-1}"
NCCL_DMABUF_ENABLE="${NCCL_DMABUF_ENABLE:-0}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
NO_DISABLE_HYBRID_KV_CACHE_MANAGER="${NO_DISABLE_HYBRID_KV_CACHE_MANAGER:-1}"
PREFILL_ENFORCE_EAGER="${PREFILL_ENFORCE_EAGER:-1}"
MODEL="${MODEL:-moonshotai/Kimi-K3}"

KV_BUFFER_SIZE="${KV_BUFFER_SIZE:-33554432}"
KV_BUFFER_DEVICE="${KV_BUFFER_DEVICE:-cpu}"
PREFILL_KV="{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"kv_producer\",\"kv_buffer_size\":${KV_BUFFER_SIZE},\"kv_buffer_device\":\"${KV_BUFFER_DEVICE}\",\"kv_load_failure_policy\":\"fail\"}"
DECODE_KV="{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"kv_consumer\",\"kv_buffer_size\":${KV_BUFFER_SIZE},\"kv_buffer_device\":\"${KV_BUFFER_DEVICE}\",\"kv_load_failure_policy\":\"fail\"}"
DECODE_CC='{"cudagraph_mode":"FULL_DECODE_ONLY"}'

echo "==> Ensuring 4 recipe pods (dummy P/D)"
kubectl apply -f "$ROOT/manifests/vllm-recipe.yaml"
kubectl -n "$NS" scale statefulset/vllm-recipe --replicas=4
kubectl -n "$NS" rollout status statefulset/vllm-recipe --timeout=30m
for i in 0 1 2 3; do
  kubectl -n "$NS" wait --for=condition=Ready "pod/vllm-recipe-$i" --timeout=30m
done
kubectl -n "$NS" get pods -l app=vllm-recipe -o wide

echo "==> Stopping prior serve / router"
for i in 0 1 2 3; do
  kubectl -n "$NS" cp "$ROOT/scripts/stop-inpod-vllm.sh" "vllm-recipe-$i:/tmp/stop-inpod-vllm.sh"
  kubectl -n "$NS" exec "vllm-recipe-$i" -- bash /tmp/stop-inpod-vllm.sh 2>/dev/null || true
  # Kill router by pidfile only (never pkill -f pattern in argv).
  kubectl -n "$NS" exec "vllm-recipe-$i" -- bash -c \
    'if [[ -f /tmp/pd-router.pid ]]; then kill "$(cat /tmp/pd-router.pid)" 2>/dev/null || true; fi' \
    2>/dev/null || true
done
sleep 4

PREFILL_IP="$(kubectl -n "$NS" get pod vllm-recipe-0 -o jsonpath='{.status.podIP}')"
DECODE_IP="$(kubectl -n "$NS" get pod vllm-recipe-2 -o jsonpath='{.status.podIP}')"
echo "==> Dummy P/D 4×8"
echo "    Prefill: pods 0-1 TEP TP=${TP_SIZE} EP=${ENABLE_EXPERT_PARALLEL} :${PREFILL_PORT}"
echo "    Decode:  pods 2-3 TEP TP=${TP_SIZE} EP=${ENABLE_EXPERT_PARALLEL} :${DECODE_PORT}"
echo "    hf-overrides layers=${NUM_LAYERS} experts=${NUM_EXPERTS}/${NUM_EXPERTS_PER_TOKEN}/${NUM_SHARED_EXPERTS}"
echo "    profiler=${ENABLE_TORCH_PROFILER} PROFILE_DIR=${PROFILE_DIR}"
echo "$PREFILL_IP" > /tmp/pd-prefill-ip
echo "$DECODE_IP" > /tmp/pd-decode-ip

for i in 0 1 2 3; do
  kubectl -n "$NS" cp "$ROOT/scripts/run-vllm-kimi-k3-recipe-dummy.sh" \
    "vllm-recipe-$i:/tmp/run-recipe-dummy.sh"
done
kubectl -n "$NS" exec vllm-recipe-0 -- bash -c \
  'cp /vllm-workspace/examples/disaggregated/disaggregated_serving/disagg_proxy_demo.py /tmp/disagg_proxy_demo.py'

start_role_rank() {
  local pod=$1 node_rank=$2 master_addr=$3 port=$4 nixl_host=$5 nixl_port=$6
  local kv_cfg=$7 max_seqs=$8 enforce_eager=$9 compilation_cfg=${10:-}

  local env_args=(
    MODEL="$MODEL"
    TP_SIZE="$TP_SIZE" PP_SIZE=1 DP_SIZE=1 NNODES="$NNODES_PER_ROLE"
    NODE_RANK="$node_rank" MASTER_ADDR="$master_addr"
    ENABLE_EXPERT_PARALLEL="$ENABLE_EXPERT_PARALLEL"
    GPU_MEM_UTIL="$GPU_MEM_UTIL" MAX_MODEL_LEN="$MAX_MODEL_LEN"
    MAX_NUM_BATCHED_TOKENS="$MAX_NUM_BATCHED_TOKENS"
    MAX_NUM_SEQS="$max_seqs"
    NUM_LAYERS="$NUM_LAYERS" NUM_EXPERTS="$NUM_EXPERTS"
    NUM_EXPERTS_PER_TOKEN="$NUM_EXPERTS_PER_TOKEN"
    NUM_SHARED_EXPERTS="$NUM_SHARED_EXPERTS"
    VLLM_USE_RUST_FRONTEND="$VLLM_USE_RUST_FRONTEND"
    VLLM_USE_V2_MODEL_RUNNER="$VLLM_USE_V2_MODEL_RUNNER"
    NCCL_DMABUF_ENABLE="$NCCL_DMABUF_ENABLE"
    PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF"
    NO_DISABLE_HYBRID_KV_CACHE_MANAGER="$NO_DISABLE_HYBRID_KV_CACHE_MANAGER"
    ENABLE_TORCH_PROFILER="$ENABLE_TORCH_PROFILER"
    PROFILE_DIR="$PROFILE_DIR"
    VLLM_RPC_TIMEOUT="$VLLM_RPC_TIMEOUT"
    VLLM_SSM_CONV_STATE_LAYOUT=DS
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
    bash -c 'rm -f /tmp/vllm-recipe.log; nohup bash /tmp/run-recipe-dummy.sh > /tmp/vllm-recipe.log 2>&1 & echo started=$!'
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

echo "==> Waiting for prefill/decode /health"
for n in $(seq 1 180); do
  p_ok=0; d_ok=0
  kubectl -n "$NS" exec vllm-recipe-0 -- curl -sf -m 2 "http://127.0.0.1:${PREFILL_PORT}/health" >/dev/null 2>&1 && p_ok=1
  kubectl -n "$NS" exec vllm-recipe-2 -- curl -sf -m 2 "http://127.0.0.1:${DECODE_PORT}/health" >/dev/null 2>&1 && d_ok=1
  if [[ $p_ok -eq 1 && $d_ok -eq 1 ]]; then
    echo "Healthy poll=$n"
    break
  fi
  if (( n % 6 == 0 )); then
    echo "still loading poll=$n p=$p_ok d=$d_ok"
    for rank in 0 1 2 3; do
      kubectl -n "$NS" exec "vllm-recipe-$rank" -- bash -c \
        'grep -E "Launching|Application startup|Error|OOM|AssertionError" /tmp/vllm-recipe.log | tail -4' 2>/dev/null || true
    done
  fi
  for rank in 0 1 2 3; do
    if ! kubectl -n "$NS" exec "vllm-recipe-$rank" -- pgrep -f 'vllm serve' >/dev/null 2>&1; then
      if kubectl -n "$NS" exec "vllm-recipe-$rank" -- grep -qE 'CUDA out of memory|Engine core initialization failed|AssertionError|ValueError|ValidationError' /tmp/vllm-recipe.log 2>/dev/null; then
        echo "serve died on rank=$rank" >&2
        kubectl -n "$NS" exec "vllm-recipe-$rank" -- tail -40 /tmp/vllm-recipe.log >&2 || true
        exit 1
      fi
    fi
  done
  sleep 10
done
kubectl -n "$NS" exec vllm-recipe-0 -- curl -sf -m 2 "http://127.0.0.1:${PREFILL_PORT}/health" >/dev/null
kubectl -n "$NS" exec vllm-recipe-2 -- curl -sf -m 2 "http://127.0.0.1:${DECODE_PORT}/health" >/dev/null

echo "==> Starting router :$ROUTER_PORT"
kubectl -n "$NS" exec vllm-recipe-0 -- bash -c "
  if [[ -f /tmp/pd-router.pid ]]; then kill \"\$(cat /tmp/pd-router.pid)\" 2>/dev/null || true; fi
  nohup python3 /tmp/disagg_proxy_demo.py \
    --model ${MODEL} \
    --prefill ${PREFILL_IP}:${PREFILL_PORT} \
    --decode ${DECODE_IP}:${DECODE_PORT} \
    --port ${ROUTER_PORT} \
    > /tmp/pd-router.log 2>&1 &
  echo \$! > /tmp/pd-router.pid
  echo router=\$(cat /tmp/pd-router.pid)
"
sleep 3
echo "==> Prefill models:"
kubectl -n "$NS" exec vllm-recipe-0 -- curl -sS "http://127.0.0.1:${PREFILL_PORT}/v1/models" | head -c 400 || true
echo
echo "==> Dummy P/D ready (router :${ROUTER_PORT}, profile via profile-pd-short-query.sh)"
