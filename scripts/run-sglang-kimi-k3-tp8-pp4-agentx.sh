#!/usr/bin/env bash
# In-pod serve script for Kimi-K3 on SGLang — TP8×PP4 AgentX config (4×8 H200).
#
# Based on the upstream H200 high-throughput recipe adapted for PP4:
#   https://docs.sglang.io/cookbook/autoregressive/Moonshotai/Kimi-K3#hw=h200&pdMode=unified&strategy=high-throughput
#
# Includes: DSpark speculative decoding, extra_buffer_lazy mamba cache strategy,
# fused gate radix, JIT attention residual mode, 262K context.
#
# Required env:
#   DIST_INIT_ADDR     rank-0 IP
#   NODE_RANK          0..3

set -euo pipefail

MODEL="${MODEL:-/models/Kimi-K3}"
NNODES="${NNODES:-4}"
TP_SIZE="${TP_SIZE:-8}"
PP_SIZE="${PP_SIZE:-4}"
PORT="${PORT:-30000}"
DIST_PORT="${DIST_PORT:-20000}"
NODE_RANK="${NODE_RANK:-}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.90}"
MAMBA_FULL_MEMORY_RATIO="${MAMBA_FULL_MEMORY_RATIO:-0.17}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-262144}"
MAMBA_SSM_DTYPE="${MAMBA_SSM_DTYPE:-bfloat16}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-48}"
# DSpark speculative decoding (set ENABLE_DSPARK=0 to disable)
ENABLE_DSPARK="${ENABLE_DSPARK:-1}"
DSPARK_MODEL="${DSPARK_MODEL:-RadixArk/Kimi-K3-DSpark}"
DSPARK_BLOCK_SIZE="${DSPARK_BLOCK_SIZE:-7}"

if [[ "$NNODES" -gt 1 && -z "$NODE_RANK" ]]; then
  echo "NODE_RANK required when NNODES>1" >&2
  exit 1
fi
NODE_RANK="${NODE_RANK:-0}"

if [[ "$NNODES" -gt 1 ]]; then
  DIST_INIT_ADDR="${DIST_INIT_ADDR:?DIST_INIT_ADDR required for multi-node}"
else
  DIST_INIT_ADDR="${DIST_INIT_ADDR:-127.0.0.1}"
fi

MARKER="${MODEL}/.download-complete"
if [[ ! -d "$MODEL" ]]; then
  echo "MODEL path missing: $MODEL (is hostPath /models populated?)" >&2
  exit 1
fi
if [[ ! -f "$MARKER" ]]; then
  echo "Download marker missing: $MARKER" >&2
  exit 1
fi
if [[ ! -f "$MODEL/config.json" ]]; then
  echo "config.json missing under $MODEL" >&2
  exit 1
fi

# NIC detection for cross-node comms
detect_iface() {
  if [[ -n "${IFACE_NAME:-}" ]]; then
    echo "$IFACE_NAME"
    return
  fi
  local iface
  for iface in $(ls /sys/class/net 2>/dev/null | grep -E '^enp' | sort || true); do
    echo "$iface"
    return
  done
  for iface in $(ls /sys/class/net 2>/dev/null | grep -E '^(eth|bond)' || true); do
    echo "$iface"
    return
  done
  echo "eth0"
}

IFACE_NAME="$(detect_iface)"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-$IFACE_NAME}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-$IFACE_NAME}"
export SGLANG_HOST_IP="${SGLANG_HOST_IP:-$(hostname -I | awk '{print $1}')}"

# Recipe env vars (from upstream H200 high-throughput recipe)
export NCCL_MNNVL_ENABLE="${NCCL_MNNVL_ENABLE:-1}"
export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK="${SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK:-0}"
# Upstream optimizations
export SGLANG_K3_ATTN_RES_MODE="${SGLANG_K3_ATTN_RES_MODE:-jit}"
export SGLANG_MOE_FUSED_GATE_RADIX="${SGLANG_MOE_FUSED_GATE_RADIX:-1}"

# Cross-node PP P2P (RoCE IPv4 GID fix)
export NCCL_DMABUF_ENABLE="${NCCL_DMABUF_ENABLE:-0}"
export NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5_5}"
export NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-3}"

export HF_HOME="${HF_HOME:-/models/hf}"
export PYTHONUNBUFFERED=1

# ---------- Metadata banner ----------
echo "Launching SGLang serve (Kimi-K3) — AgentX PP4 config:"
echo "  MODEL=$MODEL TP=$TP_SIZE PP=$PP_SIZE"
echo "  NNODES=$NNODES NODE_RANK=$NODE_RANK DIST_INIT_ADDR=$DIST_INIT_ADDR:$DIST_PORT"
echo "  IFACE=$GLOO_SOCKET_IFNAME HOST_IP=$SGLANG_HOST_IP"
echo "  mem-fraction-static=$MEM_FRACTION_STATIC mamba-full-memory-ratio=$MAMBA_FULL_MEMORY_RATIO"
echo "  context-length=$CONTEXT_LENGTH mamba-ssm-dtype=$MAMBA_SSM_DTYPE max-running-requests=$MAX_RUNNING_REQUESTS"
echo "  moe-runner-backend=marlin decode-attention-backend=flashmla enable-symm-mem=true"
echo "  mamba-radix-cache-strategy=extra_buffer_lazy dist-timeout=3600"
echo "  dspark=$ENABLE_DSPARK model=$DSPARK_MODEL block-size=$DSPARK_BLOCK_SIZE"
echo "  port=$PORT"
echo "  NCCL_MNNVL_ENABLE=$NCCL_MNNVL_ENABLE NCCL_CUMEM_ENABLE=$NCCL_CUMEM_ENABLE"
echo "  NCCL_DMABUF_ENABLE=$NCCL_DMABUF_ENABLE NCCL_IB_HCA=$NCCL_IB_HCA NCCL_IB_GID_INDEX=$NCCL_IB_GID_INDEX"
echo "  PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"
echo "  SGLANG_K3_ATTN_RES_MODE=$SGLANG_K3_ATTN_RES_MODE SGLANG_MOE_FUSED_GATE_RADIX=$SGLANG_MOE_FUSED_GATE_RADIX"
cat "$MARKER" || true

# ---------- Build serve args ----------
SERVE_ARGS=(
  --trust-remote-code
  --model-path "$MODEL"
  --tp-size "$TP_SIZE"
  --pp-size "$PP_SIZE"
  --nnodes "$NNODES"
  --node-rank "$NODE_RANK"
  --dist-init-addr "${DIST_INIT_ADDR}:${DIST_PORT}"
  --moe-runner-backend marlin
  --decode-attention-backend flashmla
  --enable-symm-mem
  --mem-fraction-static "$MEM_FRACTION_STATIC"
  --reasoning-parser kimi_k3
  --tool-call-parser kimi_k3
  --mamba-full-memory-ratio "$MAMBA_FULL_MEMORY_RATIO"
  --mamba-ssm-dtype "$MAMBA_SSM_DTYPE"
  --mamba-radix-cache-strategy extra_buffer_lazy
  --context-length "$CONTEXT_LENGTH"
  --max-running-requests "$MAX_RUNNING_REQUESTS"
  --dist-timeout 3600
  --enable-cache-report
  --host 0.0.0.0
  --port "$PORT"
)

# DSpark speculative decoding
if [[ "$ENABLE_DSPARK" == "1" ]]; then
  SERVE_ARGS+=(
    --speculative-algorithm DSPARK
    --speculative-draft-model-path "$DSPARK_MODEL"
    --speculative-dspark-block-size "$DSPARK_BLOCK_SIZE"
    --enable-linear-replayssm-spec
  )
fi

# ---------- Run ----------
exec sglang serve "${SERVE_ARGS[@]}"
