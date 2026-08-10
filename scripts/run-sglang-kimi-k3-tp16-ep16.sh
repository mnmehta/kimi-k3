#!/usr/bin/env bash
# In-pod serve script for Kimi-K3 on SGLang — H200 low-latency (TP16+EP16).
#
# Exact command from:
#   https://docs.sglang.io/cookbook/autoregressive/Moonshotai/Kimi-K3#hw=h200&pdMode=unified&strategy=low-latency&spec=none&hicache=off
#
#   NCCL_MNNVL_ENABLE=1 NCCL_CUMEM_ENABLE=1 SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0 \
#   sglang serve \
#     --trust-remote-code --model-path moonshotai/Kimi-K3 \
#     --tp-size 16 --nnodes 2 --node-rank <N> --dist-init-addr <ip>:20000 \
#     --ep-size 16 --moe-runner-backend marlin --decode-attention-backend flashmla \
#     --enable-symm-mem --mem-fraction-static 0.85 \
#     --reasoning-parser kimi_k3 --tool-call-parser kimi_k3 \
#     --mamba-full-memory-ratio 0.45 --host 0.0.0.0 --port 30000
#
# Required env:
#   DIST_INIT_ADDR     rank-0 IP
#   NODE_RANK          0 or 1

set -euo pipefail

MODEL="${MODEL:-/models/Kimi-K3}"
NNODES="${NNODES:-2}"
TP_SIZE="${TP_SIZE:-16}"
EP_SIZE="${EP_SIZE:-16}"
PORT="${PORT:-30000}"
DIST_PORT="${DIST_PORT:-20000}"
NODE_RANK="${NODE_RANK:-}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.90}"
MAMBA_FULL_MEMORY_RATIO="${MAMBA_FULL_MEMORY_RATIO:-0.90}"
#KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8_e4m3}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-8192}"
MAMBA_SSM_DTYPE="${MAMBA_SSM_DTYPE:-bfloat16}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-256}"

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

# Recipe env vars (from H200 docs)
export NCCL_MNNVL_ENABLE="${NCCL_MNNVL_ENABLE:-1}"
export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-1}"
export SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK="${SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK:-0}"

# Cross-node comms
[[ -n "${NCCL_IB_HCA:-}" ]] && export NCCL_IB_HCA
[[ -n "${NCCL_IB_GID_INDEX:-}" ]] && export NCCL_IB_GID_INDEX

export HF_HOME="${HF_HOME:-/models/hf}"
export PYTHONUNBUFFERED=1

# ---------- Metadata banner ----------
echo "Launching SGLang serve (Kimi-K3) — H200 low-latency TP16+EP16:"
echo "  MODEL=$MODEL TP=$TP_SIZE EP=$EP_SIZE"
echo "  NNODES=$NNODES NODE_RANK=$NODE_RANK DIST_INIT_ADDR=$DIST_INIT_ADDR:$DIST_PORT"
echo "  IFACE=$GLOO_SOCKET_IFNAME HOST_IP=$SGLANG_HOST_IP"
echo "  mem-fraction-static=$MEM_FRACTION_STATIC mamba-full-memory-ratio=$MAMBA_FULL_MEMORY_RATIO kv-cache-dtype=$KV_CACHE_DTYPE"
echo "  context-length=$CONTEXT_LENGTH mamba-ssm-dtype=$MAMBA_SSM_DTYPE max-running-requests=$MAX_RUNNING_REQUESTS"
echo "  moe-runner-backend=marlin decode-attention-backend=flashmla enable-symm-mem=true"
echo "  port=$PORT"
echo "  NCCL_MNNVL_ENABLE=$NCCL_MNNVL_ENABLE NCCL_CUMEM_ENABLE=$NCCL_CUMEM_ENABLE"
echo "  NCCL_IB_HCA=${NCCL_IB_HCA:-<unset>} NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-<unset>}"
echo "  SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=$SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK"
cat "$MARKER" || true

# ---------- Run ----------
exec sglang serve \
  --trust-remote-code \
  --model-path "$MODEL" \
  --tp-size "$TP_SIZE" \
  --nnodes "$NNODES" \
  --node-rank "$NODE_RANK" \
  --dist-init-addr "${DIST_INIT_ADDR}:${DIST_PORT}" \
  --ep-size "$EP_SIZE" \
  --moe-runner-backend marlin \
  --decode-attention-backend flashmla \
  --enable-symm-mem \
  --mem-fraction-static "$MEM_FRACTION_STATIC" \
  --reasoning-parser kimi_k3 \
  --tool-call-parser kimi_k3 \
  --kv-cache-dtype "$KV_CACHE_DTYPE" \
  --mamba-ssm-dtype "$MAMBA_SSM_DTYPE" \
  --mamba-full-memory-ratio "$MAMBA_FULL_MEMORY_RATIO" \
  --context-length "$CONTEXT_LENGTH" \
  --max-running-requests "$MAX_RUNNING_REQUESTS" \
  --host 0.0.0.0 \
  --port "$PORT"
