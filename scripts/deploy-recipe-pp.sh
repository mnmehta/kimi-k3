#!/usr/bin/env bash
# Deploy multi-node REAL-weight recipe with TP=8, PP=2 (same 2×8 GPU STS).
# Recipe strategy: Multi-Node TP + Pipeline Parallel
#   https://recipes.vllm.ai/moonshotai/Kimi-K3
#
# Reuses manifests/vllm-recipe.yaml + scripts/run-vllm-kimi-k3-recipe.sh.
# Stops any prior TP16 serve on those pods before starting.
#
# Usage:
#   export KUBECONFIG=...
#   ./scripts/deploy-recipe-pp.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export TP_SIZE="${TP_SIZE:-8}"
export PP_SIZE="${PP_SIZE:-2}"
export NNODES="${NNODES:-2}"
# Cross-node PP P2P failed on default RoCE IPv6 link-local GIDs (mlx5_0 /
# fe80::… → IBV_WC_RETRY_EXC_ERR). Prefer the IPv4-mapped RoCEv2 GID on mlx5_5.
export NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5_5}"
export NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-3}"
# Do NOT set NCCL_IB_DISABLE — keep InfiniBand/RoCE enabled.
unset NCCL_IB_DISABLE || true
# PP frees headroom vs TP16; raise seq budget for throughput sweeps.
export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-}"
export NCCL_DMABUF_ENABLE="${NCCL_DMABUF_ENABLE:-0}"

echo "==> Parallelism: TP=${TP_SIZE} PP=${PP_SIZE} NNODES=${NNODES} NCCL_IB_HCA=${NCCL_IB_HCA} NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX}"
exec "$ROOT/scripts/deploy-recipe.sh"
