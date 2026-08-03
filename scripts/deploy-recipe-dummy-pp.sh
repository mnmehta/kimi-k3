#!/usr/bin/env bash
# Deploy multi-node DUMMY-weight recipe with TP=8, PP=2 (same 2×8 GPU STS).
# Recipe strategy: Multi-Node TP + Pipeline Parallel
#
# Usage:
#   export KUBECONFIG=...
#   ./scripts/deploy-recipe-dummy-pp.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export TP_SIZE="${TP_SIZE:-8}"
export PP_SIZE="${PP_SIZE:-2}"
export NNODES="${NNODES:-2}"
# Same cross-node PP RoCE fix as real-weight deploy-recipe-pp.sh.
export NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5_5}"
export NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-3}"
unset NCCL_IB_DISABLE || true
export NCCL_DMABUF_ENABLE="${NCCL_DMABUF_ENABLE:-0}"

echo "==> Parallelism: TP=${TP_SIZE} PP=${PP_SIZE} NNODES=${NNODES} (dummy) NCCL_IB_HCA=${NCCL_IB_HCA} GID=${NCCL_IB_GID_INDEX}"
exec "$ROOT/scripts/deploy-recipe-dummy.sh"
