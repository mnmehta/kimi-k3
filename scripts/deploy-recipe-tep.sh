#!/usr/bin/env bash
# Deploy multi-node REAL-weight recipe with TP=16 + Expert Parallel (TEP16).
# Recipe strategy: Multi-Node Tensor + Expert Parallel
#   https://recipes.vllm.ai/moonshotai/Kimi-K3?strategy=multi_node_tep
#
# Same 2×8 GPU STS as TP16; adds --enable-expert-parallel.
# Reuses manifests/vllm-recipe.yaml + scripts/run-vllm-kimi-k3-recipe.sh.
#
# Usage:
#   export KUBECONFIG=/Users/mimehta/kubeconfigs/kubeconfig.fozzie
#   ./scripts/deploy-recipe-tep.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export TP_SIZE="${TP_SIZE:-16}"
export PP_SIZE="${PP_SIZE:-1}"
export DP_SIZE="${DP_SIZE:-1}"
export NNODES="${NNODES:-2}"
export ENABLE_EXPERT_PARALLEL="${ENABLE_EXPERT_PARALLEL:-1}"
# Match TP16 real-weight sweep knobs for an apples-to-apples compare.
export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.97}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-128}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
export VLLM_USE_RUST_FRONTEND="${VLLM_USE_RUST_FRONTEND:-0}"
export VLLM_USE_V2_MODEL_RUNNER="${VLLM_USE_V2_MODEL_RUNNER:-1}"
export NCCL_DMABUF_ENABLE="${NCCL_DMABUF_ENABLE:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "==> Parallelism: TEP TP=${TP_SIZE} EP=${ENABLE_EXPERT_PARALLEL} NNODES=${NNODES}"
exec "$ROOT/scripts/deploy-recipe.sh"
