#!/usr/bin/env bash
# Deploy multi-node REAL-weight recipe with TP=8, DP=2 (same 2×8 GPU STS).
# Recipe strategy: Multi-Node TP + Data Parallel
#   https://recipes.vllm.ai/moonshotai/Kimi-K3?strategy=multi_node_tp_dp
#
# One full model replica per node (TP inside the node; DP across nodes).
# Reuses manifests/vllm-recipe.yaml + scripts/run-vllm-kimi-k3-recipe.sh.
#
# Usage:
#   export KUBECONFIG=/Users/mimehta/kubeconfigs/kubeconfig.fozzie
#   ./scripts/deploy-recipe-dp.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export TP_SIZE="${TP_SIZE:-8}"
export PP_SIZE="${PP_SIZE:-1}"
export DP_SIZE="${DP_SIZE:-2}"
export DP_SIZE_LOCAL="${DP_SIZE_LOCAL:-1}"
export NNODES="${NNODES:-2}"
export DP_RPC_PORT="${DP_RPC_PORT:-13345}"
# Recipe enables Rust frontend + V2 runner; leave off by default (override OK).
export VLLM_USE_RUST_FRONTEND="${VLLM_USE_RUST_FRONTEND:-0}"
export VLLM_USE_V2_MODEL_RUNNER="${VLLM_USE_V2_MODEL_RUNNER:-0}"
export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.97}"
# H200 TP8 replica is ~135.7 GiB — only ~150–200 MiB free after load.
# Default FlashInfer workspace is ~394 MiB (won't fit). Shrink workspace +
# batched tokens so profile_run can size MoE scratch; pin a small KV so
# 1000/1000 still fits (max-model-len 2048).
export VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE="${VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE:-67108864}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-2}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-256}"
# ~0.15 GiB — need ≥0.12 GiB for max-model-len 2048 (1000/1000).
export KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES:-160000000}"
# Kimi-K3 mm encoder profile alone tries ~396 MiB — more than H200 free after TP8 weights.
export SKIP_MM_PROFILING="${SKIP_MM_PROFILING:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NCCL_DMABUF_ENABLE="${NCCL_DMABUF_ENABLE:-0}"

echo "==> Parallelism: TP=${TP_SIZE} DP=${DP_SIZE} DP_LOCAL=${DP_SIZE_LOCAL} NNODES=${NNODES}"
exec "$ROOT/scripts/deploy-recipe.sh"
