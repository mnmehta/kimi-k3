#!/usr/bin/env bash
# Deploy multi-node DUMMY-weight recipe with TP=8, DP=2 (same 2×8 GPU STS).
# Recipe strategy: Multi-Node TP + Data Parallel
#   https://recipes.vllm.ai/moonshotai/Kimi-K3?strategy=multi_node_tp_dp
#
# Usage:
#   export KUBECONFIG=/Users/mimehta/kubeconfigs/kubeconfig.fozzie
#   ./scripts/deploy-recipe-dummy-dp.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export TP_SIZE="${TP_SIZE:-8}"
export PP_SIZE="${PP_SIZE:-1}"
export DP_SIZE="${DP_SIZE:-2}"
export DP_SIZE_LOCAL="${DP_SIZE_LOCAL:-1}"
export NNODES="${NNODES:-2}"
export DP_RPC_PORT="${DP_RPC_PORT:-13345}"
# Nested --hf-overrides JSON breaks the Rust frontend --args-json parser.
# Keep Python frontend for shallow dummy / profiler sweeps (override with =1 if needed).
export VLLM_USE_RUST_FRONTEND="${VLLM_USE_RUST_FRONTEND:-0}"

echo "==> Parallelism: TP=${TP_SIZE} DP=${DP_SIZE} DP_LOCAL=${DP_SIZE_LOCAL} NNODES=${NNODES} (dummy) RUST_FRONTEND=${VLLM_USE_RUST_FRONTEND}"
exec "$ROOT/scripts/deploy-recipe-dummy.sh"
