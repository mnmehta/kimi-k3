#!/usr/bin/env bash
# Deploy multi-node DUMMY-weight recipe with TP=16 + Expert Parallel (TEP16).
# Recipe strategy: Multi-Node Tensor + Expert Parallel
#   https://recipes.vllm.ai/moonshotai/Kimi-K3?strategy=multi_node_tep
#
# Usage:
#   export KUBECONFIG=...
#   ENABLE_TORCH_PROFILER=1 PROFILE_DIR=/tmp/vllm_profile/tep16 \
#     NUM_LAYERS=4 NUM_EXPERTS=8 NUM_EXPERTS_PER_TOKEN=2 NUM_SHARED_EXPERTS=1 \
#     MAX_MODEL_LEN=1024 ./scripts/deploy-recipe-dummy-tep.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export TP_SIZE="${TP_SIZE:-16}"
export PP_SIZE="${PP_SIZE:-1}"
export DP_SIZE="${DP_SIZE:-1}"
export NNODES="${NNODES:-2}"
export ENABLE_EXPERT_PARALLEL="${ENABLE_EXPERT_PARALLEL:-1}"

echo "==> Parallelism: TEP TP=${TP_SIZE} EP=${ENABLE_EXPERT_PARALLEL} NNODES=${NNODES} (dummy)"
exec "$ROOT/scripts/deploy-recipe-dummy.sh"
