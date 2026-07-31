#!/usr/bin/env bash
# Deploy multi-node DUMMY-weight recipe with TP=8, PP=2 (same 2×8 GPU STS).
# Recipe strategy: Multi-Node TP + Pipeline Parallel
#
# Usage:
#   export KUBECONFIG=/Users/mimehta/kubeconfigs/kubeconfig.fozzie
#   ./scripts/deploy-recipe-dummy-pp.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export TP_SIZE="${TP_SIZE:-8}"
export PP_SIZE="${PP_SIZE:-2}"
export NNODES="${NNODES:-2}"

echo "==> Parallelism: TP=${TP_SIZE} PP=${PP_SIZE} NNODES=${NNODES} (dummy)"
exec "$ROOT/scripts/deploy-recipe-dummy.sh"
