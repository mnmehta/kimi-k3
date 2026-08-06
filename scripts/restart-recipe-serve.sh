#!/usr/bin/env bash
# Restart vLLM serve in place (no StatefulSet pod recreate). Clears GPU/CPU KV
# / prefix cache by process restart. Same recipe merge as deploy.sh.
#
# Usage:
#   ./scripts/restart-recipe-serve.sh pp2-humming-agentx-kv-offload
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export KUBECONFIG="${KUBECONFIG:-/Users/mimehta/kubeconfigs/kubeconfig.fozzie}"
export DEPLOY_SKIP_POD_ROLLOUT=1
# Weights already verified on first bring-up.
export DEPLOY_VERIFY_WEIGHTS="${DEPLOY_VERIFY_WEIGHTS:-0}"

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <recipe|config.yaml> [more.yaml...]" >&2
  exit 2
fi

exec bash "$ROOT/scripts/deploy.sh" "$@"
