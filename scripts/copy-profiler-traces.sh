#!/usr/bin/env bash
# Copy torch.profiler traces from recipe pods to host profiler-results/<CONFIG>/.
#
# Usage:
#   CONFIG=tp16 PROFILE_DIR=/tmp/vllm_profile/tp16 NNODES=2 \
#     ./scripts/copy-profiler-traces.sh
#
# Optional:
#   META_EXTRA='deploy=./scripts/deploy-recipe-dummy.sh'

set -euo pipefail

NS="${NS:-kimi-k3}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${CONFIG:?CONFIG required (e.g. tp16)}"
PROFILE_DIR="${PROFILE_DIR:?PROFILE_DIR required (in-pod path)}"
NNODES="${NNODES:-2}"
OUT_ROOT="${OUT_ROOT:-$ROOT/profiler-results}"
OUT="$OUT_ROOT/$CONFIG"

mkdir -p "$OUT"
echo "==> Copying profiler traces CONFIG=$CONFIG NNODES=$NNODES"
echo "    PROFILE_DIR=$PROFILE_DIR -> $OUT"

for i in $(seq 0 $((NNODES - 1))); do
  dest="$OUT/rank-$i"
  mkdir -p "$dest"
  # tar stream whatever exists under PROFILE_DIR
  if kubectl -n "$NS" exec "vllm-recipe-$i" -- bash -c "test -d '$PROFILE_DIR' && ls -A '$PROFILE_DIR' | grep -q ."; then
    kubectl -n "$NS" exec "vllm-recipe-$i" -- tar -C "$PROFILE_DIR" -czf - . >"/tmp/profiler-${CONFIG}-rank-${i}.tgz"
    tar -C "$dest" -xzf "/tmp/profiler-${CONFIG}-rank-${i}.tgz"
    echo "    rank-$i traces: $(find "$dest" -type f | wc -l | tr -d ' ') files"
  else
    echo "    rank-$i: empty or missing $PROFILE_DIR" >&2
  fi
  # Archive serve log before the next config overwrites /tmp/vllm-recipe.log
  log_out="$OUT/vllm-recipe-$i.log"
  if kubectl -n "$NS" exec "vllm-recipe-$i" -- test -f /tmp/vllm-recipe.log; then
    kubectl -n "$NS" exec "vllm-recipe-$i" -- cat /tmp/vllm-recipe.log >"$log_out"
    echo "    rank-$i serve log -> $log_out ($(wc -c <"$log_out" | tr -d ' ') bytes)"
  else
    echo "    rank-$i: /tmp/vllm-recipe.log missing" >&2
  fi
done

{
  echo "config=$CONFIG"
  echo "utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "git_sha=$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo n/a)"
  echo "profile_dir=$PROFILE_DIR"
  echo "nnodes=$NNODES"
  echo "num_layers=${NUM_LAYERS:-}"
  echo "num_experts=${NUM_EXPERTS:-}"
  echo "num_experts_per_token=${NUM_EXPERTS_PER_TOKEN:-}"
  echo "num_shared_experts=${NUM_SHARED_EXPERTS:-}"
  echo "max_model_len=${MAX_MODEL_LEN:-}"
  echo "prompt=10/5 warmup=1"
  echo "meta_extra=${META_EXTRA:-}"
  echo
  echo "== non-default args (rank-0 log) =="
  kubectl -n "$NS" exec vllm-recipe-0 -- bash -c \
    'grep -E "Launching|hf-overrides|non-default args|profiler" /tmp/vllm-recipe.log | head -40' 2>/dev/null || true
} >"$OUT/meta.txt"

echo "==> Host layout"
find "$OUT" -type f | head -80
gz=$(find "$OUT" -name '*.pt.trace.json.gz' | wc -l | tr -d ' ')
echo "gzipped traces: $gz"
if [[ "$gz" -lt 1 ]]; then
  echo "WARNING: no *.pt.trace.json.gz found under $OUT" >&2
  exit 1
fi
echo "Done: $OUT"
