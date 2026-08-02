#!/usr/bin/env bash
# Copy a finished in-pod sweep RESULT_DIR to a local bench-results/ folder,
# and always archive both ranks' vLLM serve logs alongside the metrics.
#
# Usage:
#   RESULT=/tmp/vllm-bench-sweep-pp2-256 \
#   LOCAL_OUT=bench-results/conc-sweep-pp2-1000-1000 \
#   ./scripts/copy-sweep-results.sh
#
# Optional:
#   NS=kimi-k3
#   NNODES=2

set -euo pipefail

export KUBECONFIG="${KUBECONFIG:-/Users/mimehta/kubeconfigs/kubeconfig.fozzie}"
NS="${NS:-kimi-k3}"
NNODES="${NNODES:-4}"
RESULT="${RESULT:?RESULT required (in-pod sweep dir, e.g. /tmp/vllm-bench-sweep-pp2-256)}"
LOCAL_OUT="${LOCAL_OUT:?LOCAL_OUT required (host bench-results path)}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Allow relative LOCAL_OUT from repo root
if [[ "$LOCAL_OUT" != /* ]]; then
  LOCAL_OUT="$ROOT/$LOCAL_OUT"
fi

mkdir -p "$LOCAL_OUT"

echo "==> Copying sweep results from $RESULT -> $LOCAL_OUT"
kubectl -n "$NS" exec vllm-recipe-0 -- tar -C "$RESULT" -czf - . > /tmp/sweep-results.tgz
tar -C "$LOCAL_OUT" -xzf /tmp/sweep-results.tgz

echo "==> Copying vLLM serve logs (all ranks)"
for i in $(seq 0 $((NNODES - 1))); do
  out="$LOCAL_OUT/vllm-recipe-$i.log"
  if kubectl -n "$NS" exec "vllm-recipe-$i" -- test -f /tmp/vllm-recipe.log; then
    kubectl -n "$NS" exec "vllm-recipe-$i" -- cat /tmp/vllm-recipe.log >"$out"
    echo "    rank-$i -> $out ($(wc -c <"$out") bytes)"
  else
    echo "    rank-$i: /tmp/vllm-recipe.log missing" >&2
  fi
done

echo "==> Done"
ls -la "$LOCAL_OUT" | head -40
