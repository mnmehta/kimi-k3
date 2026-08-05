#!/usr/bin/env bash
# Run NVIDIA AIPerf InferenceX AgentX MVP against the local recipe serve.
# Docs: https://docs.nvidia.com/aiperf/dev/tutorials/datasets-inputs/inference-x-agent-x-mvp-benchmark
#
# Prerequisites:
#   - ./scripts/deploy.sh pp2-humming-agentx  (256k max-model-len)
#   - kubectl -n kimi-k3 port-forward pod/vllm-recipe-0 8000:8000
#   - AgentX needs AIPerf ≥ 0.12.dev, e.g.:
#       reports/.venv/bin/pip install 'aiperf==0.12.0.dev20260803'
#
# Default: concurrency 1 (KV ~1.1M tokens @ 256k on PP2 Humming).
# After the run, archives vllm-recipe-*.log from each rank into ARTIFACT_DIR
# (same convention as scripts/copy-sweep-results.sh).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
URL="${URL:-127.0.0.1:8000}"
MODEL="${MODEL:-/models/Kimi-K3}"
TOKENIZER="${TOKENIZER:-moonshotai/Kimi-K3}"
CONCURRENCY="${CONCURRENCY:-1}"
MAX_CONTEXT_LENGTH="${MAX_CONTEXT_LENGTH:-262144}"
BENCHMARK_DURATION="${BENCHMARK_DURATION:-900}"
RANDOM_SEED="${RANDOM_SEED:-20260804}"
# Date-pinned with-subagents corpus, 256k-filtered (matches MAX_MODEL_LEN).
PUBLIC_DATASET="${PUBLIC_DATASET:-semianalysis_cc_traces_weka_with_subagents_256k}"
ARTIFACT_DIR="${ARTIFACT_DIR:-$ROOT/bench-results/agentx-mvp-pp2-humming-c${CONCURRENCY}}"
NS="${NS:-kimi-k3}"
NNODES="${NNODES:-2}"

pick_aiperf() {
  if command -v aiperf >/dev/null 2>&1; then
    echo aiperf
    return
  fi
  if [[ -x "$ROOT/reports/.venv/bin/aiperf" ]]; then
    echo "$ROOT/reports/.venv/bin/aiperf"
    return
  fi
  if python3 -c 'import aiperf' 2>/dev/null; then
    echo "python3 -m aiperf"
    return
  fi
  return 1
}

archive_vllm_logs() {
  local out_dir="$1"
  echo "==> Copying vLLM serve logs into $out_dir"
  for i in $(seq 0 $((NNODES - 1))); do
    local out="$out_dir/vllm-recipe-$i.log"
    if kubectl -n "$NS" exec "vllm-recipe-$i" -- test -f /tmp/vllm-recipe.log; then
      kubectl -n "$NS" exec "vllm-recipe-$i" -- cat /tmp/vllm-recipe.log >"$out"
      echo "    rank-$i -> $out ($(wc -c <"$out") bytes)"
    else
      echo "    rank-$i: /tmp/vllm-recipe.log missing" >&2
    fi
  done
}

if ! AIPERF="$(pick_aiperf)"; then
  echo "aiperf not found. Install with:" >&2
  echo "  python3 -m pip install aiperf" >&2
  echo "  # or: reports/.venv/bin/pip install aiperf" >&2
  exit 1
fi

mkdir -p "$ARTIFACT_DIR"
export AIPERF_DATASET_CONFIGURATION_TIMEOUT="${AIPERF_DATASET_CONFIGURATION_TIMEOUT:-1800}"
export AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT="${AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT:-1800}"

echo "==> aiperf=$AIPERF"
echo "==> url=$URL model=$MODEL tokenizer=$TOKENIZER"
echo "==> concurrency=$CONCURRENCY max_context_length=$MAX_CONTEXT_LENGTH duration=${BENCHMARK_DURATION}s"
echo "==> dataset=$PUBLIC_DATASET artifact_dir=$ARTIFACT_DIR"

set +e
# shellcheck disable=SC2086
$AIPERF profile \
  --scenario inferencex-agentx-mvp \
  --url "$URL" \
  --model "$MODEL" \
  --tokenizer "$TOKENIZER" \
  --tokenizer-trust-remote-code \
  --max-context-length "$MAX_CONTEXT_LENGTH" \
  --endpoint-type chat \
  --public-dataset "$PUBLIC_DATASET" \
  --concurrency "$CONCURRENCY" \
  --use-server-token-count \
  --streaming \
  --extra-inputs ignore_eos:true \
  --cache-bust first_turn_prefix \
  --system-idle-gap-cap-seconds 10 \
  --trajectory-start-min-ratio 0.0 \
  --trajectory-start-max-ratio 1.0 \
  --benchmark-duration "$BENCHMARK_DURATION" \
  --random-seed "$RANDOM_SEED" \
  --artifact-dir "$ARTIFACT_DIR" \
  --ui simple
rc=$?
set -e

archive_vllm_logs "$ARTIFACT_DIR"
exit "$rc"
