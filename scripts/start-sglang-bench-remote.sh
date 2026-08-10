#!/usr/bin/env bash
# Launch concurrency sweep from a remote bench pod against an SGLang endpoint.
# Runs inside the vllm-bench pod (no GPU, host networking).
#
# Usage (from host):
#   SGLANG_IP=10.202.210.23 \
#   kubectl -n kimi-k3 exec vllm-bench -- bash /tmp/start-sglang-bench-remote.sh
#
# Or override:
#   SGLANG_IP=... CONCURRENCIES="1 2 4 8 16 32" RESULT_DIR=/tmp/... \
#   kubectl -n kimi-k3 exec vllm-bench -- bash /tmp/start-sglang-bench-remote.sh

set -euo pipefail

SGLANG_IP="${SGLANG_IP:?SGLANG_IP required (rank-0 pod IP)}"
PORT="${PORT:-30000}"
RESULT="${RESULT_DIR:-/models/vllm-bench-results/sglang/conc-sweep-tp16-ep16-1000-1000-fp8-kv}"
MODEL="${MODEL:-/models/Kimi-K3}"
CONCURRENCIES="${CONCURRENCIES:-1 2 4 8 16 32 64 128 256 512}"

BASE_URL="http://${SGLANG_IP}:${PORT}"

echo "==> Checking SGLang health at $BASE_URL"
if ! curl -sf -m 10 "$BASE_URL/health" >/dev/null 2>&1; then
  echo "SGLang not healthy at $BASE_URL" >&2
  exit 1
fi
echo "    Healthy"
curl -sS "$BASE_URL/v1/models" 2>&1 | head -5
echo

if [[ -f "$RESULT/sweep.pid" ]]; then
  old=$(cat "$RESULT/sweep.pid" || true)
  [[ -n "${old:-}" ]] && kill -9 "$old" 2>/dev/null || true
fi
sleep 1
rm -rf "$RESULT"
mkdir -p "$RESULT"

# Use Python bench to avoid Rust binary's fatal /tokenize error on SGLang.
BENCH_CMD="${BENCH_CMD:-python3 -m vllm.benchmarks.serve}"

nohup env \
  RESULT_DIR="$RESULT" \
  MODEL="$MODEL" \
  CONCURRENCIES="$CONCURRENCIES" \
  BASE_URL="$BASE_URL" \
  BENCH_CMD="$BENCH_CMD" \
  INPUT_LEN=1000 \
  OUTPUT_LEN=1000 \
  TARGET_SECS=120 \
  bash /tmp/bench-concurrency-sweep.sh >"$RESULT/nohup.out" 2>&1 &
echo $! >"$RESULT/sweep.pid"
echo "SWEEP_PID=$(cat "$RESULT/sweep.pid") RESULT_DIR=$RESULT BASE_URL=$BASE_URL"
sleep 4
head -30 "$RESULT/nohup.out" || true
kill -0 "$(cat "$RESULT/sweep.pid")" && echo ALIVE || echo DEAD
