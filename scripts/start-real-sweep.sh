#!/usr/bin/env bash
# In-pod launcher for real-weight concurrency sweep. Avoids kubectl-exec argv
# matching kill patterns.
set -euo pipefail

RESULT="${RESULT_DIR:-/tmp/vllm-bench-sweep-real-256}"
MODEL="${MODEL:-/models/Kimi-K3}"

# Stop prior sweep by pidfile only
if [[ -f "$RESULT/sweep.pid" ]]; then
  old=$(cat "$RESULT/sweep.pid" || true)
  if [[ -n "${old:-}" ]]; then
    kill "$old" 2>/dev/null || true
    sleep 1
    kill -9 "$old" 2>/dev/null || true
  fi
fi

# Stop leftover bench children by exact binary path
pkill -9 -f '/usr/local/bin/vllm bench' 2>/dev/null || true
pkill -9 -f '/vllm/vllm-rs' 2>/dev/null || true
sleep 2

rm -rf "$RESULT"
mkdir -p "$RESULT"

nohup env \
  RESULT_DIR="$RESULT" \
  MODEL="$MODEL" \
  CONCURRENCIES='1 2 4 8 16 32 64 128 256' \
  INPUT_LEN=1000 \
  OUTPUT_LEN=1000 \
  TARGET_SECS=120 \
  bash /tmp/bench-concurrency-sweep.sh >"$RESULT/nohup.out" 2>&1 &
echo $! >"$RESULT/sweep.pid"
echo "SWEEP_PID=$(cat "$RESULT/sweep.pid")"
sleep 3
head -25 "$RESULT/nohup.out"
kill -0 "$(cat "$RESULT/sweep.pid")" && echo ALIVE || echo DEAD
