#!/usr/bin/env bash
set -euo pipefail
RESULT="${RESULT_DIR:-/tmp/vllm-bench-sweep-pp2-256}"
MODEL="${MODEL:-/models/Kimi-K3}"

if [[ -f "$RESULT/sweep.pid" ]]; then
  old=$(cat "$RESULT/sweep.pid" || true)
  [[ -n "${old:-}" ]] && kill -9 "$old" 2>/dev/null || true
fi
sleep 1
rm -rf "$RESULT"
mkdir -p "$RESULT"

nohup env \
  RESULT_DIR="$RESULT" \
  MODEL="$MODEL" \
  CONCURRENCIES='1 2 4 8 16 32 64 128 256' \
  INPUT_LEN=1000 OUTPUT_LEN=1000 TARGET_SECS=120 \
  bash /tmp/bench-concurrency-sweep.sh >"$RESULT/nohup.out" 2>&1 &
echo $! >"$RESULT/sweep.pid"
echo "SWEEP_PID=$(cat "$RESULT/sweep.pid")"
sleep 4
head -30 "$RESULT/nohup.out" || true
kill -0 "$(cat "$RESULT/sweep.pid")" && echo ALIVE || echo DEAD
