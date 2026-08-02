#!/usr/bin/env bash
# Compatibility wrapper — bench against P/D router on :8000
set -euo pipefail
export RESULT_DIR="${RESULT_DIR:-/tmp/vllm-bench-sweep-pd-512}"
export CONCURRENCIES="${CONCURRENCIES:-1 2 4 8 16 32 64 128 256 512}"
export BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
exec bash /tmp/start-sweep.sh
