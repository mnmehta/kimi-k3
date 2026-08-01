#!/usr/bin/env bash
# Compatibility wrappers — prefer scripts/start-sweep.sh
set -euo pipefail
export RESULT_DIR="${RESULT_DIR:-/tmp/vllm-bench-sweep-tep-512}"
export CONCURRENCIES="${CONCURRENCIES:-1 2 4 8 16 32 64 128 256 512}"
exec bash /tmp/start-sweep.sh
