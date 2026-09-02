#!/usr/bin/env bash
# Stop SGLang processes inside a pod (counterpart to stop-inpod-vllm.sh).
set -euo pipefail

# Kill sglang.launch_server and any child processes
pkill -f 'sglang.launch_server' 2>/dev/null || true
pkill -f 'sglang.srt.server' 2>/dev/null || true
sleep 1
pkill -9 -f 'sglang.launch_server' 2>/dev/null || true
pkill -9 -f 'sglang.srt.server' 2>/dev/null || true

echo "SGLang processes stopped"
