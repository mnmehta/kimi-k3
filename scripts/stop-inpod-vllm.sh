#!/usr/bin/env bash
# Stop in-pod vLLM serve / bench without argv self-match issues.
# Copy into pod and run: bash /tmp/stop-inpod-vllm.sh
set +e

kill_matching() {
  local needle="$1"
  ps -eo pid=,args= | while read -r pid args; do
    case "$args" in
      *"$needle"*)
        # never kill this stop script
        case "$args" in
          *stop-inpod-vllm*) continue ;;
        esac
        echo "kill $pid :: ${args:0:120}"
        kill -9 "$pid" 2>/dev/null
        ;;
    esac
  done
}

# Sweep / bench first
[[ -f /tmp/vllm-bench-sweep-real-256/sweep.pid ]] && kill -9 "$(cat /tmp/vllm-bench-sweep-real-256/sweep.pid)" 2>/dev/null
kill_matching 'vllm-bench-sweep'
kill_matching 'vllm-rs bench'
kill_matching 'bench-concurrency-sweep'

# Serve / engine workers
kill_matching '/usr/local/bin/vllm serve'
kill_matching 'VLLM::Worker'
kill_matching 'EngineCore'
kill_matching 'multiproc_executor'
kill_matching 'run-recipe.sh'

sleep 3
echo '--- remaining ---'
ps -eo pid=,args= | grep -E 'vllm serve|vllm-rs|EngineCore|Worker_TP|Worker_PP|bench-concurrency' | grep -v grep || echo NONE
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
