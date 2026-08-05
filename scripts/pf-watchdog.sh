#!/usr/bin/env bash
# Keep kubectl port-forward alive for AgentX runs.
set -euo pipefail

export KUBECONFIG="${KUBECONFIG:-/Users/mimehta/kubeconfigs/kubeconfig.fozzie}"
NS="${NS:-kimi-k3}"
POD="${POD:-vllm-recipe-0}"
LOCAL_PORT="${LOCAL_PORT:-18000}"
REMOTE_PORT="${REMOTE_PORT:-8000}"
URL="127.0.0.1:${LOCAL_PORT}"
LOG="${LOG:-/Users/mimehta/kimi-k3/bench-results/pf-agentx-watchdog.log}"
PIDFILE="${PIDFILE:-/Users/mimehta/kimi-k3/bench-results/pf-agentx.pid}"

mkdir -p "$(dirname "$LOG")"
: >"$LOG"

start_pf() {
  pkill -f "port-forward.*${POD}.*${LOCAL_PORT}:${REMOTE_PORT}" 2>/dev/null || true
  sleep 1
  kubectl -n "$NS" port-forward "pod/${POD}" "${LOCAL_PORT}:${REMOTE_PORT}" >>"$LOG" 2>&1 &
  echo $! >"$PIDFILE"
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) started pf pid=$(cat "$PIDFILE")" >>"$LOG"
}

echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) watchdog start" >>"$LOG"
start_pf

while true; do
  if ! curl -sf --max-time 2 "http://${URL}/v1/models" >/dev/null; then
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) unhealthy; restarting pf" >>"$LOG"
    start_pf
    # wait for ready
    for _ in $(seq 1 20); do
      if curl -sf --max-time 2 "http://${URL}/v1/models" >/dev/null; then
        echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) pf healthy again" >>"$LOG"
        break
      fi
      sleep 1
    done
  fi
  sleep 3
done
