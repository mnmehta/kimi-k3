#!/usr/bin/env bash
# Sequential AgentX MVP runs at concurrency 2, 3, 4.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export KUBECONFIG="${KUBECONFIG:-/Users/mimehta/kubeconfigs/kubeconfig.fozzie}"
export URL="${URL:-127.0.0.1:18000}"
export PATH="$ROOT/reports/.venv/bin:$PATH"
LOGDIR="$ROOT/bench-results"
PF_LOG="$LOGDIR/pf-agentx-18000.log"
MASTER_LOG="$LOGDIR/agentx-mvp-c234.master.log"

ensure_pf() {
  if curl -sf --max-time 3 "http://${URL}/v1/models" >/dev/null; then
    return 0
  fi
  echo "Port-forward down; restarting..." | tee -a "$MASTER_LOG"
  pkill -f "port-forward.*vllm-recipe-0.*18000:8000" 2>/dev/null || true
  sleep 1
  nohup kubectl -n kimi-k3 port-forward pod/vllm-recipe-0 18000:8000 \
    >"$PF_LOG" 2>&1 &
  disown || true
  for _ in $(seq 1 30); do
    if curl -sf --max-time 2 "http://${URL}/v1/models" >/dev/null; then
      echo "Port-forward ready." | tee -a "$MASTER_LOG"
      return 0
    fi
    sleep 1
  done
  echo "ERROR: port-forward failed to become ready" | tee -a "$MASTER_LOG"
  return 1
}

{
  echo "===== MASTER START $(date -u +%Y-%m-%dT%H:%M:%SZ) ====="
  ensure_pf
  for C in 2 3 4; do
    echo "===== START C=$C $(date -u +%Y-%m-%dT%H:%M:%SZ) ====="
    ensure_pf
    # Remove incomplete prior artifact if export never finished
    ART="$ROOT/bench-results/agentx-mvp-pp2-humming-c${C}"
    if [[ -d "$ART" && ! -f "$ART/profile_export_aiperf.json" ]]; then
      echo "Removing incomplete artifact dir $ART"
      rm -rf "$ART"
    fi
    set +e
    CONCURRENCY=$C URL="$URL" \
      "$ROOT/scripts/run-agentx-mvp.sh" \
      >"$LOGDIR/agentx-mvp-c${C}.run.log" 2>&1
    rc=$?
    set -e
    echo "===== DONE C=$C exit=$rc $(date -u +%Y-%m-%dT%H:%M:%SZ) ====="
    if [[ $rc -ne 0 ]]; then
      echo "ERROR: C=$C failed; aborting remaining concurrencies"
      exit $rc
    fi
    ls -la "$ART" || true
  done
  echo "===== ALL DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) ====="
} 2>&1 | tee -a "$MASTER_LOG"
