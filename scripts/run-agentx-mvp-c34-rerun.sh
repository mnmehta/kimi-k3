#!/usr/bin/env bash
# Re-run AgentX MVP at concurrency 3 and 4 (after port-forward failures).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export KUBECONFIG="${KUBECONFIG:-/Users/mimehta/kubeconfigs/kubeconfig.fozzie}"
export URL="${URL:-127.0.0.1:18000}"
export PATH="$ROOT/reports/.venv/bin:$PATH"
LOGDIR="$ROOT/bench-results"
MASTER_LOG="$LOGDIR/agentx-mvp-c34-rerun.master.log"

ensure_ready() {
  for _ in $(seq 1 60); do
    if curl -sf --max-time 2 "http://${URL}/v1/models" >/dev/null; then
      return 0
    fi
    sleep 1
  done
  echo "ERROR: endpoint not ready at $URL" | tee -a "$MASTER_LOG"
  return 1
}

{
  echo "===== RERUN START $(date -u +%Y-%m-%dT%H:%M:%SZ) ====="
  ensure_ready
  for C in 3 4; do
    echo "===== START C=$C $(date -u +%Y-%m-%dT%H:%M:%SZ) ====="
    ensure_ready
    ART="$ROOT/bench-results/agentx-mvp-pp2-humming-c${C}"
    rm -rf "$ART"
    set +e
    CONCURRENCY=$C URL="$URL" \
      "$ROOT/scripts/run-agentx-mvp.sh" \
      >"$LOGDIR/agentx-mvp-c${C}.run.log" 2>&1
    rc=$?
    set -e
    echo "===== DONE C=$C exit=$rc $(date -u +%Y-%m-%dT%H:%M:%SZ) ====="
    if [[ $rc -ne 0 ]]; then
      echo "ERROR: C=$C failed"
      exit $rc
    fi
    # Fail if error rate too high (port-forward died mid-run again)
    python3 - <<PY
import json
from pathlib import Path
p = Path("$ART") / "profile_export_aiperf.json"
d = json.loads(p.read_text())
ok = (d.get("request_count") or {}).get("avg") or 0
err = (d.get("error_request_count") or {}).get("avg") or 0
rate = (d.get("request_error_rate") or {}).get("avg") or 0
print(f"C=$C ok={ok} err={err} error_rate={rate:.2f}%")
if rate > 10:
    raise SystemExit(f"ERROR: C=$C error_rate {rate:.1f}% > 10% — likely port-forward failure")
PY
    ls -la "$ART"
  done
  echo "===== ALL DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) ====="
} 2>&1 | tee "$MASTER_LOG"
