#!/usr/bin/env bash
# Wait for 4-node P/D (TEP16 prefill + TEP16 decode) /health, start router,
# run C=1..512 sweep, copy results.
#
# Prefill head: vllm-recipe-0 :8001
# Decode head:  vllm-recipe-2 :8002
# Router:       vllm-recipe-0 :8000
set -euo pipefail
export KUBECONFIG="${KUBECONFIG:-/Users/mimehta/kubeconfigs/kubeconfig.fozzie}"
NS="${NS:-kimi-k3}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PREFILL_PORT="${PREFILL_PORT:-8001}"
DECODE_PORT="${DECODE_PORT:-8002}"
ROUTER_PORT="${ROUTER_PORT:-8000}"
PREFILL_POD="${PREFILL_POD:-vllm-recipe-0}"
DECODE_POD="${DECODE_POD:-vllm-recipe-2}"
PREFILL_IP="${PREFILL_IP:-$(cat /tmp/pd-prefill-ip 2>/dev/null || true)}"
DECODE_IP="${DECODE_IP:-$(cat /tmp/pd-decode-ip 2>/dev/null || true)}"
PREFILL_IP="${PREFILL_IP:-$(kubectl -n "$NS" get pod "$PREFILL_POD" -o jsonpath='{.status.podIP}')}"
DECODE_IP="${DECODE_IP:-$(kubectl -n "$NS" get pod "$DECODE_POD" -o jsonpath='{.status.podIP}')}"

echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) waiting for P/D /health (prefill=$PREFILL_POD:$PREFILL_PORT decode=$DECODE_POD:$DECODE_PORT)"
for n in $(seq 1 360); do
  p_ok=0; d_ok=0
  kubectl -n "$NS" exec "$PREFILL_POD" -- curl -sf -m 2 "http://127.0.0.1:${PREFILL_PORT}/health" >/dev/null 2>&1 && p_ok=1
  kubectl -n "$NS" exec "$DECODE_POD" -- curl -sf -m 2 "http://127.0.0.1:${DECODE_PORT}/health" >/dev/null 2>&1 && d_ok=1
  if [[ $p_ok -eq 1 && $d_ok -eq 1 ]]; then
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) both healthy poll=$n"
    for pod in "$PREFILL_POD" "$DECODE_POD"; do
      echo "--- $pod ---"
      kubectl -n "$NS" exec "$pod" -- bash -c \
        'grep -E "kv_role|Available KV|GPU KV cache|Maximum concurrency|Application startup|Expert parallelism|TP=|nnodes" /tmp/vllm-recipe.log | tail -20' || true
    done
    break
  fi
  if (( n % 6 == 0 )); then
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) still loading poll=$n p=$p_ok d=$d_ok"
    for pod in vllm-recipe-0 vllm-recipe-1 vllm-recipe-2 vllm-recipe-3; do
      echo "--- $pod ---"
      kubectl -n "$NS" exec "$pod" -- bash -c \
        'grep -E "Launching|kv_role|Model loading|Available KV|Application startup|CUDA out of memory|ValueError|Error|NCCL|Rank" /tmp/vllm-recipe.log | tail -6' 2>/dev/null || true
    done
  fi
  # Fatal if a serve died after OOM / init failure
  for rank in 0 1 2 3; do
    if ! kubectl -n "$NS" exec "vllm-recipe-$rank" -- pgrep -f 'vllm serve' >/dev/null 2>&1; then
      if kubectl -n "$NS" exec "vllm-recipe-$rank" -- grep -qE 'CUDA out of memory|Engine core initialization failed|ValueError: To serve at least one request|ValidationError|NixlConnector is incompatible|VLLM_SSM_CONV_STATE_LAYOUT' /tmp/vllm-recipe.log 2>/dev/null; then
        echo "serve died on rank=$rank" >&2
        kubectl -n "$NS" exec "vllm-recipe-$rank" -- tail -40 /tmp/vllm-recipe.log >&2 || true
        exit 1
      fi
    fi
  done
  sleep 10
done
kubectl -n "$NS" exec "$PREFILL_POD" -- curl -sf -m 2 "http://127.0.0.1:${PREFILL_PORT}/health" >/dev/null
kubectl -n "$NS" exec "$DECODE_POD" -- curl -sf -m 2 "http://127.0.0.1:${DECODE_PORT}/health" >/dev/null

echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) starting router :$ROUTER_PORT -> P ${PREFILL_IP}:$PREFILL_PORT D ${DECODE_IP}:$DECODE_PORT"
kubectl -n "$NS" exec "$PREFILL_POD" -- bash -c \
  'cp /vllm-workspace/examples/disaggregated/disaggregated_serving/disagg_proxy_demo.py /tmp/disagg_proxy_demo.py'
# Kill prior router via pidfile only — never pkill -f a pattern that appears in
# this bash -c argv (pkill would SIGTERM itself → exit 143).
kubectl -n "$NS" exec "$PREFILL_POD" -- bash -c "
  if [[ -f /tmp/pd-router.pid ]]; then kill \"\$(cat /tmp/pd-router.pid)\" 2>/dev/null || true; fi
  nohup python3 /tmp/disagg_proxy_demo.py \
    --model /models/Kimi-K3 \
    --prefill ${PREFILL_IP}:${PREFILL_PORT} \
    --decode ${DECODE_IP}:${DECODE_PORT} \
    --port ${ROUTER_PORT} \
    > /tmp/pd-router.log 2>&1 &
  echo \$! > /tmp/pd-router.pid
  echo router=\$(cat /tmp/pd-router.pid)
"
sleep 4
kubectl -n "$NS" exec "$PREFILL_POD" -- curl -sS "http://127.0.0.1:${ROUTER_PORT}/v1/models" | head -c 500 || {
  echo "router /v1/models failed" >&2
  kubectl -n "$NS" exec "$PREFILL_POD" -- tail -50 /tmp/pd-router.log >&2 || true
  exit 1
}
echo

echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) launching P/D sweep C=1..512 via router"
kubectl -n "$NS" cp "$ROOT/scripts/bench-concurrency-sweep.sh" "$PREFILL_POD:/tmp/bench-concurrency-sweep.sh"
kubectl -n "$NS" cp "$ROOT/scripts/start-sweep.sh" "$PREFILL_POD:/tmp/start-sweep.sh"
kubectl -n "$NS" cp "$ROOT/scripts/start-pd-sweep.sh" "$PREFILL_POD:/tmp/start-pd-sweep.sh"
kubectl -n "$NS" exec "$PREFILL_POD" -- bash /tmp/start-pd-sweep.sh

echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) monitoring sweep"
RESULT=/tmp/vllm-bench-sweep-pd-512 INTERVAL_S=60 "$ROOT/scripts/monitor-pd-sweep.sh"

echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) copying results"
RESULT=/tmp/vllm-bench-sweep-pd-512 LOCAL_OUT=bench-results/conc-sweep-pd-1000-1000 \
  NNODES=4 "$ROOT/scripts/copy-sweep-results.sh"
kubectl -n "$NS" exec "$PREFILL_POD" -- cat /tmp/pd-router.log \
  > "$ROOT/bench-results/conc-sweep-pd-1000-1000/pd-router.log" 2>/dev/null || true
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) sweep artifacts ready"
ls -la "$ROOT/bench-results/conc-sweep-pd-1000-1000" | head -30
