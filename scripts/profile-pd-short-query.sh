#!/usr/bin/env bash
# P/D profiler capture: start/stop on prefill+decode engines; request via router.
#
# Run on the host (uses kubectl). Engines must already be up with --profiler-config.
#
# Usage:
#   PROFILE_DIR=/tmp/vllm_profile/pd ./scripts/profile-pd-short-query.sh

set -euo pipefail

NS="${NS:-kimi-k3}"
PREFILL_POD="${PREFILL_POD:-vllm-recipe-0}"
DECODE_POD="${DECODE_POD:-vllm-recipe-2}"
PREFILL_PORT="${PREFILL_PORT:-8001}"
DECODE_PORT="${DECODE_PORT:-8002}"
ROUTER_PORT="${ROUTER_PORT:-8000}"
PROFILE_DIR="${PROFILE_DIR:-/tmp/vllm_profile/pd}"
PROMPT_TOKENS="${PROMPT_TOKENS:-10}"
MAX_TOKENS="${MAX_TOKENS:-5}"
WARMUP="${WARMUP:-1}"

MODEL="${MODEL:-}"
if [[ -z "$MODEL" || "$MODEL" == "auto" ]]; then
  MODEL="$(kubectl -n "$NS" exec "$PREFILL_POD" -- curl -sS "http://127.0.0.1:${PREFILL_PORT}/v1/models" \
    | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["data"][0]["id"])')"
fi
echo "MODEL=$MODEL PROFILE_DIR=$PROFILE_DIR"

PROMPT_IDS=$(kubectl -n "$NS" exec "$PREFILL_POD" -- \
  curl -sS "http://127.0.0.1:${PREFILL_PORT}/tokenize" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"${MODEL}\",\"prompt\":\"The quick brown fox jumps over the lazy dog again and again near the river bank under the bright afternoon sun while birds sing.\"}" \
  | python3 -c "
import json,sys
data=json.load(sys.stdin)
ids=data.get('tokens') or data.get('token_ids') or data.get('ids')
if ids is None:
    raise SystemExit(f'unexpected tokenize response: {data}')
n=int('${PROMPT_TOKENS}')
ids=list(ids)[:n]
if len(ids) < n:
    raise SystemExit(f'prompt too short after tokenize: got {len(ids)}')
print(json.dumps(ids))
")
echo "prompt_token_ids=$PROMPT_IDS"

complete() {
  local url=$1
  kubectl -n "$NS" exec "$PREFILL_POD" -- curl -sS "$url" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"${MODEL}\",\"prompt\":${PROMPT_IDS},\"max_tokens\":${MAX_TOKENS},\"temperature\":0}"
}

if [[ "$WARMUP" == "1" ]]; then
  echo "Warmup via router :$ROUTER_PORT"
  complete "http://127.0.0.1:${ROUTER_PORT}/v1/completions" >/dev/null
fi

echo "POST /start_profile on prefill :$PREFILL_PORT and decode :$DECODE_PORT"
kubectl -n "$NS" exec "$PREFILL_POD" -- curl -sS -X POST "http://127.0.0.1:${PREFILL_PORT}/start_profile"
echo
kubectl -n "$NS" exec "$DECODE_POD" -- curl -sS -X POST "http://127.0.0.1:${DECODE_PORT}/start_profile"
echo

echo "Profiled completion via router"
complete "http://127.0.0.1:${ROUTER_PORT}/v1/completions" | python3 -c '
import json,sys
r=json.load(sys.stdin)
print(json.dumps(r, indent=2, ensure_ascii=False)[:2000])
u=r.get("usage") or {}
print(f"usage: prompt_tokens={u.get(\"prompt_tokens\")} completion_tokens={u.get(\"completion_tokens\")}")
'

echo "POST /stop_profile (both engines; may take a while)"
kubectl -n "$NS" exec "$PREFILL_POD" -- curl -sS -X POST "http://127.0.0.1:${PREFILL_PORT}/stop_profile"
echo
kubectl -n "$NS" exec "$DECODE_POD" -- curl -sS -X POST "http://127.0.0.1:${DECODE_PORT}/stop_profile"
echo

for pod in "$PREFILL_POD" "$DECODE_POD" vllm-recipe-1 vllm-recipe-3; do
  echo "--- $pod $PROFILE_DIR ---"
  kubectl -n "$NS" exec "$pod" -- bash -c "ls -lah '$PROFILE_DIR' 2>/dev/null | head -20 || echo missing"
done
