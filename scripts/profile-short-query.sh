#!/usr/bin/env bash
# Capture a torch.profiler chrome trace for one short completion:
#   ~10 prompt tokens in, 5 completion tokens out.
#
# Prerequisites: vllm serve started with --profiler-config (see run-vllm-kimi-k3-dummy.sh)
#
# Usage (inside the pod):
#   bash /tmp/profile-short-query.sh
# From host:
#   kubectl -n kimi-k3 cp scripts/profile-short-query.sh vllm:/tmp/profile-short-query.sh
#   kubectl -n kimi-k3 exec vllm -- bash /tmp/profile-short-query.sh
#   kubectl -n kimi-k3 cp vllm:/tmp/vllm_profile ./vllm_profile

set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
MODEL="${MODEL:-moonshotai/Kimi-K3}"
PROMPT_TOKENS="${PROMPT_TOKENS:-10}"
MAX_TOKENS="${MAX_TOKENS:-5}"
PROFILE_DIR="${PROFILE_DIR:-/tmp/vllm_profile}"
WARMUP="${WARMUP:-1}"

echo "Building a ${PROMPT_TOKENS}-token prompt via /tokenize ..."
# Use a long wordy prompt, then trim token ids to exactly PROMPT_TOKENS.
PROMPT_TEXT="${PROMPT_TEXT:-The quick brown fox jumps over the lazy dog again and again near the river bank under the bright afternoon sun while birds sing.}"
PROMPT_IDS=$(curl -sS "$BASE_URL/tokenize" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"${MODEL}\",\"prompt\":\"${PROMPT_TEXT}\"}" \
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

if [[ "$WARMUP" == "1" ]]; then
  echo "Warmup request (outside profiler) ..."
  curl -sS "$BASE_URL/v1/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"${MODEL}\",\"prompt\":${PROMPT_IDS},\"max_tokens\":${MAX_TOKENS},\"temperature\":0}" \
    >/dev/null
fi

mkdir -p "$PROFILE_DIR"
echo "POST /start_profile"
curl -sS -X POST "$BASE_URL/start_profile"
echo

echo "Profiled completion: ${PROMPT_TOKENS} in / ${MAX_TOKENS} out"
RESP_FILE=$(mktemp)
curl -sS "$BASE_URL/v1/completions" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"${MODEL}\",\"prompt\":${PROMPT_IDS},\"max_tokens\":${MAX_TOKENS},\"temperature\":0}" \
  >"$RESP_FILE"
python3 - "$RESP_FILE" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
print(json.dumps(r, indent=2, ensure_ascii=False))
u=r.get("usage") or {}
print(f"usage: prompt_tokens={u.get('prompt_tokens')} completion_tokens={u.get('completion_tokens')}")
PY
rm -f "$RESP_FILE"

echo "POST /stop_profile (may take a bit to flush) ..."
curl -sS -X POST "$BASE_URL/stop_profile"
echo
echo "Traces under $PROFILE_DIR :"
ls -lah "$PROFILE_DIR" || true
find "$PROFILE_DIR" -type f | head -50

