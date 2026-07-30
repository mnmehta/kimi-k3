#!/usr/bin/env bash
# Resume concurrency sweep from remaining tiers (assumes conc-1 already done).
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
MODEL="${MODEL:-moonshotai/Kimi-K3}"
INPUT_LEN="${INPUT_LEN:-1000}"
OUTPUT_LEN="${OUTPUT_LEN:-1000}"
TARGET_SECS="${TARGET_SECS:-120}"
RESULT_DIR="${RESULT_DIR:-/tmp/vllm-bench-sweep}"
SUMMARY="$RESULT_DIR/summary.tsv"
CONCURRENCIES=(${CONCURRENCIES:-2 4 8 16 32})

mkdir -p "$RESULT_DIR"
if [[ ! -f "$SUMMARY" ]]; then
  echo -e "concurrency\tnum_prompts\tduration_s\trequest_throughput\toutput_tok_s\tmean_ttft_ms\tmean_tpot_ms" >"$SUMMARY"
fi

append_summary() {
  local json="$1" c="$2" n="$3"
  python3 - "$json" "$c" "$n" <<'PY' | tee -a "$SUMMARY"
import json, sys
path, conc, nprompts = sys.argv[1], sys.argv[2], sys.argv[3]
d = json.load(open(path))
dur = float(d.get("duration", 0))
print(
    f"{conc}\t{nprompts}\t{dur:.2f}\t"
    f"{d.get('request_throughput', '')}\t{d.get('output_throughput', '')}\t"
    f"{d.get('mean_ttft_ms', '')}\t{d.get('mean_tpot_ms', '')}"
)
waves = max(1.0, float(nprompts) / float(conc))
print(f"observed_avg_request_s={dur / waves:.4f}", file=sys.stderr)
PY
}

LATENCY_S="${LATENCY_S:-}"
if [[ -z "$LATENCY_S" && -f "$RESULT_DIR/conc-1.json" ]]; then
  LATENCY_S=$(python3 -c 'import json; d=json.load(open("'"$RESULT_DIR"'/conc-1.json")); print(float(d["duration"])/max(1,int(d.get("num_prompts",1))))')
fi
LATENCY_S="${LATENCY_S:-3.4}"
echo "resume seed_latency_s=${LATENCY_S} concs=${CONCURRENCIES[*]}"

for c in "${CONCURRENCIES[@]}"; do
  if [[ -f "$RESULT_DIR/conc-${c}.json" ]]; then
    echo "== skip concurrency=${c} (result exists) =="
    continue
  fi
  est_lat=$(python3 -c "print(max(0.5, ${LATENCY_S} * (1.0 + 0.15 * (${c} - 1) ** 0.5)))")
  waves=$(python3 -c "import math; print(max(2, int(math.ceil(${TARGET_SECS} / ${est_lat}))))")
  num_prompts=$((c * waves))
  echo
  echo "== concurrency=${c} num_prompts=${num_prompts} est_lat=${est_lat}s target=${TARGET_SECS}s =="
  OUT_JSON="conc-${c}.json"
  LOG="$RESULT_DIR/conc-${c}.log"
  set +e
  vllm bench serve \
    --backend openai \
    --base-url "$BASE_URL" \
    --model "$MODEL" \
    --endpoint /v1/completions \
    --dataset-name random \
    --random-input-len "$INPUT_LEN" \
    --random-output-len "$OUTPUT_LEN" \
    --ignore-eos \
    --request-rate inf \
    --max-concurrency "$c" \
    --num-prompts "$num_prompts" \
    --num-warmups 0 \
    --save-result \
    --result-dir "$RESULT_DIR" \
    --result-filename "$OUT_JSON" \
    2>&1 | tee "$LOG"
  RC=${PIPESTATUS[0]}
  set -e
  if [[ -f "$RESULT_DIR/$OUT_JSON" ]]; then
    append_summary "$RESULT_DIR/$OUT_JSON" "$c" "$num_prompts"
    LATENCY_S=$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); waves=max(1.0,float(sys.argv[2])/float(sys.argv[3])); print(float(d.get("duration",4))/waves)' "$RESULT_DIR/$OUT_JSON" "$num_prompts" "$c")
  else
    echo -e "${c}\t${num_prompts}\tFAIL_rc=${RC}\t\t\t\t" | tee -a "$SUMMARY"
  fi
done

echo
echo "== sweep complete; results in ${RESULT_DIR} =="
cat "$SUMMARY"
