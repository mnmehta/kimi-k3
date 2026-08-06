#!/usr/bin/env bash
# vllm bench concurrency sweep: 1000/1000 tokens, ~2 min per concurrency tier.
# Powers of two from 1 .. 512. Uses a random --seed so prompts differ across runs.
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
MODEL="${MODEL:-moonshotai/Kimi-K3}"
INPUT_LEN="${INPUT_LEN:-1000}"
OUTPUT_LEN="${OUTPUT_LEN:-1000}"
TARGET_SECS="${TARGET_SECS:-120}"
RESULT_DIR="${RESULT_DIR:-/tmp/vllm-bench-sweep-512}"
CONCURRENCIES=(${CONCURRENCIES:-1 2 4 8 16 32 64 128 256 512})
# Fresh random seed for this sweep (override with SEED=...).
SEED="${SEED:-$(python3 -c 'import random; print(random.randint(1, 2_000_000_000))')}"
# Bench command: default to Rust binary, fall back to Python for servers
# whose /tokenize endpoint is broken (e.g. SGLang + Kimi-K3 tiktoken).
BENCH_CMD="${BENCH_CMD:-vllm bench serve}"

mkdir -p "$RESULT_DIR"
SUMMARY="$RESULT_DIR/summary.tsv"
echo -e "concurrency\tnum_prompts\tseed\tduration_s\trequest_throughput\toutput_tok_s\tmean_ttft_ms\tmean_tpot_ms\tmean_e2el_ms" >"$SUMMARY"
echo "$SEED" >"$RESULT_DIR/seed.txt"
echo "== sweep seed=${SEED} result_dir=${RESULT_DIR} concs=${CONCURRENCIES[*]} =="

append_summary() {
  local json="$1" c="$2" n="$3" seed="$4"
  python3 - "$json" "$c" "$n" "$seed" <<'PY' | tee -a "$SUMMARY"
import json, sys
path, conc, nprompts, seed = sys.argv[1:5]
d = json.load(open(path))
dur = float(d.get("duration", 0))
print(
    f"{conc}\t{nprompts}\t{seed}\t{dur:.2f}\t"
    f"{d.get('request_throughput', '')}\t{d.get('output_throughput', '')}\t"
    f"{d.get('mean_ttft_ms', '')}\t{d.get('mean_tpot_ms', '')}\t"
    f"{d.get('mean_e2el_ms', '')}"
)
numeric_conc = float(''.join(c for c in conc if c.isdigit() or c == '.') or '1')
waves = max(1.0, float(nprompts) / numeric_conc)
print(f"observed_avg_request_s={dur / waves:.4f}", file=sys.stderr)
PY
}

run_bench() {
  local c="$1" n="$2" out_json="$3" seed="$4" log="$5"
  set +e
  $BENCH_CMD \
    --backend openai \
    --base-url "$BASE_URL" \
    --model "$MODEL" \
    --trust-remote-code \
    --endpoint /v1/completions \
    --dataset-name random \
    --random-input-len "$INPUT_LEN" \
    --random-output-len "$OUTPUT_LEN" \
    --seed "$seed" \
    --ignore-eos \
    --request-rate inf \
    --max-concurrency "$c" \
    --num-prompts "$n" \
    --num-warmups 0 \
    --save-result \
    --result-dir "$RESULT_DIR" \
    --result-filename "$out_json" \
    2>&1 | tee "$log"
  echo "${PIPESTATUS[0]}" >"${log}.rc"
  set -e
}

# Probe with unique seed
echo "== probe latency (1x ${INPUT_LEN}/${OUTPUT_LEN}) seed=${SEED} =="
run_bench 1 1 "probe.json" "$SEED" "$RESULT_DIR/probe.log"
LATENCY_S=4.0
if [[ -f "$RESULT_DIR/probe.json" ]]; then
  LATENCY_S=$(python3 -c 'import json; print(float(json.load(open("'"$RESULT_DIR"'/probe.json")).get("duration", 4.0)))')
fi
echo "seed_latency_s=${LATENCY_S}"

for c in "${CONCURRENCIES[@]}"; do
  # Unique seed per tier so prompts never collide with prior tiers/runs.
  tier_seed=$((SEED + c * 10007))
  est_lat=$(python3 -c "print(max(0.5, ${LATENCY_S} * (1.0 + 0.12 * (${c} - 1) ** 0.5)))")
  waves=$(python3 -c "import math; print(max(2, int(math.ceil(${TARGET_SECS} / ${est_lat}))))")
  num_prompts=$((c * waves))

  echo
  echo "== concurrency=${c} num_prompts=${num_prompts} seed=${tier_seed} est_lat=${est_lat}s target=${TARGET_SECS}s =="
  OUT_JSON="conc-${c}.json"
  LOG="$RESULT_DIR/conc-${c}.log"
  run_bench "$c" "$num_prompts" "$OUT_JSON" "$tier_seed" "$LOG"
  RC=$(cat "${LOG}.rc")

  if [[ -f "$RESULT_DIR/$OUT_JSON" ]]; then
    append_summary "$RESULT_DIR/$OUT_JSON" "$c" "$num_prompts" "$tier_seed"
    LATENCY_S=$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); waves=max(1.0,float(sys.argv[2])/float(sys.argv[3])); print(float(d.get("duration",4))/waves)' "$RESULT_DIR/$OUT_JSON" "$num_prompts" "$c")
    # If we undershot badly (<75% of target), top up with another run at same conc using a new seed.
    dur=$(python3 -c 'import json; print(float(json.load(open("'"$RESULT_DIR/$OUT_JSON"'"))["duration"]))')
    if python3 -c "import sys; sys.exit(0 if float('$dur') < 0.75 * $TARGET_SECS else 1)"; then
      extra_waves=$(python3 -c "import math; print(max(2, int(math.ceil(($TARGET_SECS - $dur) / max(0.5, $LATENCY_S)))))")
      extra_n=$((c * extra_waves))
      topup_seed=$((tier_seed + 1))
      echo "== top-up concurrency=${c} extra_prompts=${extra_n} seed=${topup_seed} (first run only ${dur}s) =="
      TOP_JSON="conc-${c}-topup.json"
      run_bench "$c" "$extra_n" "$TOP_JSON" "$topup_seed" "$RESULT_DIR/conc-${c}-topup.log"
      # Keep primary result; note top-up separately if present.
      if [[ -f "$RESULT_DIR/$TOP_JSON" ]]; then
        append_summary "$RESULT_DIR/$TOP_JSON" "${c}-topup" "$extra_n" "$topup_seed"
        LATENCY_S=$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); waves=max(1.0,float(sys.argv[2])/float(sys.argv[3])); print(float(d.get("duration",4))/waves)' "$RESULT_DIR/$TOP_JSON" "$extra_n" "$c")
      fi
    fi
  else
    echo -e "${c}\t${num_prompts}\t${tier_seed}\tFAIL_rc=${RC}\t\t\t\t\t" | tee -a "$SUMMARY"
  fi
done

echo
echo "== sweep complete; seed=${SEED}; results in ${RESULT_DIR} =="
cat "$SUMMARY"
