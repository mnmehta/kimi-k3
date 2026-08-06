#!/usr/bin/env bash
# Gemma-4 FP8 concurrency sweep from the in-cluster vllm-bench pod, updating
# a Quarto report after each finished concurrency.
#
# Prerequisites:
#   - ./scripts/deploy.sh gemma4-fp8-1gpu   (serve healthy on vllm-recipe-0)
#   - vllm-bench pod Ready
#
# Optional:
#   CONCURRENCIES="1 2 4 8 16 32 64"
#   INPUT_LEN=15000 OUTPUT_LEN=900
#   TARGET_SECS=120
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export KUBECONFIG="${KUBECONFIG:-/Users/mimehta/kubeconfigs/kubeconfig.fozzie}"
NS="${NS:-kimi-k3}"
BENCH_POD="${BENCH_POD:-vllm-bench}"
URL="${URL:-http://vllm-recipe-0.vllm-recipe:8000}"
# Served model id may be the path; probe /v1/models if unset.
MODEL="${MODEL:-/models/gemma-4-26B-A4B-it-FP8-dynamic}"
INPUT_LEN="${INPUT_LEN:-15000}"
OUTPUT_LEN="${OUTPUT_LEN:-900}"
TARGET_SECS="${TARGET_SECS:-120}"
CONCURRENCIES="${CONCURRENCIES:-1 2 4 8 16 32 64}"
RESULT_DIR_POD="${RESULT_DIR_POD:-/tmp/vllm-bench-gemma4-fp8-15k-900}"
LOCAL_OUT="${LOCAL_OUT:-$ROOT/bench-results/gemma4-fp8-1gpu-15k-900}"
REPORT_QMD="${REPORT_QMD:-$ROOT/reports/gemma4-fp8-1gpu-15k-900.qmd}"
MASTER_LOG="${MASTER_LOG:-$ROOT/bench-results/gemma4-fp8-1gpu-15k-900.master.log}"
SEED="${SEED:-$(python3 -c 'import random; print(random.randint(1, 2_000_000_000))')}"

mkdir -p "$LOCAL_OUT" "$ROOT/bench-results"
: >"$MASTER_LOG"
log() { echo "$*" | tee -a "$MASTER_LOG"; }

ensure_endpoint() {
  kubectl -n "$NS" exec "$BENCH_POD" -- bash -lc "
    set -euo pipefail
    curl -sf --max-time 10 '$URL/v1/models' | head -c 400
    echo
  "
}

# Resolve served model id from /v1/models if needed
resolve_model() {
  local id
  id="$(kubectl -n "$NS" exec "$BENCH_POD" -- bash -lc \
    "curl -sf '$URL/v1/models' | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d[\"data\"][0][\"id\"])'")"
  MODEL="$id"
  log "MODEL=$MODEL"
}

render_report() {
  log "==> Rendering $REPORT_QMD"
  if [[ -x "$ROOT/reports/bin/quarto" ]]; then
    (cd "$ROOT" && reports/bin/quarto render "$REPORT_QMD" --to html) 2>&1 | tee -a "$MASTER_LOG" | tail -20
  else
    log "WARN: quarto CLI missing; skip render"
  fi
}

copy_tier() {
  local c="$1"
  mkdir -p "$LOCAL_OUT"
  kubectl -n "$NS" exec "$BENCH_POD" -- bash -lc "
    ls -la '$RESULT_DIR_POD'/conc-${c}.json '$RESULT_DIR_POD'/conc-${c}.log 2>/dev/null || true
  "
  kubectl -n "$NS" cp "$BENCH_POD:$RESULT_DIR_POD/conc-${c}.json" "$LOCAL_OUT/conc-${c}.json" 2>/dev/null || true
  kubectl -n "$NS" cp "$BENCH_POD:$RESULT_DIR_POD/conc-${c}.log" "$LOCAL_OUT/conc-${c}.log" 2>/dev/null || true
  # Keep summary.tsv / seed fresh
  kubectl -n "$NS" cp "$BENCH_POD:$RESULT_DIR_POD/summary.tsv" "$LOCAL_OUT/summary.tsv" 2>/dev/null || true
  kubectl -n "$NS" cp "$BENCH_POD:$RESULT_DIR_POD/seed.txt" "$LOCAL_OUT/seed.txt" 2>/dev/null || true
}

run_sweep_in_pod() {
  # Upload sweep script and run with our knobs. Stop after each C by driving
  # the loop from the host so we can re-render the report.
  kubectl -n "$NS" cp "$ROOT/scripts/bench-concurrency-sweep.sh" "$BENCH_POD:/tmp/bench-concurrency-sweep.sh"
  kubectl -n "$NS" exec "$BENCH_POD" -- bash -lc "rm -rf '$RESULT_DIR_POD' && mkdir -p '$RESULT_DIR_POD'"

  # Probe once
  log "===== PROBE $(date -u +%Y-%m-%dT%H:%M:%SZ) ====="
  kubectl -n "$NS" exec "$BENCH_POD" -- bash -lc "
    set -euo pipefail
    export BASE_URL='$URL' MODEL='$MODEL' INPUT_LEN='$INPUT_LEN' OUTPUT_LEN='$OUTPUT_LEN'
    export RESULT_DIR='$RESULT_DIR_POD' SEED='$SEED' TARGET_SECS='$TARGET_SECS'
    export CONCURRENCIES='1'
    # Only probe path: run first part manually
    vllm bench serve \
      --backend openai --base-url \"\$BASE_URL\" --model \"\$MODEL\" \
      --endpoint /v1/completions --dataset-name random \
      --random-input-len \"\$INPUT_LEN\" --random-output-len \"\$OUTPUT_LEN\" \
      --seed \"\$SEED\" --ignore-eos --request-rate inf \
      --max-concurrency 1 --num-prompts 1 --num-warmups 0 \
      --save-result --result-dir \"\$RESULT_DIR\" --result-filename probe.json \
      > \"\$RESULT_DIR/probe.log\" 2>&1 || true
    echo SEED > \"\$RESULT_DIR/seed.txt\"
    echo -e \"concurrency\tnum_prompts\tseed\tduration_s\trequest_throughput\toutput_tok_s\tmean_ttft_ms\tmean_tpot_ms\tmean_e2el_ms\" > \"\$RESULT_DIR/summary.tsv\"
    echo \"\$SEED\" > \"\$RESULT_DIR/seed.txt\"
  "
  kubectl -n "$NS" cp "$BENCH_POD:$RESULT_DIR_POD/probe.json" "$LOCAL_OUT/probe.json" 2>/dev/null || true
  kubectl -n "$NS" cp "$BENCH_POD:$RESULT_DIR_POD/probe.log" "$LOCAL_OUT/probe.log" 2>/dev/null || true

  LATENCY_S=4.0
  if [[ -f "$LOCAL_OUT/probe.json" ]]; then
    LATENCY_S=$(python3 -c 'import json; print(float(json.load(open("'"$LOCAL_OUT"'/probe.json")).get("duration", 4.0)))')
  fi
  log "seed_latency_s=$LATENCY_S seed=$SEED"

  for c in $CONCURRENCIES; do
    log "===== START C=$c $(date -u +%Y-%m-%dT%H:%M:%SZ) ====="
    tier_seed=$((SEED + c * 10007))
    est_lat=$(python3 -c "print(max(0.5, ${LATENCY_S} * (1.0 + 0.12 * (${c} - 1) ** 0.5)))")
    waves=$(python3 -c "import math; print(max(2, int(math.ceil(${TARGET_SECS} / ${est_lat}))))")
    num_prompts=$((c * waves))
    log "C=$c num_prompts=$num_prompts tier_seed=$tier_seed est_lat=$est_lat"

    set +e
    kubectl -n "$NS" exec "$BENCH_POD" -- bash -lc "
      set -uo pipefail
      vllm bench serve \
        --backend openai \
        --base-url '$URL' \
        --model '$MODEL' \
        --endpoint /v1/completions \
        --dataset-name random \
        --random-input-len '$INPUT_LEN' \
        --random-output-len '$OUTPUT_LEN' \
        --seed '$tier_seed' \
        --ignore-eos \
        --request-rate inf \
        --max-concurrency '$c' \
        --num-prompts '$num_prompts' \
        --num-warmups 0 \
        --save-result \
        --result-dir '$RESULT_DIR_POD' \
        --result-filename 'conc-${c}.json' \
        > '$RESULT_DIR_POD/conc-${c}.log' 2>&1
      rc=\$?
      python3 - <<'PY'
import json
from pathlib import Path
p = Path('$RESULT_DIR_POD') / 'conc-${c}.json'
summary = Path('$RESULT_DIR_POD') / 'summary.tsv'
if not p.is_file():
    raise SystemExit('missing result json')
d = json.loads(p.read_text())
failed = int(d.get('failed') or 0)
completed = int(d.get('completed') or 0)
line = (
    f\"${c}\t${num_prompts}\t${tier_seed}\t{float(d.get('duration', 0)):.2f}\t\"
    f\"{d.get('request_throughput', '')}\t{d.get('output_throughput', '')}\t\"
    f\"{d.get('mean_ttft_ms', '')}\t{d.get('mean_tpot_ms', '')}\t\"
    f\"{d.get('mean_e2el_ms', '')}\"
)
with summary.open('a') as f:
    f.write(line + '\n')
print(line)
print(f'completed={completed} failed={failed}')
# Prefer result metrics over flaky kubectl/bench exit codes.
if failed > 0 or completed <= 0:
    raise SystemExit(f'bench failed: completed={completed} failed={failed}')
PY
    " 2>&1 | tee -a "$MASTER_LOG"
    krc=${PIPESTATUS[0]}
    set -e
    copy_tier "$c"
    # Success = local result json with 0 failures (kubectl exit can be flaky with long output).
    rc=0
    if [[ ! -f "$LOCAL_OUT/conc-${c}.json" ]]; then
      rc=1
    else
      failed=$(python3 -c 'import json; print(int(json.load(open("'"$LOCAL_OUT"'/conc-'"$c"'.json")).get("failed") or 0))')
      completed=$(python3 -c 'import json; print(int(json.load(open("'"$LOCAL_OUT"'/conc-'"$c"'.json")).get("completed") or 0))')
      if [[ "$failed" -gt 0 || "$completed" -le 0 ]]; then
        rc=1
      fi
    fi
    render_report
    log "===== DONE C=$c exit=$rc kubectl_rc=$krc failed=${failed:-?} completed=${completed:-?} $(date -u +%Y-%m-%dT%H:%M:%SZ) ====="
    if [[ $rc -ne 0 ]]; then
      log "ERROR: C=$c failed; stopping sweep"
      exit "$rc"
    fi
  done
}

{
  log "===== MASTER START $(date -u +%Y-%m-%dT%H:%M:%SZ) ====="
  kubectl -n "$NS" wait --for=condition=Ready "pod/$BENCH_POD" --timeout=10m
  ensure_endpoint
  resolve_model
  run_sweep_in_pod
  log "===== ALL DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) ====="
} 2>&1 | tee -a "$MASTER_LOG"
