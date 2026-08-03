#!/usr/bin/env bash
# Minimal-depth torch.profiler sweep across five H200 recipe strategies.
# See profiler_sweep.md.
#
# Usage:
#   export KUBECONFIG=/Users/mimehta/kubeconfigs/kubeconfig.fozzie
#   ./scripts/run-profiler-sweep.sh
#   CONFIGS="tp16 tep16" ./scripts/run-profiler-sweep.sh   # subset

set -euo pipefail

export KUBECONFIG="${KUBECONFIG:-/Users/mimehta/kubeconfigs/kubeconfig.fozzie}"
NS="${NS:-kimi-k3}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="${LOG:-/tmp/profiler-sweep.log}"

export NUM_LAYERS="${NUM_LAYERS:-4}"
# Single-pod smoke used 8 experts (TP=2, no EP). Multi-node TP16/EP16 needs
# num_experts divisible by EP (and large enough for marlin MXFP4 repack) → 32.
export NUM_EXPERTS="${NUM_EXPERTS:-32}"
export NUM_EXPERTS_PER_TOKEN="${NUM_EXPERTS_PER_TOKEN:-2}"
export NUM_SHARED_EXPERTS="${NUM_SHARED_EXPERTS:-1}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-1024}"
export ENABLE_TORCH_PROFILER=1
export VLLM_RPC_TIMEOUT="${VLLM_RPC_TIMEOUT:-1800000}"
# H200 ≠ SM100; do not enable K3 latent-MoE tail fusion.
export VLLM_ENABLE_K3_LATENT_MOE_TAIL_FUSION="${VLLM_ENABLE_K3_LATENT_MOE_TAIL_FUSION:-0}"

# Prefer 2-node configs first so a missing 4th node does not block the rest;
# P/D last (needs 4 Ready nodes). Override with CONFIGS=...
CONFIGS="${CONFIGS:-tp16 tep16 pp2 dp2 pd}"

exec > >(tee -a "$LOG") 2>&1
echo "===== profiler sweep start $(date -u +%Y-%m-%dT%H:%M:%SZ) ====="
echo "CONFIGS=$CONFIGS layers=$NUM_LAYERS experts=$NUM_EXPERTS/$NUM_EXPERTS_PER_TOKEN/$NUM_SHARED_EXPERTS"

stop_all() {
  local n=${1:-4}
  echo "==> Stopping serves on ranks 0..$((n-1))"
  for i in $(seq 0 $((n - 1))); do
    if kubectl -n "$NS" get "pod/vllm-recipe-$i" >/dev/null 2>&1; then
      kubectl -n "$NS" cp "$ROOT/scripts/stop-inpod-vllm.sh" "vllm-recipe-$i:/tmp/stop-inpod-vllm.sh"
      kubectl -n "$NS" exec "vllm-recipe-$i" -- bash /tmp/stop-inpod-vllm.sh 2>/dev/null || true
      kubectl -n "$NS" exec "vllm-recipe-$i" -- bash -c \
        'if [[ -f /tmp/pd-router.pid ]]; then kill "$(cat /tmp/pd-router.pid)" 2>/dev/null || true; fi' \
        2>/dev/null || true
    fi
  done
  sleep 3
}

capture_2node() {
  local config=$1
  local profile_dir="/tmp/vllm_profile/${config}"
  echo "==> Capture $config via profile-short-query.sh"
  kubectl -n "$NS" cp "$ROOT/scripts/profile-short-query.sh" vllm-recipe-0:/tmp/profile-short-query.sh
  kubectl -n "$NS" exec vllm-recipe-0 -- bash -c \
    "MODEL=auto PROFILE_DIR=$profile_dir bash /tmp/profile-short-query.sh"
  CONFIG="$config" PROFILE_DIR="$profile_dir" NNODES=2 \
    NUM_LAYERS="$NUM_LAYERS" NUM_EXPERTS="$NUM_EXPERTS" \
    NUM_EXPERTS_PER_TOKEN="$NUM_EXPERTS_PER_TOKEN" NUM_SHARED_EXPERTS="$NUM_SHARED_EXPERTS" \
    MAX_MODEL_LEN="$MAX_MODEL_LEN" \
    META_EXTRA="config=$config" \
    "$ROOT/scripts/copy-profiler-traces.sh"
}

wait_ready_nodes() {
  local need=$1
  local max_wait=${2:-1800}
  local elapsed=0
  while true; do
    local ready
    ready=$(kubectl get nodes --no-headers 2>/dev/null | awk '$2=="Ready"{c++} END{print c+0}')
    echo "    Ready nodes=$ready (need $need) elapsed=${elapsed}s"
    if [[ "$ready" -ge "$need" ]]; then
      return 0
    fi
    if [[ "$elapsed" -ge "$max_wait" ]]; then
      echo "timed out waiting for $need Ready nodes" >&2
      return 1
    fi
    sleep 30
    elapsed=$((elapsed + 30))
  done
}

run_pd() {
  export PROFILE_DIR=/tmp/vllm_profile/pd
  echo "==> Waiting for 4 Ready nodes for P/D"
  wait_ready_nodes 4 3600
  echo "==> Deploy dummy P/D"
  "$ROOT/scripts/deploy-recipe-dummy-pd.sh"
  echo "==> Capture P/D"
  PROFILE_DIR="$PROFILE_DIR" "$ROOT/scripts/profile-pd-short-query.sh"
  CONFIG=pd PROFILE_DIR="$PROFILE_DIR" NNODES=4 \
    NUM_LAYERS="$NUM_LAYERS" NUM_EXPERTS="$NUM_EXPERTS" \
    NUM_EXPERTS_PER_TOKEN="$NUM_EXPERTS_PER_TOKEN" NUM_SHARED_EXPERTS="$NUM_SHARED_EXPERTS" \
    MAX_MODEL_LEN="$MAX_MODEL_LEN" \
    META_EXTRA="deploy=deploy-recipe-dummy-pd.sh" \
    "$ROOT/scripts/copy-profiler-traces.sh"
  stop_all 4
}

run_tp16() {
  export PROFILE_DIR=/tmp/vllm_profile/tp16
  export NNODES=2
  kubectl -n "$NS" scale statefulset/vllm-recipe --replicas=2
  "$ROOT/scripts/deploy-recipe-dummy.sh"
  capture_2node tp16
  stop_all 2
}

run_tep16() {
  export PROFILE_DIR=/tmp/vllm_profile/tep16
  export NNODES=2
  kubectl -n "$NS" scale statefulset/vllm-recipe --replicas=2
  "$ROOT/scripts/deploy-recipe-dummy-tep.sh"
  capture_2node tep16
  stop_all 2
}

run_pp2() {
  export PROFILE_DIR=/tmp/vllm_profile/pp2
  export NNODES=2
  # PP may need slightly larger seq budget; keep model-len 1024.
  export MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
  kubectl -n "$NS" scale statefulset/vllm-recipe --replicas=2
  "$ROOT/scripts/deploy-recipe-dummy-pp.sh"
  capture_2node pp2
  stop_all 2
  unset MAX_NUM_SEQS
}

run_dp2() {
  export PROFILE_DIR=/tmp/vllm_profile/dp2
  export NNODES=2
  # Dummy DP2 with shallow model: keep modest seqs.
  export MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
  kubectl -n "$NS" scale statefulset/vllm-recipe --replicas=2
  "$ROOT/scripts/deploy-recipe-dummy-dp.sh"
  capture_2node dp2
  stop_all 2
  unset MAX_NUM_SEQS
}

# Quiet cluster first
stop_all 4

for cfg in $CONFIGS; do
  echo
  echo "########## CONFIG=$cfg $(date -u +%Y-%m-%dT%H:%M:%SZ) ##########"
  case "$cfg" in
    pd) run_pd ;;
    tp16) run_tp16 ;;
    tep16) run_tep16 ;;
    pp2) run_pp2 ;;
    dp2) run_dp2 ;;
    *) echo "unknown config: $cfg" >&2; exit 1 ;;
  esac
done

mkdir -p "$ROOT/profiler-results"
{
  echo "# Profiler sweep results"
  echo
  echo "- Started/finished: see /tmp/profiler-sweep.log"
  echo "- Architecture: NUM_LAYERS=$NUM_LAYERS NUM_EXPERTS=$NUM_EXPERTS NUM_EXPERTS_PER_TOKEN=$NUM_EXPERTS_PER_TOKEN NUM_SHARED_EXPERTS=$NUM_SHARED_EXPERTS"
  echo "- Protocol: warmup + 10 in / 5 out"
  echo "- Configs: $CONFIGS"
  echo
  echo "## Layout"
  echo '```'
  find "$ROOT/profiler-results" -type f \( -name '*.gz' -o -name meta.txt \) | sort
  echo '```'
} >"$ROOT/profiler-results/README.md"

echo "===== profiler sweep done $(date -u +%Y-%m-%dT%H:%M:%SZ) ====="
ls -la "$ROOT/profiler-results"
