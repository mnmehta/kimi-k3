#!/usr/bin/env bash
# Download moonshotai/Kimi-K3 onto shared storage for the recipe pods.
#
# STORAGE_BACKEND:
#   auto      try PVC (shared-vast); if not Bound quickly, fall back to hostPath
#   pvc       require PVC Bound, then run manifests/model-download-job.yaml
#   hostpath  download on each vllm-recipe pod into container-local /models
#             (name kept for compatibility; not a hostPath mount — see STORAGE.md)
#
# Usage:
#   export KUBECONFIG=/Users/mimehta/kubeconfigs/kubeconfig.fozzie
#   ./scripts/download-model.sh
#   STORAGE_BACKEND=hostpath FOLLOW=1 ./scripts/download-model.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export KUBECONFIG="${KUBECONFIG:-/Users/mimehta/kubeconfigs/kubeconfig.fozzie}"
NS="${NS:-kimi-k3}"
JOB="${JOB:-kimi-k3-model-download}"
PVC="${PVC:-kimi-k3-model}"
STORAGE_BACKEND="${STORAGE_BACKEND:-auto}"
FOLLOW="${FOLLOW:-1}"
REPLACE_JOB="${REPLACE_JOB:-1}"
PVC_WAIT_S="${PVC_WAIT_S:-120}"
NNODES="${NNODES:-2}"

apply_namespace() {
  kubectl apply -f "$ROOT/manifests/namespace.yaml"
}

publish_script_cm() {
  kubectl -n "$NS" create configmap kimi-k3-model-download \
    --from-file=download-model-inpod.sh="$ROOT/scripts/download-model-inpod.sh" \
    --dry-run=client -o yaml | kubectl apply -f -
}

wait_pvc_bound() {
  local timeout_s="$1"
  local end=$((SECONDS + timeout_s))
  while (( SECONDS < end )); do
    local phase
    phase="$(kubectl -n "$NS" get pvc "$PVC" -o jsonpath='{.status.phase}' 2>/dev/null || true)"
    if [[ "$phase" == "Bound" ]]; then
      return 0
    fi
    # Surface auth/provisioner errors early.
    if kubectl -n "$NS" describe pvc "$PVC" 2>/dev/null | grep -q 'ProvisioningFailed'; then
      echo "PVC provisioning failed:" >&2
      kubectl -n "$NS" describe pvc "$PVC" | sed -n '/Events:/,$p' >&2 || true
      return 1
    fi
    sleep 5
  done
  echo "PVC $PVC not Bound within ${timeout_s}s (phase=$(kubectl -n "$NS" get pvc "$PVC" -o jsonpath='{.status.phase}' 2>/dev/null || echo missing))" >&2
  return 1
}

download_via_pvc() {
  echo "==> STORAGE_BACKEND=pvc"
  apply_namespace
  kubectl apply -f "$ROOT/manifests/model-pvc.yaml"
  wait_pvc_bound 1800
  kubectl -n "$NS" get pvc "$PVC"
  publish_script_cm

  if kubectl -n "$NS" get job "$JOB" >/dev/null 2>&1; then
    if [[ "$REPLACE_JOB" == "1" ]]; then
      echo "==> Deleting existing Job $JOB"
      kubectl -n "$NS" delete job "$JOB" --wait=true
    else
      echo "Job $JOB already exists (set REPLACE_JOB=1 to recreate)" >&2
      exit 1
    fi
  fi

  echo "==> Starting download Job"
  kubectl apply -f "$ROOT/manifests/model-download-job.yaml"
  kubectl -n "$NS" wait --for=condition=Ready pod -l job-name="$JOB" --timeout=30m || true
  kubectl -n "$NS" get pods -l job-name="$JOB" -o wide

  if [[ "$FOLLOW" == "1" ]]; then
    echo "==> Streaming Job logs (Ctrl-C stops follow only)"
    kubectl -n "$NS" logs -f "job/$JOB" --all-containers=true || true
  fi

  echo "==> Waiting for Job complete (1.5TB+; can take hours)"
  kubectl -n "$NS" wait --for=condition=complete "job/$JOB" --timeout=24h
  kubectl -n "$NS" get job "$JOB"
}

ensure_recipe_pods() {
  echo "==> Ensuring recipe StatefulSet (container-local /models)"
  kubectl apply -f "$ROOT/manifests/vllm-recipe.yaml"
  kubectl -n "$NS" scale statefulset/vllm-recipe --replicas="$NNODES"
  kubectl -n "$NS" rollout status statefulset/vllm-recipe --timeout=30m
  for i in $(seq 0 $((NNODES - 1))); do
    kubectl -n "$NS" wait --for=condition=Ready "pod/vllm-recipe-$i" --timeout=30m
  done
  kubectl -n "$NS" get pods -l app=vllm-recipe -o wide
  echo "==> Storage check (need multi-TB free on /models)"
  for i in $(seq 0 $((NNODES - 1))); do
    kubectl -n "$NS" exec "vllm-recipe-$i" -- bash -c \
      'mkdir -p /models; df -h /models; findmnt -T /models || true'
  done
}

download_via_hostpath() {
  echo "==> STORAGE_BACKEND=hostpath (container-local /models on ~28T overlay)"
  echo "    See manifests/STORAGE.md for PVC status / why hostPath /var/lib is unsafe."
  apply_namespace
  ensure_recipe_pods

  echo "==> Copying in-pod download script to each rank"
  for i in $(seq 0 $((NNODES - 1))); do
    kubectl -n "$NS" cp \
      "$ROOT/scripts/download-model-inpod.sh" \
      "vllm-recipe-$i:/tmp/download-model-inpod.sh"
  done

  echo "==> Starting downloads on each recipe pod (parallel)"
  for i in $(seq 0 $((NNODES - 1))); do
    # NOTE: do not `pkill -f download-model-inpod.sh` here — it matches this
    # kubectl exec argv and SIGTERMs the launcher (exit 143).
    kubectl -n "$NS" exec "vllm-recipe-$i" -- bash -c \
      'if [[ -f /tmp/model-download.pid ]] && kill -0 "$(cat /tmp/model-download.pid)" 2>/dev/null; then
         echo "download already running pid=$(cat /tmp/model-download.pid)"
         exit 0
       fi
       rm -f /tmp/model-download.log /tmp/model-download.pid
       MODEL_ROOT=/models LOCAL_DIR=/models/Kimi-K3 HF_HOME=/models/hf \
         nohup bash /tmp/download-model-inpod.sh > /tmp/model-download.log 2>&1 &
       echo $! > /tmp/model-download.pid
       echo started download on rank-'"$i"' pid=$(cat /tmp/model-download.pid)'
  done

  if [[ "$FOLLOW" == "1" ]]; then
    echo "==> Tailing rank-0 download log (Ctrl-C stops follow only)"
    kubectl -n "$NS" exec vllm-recipe-0 -- tail -f /tmp/model-download.log || true
  fi

  echo "==> Waiting for .download-complete on every rank"
  local pending=1
  while [[ "$pending" == "1" ]]; do
    pending=0
    for i in $(seq 0 $((NNODES - 1))); do
      if ! kubectl -n "$NS" exec "vllm-recipe-$i" -- \
          test -f /models/Kimi-K3/.download-complete 2>/dev/null; then
        pending=1
        local lines
        lines="$(kubectl -n "$NS" exec "vllm-recipe-$i" -- \
          bash -c 'tail -n 3 /tmp/model-download.log 2>/dev/null || echo "(no log yet)"' 2>/dev/null || true)"
        echo "  rank-$i: still downloading... $lines"
      else
        echo "  rank-$i: complete"
        kubectl -n "$NS" exec "vllm-recipe-$i" -- cat /models/Kimi-K3/.download-complete || true
      fi
    done
    if [[ "$pending" == "1" ]]; then
      sleep 60
    fi
  done
}

case "$STORAGE_BACKEND" in
  pvc)
    download_via_pvc
    ;;
  hostpath)
    download_via_hostpath
    ;;
  auto)
    apply_namespace
    echo "==> STORAGE_BACKEND=auto — trying PVC first (${PVC_WAIT_S}s)"
    kubectl apply -f "$ROOT/manifests/model-pvc.yaml" || true
    if wait_pvc_bound "$PVC_WAIT_S"; then
      download_via_pvc
    else
      echo "==> Falling back to hostPath"
      download_via_hostpath
    fi
    ;;
  *)
    echo "Unknown STORAGE_BACKEND=$STORAGE_BACKEND (use auto|pvc|hostpath)" >&2
    exit 1
    ;;
esac

echo
echo "Weights ready. Next: ./scripts/deploy-recipe.sh"
echo "Storage notes: manifests/STORAGE.md"
