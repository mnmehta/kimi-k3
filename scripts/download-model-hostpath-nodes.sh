#!/usr/bin/env bash
# Download a Hugging Face model onto hostPath /mnt/local/kimi-k3/models on
# each listed node (no GPU required — works while HPC holds GPUs).
#
# Usage:
#   export KUBECONFIG=...
#   MODEL_ID=RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic \
#   LOCAL_NAME=gemma-4-26B-A4B-it-FP8-dynamic \
#   EXPECTED_SHARDS=0 \
#   NODES="g1191e4 g13c364 gc37d78 gf2a612" \
#   ./scripts/download-model-hostpath-nodes.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NS="${NS:-kimi-k3}"
MODEL_ID="${MODEL_ID:?MODEL_ID required}"
LOCAL_NAME="${LOCAL_NAME:-$(basename "$MODEL_ID")}"
LOCAL_DIR="/models/${LOCAL_NAME}"
EXPECTED_SHARDS="${EXPECTED_SHARDS:-0}"
NODES="${NODES:-g1191e4 g13c364 gc37d78 gf2a612}"
IMAGE="${IMAGE:-vllm/vllm-openai:kimi-k3}"

echo "==> MODEL_ID=$MODEL_ID"
echo "    LOCAL_DIR(hostPath)=/mnt/local/kimi-k3/models/${LOCAL_NAME}"
echo "    NODES=$NODES EXPECTED_SHARDS=$EXPECTED_SHARDS"

pod_name_for_node() {
  echo "model-dl-$(echo "$1" | tr -cd 'A-Za-z0-9-' | tr '[:upper:]' '[:lower:]')"
}

launch_pod() {
  local node=$1
  local name
  name="$(pod_name_for_node "$node")"
  kubectl -n "$NS" delete pod "$name" --ignore-not-found --wait=false >/dev/null 2>&1 || true
  local i
  for i in $(seq 1 60); do
    kubectl -n "$NS" get pod "$name" >/dev/null 2>&1 || break
    sleep 1
  done
  kubectl -n "$NS" apply -f - >/dev/null <<EOF
apiVersion: v1
kind: Pod
metadata:
  name: $name
  namespace: $NS
  labels:
    app: model-dl
spec:
  restartPolicy: Never
  nodeName: $node
  hostNetwork: true
  dnsPolicy: ClusterFirstWithHostNet
  containers:
  - name: dl
    image: $IMAGE
    imagePullPolicy: IfNotPresent
    command: ["sleep", "infinity"]
    securityContext:
      privileged: true
    volumeMounts:
    - name: model
      mountPath: /models
    resources:
      requests:
        cpu: "4"
        memory: 16Gi
        ephemeral-storage: 20Gi
  volumes:
  - name: model
    hostPath:
      path: /mnt/local/kimi-k3/models
      type: DirectoryOrCreate
  tolerations:
  - operator: Exists
EOF
  kubectl -n "$NS" wait --for=condition=Ready "pod/$name" --timeout=15m >/dev/null
  printf '%s\n' "$name"
}

echo "==> Launching no-GPU download pods + copying script"
for node in $NODES; do
  pod="$(launch_pod "$node")"
  kubectl -n "$NS" cp "$ROOT/scripts/download-model-inpod.sh" "$pod:/tmp/download-model-inpod.sh"
  echo "  $node -> $pod"
done

echo "==> Starting downloads (parallel)"
for node in $NODES; do
  pod="$(pod_name_for_node "$node")"
  kubectl -n "$NS" exec "$pod" -- bash -c "
    if [[ -f /tmp/model-download.pid ]] && kill -0 \"\$(cat /tmp/model-download.pid)\" 2>/dev/null; then
      echo already-running pid=\$(cat /tmp/model-download.pid)
      exit 0
    fi
    rm -f /tmp/model-download.log /tmp/model-download.pid
    MODEL_ID='$MODEL_ID' MODEL_ROOT=/models LOCAL_DIR='$LOCAL_DIR' \
      HF_HOME=/models/hf EXPECTED_SHARDS='$EXPECTED_SHARDS' \
      nohup bash /tmp/download-model-inpod.sh > /tmp/model-download.log 2>&1 &
    echo \$! > /tmp/model-download.pid
    echo started node=$node pod=$pod pid=\$(cat /tmp/model-download.pid)
  "
done

echo "==> Waiting for .download-complete on every node"
pending=1
while [[ "$pending" == "1" ]]; do
  pending=0
  for node in $NODES; do
    pod="$(pod_name_for_node "$node")"
    if kubectl -n "$NS" exec "$pod" -- test -f "${LOCAL_DIR}/.download-complete" 2>/dev/null; then
      echo "  $node: complete"
      kubectl -n "$NS" exec "$pod" -- cat "${LOCAL_DIR}/.download-complete" || true
      kubectl -n "$NS" exec "$pod" -- du -sh "$LOCAL_DIR" || true
    else
      pending=1
      lines="$(kubectl -n "$NS" exec "$pod" -- bash -c 'tail -n 2 /tmp/model-download.log 2>/dev/null || echo "(no log yet)"' 2>/dev/null || true)"
      echo "  $node: downloading... $lines"
    fi
  done
  if [[ "$pending" == "1" ]]; then
    sleep 60
  fi
done

echo "==> All nodes have ${LOCAL_DIR}/.download-complete"
echo "    Download pods left running (app=model-dl); delete when done:"
echo "    kubectl -n $NS delete pod -l app=model-dl"
