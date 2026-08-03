#!/usr/bin/env bash
# Copy hostPath /mnt/local/kimi-k3/models/Kimi-K3 onto nodes that lack it.
# Uses short-lived no-GPU pods (so HPC/GPU occupancy does not block the copy).
#
# Usage:
#   export KUBECONFIG=...
#   ./scripts/sync-model-to-nodes.sh
#   SRC_POD=vllm-recipe-0 DST_NODES="g11cd44 gf2a612" ./scripts/sync-model-to-nodes.sh

set -euo pipefail

NS="${NS:-kimi-k3}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${PORT:-19001}"
IMAGE="${IMAGE:-vllm/vllm-openai:kimi-k3}"
SRC_POD="${SRC_POD:-vllm-recipe-0}"
DST_NODES="${DST_NODES:-g11cd44 gf2a612}"

if ! kubectl -n "$NS" exec "$SRC_POD" -- test -f /models/Kimi-K3/.download-complete; then
  echo "source $SRC_POD missing /models/Kimi-K3/.download-complete" >&2
  exit 1
fi
SRC_IP="$(kubectl -n "$NS" get pod "$SRC_POD" -o jsonpath='{.status.podIP}')"
SRC_NODE="$(kubectl -n "$NS" get pod "$SRC_POD" -o jsonpath='{.spec.nodeName}')"
echo "==> Source $SRC_POD on $SRC_NODE ($SRC_IP)"

launch_recv_pod() {
  local node=$1
  local name="model-sync-$(echo "$node" | tr -cd 'A-Za-z0-9-' | tr '[:upper:]' '[:lower:]')"
  kubectl -n "$NS" delete pod "$name" --ignore-not-found --wait=true 2>/dev/null || true
  kubectl -n "$NS" apply -f - <<EOF
apiVersion: v1
kind: Pod
metadata:
  name: $name
  namespace: $NS
  labels:
    app: model-sync
spec:
  restartPolicy: Never
  nodeName: $node
  hostNetwork: true
  dnsPolicy: ClusterFirstWithHostNet
  containers:
  - name: sync
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
        cpu: "2"
        memory: 8Gi
        ephemeral-storage: 20Gi
  volumes:
  - name: model
    hostPath:
      path: /mnt/local/kimi-k3/models
      type: DirectoryOrCreate
  tolerations:
  - operator: Exists
EOF
  kubectl -n "$NS" wait --for=condition=Ready "pod/$name" --timeout=10m
  echo "$name"
}

for node in $DST_NODES; do
  echo
  echo "==> Destination node $node"
  recv="$(launch_recv_pod "$node")"
  kubectl -n "$NS" exec "$recv" -- bash -c \
    "pkill -f sync-model-recv 2>/dev/null || true; rm -rf /models/Kimi-K3; mkdir -p /models"

  kubectl -n "$NS" exec "$recv" -- bash -c "
    cat > /tmp/sync-model-recv.py <<'PY'
import socket, subprocess, sys
port = int(sys.argv[1])
s = socket.socket()
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(('0.0.0.0', port))
s.listen(1)
print(f'recv listening :{port}', flush=True)
conn, addr = s.accept()
print(f'recv accepted {addr}', flush=True)
p = subprocess.Popen(['tar', '-C', '/models', '-xf', '-'], stdin=subprocess.PIPE)
n = 0
while True:
    buf = conn.recv(8 * 1024 * 1024)
    if not buf:
        break
    p.stdin.write(buf)
    n += len(buf)
    if n and n % (16 * 1024**3) < 8 * 1024 * 1024:
        print(f'recv GiB={n/1024**3:.1f}', flush=True)
p.stdin.close()
rc = p.wait()
conn.close(); s.close()
print(f'recv done GiB={n/1024**3:.2f} tar_rc={rc}', flush=True)
sys.exit(rc)
PY
    nohup python3 /tmp/sync-model-recv.py $PORT > /tmp/sync-model-recv.log 2>&1 &
    echo \$! > /tmp/sync-model-recv.pid
    sleep 1
  "
  DST_IP="$(kubectl -n "$NS" get pod "$recv" -o jsonpath='{.status.podIP}')"
  echo "    streaming $SRC_IP → $DST_IP:$PORT (pod $recv)"

  kubectl -n "$NS" exec "$SRC_POD" -- bash -c "
    cat > /tmp/sync-model-send.py <<'PY'
import socket, subprocess, sys, time
host, port = sys.argv[1], int(sys.argv[2])
for attempt in range(90):
    try:
        s = socket.create_connection((host, port), timeout=30)
        break
    except OSError as e:
        print(f'connect retry {attempt}: {e}', flush=True)
        time.sleep(2)
else:
    raise SystemExit('connect failed')
print(f'send connected {host}:{port}', flush=True)
p = subprocess.Popen(['tar', '-C', '/models', '-cf', '-', 'Kimi-K3'], stdout=subprocess.PIPE)
n = 0
assert p.stdout is not None
while True:
    buf = p.stdout.read(8 * 1024 * 1024)
    if not buf:
        break
    s.sendall(buf)
    n += len(buf)
    if n and n % (16 * 1024**3) < 8 * 1024 * 1024:
        print(f'send GiB={n/1024**3:.1f}', flush=True)
rc = p.wait(); s.close()
print(f'send done GiB={n/1024**3:.2f} tar_rc={rc}', flush=True)
sys.exit(rc)
PY
    python3 /tmp/sync-model-send.py '$DST_IP' $PORT
  "

  for _ in $(seq 1 900); do
    if kubectl -n "$NS" exec "$recv" -- bash -c \
        '! kill -0 "$(cat /tmp/sync-model-recv.pid 2>/dev/null)" 2>/dev/null'; then
      break
    fi
    if (( _ % 12 == 0 )); then
      kubectl -n "$NS" exec "$recv" -- bash -c \
        'tail -n 1 /tmp/sync-model-recv.log; du -sh /models/Kimi-K3 2>/dev/null || true'
    fi
    sleep 10
  done
  kubectl -n "$NS" exec "$recv" -- tail -n 8 /tmp/sync-model-recv.log

  kubectl -n "$NS" cp "$ROOT/scripts/download-model-inpod.sh" "$recv:/tmp/download-model-inpod.sh"
  kubectl -n "$NS" exec "$recv" -- bash -c \
    'MODEL_ROOT=/models LOCAL_DIR=/models/Kimi-K3 HF_HOME=/models/hf \
       bash /tmp/download-model-inpod.sh' | tail -25

  echo "    verified on $node — leaving pod $recv (delete later if desired)"
done

echo
echo "==> hostPath check"
for node in $SRC_NODE $DST_NODES; do
  kubectl run "verify-$node" -n "$NS" --rm -i --restart=Never --overrides="
{\"spec\":{\"nodeName\":\"$node\",\"hostNetwork\":true,\"containers\":[{\"name\":\"c\",\"image\":\"busybox:1.36\",\"command\":[\"sh\",\"-c\",\"du -sh /h/kimi-k3/models/Kimi-K3; test -f /h/kimi-k3/models/Kimi-K3/.download-complete && head -1 /h/kimi-k3/models/Kimi-K3/.download-complete || echo NO_MARKER\"],\"volumeMounts\":[{\"name\":\"h\",\"mountPath\":\"/h\"}],\"resources\":{\"requests\":{\"cpu\":\"10m\",\"memory\":\"32Mi\"}}}],\"volumes\":[{\"name\":\"h\",\"hostPath\":{\"path\":\"/mnt/local\",\"type\":\"DirectoryOrCreate\"}}],\"tolerations\":[{\"operator\":\"Exists\"}]}}
" --image=busybox:1.36 --timeout=90s 2>&1 | grep -vE 'warning:|If you|pod .* deleted|Unknown stream' | sed "s/^/[$node] /"
done
echo "Done."
