#!/usr/bin/env bash
# Copy /models/Kimi-K3 between recipe pods over the cluster network (tar stream).
# Faster than re-downloading ~1.5 TiB from Hugging Face when some ranks already
# have a verified hostPath copy.
#
# Usage:
#   export KUBECONFIG=...
#   ./scripts/sync-model-hostpath.sh              # auto: sources → empty ranks
#   SRC_RANK=0 DST_RANKS="2 3" ./scripts/sync-model-hostpath.sh
#   PORT=19001 ./scripts/sync-model-hostpath.sh

set -euo pipefail

NS="${NS:-kimi-k3}"
PORT="${PORT:-19001}"
MODEL_DIR="${MODEL_DIR:-/models/Kimi-K3}"
MARKER="${MARKER:-${MODEL_DIR}/.download-complete}"

PODS=()
while IFS= read -r p; do
  [[ -n "$p" ]] && PODS+=("$p")
done < <(kubectl -n "$NS" get pods -l app=vllm-recipe \
  -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}' | sort -V)
if [[ ${#PODS[@]} -lt 2 ]]; then
  echo "need ≥2 vllm-recipe pods; got ${#PODS[@]}" >&2
  exit 1
fi

echo "==> Recipe pods"
kubectl -n "$NS" get pods -l app=vllm-recipe -o wide

have_marker() {
  local pod=$1
  kubectl -n "$NS" exec "$pod" -- test -f "$MARKER" 2>/dev/null
}

if [[ -n "${SRC_RANK:-}" ]]; then
  SRC_POD="vllm-recipe-${SRC_RANK}"
else
  SRC_POD=""
  for p in "${PODS[@]}"; do
    if have_marker "$p"; then SRC_POD=$p; break; fi
  done
fi
if [[ -z "$SRC_POD" ]]; then
  echo "no source pod with $MARKER; run ./scripts/download-model.sh first" >&2
  exit 1
fi

if [[ -n "${DST_RANKS:-}" ]]; then
  DST_PODS=()
  for r in $DST_RANKS; do DST_PODS+=("vllm-recipe-$r"); done
else
  DST_PODS=()
  for p in "${PODS[@]}"; do
    [[ "$p" == "$SRC_POD" ]] && continue
    if have_marker "$p"; then
      echo "  $p: already complete (skip)"
    else
      DST_PODS+=("$p")
    fi
  done
fi

if [[ ${#DST_PODS[@]} -eq 0 ]]; then
  echo "nothing to sync — all ranks have $MARKER"
  exit 0
fi

SRC_IP="$(kubectl -n "$NS" get pod "$SRC_POD" -o jsonpath='{.status.podIP}')"
echo "==> Source $SRC_POD ($SRC_IP) → ${DST_PODS[*]}  port=$PORT"

# One listener per destination (sequential) to keep the source disk streaming simple.
for dst in "${DST_PODS[@]}"; do
  echo
  echo "==> Syncing → $dst"
  # Kill prior receiver via pidfile only. Never pkill -f a pattern that appears
  # in this bash -c argv (pkill would SIGTERM itself → exit 143).
  kubectl -n "$NS" exec "$dst" -- bash -c \
    "if [[ -f /tmp/sync-model-recv.pid ]]; then kill \"\$(cat /tmp/sync-model-recv.pid)\" 2>/dev/null || true; fi; mkdir -p /models; rm -rf '$MODEL_DIR'"

  # Start receiver first (blocks until stream ends).
  kubectl -n "$NS" exec "$dst" -- bash -c "
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
    if n % (8 * 1024**3) < 8 * 1024 * 1024:
        print(f'recv bytes={n}', flush=True)
p.stdin.close()
rc = p.wait()
conn.close()
s.close()
print(f'recv done bytes={n} tar_rc={rc}', flush=True)
sys.exit(rc)
PY
    nohup python3 /tmp/sync-model-recv.py $PORT > /tmp/sync-model-recv.log 2>&1 &
    echo \$! > /tmp/sync-model-recv.pid
    sleep 1
    echo recv_pid=\$(cat /tmp/sync-model-recv.pid)
  "

  DST_IP="$(kubectl -n "$NS" get pod "$dst" -o jsonpath='{.status.podIP}')"
  echo "    pushing $SRC_POD → $DST_IP:$PORT"

  kubectl -n "$NS" exec "$SRC_POD" -- bash -c "
    cat > /tmp/sync-model-send.py <<'PY'
import socket, subprocess, sys, time
host, port = sys.argv[1], int(sys.argv[2])
for attempt in range(60):
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
    if n % (8 * 1024**3) < 8 * 1024 * 1024:
        print(f'send bytes={n}', flush=True)
rc = p.wait()
s.close()
print(f'send done bytes={n} tar_rc={rc}', flush=True)
sys.exit(rc)
PY
    python3 /tmp/sync-model-send.py '$DST_IP' $PORT 2>&1 | tee /tmp/sync-model-send.log
  "

  echo "    waiting for receiver on $dst"
  for _ in $(seq 1 720); do
    if kubectl -n "$NS" exec "$dst" -- bash -c \
        '! kill -0 "$(cat /tmp/sync-model-recv.pid 2>/dev/null)" 2>/dev/null'; then
      break
    fi
    if (( _ % 6 == 0 )); then
      kubectl -n "$NS" exec "$dst" -- bash -c \
        'tail -n 2 /tmp/sync-model-recv.log 2>/dev/null; du -sh /models/Kimi-K3 2>/dev/null || true'
    fi
    sleep 10
  done
  kubectl -n "$NS" exec "$dst" -- tail -n 5 /tmp/sync-model-recv.log || true

  echo "    verifying $dst"
  kubectl -n "$NS" cp "$(dirname "$0")/download-model-inpod.sh" "$dst:/tmp/download-model-inpod.sh"
  kubectl -n "$NS" exec "$dst" -- bash -c \
    'MODEL_ROOT=/models LOCAL_DIR=/models/Kimi-K3 HF_HOME=/models/hf \
       bash /tmp/download-model-inpod.sh' | tail -20
  echo "    $dst OK"
done

echo
echo "==> All destinations verified"
for p in "${PODS[@]}"; do
  kubectl -n "$NS" exec "$p" -- bash -c \
    "echo -n '$p '; test -f $MARKER && cat $MARKER | head -1 || echo MISSING; du -sh $MODEL_DIR 2>/dev/null"
done
