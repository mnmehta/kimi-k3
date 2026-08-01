#!/usr/bin/env bash
# In-pod serve script for real Kimi-K3 weights (multi-node recipe).
# Recipe: https://recipes.vllm.ai/moonshotai/Kimi-K3
#
# Parallelism presets (via env):
#   TP16 (default):  TP_SIZE=16 PP_SIZE=1  (multi-node TP across 2×8 GPUs)
#   TP8×PP2:         TP_SIZE=8  PP_SIZE=2
#
# Expects weights at MODEL (default /models/Kimi-K3).
#
# Required env:
#   MASTER_ADDR   rank-0 host/IP
#   NODE_RANK     0 or 1
#
# Optional:
#   MODEL         default /models/Kimi-K3
#   LOAD_FORMAT   default empty (vLLM auto); set e.g. fastsafetensors
#   TP_SIZE / PP_SIZE / NNODES / GPUS_PER_NODE

set -euo pipefail

MODEL="${MODEL:-/models/Kimi-K3}"
NNODES="${NNODES:-2}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
# Default: pure multi-node TP (TP = nnodes × GPUs/node). For TP8×PP2 set:
#   TP_SIZE=8 PP_SIZE=2
PP_SIZE="${PP_SIZE:-1}"
TP_SIZE="${TP_SIZE:-$((NNODES * GPUS_PER_NODE / PP_SIZE))}"
NODE_RANK="${NODE_RANK:?NODE_RANK required (0 or 1)}"
MASTER_ADDR="${MASTER_ADDR:?MASTER_ADDR required}"
PORT="${PORT:-8000}"
LOAD_FORMAT="${LOAD_FORMAT:-}"

GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.97}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-128}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
MOE_BACKEND="${MOE_BACKEND:-marlin}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-FLASHMLA}"

MARKER="${MODEL}/.download-complete"
if [[ ! -d "$MODEL" ]]; then
  echo "MODEL path missing: $MODEL (is hostPath /models populated?)" >&2
  exit 1
fi
if [[ ! -f "$MARKER" ]]; then
  echo "Download marker missing: $MARKER" >&2
  echo "Run ./scripts/download-model.sh and wait for completion." >&2
  exit 1
fi
if [[ ! -f "$MODEL/config.json" ]]; then
  echo "config.json missing under $MODEL" >&2
  exit 1
fi

detect_iface() {
  if [[ -n "${IFACE_NAME:-}" ]]; then
    echo "$IFACE_NAME"
    return
  fi
  local iface
  for iface in $(ls /sys/class/net 2>/dev/null | grep -E '^enp' | sort || true); do
    echo "$iface"
    return
  done
  for iface in $(ls /sys/class/net 2>/dev/null | grep -E '^(eth|bond)' || true); do
    echo "$iface"
    return
  done
  for iface in $(ls /sys/class/net 2>/dev/null | grep -vE '^(lo|docker|cni|flannel|veth|cali|tunl|lxc|cilium|ibs|ib|mlx)' || true); do
    echo "$iface"
    return
  done
  echo "eth0"
}

IFACE_NAME="$(detect_iface)"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-$IFACE_NAME}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-$IFACE_NAME}"
if [[ -w /proc/sys/net/ipv6/conf/all/disable_ipv6 ]]; then
  echo 1 > /proc/sys/net/ipv6/conf/all/disable_ipv6 || true
  echo 1 > /proc/sys/net/ipv6/conf/default/disable_ipv6 || true
  for f in /proc/sys/net/ipv6/conf/*/disable_ipv6; do
    echo 1 > "$f" 2>/dev/null || true
  done
fi
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-3600}"
export VLLM_USE_V2_MODEL_RUNNER="${VLLM_USE_V2_MODEL_RUNNER:-1}"
export VLLM_USE_RUST_FRONTEND="${VLLM_USE_RUST_FRONTEND:-0}"
# Empty PYTORCH_CUDA_ALLOC_CONF disables the default; expandable_segments can
# fail hard under PP peak-load when free memory is tiny.
if [[ "${PYTORCH_CUDA_ALLOC_CONF+x}" = x && -z "${PYTORCH_CUDA_ALLOC_CONF}" ]]; then
  unset PYTORCH_CUDA_ALLOC_CONF
elif [[ -z "${PYTORCH_CUDA_ALLOC_CONF+x}" ]]; then
  export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
fi
# Recipe note: mlx5 dmabuf registration (errno 524) → fall back to nvidia_peermem.
export NCCL_DMABUF_ENABLE="${NCCL_DMABUF_ENABLE:-0}"
# PP cross-node P2P has failed on IPv6 link-local RoCE GIDs (IBV_WC_RETRY_EXC_ERR).
# Optional overrides (set via deploy env):
#   NCCL_IB_DISABLE=1          # force Socket (debug / workaround)
#   NCCL_IB_GID_INDEX=3        # prefer IPv4-mapped RoCEv2 GID when present
#   NCCL_IB_HCA=mlx5_5         # HCA that carries the IPv4 GID on these nodes
[[ -n "${NCCL_IB_DISABLE:-}" ]] && export NCCL_IB_DISABLE
[[ -n "${NCCL_IB_GID_INDEX:-}" ]] && export NCCL_IB_GID_INDEX
[[ -n "${NCCL_IB_HCA:-}" ]] && export NCCL_IB_HCA
export VLLM_ENABLE_K3_LATENT_MOE_TAIL_FUSION="${VLLM_ENABLE_K3_LATENT_MOE_TAIL_FUSION:-0}"
export HF_HOME="${HF_HOME:-/models/hf}"
export PYTHONUNBUFFERED=1

# MLA dcp_world_size lazy-init workaround (same as dummy harness).
MLA_PY="/usr/local/lib/python3.12/dist-packages/vllm/models/kimi_k3/nvidia/mla.py"
if [[ -f "$MLA_PY" ]] && ! grep -q "Kimi MLA bypasses MLAAttention.forward lazy-init" "$MLA_PY"; then
  python3 - <<'PY'
from pathlib import Path
path = Path("/usr/local/lib/python3.12/dist-packages/vllm/models/kimi_k3/nvidia/mla.py")
src = path.read_text()
needle = """            latent_out, _lse = self.impl.forward_mqa(  # type: ignore[attr-defined]
                mqa_q, self._attn_read_kv_cache(), attn_metadata, self
            )"""
patch = """            # Kimi MLA bypasses MLAAttention.forward lazy-init; ensure DCP size
            # is set before FlashAttnMLA (cp_world_size must be > 0).
            if getattr(self.impl, "dcp_world_size", 1) == -1:
                from vllm.distributed.parallel_state import get_dcp_group
                try:
                    self.impl.dcp_world_size = get_dcp_group().world_size
                except AssertionError:
                    self.impl.dcp_world_size = 1
            latent_out, _lse = self.impl.forward_mqa(  # type: ignore[attr-defined]
                mqa_q, self._attn_read_kv_cache(), attn_metadata, self
            )"""
if needle not in src:
    raise SystemExit(f"patch needle missing in {path}")
path.write_text(src.replace(needle, patch, 1))
print(f"patched {path}")
PY
fi

SHM_PY="/usr/local/lib/python3.12/dist-packages/vllm/distributed/device_communicators/shm_broadcast.py"
if [[ -f "$SHM_PY" ]] && ! grep -q "Kimi-K3 harness: retry ZMQ bind" "$SHM_PY"; then
  python3 - <<'PY'
from pathlib import Path
path = Path("/usr/local/lib/python3.12/dist-packages/vllm/distributed/device_communicators/shm_broadcast.py")
src = path.read_text()
needle = """            remote_subscribe_port = get_open_port()
            if is_valid_ipv6_address(connect_ip):
                self.remote_socket.setsockopt(IPV6, 1)
                remote_addr_ipv6 = True
                connect_ip = f"[{connect_ip}]"
            socket_addr = f"tcp://{connect_ip}:{remote_subscribe_port}"
            self.remote_socket.bind(socket_addr)
            remote_subscribe_addr = f"tcp://{connect_ip}:{remote_subscribe_port}"
"""
patch = """            # Kimi-K3 harness: retry ZMQ bind on EADDRINUSE (get_open_port TOCTOU).
            _mq_connect_ip = connect_ip
            if is_valid_ipv6_address(_mq_connect_ip):
                self.remote_socket.setsockopt(IPV6, 1)
                remote_addr_ipv6 = True
                _mq_connect_ip = f"[{_mq_connect_ip}]"
            last_err = None
            for _attempt in range(64):
                remote_subscribe_port = get_open_port()
                socket_addr = f"tcp://{_mq_connect_ip}:{remote_subscribe_port}"
                try:
                    self.remote_socket.bind(socket_addr)
                    last_err = None
                    break
                except zmq.ZMQError as e:
                    last_err = e
                    if getattr(e, "errno", None) != zmq.EADDRINUSE:
                        raise
            if last_err is not None:
                raise last_err
            connect_ip = _mq_connect_ip
            remote_subscribe_addr = f"tcp://{connect_ip}:{remote_subscribe_port}"
"""
if needle not in src:
    raise SystemExit(f"zmq bind patch needle missing in {path}")
path.write_text(src.replace(needle, patch, 1))
print(f"patched {path}")
PY
fi

SERVE_ARGS=(
  --host 0.0.0.0
  --port "$PORT"
  --trust-remote-code
  --tensor-parallel-size "$TP_SIZE"
  --nnodes "$NNODES"
  --node-rank "$NODE_RANK"
  --master-addr "$MASTER_ADDR"
  --gpu-memory-utilization "$GPU_MEM_UTIL"
  --max-num-seqs "$MAX_NUM_SEQS"
  --max-model-len "$MAX_MODEL_LEN"
  --moe-backend "$MOE_BACKEND"
  --disable-custom-all-reduce
  --no-enable-flashinfer-autotune
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
  --attention-backend "$ATTENTION_BACKEND"
  --enable-auto-tool-choice
  --tool-call-parser kimi_k3
  --reasoning-parser kimi_k3
)

if [[ "$PP_SIZE" -gt 1 ]]; then
  SERVE_ARGS+=(--pipeline-parallel-size "$PP_SIZE")
fi

if [[ -n "$LOAD_FORMAT" ]]; then
  SERVE_ARGS+=(--load-format "$LOAD_FORMAT")
fi

if [[ "$NODE_RANK" != "0" ]]; then
  SERVE_ARGS+=(--headless)
fi

echo "Launching recipe-shaped REAL-WEIGHT serve:"
echo "  MODEL=$MODEL TP=$TP_SIZE PP=$PP_SIZE NNODES=$NNODES NODE_RANK=$NODE_RANK MASTER_ADDR=$MASTER_ADDR"
echo "  IFACE=$GLOO_SOCKET_IFNAME LOAD_FORMAT=${LOAD_FORMAT:-<auto>}"
echo "  max-num-seqs=$MAX_NUM_SEQS max-model-len=$MAX_MODEL_LEN"
cat "$MARKER" || true

exec vllm serve "$MODEL" "${SERVE_ARGS[@]}"
