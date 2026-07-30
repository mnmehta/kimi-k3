#!/usr/bin/env bash
# In-pod serve script mimicking the multi-node TP recipe at:
#   https://recipes.vllm.ai/moonshotai/Kimi-K3
#
# Differences from the published recipe (intentional for this cluster smoke):
#   - --load-format dummy (full HF architecture; no weight download)
#   - rank > 0 adds --headless (recipe: workers are headless)
#
# Optional shallow overrides for tiny smoke fits (off by default):
#   NUM_LAYERS / NUM_EXPERTS / NUM_EXPERTS_PER_TOKEN / NUM_SHARED_EXPERTS
#
# Required env:
#   MASTER_ADDR   rank-0 host/IP
#   NODE_RANK     0 or 1
#
# Optional env overrides match recipe knobs (defaults below).

set -euo pipefail

MODEL="${MODEL:-moonshotai/Kimi-K3}"
NNODES="${NNODES:-2}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
TP_SIZE="${TP_SIZE:-$((NNODES * GPUS_PER_NODE))}"
NODE_RANK="${NODE_RANK:?NODE_RANK required (0 or 1)}"
MASTER_ADDR="${MASTER_ADDR:?MASTER_ADDR required}"
PORT="${PORT:-8000}"

# Optional shallow overrides — unset by default so dummy loads full Kimi-K3 config.
NUM_LAYERS="${NUM_LAYERS:-}"
NUM_EXPERTS="${NUM_EXPERTS:-}"
NUM_EXPERTS_PER_TOKEN="${NUM_EXPERTS_PER_TOKEN:-}"
NUM_SHARED_EXPERTS="${NUM_SHARED_EXPERTS:-}"

# Recipe defaults
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.97}"
# Full-weight dummy leaves ~161 Mamba blocks on 8×H200 @ 0.97 util;
# stay under that so CUDA graph capture can succeed.
MAX_NUM_SEQS="${MAX_NUM_SEQS:-128}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
MOE_BACKEND="${MOE_BACKEND:-marlin}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-FLASHMLA}"

detect_iface() {
  if [[ -n "${IFACE_NAME:-}" ]]; then
    echo "$IFACE_NAME"
    return
  fi
  local iface
  # Prefer Ethernet for Gloo/NCCL *socket* bootstrap. IB ifaces (ibs*) often
  # only have IPv6 link-local and break multi-node Gloo (fe80::... timeouts).
  # NCCL still auto-discovers IB devices for GPU collectives.
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
# Avoid Gloo AF_INET vs AF_INET6 mismatches (common on hosts with IB link-local).
if [[ -w /proc/sys/net/ipv6/conf/all/disable_ipv6 ]]; then
  echo 1 > /proc/sys/net/ipv6/conf/all/disable_ipv6 || true
  echo 1 > /proc/sys/net/ipv6/conf/default/disable_ipv6 || true
  for f in /proc/sys/net/ipv6/conf/*/disable_ipv6; do
    echo 1 > "$f" 2>/dev/null || true
  done
fi
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-3600}"
export VLLM_USE_V2_MODEL_RUNNER="${VLLM_USE_V2_MODEL_RUNNER:-1}"
# Nested --hf-overrides JSON currently breaks the Rust frontend --args-json parser.
export VLLM_USE_RUST_FRONTEND="${VLLM_USE_RUST_FRONTEND:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export VLLM_ENABLE_K3_LATENT_MOE_TAIL_FUSION="${VLLM_ENABLE_K3_LATENT_MOE_TAIL_FUSION:-0}"
export PYTHONUNBUFFERED=1

# MLA dcp_world_size lazy-init workaround (same as smoke script).
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

# get_open_port() has a TOCTOU race under multi-proc hostNetwork TP: many workers
# pick the same free port then ZMQ bind fails with EADDRINUSE. Retry bind.
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
# The original block sets remote_addr_ipv6 inside the ipv6 branch before bind;
# our patch also sets it. Avoid double-assign by matching exact needle only once.
if needle not in src:
    raise SystemExit(f"zmq bind patch needle missing in {path}")
# Original code already may have set remote_addr_ipv6 = False earlier; duplicate is fine.
path.write_text(src.replace(needle, patch, 1))
print(f"patched {path}")
PY
fi

SERVE_ARGS=(
  --host 0.0.0.0
  --port "$PORT"
  --trust-remote-code
  --load-format dummy
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

# Only apply architecture overrides when explicitly requested (shallow smoke).
if [[ -n "$NUM_LAYERS" || -n "$NUM_EXPERTS" || -n "$NUM_EXPERTS_PER_TOKEN" || -n "$NUM_SHARED_EXPERTS" ]]; then
  text_cfg="{"
  sep=""
  [[ -n "$NUM_LAYERS" ]] && { text_cfg+="${sep}\"num_hidden_layers\": ${NUM_LAYERS}"; sep=", "; }
  [[ -n "$NUM_EXPERTS" ]] && { text_cfg+="${sep}\"num_experts\": ${NUM_EXPERTS}"; sep=", "; }
  [[ -n "$NUM_EXPERTS_PER_TOKEN" ]] && { text_cfg+="${sep}\"num_experts_per_token\": ${NUM_EXPERTS_PER_TOKEN}"; sep=", "; }
  [[ -n "$NUM_SHARED_EXPERTS" ]] && { text_cfg+="${sep}\"num_shared_experts\": ${NUM_SHARED_EXPERTS}"; sep=", "; }
  text_cfg+="}"
  HF_OVERRIDES="{\"text_config\": ${text_cfg}}"
  SERVE_ARGS+=(--hf-overrides "$HF_OVERRIDES")
  OVERRIDE_DESC="hf-overrides=$HF_OVERRIDES"
else
  OVERRIDE_DESC="full HF config (no hf-overrides)"
fi

if [[ "$NODE_RANK" != "0" ]]; then
  SERVE_ARGS+=(--headless)
fi

echo "Launching recipe-shaped dummy serve:"
echo "  MODEL=$MODEL TP=$TP_SIZE NNODES=$NNODES NODE_RANK=$NODE_RANK MASTER_ADDR=$MASTER_ADDR"
echo "  IFACE=$GLOO_SOCKET_IFNAME $OVERRIDE_DESC"
echo "  max-num-seqs=$MAX_NUM_SEQS"

exec vllm serve "$MODEL" "${SERVE_ARGS[@]}"
