#!/usr/bin/env bash
# In-pod serve script for real Kimi-K3 weights (multi-node recipe).
# Recipe: https://recipes.vllm.ai/moonshotai/Kimi-K3
#
# Parallelism presets (via env):
#   TP16 (default):  TP_SIZE=16 PP_SIZE=1 DP_SIZE=1  (multi-node TP across 2×8)
#   TEP16:           TP_SIZE=16 + ENABLE_EXPERT_PARALLEL=1
#     https://recipes.vllm.ai/moonshotai/Kimi-K3?strategy=multi_node_tep
#   TP8×PP2:         TP_SIZE=8  PP_SIZE=2 DP_SIZE=1
#   TP8×DP2:         TP_SIZE=8  PP_SIZE=1 DP_SIZE=2 DP_SIZE_LOCAL=1
#     https://recipes.vllm.ai/moonshotai/Kimi-K3?strategy=multi_node_tp_dp
#   P/D role:        NNODES=1 TP_SIZE=8 + KV_TRANSFER_CONFIG / NIXL env
#     https://recipes.vllm.ai/moonshotai/Kimi-K3?strategy=pd_cluster
#
# Expects weights at MODEL (default /models/Kimi-K3).
#
# Required env:
#   MASTER_ADDR            rank-0 / DP coordinator IP (multi-node TP/PP or DP)
#   NODE_RANK              0 or 1  (TP/PP multi-node mode only; not needed for NNODES=1)
#   or DP_START_RANK       0 or 1  (DP mode; NODE_RANK optional)
#
# Optional:
#   MODEL         default /models/Kimi-K3
#   LOAD_FORMAT   default empty (vLLM auto); set e.g. fastsafetensors
#   TP_SIZE / PP_SIZE / DP_SIZE / DP_SIZE_LOCAL / DP_RPC_PORT / NNODES

set -euo pipefail

MODEL="${MODEL:-/models/Kimi-K3}"
NNODES="${NNODES:-2}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
PP_SIZE="${PP_SIZE:-1}"
DP_SIZE="${DP_SIZE:-1}"
DP_SIZE_LOCAL="${DP_SIZE_LOCAL:-1}"
DP_START_RANK="${DP_START_RANK:-0}"
DP_RPC_PORT="${DP_RPC_PORT:-13345}"
# Default TP: full multi-node TP, unless DP>1 (then one replica per node → TP=GPUs/node).
if [[ "$DP_SIZE" -gt 1 ]]; then
  TP_SIZE="${TP_SIZE:-$GPUS_PER_NODE}"
else
  TP_SIZE="${TP_SIZE:-$((NNODES * GPUS_PER_NODE / PP_SIZE))}"
fi
PORT="${PORT:-8000}"
LOAD_FORMAT="${LOAD_FORMAT:-}"
NODE_RANK="${NODE_RANK:-}"
# Single-node roles (P/D) need neither MASTER_ADDR nor NODE_RANK.
if [[ "$NNODES" -gt 1 || "$DP_SIZE" -gt 1 ]]; then
  MASTER_ADDR="${MASTER_ADDR:?MASTER_ADDR required for multi-node / DP}"
else
  MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
fi
DATA_PARALLEL_ADDRESS="${DATA_PARALLEL_ADDRESS:-$MASTER_ADDR}"
if [[ "$DP_SIZE" -le 1 && "$NNODES" -gt 1 && -z "$NODE_RANK" ]]; then
  echo "NODE_RANK required when DP_SIZE=1 and NNODES>1" >&2
  exit 1
fi
# Prefer explicit NODE_RANK; else align with DP_START_RANK for logging.
NODE_RANK="${NODE_RANK:-$DP_START_RANK}"

GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.97}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-128}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
# Defaults empty: Kimi recipes set these via base.yaml; Gemma/other models leave unset
# so vLLM picks a valid backend. Do NOT default to FLASHMLA/marlin here — that breaks
# non-MLA models when a recipe intentionally unsets the knobs.
MOE_BACKEND="${MOE_BACKEND:-}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-}"
# Optional hard KV reservation (bytes). When set, skips util-based profiling.
KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES:-}"
# Classic weight CPU offload (GiB per GPU virtual extension via --cpu-offload-gb).
CPU_OFFLOAD_GB="${CPU_OFFLOAD_GB:-}"
# Prefix caching (required for OffloadingConnector on hybrid Mamba+Attention).
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-0}"
# Skip multimodal encoder profiling (can need ~400 MiB on Kimi-K3).
SKIP_MM_PROFILING="${SKIP_MM_PROFILING:-0}"
# Multi-node TEP: --enable-expert-parallel (recipe multi_node_tep).
ENABLE_EXPERT_PARALLEL="${ENABLE_EXPERT_PARALLEL:-0}"
# Optional: skip loading expert weights not owned by this EP rank.
ENABLE_EP_WEIGHT_FILTER="${ENABLE_EP_WEIGHT_FILTER:-0}"
# Prefill/Decode disaggregation (NIXL). Example:
#   KV_TRANSFER_CONFIG='{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_load_failure_policy":"fail"}'
KV_TRANSFER_CONFIG="${KV_TRANSFER_CONFIG:-}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
# JSON string for --compilation-config (decode often uses FULL_DECODE_ONLY).
COMPILATION_CONFIG="${COMPILATION_CONFIG:-}"
# Recipe P/D uses --no-disable-hybrid-kv-cache-manager.
NO_DISABLE_HYBRID_KV_CACHE_MANAGER="${NO_DISABLE_HYBRID_KV_CACHE_MANAGER:-0}"

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
# Prefill/Decode hybrid Mamba: 3-read conv transfer needs DS layout.
if [[ -n "${KV_TRANSFER_CONFIG:-}" ]]; then
  export VLLM_SSM_CONV_STATE_LAYOUT="${VLLM_SSM_CONV_STATE_LAYOUT:-DS}"
elif [[ -n "${VLLM_SSM_CONV_STATE_LAYOUT:-}" ]]; then
  export VLLM_SSM_CONV_STATE_LAYOUT
fi
# Default is ~394 MiB; TP8 on H200 only has ~200–260 MiB free after weights.
# Callers (deploy-recipe-dp.sh) should set a smaller value when needed.
if [[ -n "${VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE:-}" ]]; then
  export VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE
fi
# Empty PYTORCH_CUDA_ALLOC_CONF disables the default; expandable_segments can
# fail hard under PP peak-load when free memory is tiny.
# NixlConnector rejects expandable_segments unless --enable-cumem-allocator
# (or sleep mode) is also on — P/D deploy enables both.
if [[ "${PYTORCH_CUDA_ALLOC_CONF+x}" = x && -z "${PYTORCH_CUDA_ALLOC_CONF}" ]]; then
  unset PYTORCH_CUDA_ALLOC_CONF
elif [[ -z "${PYTORCH_CUDA_ALLOC_CONF+x}" ]]; then
  export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
fi
ENABLE_CUMEM_ALLOCATOR="${ENABLE_CUMEM_ALLOCATOR:-0}"
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

# P/D on H200 TP8: stock vLLM rejects expandable_segments with NixlConnector
# unless --enable-cumem-allocator. CuMem then wraps weight load and OOMs
# (~137.2 GiB peak vs DP2's successful 135.7 GiB). Prefer DP2's allocator
# path: keep expandable_segments, skip CuMem, and relax the config check.
# Risk: KV page remaps can break NIXL RDMA — document if transfers fail.
VLLM_CFG_PY="/usr/local/lib/python3.12/dist-packages/vllm/config/vllm.py"
if [[ -n "${KV_TRANSFER_CONFIG:-}" && -f "$VLLM_CFG_PY" ]] \
  && ! grep -q "Kimi-K3 harness: allow expandable_segments with KV connector" "$VLLM_CFG_PY"; then
  python3 - <<'PY'
from pathlib import Path
path = Path("/usr/local/lib/python3.12/dist-packages/vllm/config/vllm.py")
src = path.read_text()
needle = """        if "expandable_segments:True" not in os.environ.get(
            "PYTORCH_CUDA_ALLOC_CONF", ""
        ):
            return
        if self.model_config is not None and (self.model_config.enable_cumem_allocator):
            return

        raise ValueError(
            f"KV connector {self.kv_transfer_config.kv_connector} is "
            "incompatible with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
"""
patch = """        if "expandable_segments:True" not in os.environ.get(
            "PYTORCH_CUDA_ALLOC_CONF", ""
        ):
            return
        if self.model_config is not None and (self.model_config.enable_cumem_allocator):
            return
        # Kimi-K3 harness: allow expandable_segments with KV connector on H200.
        # CuMem+NIXL path OOMs TP8 weight load; DP2-style expandable is required.
        return

        raise ValueError(
            f"KV connector {self.kv_transfer_config.kv_connector} is "
            "incompatible with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
"""
if needle not in src:
    raise SystemExit(f"expandable/nixl validation patch needle missing in {path}")
path.write_text(src.replace(needle, patch, 1))
print(f"patched {path}")
PY
fi

# Also skip CuMem weight pool if cumem is still enabled (belt-and-suspenders).
WORKER_PY="/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_worker.py"
if [[ -n "${KV_TRANSFER_CONFIG:-}" && -f "$WORKER_PY" ]] \
  && ! grep -q "Kimi-K3 harness: skip CuMem pool for weights" "$WORKER_PY"; then
  python3 - <<'PY'
from pathlib import Path
path = Path("/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_worker.py")
src = path.read_text()
needle = """    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        with (
            self._maybe_get_memory_pool_context(tag="weights"),
            set_current_vllm_config(self.vllm_config),
            # 20 MiB is the minimum PyTorch allows for max_split_size_mb.
            self._scoped_allocator_max_split(max_split_size_mb=20),
        ):
            self.model_runner.load_model(load_dummy_weights=load_dummy_weights)
"""
patch = """    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        # Kimi-K3 harness: skip CuMem pool for weights (KV pool unchanged).
        from contextlib import nullcontext
        with (
            nullcontext(),
            set_current_vllm_config(self.vllm_config),
            # 20 MiB is the minimum PyTorch allows for max_split_size_mb.
            self._scoped_allocator_max_split(max_split_size_mb=20),
        ):
            self.model_runner.load_model(load_dummy_weights=load_dummy_weights)
"""
if needle not in src:
    # Already patched in a prior run — OK.
    print(f"cumem weight-pool patch needle missing (maybe already patched) in {path}")
else:
    path.write_text(src.replace(needle, patch, 1))
    print(f"patched {path}")
PY
fi

# Upstream PR #50327 (merged 2026-08-03): scalar postprocess used index_fill_ on
# int32 idx_mapping (PP sentinels). Image may predate the fix — see issue #50947.
MAMBA_HYBRID_PY="/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/model_states/mamba_hybrid.py"
if [[ -f "$MAMBA_HYBRID_PY" ]] && ! grep -q "_fill_num_accepted_kernel" "$MAMBA_HYBRID_PY"; then
  python3 - <<'PY'
from pathlib import Path

path = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/model_states/mamba_hybrid.py"
)
src = path.read_text()
needle = """        # Chunked prefill does not sample a token, so num_sampled can be 0.
        # Mamba treats num_accepted_tokens=1 as the neutral non-spec value.
        if not isinstance(num_sampled, int):
            # idx_mapping may contain -1 sentinels (filtered rows) under PP; the
            # kernel skips them rather than scattering with a host-side gather.
            n = idx_mapping.shape[0]
            if n:
                _scatter_num_accepted_kernel[(n,)](
                    idx_mapping, num_sampled, self.num_accepted_tokens_gpu
                )
        else:
            # Fill with single value.
            self.num_accepted_tokens_gpu.index_fill_(
                0, idx_mapping, max(num_sampled, 1)
            )

        # Align: save the running state to the block-aligned position when
        # spec-decode acceptance leaves the sequence non-block-aligned (mirrors
        # the V1 align postprocess). num_computed_tokens already holds the
        # post-step advanced count.
        if (
            self._align_mode
            and num_computed_tokens is not None
            and self._mamba_ctx is not None
        ):
            num_reqs = idx_mapping.shape[0]
            if num_reqs:
                self._mamba_ctx.run_fused_postprocess_align(
                    num_reqs,
                    self.num_accepted_tokens_gpu,
                    self._mamba_state_idx_gpu,
                    num_computed_tokens,
                    idx_mapping,
                )


@triton.jit
def _scatter_num_accepted_kernel(
    idx_mapping_ptr,  # [num_reqs] batch_idx -> req_state_idx (-1 to skip)
    num_sampled_ptr,  # [num_reqs]
    num_accepted_ptr,  # [max_num_reqs]
):
    row = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + row)
    if req_state_idx < 0:
        return
    num_sampled = tl.load(num_sampled_ptr + row)
    tl.store(num_accepted_ptr + req_state_idx, tl.maximum(num_sampled, 1))
"""
patch = """        # Chunked prefill does not sample a token, so num_sampled can be 0.
        # Mamba treats num_accepted_tokens=1 as the neutral non-spec value.
        # Kimi-K3 harness: PR #50327 — int32 idx_mapping + PP -1 sentinels cannot
        # use index_fill_ (needs int64; negatives corrupt state). Use Triton fill.
        num_reqs = idx_mapping.shape[0]
        if not num_reqs:
            return

        if not isinstance(num_sampled, int):
            # idx_mapping may contain -1 sentinels (filtered rows) under PP; the
            # kernel skips them rather than scattering with a host-side gather.
            _scatter_num_accepted_kernel[(num_reqs,)](
                idx_mapping, num_sampled, self.num_accepted_tokens_gpu
            )
        else:
            # Fill with single value.
            _fill_num_accepted_kernel[(num_reqs,)](
                idx_mapping, self.num_accepted_tokens_gpu, max(num_sampled, 1)
            )

        # Align: save the running state to the block-aligned position when
        # spec-decode acceptance leaves the sequence non-block-aligned (mirrors
        # the V1 align postprocess). num_computed_tokens already holds the
        # post-step advanced count.
        if (
            self._align_mode
            and num_computed_tokens is not None
            and self._mamba_ctx is not None
        ):
            self._mamba_ctx.run_fused_postprocess_align(
                num_reqs,
                self.num_accepted_tokens_gpu,
                self._mamba_state_idx_gpu,
                num_computed_tokens,
                idx_mapping,
            )


@triton.jit
def _scatter_num_accepted_kernel(
    idx_mapping_ptr,  # [num_reqs] batch_idx -> req_state_idx (-1 to skip)
    num_sampled_ptr,  # [num_reqs]
    num_accepted_ptr,  # [max_num_reqs]
):
    row = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + row)
    if req_state_idx < 0:
        return
    num_sampled = tl.load(num_sampled_ptr + row)
    tl.store(num_accepted_ptr + req_state_idx, tl.maximum(num_sampled, 1))


@triton.jit
def _fill_num_accepted_kernel(
    idx_mapping_ptr,  # [num_reqs] batch_idx -> req_state_idx (-1 to skip)
    num_accepted_ptr,  # [max_num_reqs]
    num_sampled,
):
    row = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + row)
    if req_state_idx < 0:
        return
    tl.store(num_accepted_ptr + req_state_idx, num_sampled)
"""
if needle not in src:
    raise SystemExit(f"mamba_hybrid PR#50327 patch needle missing in {path}")
path.write_text(src.replace(needle, patch, 1))
print(f"patched {path} (PR #50327 int32 idx_mapping fill)")
PY
fi

SERVE_ARGS=(
  --host 0.0.0.0
  --port "$PORT"
  --trust-remote-code
  --tensor-parallel-size "$TP_SIZE"
  --gpu-memory-utilization "$GPU_MEM_UTIL"
  --max-num-seqs "$MAX_NUM_SEQS"
  --max-model-len "$MAX_MODEL_LEN"
  --disable-custom-all-reduce
  --no-enable-flashinfer-autotune
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
)

if [[ -n "${MOE_BACKEND:-}" ]]; then
  SERVE_ARGS+=(--moe-backend "$MOE_BACKEND")
fi
if [[ -n "${ATTENTION_BACKEND:-}" ]]; then
  SERVE_ARGS+=(--attention-backend "$ATTENTION_BACKEND")
fi

TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-kimi_k3}"
REASONING_PARSER="${REASONING_PARSER:-kimi_k3}"
ENABLE_AUTO_TOOL_CHOICE="${ENABLE_AUTO_TOOL_CHOICE:-1}"
if [[ "$ENABLE_AUTO_TOOL_CHOICE" == "1" ]]; then
  SERVE_ARGS+=(--enable-auto-tool-choice)
fi
if [[ -n "$TOOL_CALL_PARSER" ]]; then
  SERVE_ARGS+=(--tool-call-parser "$TOOL_CALL_PARSER")
fi
if [[ -n "$REASONING_PARSER" ]]; then
  SERVE_ARGS+=(--reasoning-parser "$REASONING_PARSER")
fi

KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-}"
BLOCK_SIZE="${BLOCK_SIZE:-}"
ENABLE_CHUNKED_PREFILL="${ENABLE_CHUNKED_PREFILL:-0}"
LIMIT_MM_PER_PROMPT="${LIMIT_MM_PER_PROMPT:-}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-}"

if [[ -n "$KV_CACHE_DTYPE" ]]; then
  SERVE_ARGS+=(--kv-cache-dtype "$KV_CACHE_DTYPE")
fi
if [[ -n "$BLOCK_SIZE" ]]; then
  SERVE_ARGS+=(--block-size "$BLOCK_SIZE")
fi
if [[ "$ENABLE_CHUNKED_PREFILL" == "1" ]]; then
  SERVE_ARGS+=(--enable-chunked-prefill)
fi
if [[ -n "$LIMIT_MM_PER_PROMPT" ]]; then
  SERVE_ARGS+=(--limit-mm-per-prompt "$LIMIT_MM_PER_PROMPT")
fi
if [[ -n "$CHAT_TEMPLATE" ]]; then
  SERVE_ARGS+=(--chat-template "$CHAT_TEMPLATE")
fi

# USE_DP_CLI=1: emit --data-parallel-* even when DP_SIZE=1 (P/D roles).
# On H200, bare NNODES=1 TP8 OOMs during MXFP4 create (~137.2 GiB) while the
# same TP8 under the DP CLI path loads at ~135.7 GiB (see DP2 bring-up).
USE_DP_CLI="${USE_DP_CLI:-0}"
if [[ "$DP_SIZE" -gt 1 || "$USE_DP_CLI" == "1" ]]; then
  # Multi-node TP + Data Parallel (one full replica per node), or single-rank DP CLI.
  # https://docs.vllm.ai/en/latest/serving/data_parallel_deployment/
  SERVE_ARGS+=(
    --data-parallel-size "$DP_SIZE"
    --data-parallel-size-local "$DP_SIZE_LOCAL"
    --data-parallel-address "${DATA_PARALLEL_ADDRESS:-127.0.0.1}"
    --data-parallel-rpc-port "$DP_RPC_PORT"
    --data-parallel-start-rank "$DP_START_RANK"
  )
  if [[ "$DP_SIZE" -gt 1 && "$DP_START_RANK" != "0" ]]; then
    SERVE_ARGS+=(--headless)
  fi
elif [[ "$NNODES" -gt 1 ]]; then
  # Multi-node TP (and optional PP) via nnodes / node-rank.
  SERVE_ARGS+=(
    --nnodes "$NNODES"
    --node-rank "$NODE_RANK"
    --master-addr "$MASTER_ADDR"
  )
  if [[ "$PP_SIZE" -gt 1 ]]; then
    SERVE_ARGS+=(--pipeline-parallel-size "$PP_SIZE")
  fi
  if [[ "$NODE_RANK" != "0" ]]; then
    SERVE_ARGS+=(--headless)
  fi
else
  # Single-node role (P/D disagg): TP within the node only.
  if [[ "$PP_SIZE" -gt 1 ]]; then
    SERVE_ARGS+=(--pipeline-parallel-size "$PP_SIZE")
  fi
fi

if [[ "$ENABLE_EXPERT_PARALLEL" == "1" ]]; then
  SERVE_ARGS+=(--enable-expert-parallel)
fi

if [[ "$ENABLE_EP_WEIGHT_FILTER" == "1" ]]; then
  SERVE_ARGS+=(--enable-ep-weight-filter)
fi

if [[ -n "$LOAD_FORMAT" ]]; then
  SERVE_ARGS+=(--load-format "$LOAD_FORMAT")
fi

if [[ -n "$KV_CACHE_MEMORY_BYTES" ]]; then
  SERVE_ARGS+=(--kv-cache-memory-bytes "$KV_CACHE_MEMORY_BYTES")
fi

if [[ -n "$CPU_OFFLOAD_GB" ]]; then
  SERVE_ARGS+=(--cpu-offload-gb "$CPU_OFFLOAD_GB")
fi

if [[ "$ENABLE_PREFIX_CACHING" == "1" ]]; then
  SERVE_ARGS+=(--enable-prefix-caching)
fi

if [[ "$SKIP_MM_PROFILING" == "1" ]]; then
  SERVE_ARGS+=(--skip-mm-profiling)
fi

if [[ -n "$KV_TRANSFER_CONFIG" ]]; then
  SERVE_ARGS+=(--kv-transfer-config "$KV_TRANSFER_CONFIG")
fi

if [[ "$ENFORCE_EAGER" == "1" ]]; then
  SERVE_ARGS+=(--enforce-eager)
fi

if [[ -n "$COMPILATION_CONFIG" ]]; then
  SERVE_ARGS+=(--compilation-config "$COMPILATION_CONFIG")
fi

if [[ "$NO_DISABLE_HYBRID_KV_CACHE_MANAGER" == "1" ]]; then
  SERVE_ARGS+=(--no-disable-hybrid-kv-cache-manager)
fi

if [[ "$ENABLE_CUMEM_ALLOCATOR" == "1" ]]; then
  SERVE_ARGS+=(--enable-cumem-allocator)
fi

echo "Launching recipe-shaped REAL-WEIGHT serve:"
echo "  MODEL=$MODEL TP=$TP_SIZE PP=$PP_SIZE DP=$DP_SIZE DP_LOCAL=$DP_SIZE_LOCAL DP_START=$DP_START_RANK"
echo "  EP=${ENABLE_EXPERT_PARALLEL} EP_WEIGHT_FILTER=${ENABLE_EP_WEIGHT_FILTER}"
echo "  NNODES=$NNODES NODE_RANK=$NODE_RANK MASTER_ADDR=$MASTER_ADDR DP_ADDR=$DATA_PARALLEL_ADDRESS"
echo "  IFACE=$GLOO_SOCKET_IFNAME LOAD_FORMAT=${LOAD_FORMAT:-<auto>}"
echo "  max-num-seqs=$MAX_NUM_SEQS max-model-len=$MAX_MODEL_LEN gpu-mem-util=$GPU_MEM_UTIL"
echo "  kv-cache-memory-bytes=${KV_CACHE_MEMORY_BYTES:-<auto>} cpu-offload-gb=${CPU_OFFLOAD_GB:-<off>} prefix-caching=${ENABLE_PREFIX_CACHING} V2_RUNNER=${VLLM_USE_V2_MODEL_RUNNER:-} RUST_FE=${VLLM_USE_RUST_FRONTEND:-}"
echo "  kv-transfer=${KV_TRANSFER_CONFIG:-<none>} enforce-eager=${ENFORCE_EAGER} hybrid-kv=${NO_DISABLE_HYBRID_KV_CACHE_MANAGER} cumem=${ENABLE_CUMEM_ALLOCATOR}"
echo "  nixl-host=${VLLM_NIXL_SIDE_CHANNEL_HOST:-<unset>} nixl-port=${VLLM_NIXL_SIDE_CHANNEL_PORT:-<unset>}"
cat "$MARKER" || true

exec vllm serve "$MODEL" "${SERVE_ARGS[@]}"
