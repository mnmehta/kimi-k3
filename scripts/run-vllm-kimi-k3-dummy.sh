#!/usr/bin/env bash
# Run moonshotai/Kimi-K3 inside the vllm pod with dummy weights on 2 GPUs (TP=2).
# Adapted from notes.md (Ashish slack thread) for TP smoke / comm profiling.
#
# Layer pattern (from text_config / linear_attn_config):
#   - layer 0: dense MLP (first_k_dense_replace=1), KDA
#   - layers >=1: MoE (+ shared experts, latent MoE)
#   - every 4th 1-indexed layer (3,7,11,...): full MLA attention
#   - other layers: KDA (KimiK3DeltaAttention)
#   - attn_res_block_size=12: residual attention blocks
# So NUM_LAYERS=12 covers dense, MoE, KDA, MLA, and one full attn_res block.
#
# Usage (from host):
#   kubectl -n kimi-k3 cp scripts/run-vllm-kimi-k3-dummy.sh vllm:/tmp/run.sh
#   kubectl -n kimi-k3 exec -it vllm -- bash /tmp/run.sh

set -euo pipefail

MODEL="${MODEL:-moonshotai/Kimi-K3}"
NUM_LAYERS="${NUM_LAYERS:-12}"
NUM_EXPERTS="${NUM_EXPERTS:-8}"
NUM_EXPERTS_PER_TOKEN="${NUM_EXPERTS_PER_TOKEN:-2}"
NUM_SHARED_EXPERTS="${NUM_SHARED_EXPERTS:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1024}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.9}"
PORT="${PORT:-8000}"
TP_SIZE="${TP_SIZE:-2}"
PROFILE_DIR="${PROFILE_DIR:-/tmp/vllm_profile}"

# Use TP_SIZE GPUs (pod has 8); keep --disable-custom-all-reduce so NCCL shows in traces.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export VLLM_ENABLE_K3_LATENT_MOE_TAIL_FUSION="${VLLM_ENABLE_K3_LATENT_MOE_TAIL_FUSION:-1}"
# stop_profile can take a while to flush chrome traces to disk
export VLLM_RPC_TIMEOUT="${VLLM_RPC_TIMEOUT:-1800000}"
export PYTHONUNBUFFERED=1

mkdir -p "$PROFILE_DIR"
# Enough active steps for 1 prefill + a few decode tokens (plus margin).
PROFILER_CONFIG="${PROFILER_CONFIG:-{\"profiler\":\"torch\",\"torch_profiler_dir\":\"${PROFILE_DIR}\",\"torch_profiler_with_stack\":true,\"active_iterations\":32}}"

# Workaround: Kimi MultiHeadLatentAttention calls impl.forward_mqa without the
# MLAAttention.forward lazy-init that sets impl.dcp_world_size from -1 -> 1.
# Full MLA layers (every 4th) then fail FlashAttn with:
#   cp_world_size must be positive ... Use 1 if CP is not enabled.
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

HF_OVERRIDES=$(cat <<EOF
{"text_config": {"num_hidden_layers": ${NUM_LAYERS}, "num_experts": ${NUM_EXPERTS}, "num_experts_per_token": ${NUM_EXPERTS_PER_TOKEN}, "num_shared_experts": ${NUM_SHARED_EXPERTS}}}
EOF
)

exec vllm serve "$MODEL" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --trust-remote-code \
  --load-format dummy \
  --hf-overrides "$HF_OVERRIDES" \
  --tensor-parallel-size "$TP_SIZE" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs 4 \
  --moe-backend marlin \
  --no-enable-flashinfer-autotune \
  --disable-custom-all-reduce \
  -cc.pass_config.fuse_allreduce_rms=False \
  --enable-auto-tool-choice \
  --tool-call-parser kimi_k3 \
  --reasoning-parser kimi_k3 \
  --profiler-config "$PROFILER_CONFIG"
