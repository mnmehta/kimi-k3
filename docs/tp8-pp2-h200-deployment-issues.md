# TP8×PP2 on H200: deployment issues

Post-mortem of bringing up [vLLM multi-node TP + pipeline parallel](https://recipes.vllm.ai/moonshotai/Kimi-K3?strategy=multi_node_tp_pp) for `moonshotai/Kimi-K3` on 2×8 H200.

Working entrypoint: [`./scripts/deploy.sh pp2`](../scripts/deploy.sh) → recipe [`configs/recipes/pp2.yaml`](../configs/recipes/pp2.yaml) → [`scripts/run-vllm-kimi-k3-recipe.sh`](../scripts/run-vllm-kimi-k3-recipe.sh).  
Results / report: [`bench-results/conc-sweep-pp2-1000-1000/`](../bench-results/conc-sweep-pp2-1000-1000/), [`reports/tp16-vs-pp2-1000-1000.qmd`](../reports/tp16-vs-pp2-1000-1000.qmd).

Siblings: [`tp16-h200-deployment-issues.md`](tp16-h200-deployment-issues.md), [`tep16-h200-deployment-issues.md`](tep16-h200-deployment-issues.md), [`tp8-dp2-h200-deployment-issues.md`](tp8-dp2-h200-deployment-issues.md), [`pp2-humming-agentx-kv-offload-deployment-issues.md`](pp2-humming-agentx-kv-offload-deployment-issues.md), [`harness-patches.md`](harness-patches.md).

## Root constraint

Same cluster / STS / image / hostPath weights as TP16. Layout differs:

| Item | Value |
|------|--------|
| Parallelism | **TP=8 × PP=2** — one PP stage per node (8 GPUs each) |
| CLI | `--tensor-parallel-size 8 --pipeline-parallel-size 2 --nnodes 2 --node-rank … --master-addr …`; rank ≠ 0 `--headless` |
| Memory | ~**98 GiB**/GPU weights → **~25–27 GiB** KV in the successful log (~**1.45M** tokens, ~**177× @ 8192**) |
| Cross-node traffic | PP **send/recv P2P** over NCCL/IB (not just TP allreduce) |

PP is what made RoCE GID selection a hard blocker; TP16 allreduce on the same fabric had already succeeded.

Shared harness fixes from TP16 still apply (storage, `enp*` iface, MLA patch, ZMQ retry, dmabuf off, latent fusion off, Rust FE off) — see [`tp16-h200-deployment-issues.md`](tp16-h200-deployment-issues.md). Below are the PP-specific fights.

## Issues (in roughly the order hit)

### 1. Dirty GPUs / OOM when switching TP16 → PP2

**Symptom:** “GPU OOM during load”, then KV-cache / NCCL failures on PP1; `nvidia-smi` still showing ~142 GB after a killed serve.

**Cause:** Leftover processes after in-pod stop; PP peak activation differs from TP16; aggressive util during early tries.

**Fix:** Hard-clean GPU compute apps before restart; settle **`GPU_MEM_UTIL=0.90`**. Allow empty `PYTORCH_CUDA_ALLOC_CONF` in PP deploy — `expandable_segments:True` can fail hard under PP peak when free memory is thin.

---

### 2. Cross-node RoCE P2P — IPv6 link-local GIDs (main PP blocker)

**Symptom:** Weights load; then PP warmup / P2P fails with:

```text
IBV_WC_RETRY_EXC_ERR
```

on **IPv6 link-local** RoCE GIDs (`fe80::…`), typically via **`mlx5_0`**.

**Cause:** Default GID selection picked link-local RoCEv2 paths that retry-exhaust on **2-rank PP send/recv**. Cross-node **TP16 allreduce** on the same nodes did **not** hit this; PP’s unbatched P2P path did.

**Interim fix (debug only):** `NCCL_IB_DISABLE=1` (Socket) — proves PP topology, **not** fair vs TP16 for interconnect-sensitive compare.

**Real fix (current [`configs/layers/strategy-pp2.yaml`](../configs/layers/strategy-pp2.yaml) defaults):**

| Env | Value |
|-----|--------|
| `NCCL_IB_HCA` | `mlx5_5` |
| `NCCL_IB_GID_INDEX` | `3` (IPv4-mapped RoCEv2 GID) |
| `NCCL_IB_DISABLE` | **unset** (keep IB) |

Baked into the deploy script comments:

```text
# Cross-node PP P2P failed on default RoCE IPv6 link-local GIDs (mlx5_0 /
# fe80::… → IBV_WC_RETRY_EXC_ERR). Prefer the IPv4-mapped RoCEv2 GID on mlx5_5.
```

---

### 3. Conservative knobs during Socket-era bring-up

**Symptom:** Even with Socket, early PP runs needed smaller budgets to stay alive while iterating NCCL.

**Temporary:** `GPU_MEM_UTIL=0.85`, `max-num-seqs=32`, `max-model-len=4096`.

**After IB GID fix:** Raise to sweep knobs — util **0.90**, seqs **256**, len **8192**, batched tokens **4096**.

---

### 4. Mid-sweep `IndexError` on `index_fill_` / int32 `idx_mapping` (not the GID bug)

**Symptom:** Serve comes up; later runtime (AgentX / PP paths that call scalar `postprocess_state`):

```text
IndexError: index_fill_(): Expected dtype int64 for index.
```

in `vllm/v1/worker/gpu/model_states/mamba_hybrid.py` (`num_accepted_tokens_gpu.index_fill_(…, idx_mapping, …)`).

**Cause:** Model Runner V2 `idx_mapping` is `torch.int32` (PP may use `-1` sentinels). `index_fill_` requires int64. Do **not** cast alone — negatives corrupt state. Upstream fix: [vllm#50327](https://github.com/vllm-project/vllm/pull/50327) (Triton `_fill_num_accepted_kernel`); report [vllm#50947](https://github.com/vllm-project/vllm/issues/50947).

**Harness:** Applied in [`scripts/run-vllm-kimi-k3-recipe.sh`](../scripts/run-vllm-kimi-k3-recipe.sh) when the image lacks the kernel — see [`harness-patches.md`](harness-patches.md) and the AgentX/offload write-up [`pp2-humming-agentx-kv-offload-deployment-issues.md`](pp2-humming-agentx-kv-offload-deployment-issues.md) §4.

**Historical:** Older `conc-sweep-pp2-1000-1000-c1-256` saw the same string; main C=1…512 report tree is the GID-fixed IB config.

---

### 5. Recipe PP util / batch overrides vs harness

Recipe `strategy_overrides.multi_node_tp_pp` on the website suggests util **0.90** and higher batched tokens in places; early recipe docker one-liners also show util **0.97**. This harness keeps **0.90** for H200 PP headroom and **4096** batched tokens for the recorded sweep (not 8192).

## Working config (H200 TP8×PP2)

| Knob | Value | Why |
|------|-------|-----|
| `TP_SIZE` / `PP_SIZE` / `NNODES` | 8 / 2 / 2 | Recipe TP+PP |
| `GPU_MEM_UTIL` | **0.90** | Headroom vs TP16’s 0.97 |
| `MAX_NUM_SEQS` | **256** | PP KV allows it |
| `MAX_MODEL_LEN` | 8192 | 1000/1000 bench |
| `MAX_NUM_BATCHED_TOKENS` | 4096 | Sweep setting |
| `NCCL_IB_HCA` | **mlx5_5** | HCA with IPv4 RoCE GID |
| `NCCL_IB_GID_INDEX` | **3** | Avoid `fe80` on mlx5_0 |
| `NCCL_IB_DISABLE` | unset | Keep InfiniBand |
| `NCCL_DMABUF_ENABLE` | 0 | Same as TP16 |
| `PYTORCH_CUDA_ALLOC_CONF` | empty via PP deploy | Avoid expandable_segments hard fail |
| V2 runner / Rust FE | 1 / 0 | Same as TP16 real |
| Attention / MoE | FLASHMLA / marlin | Recipe |

## What still isn’t recipe parity

- H200 vs GB300.
- **Fozzie-specific `NCCL_IB_HCA` / `NCCL_IB_GID_INDEX`** — not in the published recipe docker one-liner; required for PP P2P here.
- Socket (`NCCL_IB_DISABLE=1`) was only a bring-up crutch; do not use for published TP16 vs PP2 numbers.
- `max-model-len` 8192; Rust FE / latent fusion still off.
- Logs may still warn about lazy 2-rank P2P communicator creation — informational after the GID fix.
- Vision encoder: heads not divisible by TP=8 → falls back to DP for ViT (warning only).

## Timeline (condensed)

1. Reuse TP16 storage / patches / iface; switch to TP8×PP2.  
2. Hit PP P2P `IBV_WC_RETRY_EXC_ERR` on link-local RoCE GIDs.  
3. Prove topology with Socket NCCL; then pin **mlx5_5 / GID 3** and re-enable IB.  
4. Raise util/seqs; run C=1…512 concurrency sweep for the report.

## Cross-check vs TP16

| | TP16 | TP8×PP2 |
|--|------|---------|
| Hard blocker | ZMQ race, Mamba/`max-num-seqs`, storage, Gloo iface | **RoCE GID / PP P2P** |
| Util | 0.97 | 0.90 |
| Seq budget | 128 | 256 |
| KV headroom | ~2 GiB / ~6.8×@8k | ~26 GiB / ~177×@8k |
| NCCL IB HCA/GID | unset | **mlx5_5 / 3** |
| Comm stress | Cross-node TP allreduce (default RoCE OK) | Cross-node PP send/recv (needed GID pin) |
