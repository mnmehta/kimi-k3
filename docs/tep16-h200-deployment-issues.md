# TEP16 (multi-node TP + Expert Parallel) on H200: deployment issues

Post-mortem of bringing up [vLLM multi-node tensor + expert parallel](https://recipes.vllm.ai/moonshotai/Kimi-K3?strategy=multi_node_tep) for `moonshotai/Kimi-K3` on 2×8 H200.

Working entrypoint: [`scripts/deploy-recipe-tep.sh`](../scripts/deploy-recipe-tep.sh) → [`scripts/deploy-recipe.sh`](../scripts/deploy-recipe.sh) → [`scripts/run-vllm-kimi-k3-recipe.sh`](../scripts/run-vllm-kimi-k3-recipe.sh) with `ENABLE_EXPERT_PARALLEL=1`.  
Results / report: [`bench-results/conc-sweep-tep-1000-1000/`](../bench-results/conc-sweep-tep-1000-1000/), [`reports/tp16-vs-pp2-1000-1000.qmd`](../reports/tp16-vs-pp2-1000-1000.qmd).

Siblings: [`tp16-h200-deployment-issues.md`](tp16-h200-deployment-issues.md), [`tp8-pp2-h200-deployment-issues.md`](tp8-pp2-h200-deployment-issues.md), [`tp8-dp2-h200-deployment-issues.md`](tp8-dp2-h200-deployment-issues.md).

## Root constraint

| Item | Value |
|------|--------|
| Cluster | ns `kimi-k3` (`export KUBECONFIG=...`) |
| Hardware | 2×8 H200 (140 GiB); recipe target **≥8× GB300** |
| Image | `vllm/vllm-openai:kimi-k3` |
| STS | `manifests/vllm-recipe.yaml` — same as TP16 |
| Weights | hostPath `/mnt/local/kimi-k3/models` → `/models/Kimi-K3` |
| Layout | TP=16 + `--enable-expert-parallel`; rank ≠ 0 `--headless` |

TEP16 is **TP16 plus expert parallel** on the same `nnodes` / `node-rank` / `master-addr` path (not the DP flag set). Shared infra issues (storage, Gloo iface, MLA DCP, ZMQ bind retry, latent-MoE off, dmabuf) are covered in the [TP16 post-mortem](tp16-h200-deployment-issues.md); this note focuses on EP-specific observations and the H200 bring-up that reused those fixes.

## What EP changes vs TP16

**CLI delta:** `--enable-expert-parallel` (optional recipe-adjacent `--enable-ep-weight-filter` left **off** for this sweep).

**Runtime layout (from serve log):**

```text
rank N … TP rank N, EP rank N
[EP Rank 0/16] Expert parallelism is enabled. Expert placement strategy: linear.
Local/global number of experts: 56/896.
```

Workers name as `Worker_TP{i}_EP{i}`. Model load size stayed **129.75 GiB/GPU** (~76 s with warm hostPath page cache) — same as TP16.

**KV (this bring-up):**

| Metric | TEP16 | TP16 (archived) |
|--------|-------|-----------------|
| Available KV cache | **1.58 GiB** | ~1.7–2.2 GiB |
| GPU KV cache size | **52,428** tokens | ~55k tokens |
| Max concurrency @ 8192 | **6.40×** | ~6.76× |

Slightly tighter KV than TP16 at the same `gpu-memory-utilization=0.97` / `max-num-seqs=128` / `max-model-len=8192` knobs — treat as EP overhead / allocator variance, not a separate knob hunt for this evaluation.

## Issues (TEP-specific / this bring-up)

### 1. Reuse TP16 harness; only flip EP

**Symptom / risk:** Temptation to invent a second multi-node path or enable DP flags for “expert parallel”.

**Cause:** Recipe UI groups TEP next to DEP/DP strategies; H200 DEP is unsupported.

**Fix:** `ENABLE_EXPERT_PARALLEL=1` on the existing TP16 deploy path (`deploy-recipe-tep.sh`). Do **not** pass `--data-parallel-*` for TEP16.

---

### 2. In-pod restart preferred when flipping from another strategy

**Symptom:** Full STS pod delete after DP2 → cold-ish weight reload even with hostPath.

**Cause:** Pod recreate drops process state; host page cache helps but GPU reload still dominates.

**Fix:** Stop in-pod serve (`stop-inpod-vllm.sh`), copy updated `run-vllm-kimi-k3-recipe.sh`, start workers then head with TEP env. Same pattern as DP2 iteration loops.

---

### 3. Soft allocator / high watermark after graphs (non-fatal)

**Symptom:** `nvidia-smi` ~**141.8 GiB** used after CUDA graph capture (~2.4 GiB graphs) while weights logged at 129.75 GiB; expandable_segments mapping warnings may appear (same class as TP16).

**Cause:** KV + graphs + MoE workspace on top of weights under `gpu_memory_utilization=0.97`.

**Fix:** None for bring-up — engine reached `Application startup complete` and `/health`. Do not chase util below 0.97 unless KV collapses further.

---

### 4. Optional EP weight filter not required for first sweep

**Symptom / choice:** Recipe may mention EP weight filtering.

**Fix for this eval:** `ENABLE_EP_WEIGHT_FILTER=0` (default) so TEP vs TP16 differs by **one** flag (`--enable-expert-parallel`). Revisit filter if weight-load RAM or disk I/O becomes the bottleneck.

## Working config (H200 TEP16)

| Knob | Value | Why |
|------|-------|-----|
| `TP_SIZE` / `PP_SIZE` / `DP_SIZE` / `NNODES` | 16 / 1 / 1 / 2 | Recipe multi-node TEP |
| `ENABLE_EXPERT_PARALLEL` | 1 | `--enable-expert-parallel` |
| `ENABLE_EP_WEIGHT_FILTER` | 0 | Apples-to-apples vs TP16 |
| `GPU_MEM_UTIL` | 0.97 | Match TP16 sweep |
| `MAX_NUM_SEQS` | 128 | Mamba blocks (same as TP16) |
| `MAX_MODEL_LEN` | 8192 | 1000/1000 bench |
| `MAX_NUM_BATCHED_TOKENS` | 4096 | Recipe-ish |
| Attention / MoE | FLASHMLA / marlin | Recipe |
| `--disable-custom-all-reduce` | set | Recipe |
| `--no-enable-flashinfer-autotune` | set | Recipe |
| `VLLM_USE_V2_MODEL_RUNNER` | 1 | Script default |
| `VLLM_USE_RUST_FRONTEND` | 0 | Stability |
| Latent-MoE tail fusion | 0 | SM90 |
| `NCCL_DMABUF_ENABLE` | 0 | dmabuf errno 524 |
| IFACE | `enp*` | Gloo/NCCL socket |
| Patches | MLA dcp + ZMQ bind retry | Image bugs (from TP16) |

## What still isn’t recipe parity

Same gaps as TP16 (H200 vs GB300, 8192 context, `max-num-seqs` 128, Rust FE off, etc.), plus:

- EP weight filter not enabled.
- No separate EP communication tuning beyond recipe defaults / existing NCCL socket + dmabuf knobs.

## Timeline (condensed)

1. Inherit TP16 storage / Gloo / MLA / ZMQ / Mamba fixes.  
2. Add `ENABLE_EXPERT_PARALLEL` to recipe serve + `deploy-recipe-tep.sh`.  
3. In-pod stop DP2 → start TEP16 (workers first).  
4. Confirm EP map 56/896, load 129.75 GiB/GPU, KV ~1.58 GiB, `/health`.  
5. Sweep C=1…512 → `bench-results/conc-sweep-tep-1000-1000/`.
