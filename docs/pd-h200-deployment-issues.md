# Prefill/Decode disaggregation (`pd_cluster`) on H200: deployment issues

Notes from running [vLLM Prefill/Decode Disaggregation](https://recipes.vllm.ai/moonshotai/Kimi-K3?strategy=pd_cluster) for `moonshotai/Kimi-K3` on **fozzie**.

**Status (2026-08-02): measured on 4×8 H200.** Earlier 2×8 TP8-per-role adaptation OOMed during MXFP4 `create_weights` (below). Completed sweep: [`bench-results/conc-sweep-pd-1000-1000/`](../bench-results/conc-sweep-pd-1000-1000/) (C=1…512, 1000/1000 via router). Required env: `VLLM_SSM_CONV_STATE_LAYOUT=DS` for hybrid Mamba KV transfer over NIXL.

Entrypoint: [`scripts/deploy-recipe-pd.sh`](../scripts/deploy-recipe-pd.sh) → [`scripts/wait-pd-and-sweep.sh`](../scripts/wait-pd-and-sweep.sh).  
Weight sync: [`scripts/sync-model-hostpath.sh`](../scripts/sync-model-hostpath.sh).  
Report: [`reports/tp16-vs-pp2-1000-1000.qmd`](../reports/tp16-vs-pp2-1000-1000.qmd).

Siblings: [`tp16-h200-deployment-issues.md`](tp16-h200-deployment-issues.md), [`tep16-h200-deployment-issues.md`](tep16-h200-deployment-issues.md), [`tp8-pp2-h200-deployment-issues.md`](tp8-pp2-h200-deployment-issues.md), [`tp8-dp2-h200-deployment-issues.md`](tp8-dp2-h200-deployment-issues.md).

## Root constraint

| Item | Value |
|------|--------|
| Cluster | fozzie (`kubeconfig.fozzie`), ns `kimi-k3` |
| Hardware | 4×8 H200 (140 GiB); recipe target **≥8× GB300** |
| Recipe default P/D | **4 nodes / 32 GPUs** (TEP16 prefill + TEP16 decode) |
| This harness | **4 nodes / 32 GPUs** (recipe layout) |
| Image | `vllm/vllm-openai:kimi-k3` (ships `nixl` + `disagg_proxy_demo.py`) |
| Weight nodes | all four: `g1191e4`, `g13c364`, `g11cd44`, `gf2a612` (hostPath NVMe) |

### Recipe layout (H200 4×8) — current

| Role | Pods | Parallelism | HTTP | NIXL | KV role |
|------|------|-------------|------|------|---------|
| Prefill | `vllm-recipe-0,1` | TEP16 | `:8001` | host=pod0, `:5557` | `kv_producer` |
| Decode | `vllm-recipe-2,3` | TEP16 | `:8002` | host=pod2, `:5558` | `kv_consumer` |
| Router | `vllm-recipe-0` | — | `:8000` | — | `disagg_proxy_demo` |

### Abandoned adaptation (H200 2×8)

| Role | Pod | Parallelism | HTTP | NIXL | KV role |
|------|-----|-------------|------|------|---------|
| Prefill | `vllm-recipe-0` | TP8 | `:8001` | host=`pod0`, `:5557` | `kv_producer` |
| Decode | `vllm-recipe-1` | TP8 | `:8002` | host=`pod1`, `:5558` | `kv_consumer` |
| Router | `vllm-recipe-0` | — | `:8000` | — | `disagg_proxy_demo` |

## Blocker: single-node TP8 weight create OOM

**Symptom (reproducible, with or without NIXL):**

```text
torch.OutOfMemoryError: Tried to allocate 588.00 MiB.
... 137.23 GiB is allocated by PyTorch ... of which 546.94 MiB is free
```

Fails in `mxfp4.create_weights` during `make_layers` (before `Model loading took`). Same numbers with EP on/off, enforce-eager on/off, DP CLI flags at `data_parallel_size=1`, fresh pods on weight nodes.

**Control that works:** [`deploy-recipe-dp.sh`](../scripts/deploy-recipe-dp.sh) TP8×DP2 on the same nodes → `Model loading took 135.69 GiB` and `/health`.

**Implication:** P/D needs **two independent** TP8 engines (one per node). That path is exactly the configuration that OOMs. DP2 proves the weights fit only when launched as a multi-rank DP cluster (`data_parallel_size=2`).

## Other issues hit on the way

### 1. Recipe wants 32 GPUs; harness has 16

One node per role (not TEP16×2). Documented adaptation.

### 2. hostPath + `DirectoryOrCreate` on the wrong node

Deleting STS pods can reschedule onto nodes without the 1.5 T copy (`DirectoryOrCreate` makes an empty `/models`). **Fix:** nodeAffinity to `g1191e4` / `g13c364` in [`manifests/vllm-recipe.yaml`](../manifests/vllm-recipe.yaml).

### 3. NixlConnector vs `expandable_segments`

Validation rejects expandable_segments unless `--enable-cumem-allocator`. CuMem’s weight pool then forces `expandable_segments:False` and OOMs earlier. Harness patches (when `KV_TRANSFER_CONFIG` is set):

- Allow expandable_segments with KV connector (`vllm/config/vllm.py`) — DP2-style allocator for weights.
- Skip CuMem pool for weights only (`gpu_worker.load_model`); KV pool unchanged if cumem is on.

Even with those patches and **no** KV config, solo TP8 still OOMs at 137.23 GiB — so NIXL is not the root cause of the blocker.

### 4. NIXL / TP8 knobs prepared (unused until load works)

| Knob | Value |
|------|--------|
| `kv_buffer_size` | 32 MiB (default 1 GiB is too large) |
| `kv_buffer_device` | `cpu` |
| FlashInfer workspace | 64 MiB |
| `--skip-mm-profiling` | yes |
| `max-model-len` / batched tokens / seqs | 2048 / 256 / 2 |
| `kv-cache-memory-bytes` | ~160 MiB |
| EP | off (default) |
| Router | `disagg_proxy_demo.py` on `:8000` |

## What unblocks (resolved)

- Recipe-shaped P/D on **4×8 H200** (TEP16 prefill + TEP16 decode) — same shape as recipe, smaller GPUs than GB300. **Measured** 2026-08-02.
- Populate hostPath on `g11cd44` / `gf2a612` via [`sync-model-hostpath.sh`](../scripts/sync-model-hostpath.sh) (parallel 0→2 and 1→3).
- `VLLM_SSM_CONV_STATE_LAYOUT=DS` required for NIXL hybrid Mamba conv-state transfer (3-read layout).
- Solo TP8 OOM path is abandoned for this evaluation.
- Sweep data: [`bench-results/conc-sweep-pd-1000-1000/`](../bench-results/conc-sweep-pd-1000-1000/) — C=1 ≈ 35.3 tok/s, plateaus ~229 tok/s from C≈16 (KV-limited).

## Timeline (condensed)

1. Built P/D deploy / wait / sweep scripts (2×8 adaptation of 4-node recipe).  
2. Hit NIXL ↔ expandable_segments / CuMem interactions; patched harness.  
3. Hit empty hostPath after pod reschedule; pinned weight nodes.  
4. Confirmed solo TP8 OOM vs working DP2 TP8 on the same hardware → 2×8 adaptation abandoned.  
5. Scaled STS to 4 nodes; synced weights to empty ranks; redeployed recipe 4×8 P/D.  
6. **2026-08-02:** Completed concurrency sweep (C=1…512) with `VLLM_SSM_CONV_STATE_LAYOUT=DS`; results in `conc-sweep-pd-1000-1000/`.
