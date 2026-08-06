# TP8×PP2 Humming + AgentX + CPU KV offload: deployment issues

Post-mortem for AgentX MVP on TP8×PP2 Humming with native **CPU KV offload** (`OffloadingConnector`), not weight `--cpu-offload-gb`.

Working entrypoint: [`./scripts/deploy.sh pp2-humming-agentx-kv-offload`](../scripts/deploy.sh) →
[`configs/recipes/pp2-humming-agentx-kv-offload.yaml`](../configs/recipes/pp2-humming-agentx-kv-offload.yaml) →
layers `strategy-pp2` + `moe-humming` + `agentx-long-ctx` + [`kv-cpu-offload-200`](../configs/layers/kv-cpu-offload-200.yaml).

In-cluster AgentX client: [`manifests/aiperf-agentx.yaml`](../manifests/aiperf-agentx.yaml) +
[`scripts/run-agentx-mvp-inpod-c1234.sh`](../scripts/run-agentx-mvp-inpod-c1234.sh).

Baseline (no offload): recipe `pp2-humming-agentx`, results under `bench-results/agentx-mvp-pp2-humming-c{1..4}/`.
Offload sweep: `bench-results/agentx-mvp-pp2-humming-kv-offload-c{N}/`.

Siblings: [`tp8-pp2-h200-deployment-issues.md`](tp8-pp2-h200-deployment-issues.md),
[`harness-patches.md`](harness-patches.md).

## Target shape

| Item | Value |
|------|--------|
| Parallelism | TP=8 × PP=2, Humming MoE |
| Context | `MAX_MODEL_LEN=262144`, ~1.1M GPU KV tokens (~4.14× @ 262k) |
| CPU KV | `OffloadingConnector`, `cpu_bytes_to_use=214748364800` (200 GiB) |
| Prefix cache | **required** (`ENABLE_PREFIX_CACHING=1`) for hybrid block-size align |
| Client | In-pod AIPerf → `vllm-recipe-0.vllm-recipe:8000` (avoid fragile local port-forward) |

## Issues (order hit)

### 1. Wrong offload knob: weight `--cpu-offload-gb` vs KV connector

**Symptom / confusion:** “CPU offload” often means moving **weights** off GPU (`--cpu-offload-gb`). That is not AgentX KV headroom.

**Fix:** Use [`configs/layers/kv-cpu-offload-200.yaml`](../configs/layers/kv-cpu-offload-200.yaml):

```yaml
ENABLE_PREFIX_CACHING: 1
KV_TRANSFER_CONFIG: '{"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"cpu_bytes_to_use":214748364800}}'
```

Harness forwards `--kv-transfer-config` and optional `--enable-prefix-caching` from
[`scripts/run-vllm-kimi-k3-recipe.sh`](../scripts/run-vllm-kimi-k3-recipe.sh).

---

### 2. `KV_TRANSFER_CONFIG` JSON quotes stripped at deploy

**Symptom:** Invalid / empty connector config after start — JSON `"…"` characters gone.

**Cause:** Embedding JSON inside `kubectl exec … bash -c '…'` strips double quotes.

**Fix:** [`scripts/lib/deploy_multi_node.sh`](../scripts/lib/deploy_multi_node.sh) passes knobs via
`kubectl exec -- env KEY=value …` (preserves quotes). `ENABLE_PREFIX_CACHING` and
`KV_TRANSFER_CONFIG` are in the forwarded env list.

---

### 3. Hybrid OffloadingConnector requires prefix caching

**Symptom (engine init):**

```text
tokens_per_block not divisible by tokens_per_hash
```

(or equivalent assert when enabling CPU offload without prefix cache).

**Cause:** Kimi-K3 is hybrid Mamba+Attention. Offload block sizing assumes prefix-cache hash alignment;
stock hybrid defaults often leave `enable_prefix_caching=False` (opt-in only).

**Fix:** `ENABLE_PREFIX_CACHING=1` in the kv-offload layer → `--enable-prefix-caching` on serve.
Bring-up then reports GPU KV ~**1,086,240** tokens and `CPUOffloadingSpec` active; `/health` OK.

**Note:** Earlier AgentX baseline runs without this flag had **server** `prefix_cache_hits=0`;
AIPerf “theoretical” cache hits were client-side only.

---

### 4. Runtime crash: `index_fill_` expects int64 (`idx_mapping` is int32)

**Symptom (~2–3 min into AgentX C=1 profiling):**

```text
File ".../vllm/v1/worker/gpu/model_states/mamba_hybrid.py", line 314, in postprocess_state
    self.num_accepted_tokens_gpu.index_fill_(
IndexError: index_fill_(): Expected dtype int64 for index.
```

Workers die → EngineCore fails → APIServer shuts down. Pods stay `Running`; serve process is gone.
AIPerf then sees `ConnectionRefusedError` to `:8000`.

**Cause:** Model Runner V2 `InputBatch.idx_mapping` is **`torch.int32`** (and may contain **-1** PP
sentinels). The scalar branch of `MambaHybridModelState.postprocess_state` (`num_sampled` is a
plain `int`, e.g. chunked-prefill / non-last PP rank path) called `Tensor.index_fill_`, which
requires **int64**. The tensor branch uses Triton and is dtype-agnostic — only the scalar path trips.

**Do not** “fix” with only `idx_mapping.long()`: `index_fill_` does not skip `-1` sentinels and
would silently corrupt `num_accepted_tokens` under PP.

**Upstream:**

| | Link |
|--|------|
| Issue (exact traceback, Kimi-K3 PP>1) | [vllm#50947](https://github.com/vllm-project/vllm/issues/50947) (open as of 2026-08-04) |
| Fix | [vllm#50327](https://github.com/vllm-project/vllm/pull/50327) — **merged 2026-08-03** |

PR adds Triton `_fill_num_accepted_kernel` (int32-safe, skips `req_state_idx < 0`). Our cluster
image predated the merge.

**Harness fix:** Same change applied at serve start in
[`scripts/run-vllm-kimi-k3-recipe.sh`](../scripts/run-vllm-kimi-k3-recipe.sh) when
`_fill_num_accepted_kernel` is missing from `mamba_hybrid.py`. Log line:

```text
patched .../mamba_hybrid.py (PR #50327 int32 idx_mapping fill)
```

After redeploy, AgentX C=1 survived past the prior crash window with **0** `index_fill_` hits.

Related historical note (same exception string, less root-cause detail):
[`tp8-pp2-h200-deployment-issues.md`](tp8-pp2-h200-deployment-issues.md) §4.

---

### 5. Runtime crash: `cuMemcpyBatchAsync` / `swap_blocks_batch` on OffloadingConnector (PP)

**Symptom:** Serve dies; AIPerf gets `ConnectionRefusedError` (or finishes with a huge error rate if it keeps polling a dead API). Fatal on **PP1** workers during GPU→CPU store on preemption:

```text
File ".../kv_offload/cpu/gpu_worker.py", ... in transfer_async / submit_store
    self._swap_blocks_batch(...)
RuntimeError: swap_blocks_batch, .../cache_kernels.cu:150,
  cuMemcpyBatchAsync failed at index N with error 1
```

(`error 1` = `CUDA_ERROR_INVALID_VALUE` — typically an out-of-range pointer/block index. Observed `N` has been 11 or 42.)

**When we hit it:**

| Phase | Outcome |
|-------|---------|
| Early C=1 @ `MAX_NUM_BATCHED_TOKENS=32768` | Crash ~18 min in (before 8192 retry) |
| C=1 @ `8192` | **Completes** |
| C=2 chained after warm C=1 (no restart) | Crash ~20 s in |
| C=2–5 with **fresh serve restart before each C** | **Complete** (0% errors) |
| C=6 with fresh restart (2 attempts) | Crash ~4–5 min in; retry: 19 ok / 280 err (~94%) |

So restart-between-C raises the floor from “dies on first C=2 after warm cache” to a reproducible **C=6** boundary under AgentX load. It does **not** fix the underlying bug.

Scheduler dumps at crash typically show OffloadingConnector `store` / preemption metadata, often with moderate GPU KV usage (not necessarily near full). Early dumps also noted high GPU prefix hit rate with **External** (CPU-tier) prefix hits still **0.0%**.

**Likely cause (PP + OffloadingConnector):** Native CPU offload derives `num_cpu_blocks` from
each worker’s local KV tensors. PP stages own different layer sets → different per-rank CPU
capacities. Scheduler uses rank-0’s count and can allocate CPU block IDs that **exceed PP1’s
buffer** → invalid DMA in `swap_blocks_batch`.

**Upstream:**

| | Link |
|--|------|
| Exact symptom class (closed, older) | [vllm#39491](https://github.com/vllm-project/vllm/issues/39491) `cuMemcpyBatchAsync` / error 1 on OffloadingConnector |
| Long-context native offload crash (closed; MTP-related fix) | [vllm#44780](https://github.com/vllm-project/vllm/issues/44780) / [PR #44784](https://github.com/vllm-project/vllm/pull/44784) |
| **PP CPU block mismatch (open — best match)** | [vllm#50821](https://github.com/vllm-project/vllm/issues/50821) |
| Fix PR (open, unmerged as of 2026-08-05) | [vllm#50653](https://github.com/vllm-project/vllm/pull/50653) — unify `num_cpu_blocks` to min across PP ranks |

No safe one-line harness patch like #50327; need image with #50653 or run without OffloadingConnector / without PP for AgentX above C=5 until merged.

**Artifacts:** `bench-results/agentx-mvp-pp2-humming-kv-offload-c{1..5}/` (good),
`…-c2-crash/` (chained), `…-c6-crash/` / `…-c6-retry-fail/` (fresh C=6).
Report: [`reports/agentx-mvp-pp2-humming-kv-offload.qmd`](../reports/agentx-mvp-pp2-humming-kv-offload.qmd).
Runner: [`scripts/run-agentx-mvp-inpod-c1234.sh`](../scripts/run-agentx-mvp-inpod-c1234.sh) with `RESTART_BETWEEN=1` + [`scripts/restart-recipe-serve.sh`](../scripts/restart-recipe-serve.sh).

---

### 6. Humming rejects `MoEActivation.SITU` until allowlist lands

**Symptom:** Humming MoE init rejects SiTU activation used by Kimi-K3.

**Fix:** [`scripts/lib/patch_humming_situ.sh`](../scripts/lib/patch_humming_situ.sh) applied by
`deploy_multi_node.sh` when `MOE_BACKEND=humming` (mirrors [vllm#50510](https://github.com/vllm-project/vllm/pull/50510)).
`stop-inpod-vllm.sh` unpatches on stop.

---

### 7. Local port-forward drops mid AgentX (C≥3)

**Symptom:** AIPerf errors / incomplete runs when laptop `kubectl port-forward` to `:18000` drops under long AgentX replay.

**Fix:** Run AIPerf **in-cluster** (`aiperf-agentx` pod, hostNetwork, large CPU/mem) against the
in-cluster Service DNS. Prefer this for C=1–4 15‑minute profiles.

---

## Working config (kv-offload AgentX)

| Knob | Value | Why |
|------|-------|-----|
| Recipe | `pp2-humming-agentx-kv-offload` | Humming + AgentX + OffloadingConnector |
| `MAX_MODEL_LEN` | 262144 | AgentX long ctx |
| `MAX_NUM_BATCHED_TOKENS` | **8192** (was 32768; recipe override) | Retry after §5 crash; also raises GPU KV (~1.79M vs ~1.09M) via lower peak-activation reserve |
| `MAX_NUM_SEQS` | 32 | Fan-out headroom vs stock PP2 256 |
| `ENABLE_PREFIX_CACHING` | 1 | Hybrid offload block align |
| `KV_TRANSFER_CONFIG` | OffloadingConnector, 200 GiB CPU | Native KV offload — **C=1–5 OK with restart-between-C; C=6 blocked by §5 until #50653** |
| Harness patches | MLA, ZMQ, NIXL/CuMem (if set), **PR#50327 mamba**, Humming SiTU | See [`harness-patches.md`](harness-patches.md) |
| NCCL GID / HCA | mlx5_5 / 3 | Same as PP2 bring-up |
| Between-C restart | `RESTART_BETWEEN=1` | Clears GPU/CPU KV + prefix cache; required for fair C>1 (see §5) |

## Timeline (condensed)

1. Baseline AgentX C=1–4 on `pp2-humming-agentx` (no CPU KV offload; server prefix hits 0).
2. Add OffloadingConnector layer; fix env JSON quoting; enable prefix caching for hybrid assert.
3. Serve healthy; in-pod C=1 crash on `index_fill_` / int32 `idx_mapping` → harness [#50327](https://github.com/vllm-project/vllm/pull/50327).
4. Redeploy; C=1 progresses ~18 min then dies on `cuMemcpyBatchAsync` / PP1 `swap_blocks_batch` → [#50821](https://github.com/vllm-project/vllm/issues/50821) / open [#50653](https://github.com/vllm-project/vllm/pull/50653).
5. Recipe override `MAX_NUM_BATCHED_TOKENS=8192` (GPU KV ~1.79M); **C=1 completes**.
6. Chained C=2 (no restart) dies ~20 s in on same `cuMemcpy` class.
7. Sweep with **restart before each C**: **C=2–5 complete** (0% errors); **C=6** crashes twice (~4–5 min in) — reproducible AgentX ceiling under this image until #50653.
