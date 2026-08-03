# TP8×DP2 on H200: deployment issues

Post-mortem of bringing up [vLLM multi-node TP+DP](https://recipes.vllm.ai/moonshotai/Kimi-K3?strategy=multi_node_tp_dp) for `moonshotai/Kimi-K3` on 2×8 H200 (hostPath weights).

Working entrypoint: [`scripts/deploy-recipe-dp.sh`](../scripts/deploy-recipe-dp.sh) → [`scripts/run-vllm-kimi-k3-recipe.sh`](../scripts/run-vllm-kimi-k3-recipe.sh).  
Results / report: [`bench-results/conc-sweep-dp2-1000-1000/`](../bench-results/conc-sweep-dp2-1000-1000/), [`reports/tp16-vs-pp2-1000-1000.qmd`](../reports/tp16-vs-pp2-1000-1000.qmd).

Siblings: [`tp16-h200-deployment-issues.md`](tp16-h200-deployment-issues.md), [`tep16-h200-deployment-issues.md`](tep16-h200-deployment-issues.md), [`tp8-pp2-h200-deployment-issues.md`](tp8-pp2-h200-deployment-issues.md).

## Root constraint

A **TP8 replica** of MXFP4 Kimi-K3 is ~**135.7 GiB/GPU** (TP16 was ~129.8 GiB). On 140 GiB H200 that leaves only **~150–265 MiB free** after weights — before KV, FlashInfer workspace, MoE scratch, mm-encoder profiling, or Mamba blocks.

The published recipe targets **GB300**-class memory; H200 is undersized for an “open” TP8+DP2 config. Everything below is fighting that headroom.

## Issues (in roughly the order hit)

### 1. CLI mode: DP flags vs `nnodes` / `node-rank`

**Symptom:** Confusing serve args / workers not joining as DP replicas if TP/PP-style multi-node flags are mixed with DP.

**Cause:** For DP, vLLM wants `--data-parallel-size`, `--data-parallel-size-local`, `--data-parallel-address`, `--data-parallel-start-rank`, and `--headless` on non-zero DP ranks — **not** `--nnodes` / `--node-rank` / `--master-addr` (those stay on the TP/PP path).

**Fix:** Branch in `run-vllm-kimi-k3-recipe.sh` when `DP_SIZE > 1`. Deploy wrapper sets `TP_SIZE=8 DP_SIZE=2 DP_SIZE_LOCAL=1`.

---

### 2. `--gpu-memory-utilization` alone cannot buy a real KV budget

| Attempt | Result |
|--------|--------|
| `0.97` (recipe-ish) | OOM / failure inside `determine_available_memory` (or immediately after) — not enough free for profiling + KV sizing |
| `0.98`–`0.99` | Rejected: desired reserved memory **greater than free** after weight load |

**Takeaway:** Raising util past ~0.97 does not create headroom; weights already consume almost the whole device. Need smaller workspaces + pinned tiny KV, not a higher util fraction.

---

### 3. Multimodal encoder profiling wants ~396 MiB

**Symptom (repeated):**

```text
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 396.00 MiB.
GPU 0 ... of which ~20–265 MiB is free
```

Seen with recipe-like settings (`max-model-len=8192`, `max-num-seqs=256`, Rust/V2 on or off) and again after shrinking other knobs **until** mm profiling was skipped.

**Cause:** Kimi-K3 mm-encoder `profile_run` path allocates ~**396 MiB**, larger than free VRAM after TP8 weights. Shrinking FlashInfer / KV does **not** avoid this alloc.

**Fix:** `--skip-mm-profiling` (`SKIP_MM_PROFILING=1` in `deploy-recipe-dp.sh`).

---

### 4. Default FlashInfer workspace (~394 MiB) also won’t fit

**Symptom:** OOM or init failure allocating FlashInfer workspace buffer near default size (~394 MiB) while only ~200 MiB free.

**Fix:** `VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE=67108864` (64 MiB).

---

### 5. Skipping `profile_run` entirely → MoE workspace undersized at first request

**Symptom:** Engine could init with an in-image / harness patch that skipped `model_runner.profile_run()` when `--kv-cache-memory-bytes` was set, then **failed on the first real/dummy batch** because MoE workspace was locked tiny (~3.5 MiB) vs need (~219 MiB).

**Cause:** Profiling also sizes MoE scratch; skipping it leaves an unusable workspace for decode/prefill.

**Fix:** Do **not** skip the full GPU worker profile long-term. Use `--skip-mm-profiling` only, keep normal `profile_run`, and shrink `max-num-batched-tokens` (256) so profiled activations fit.

---

### 6. Pinned KV vs `max-model-len` mismatch

**Symptom:**

```text
ValueError: To serve at least one request with the model's max seq len (4096),
(0.18 GiB KV cache is needed, which is larger than the available KV cache memory (0.15 GiB).
... estimated maximum model length is 3072.
```

**Cause:** `--kv-cache-memory-bytes=160000000` (~0.15 GiB) cannot hold one sequence at `max-model-len=4096`.

**Fix:** Drop to `MAX_MODEL_LEN=2048` (enough for 1000/1000 bench) with `KV_CACHE_MEMORY_BYTES=160000000`. Intermediate try at 3072 still left almost no concurrency.

---

### 7. Mamba cache blocks vs `max-num-seqs`

**Symptom:**

```text
ValueError: max_num_seqs (256) exceeds available Mamba cache blocks (7).
Each decode sequence requires one Mamba cache block...
Please lower max_num_seqs to at most 7 ...
```

**Cause:** With a tiny KV budget, KDA/Mamba state slots collapse; recipe `max-num-seqs=256` is impossible.

**Fix:** `MAX_NUM_SEQS=2` (even 7 is optimistic for meaningful 1000/1000 concurrency).

---

### 8. Recipe Rust frontend / V2 model runner on H200

**Symptom:** Extra bring-up fragility / failed cores when enabling `VLLM_USE_RUST_FRONTEND=1` and `VLLM_USE_V2_MODEL_RUNNER=1` on the already memory-tight TP8 path.

**Fix:** Leave both **off** by default in `deploy-recipe-dp.sh` (override still allowed). TP16/PP2 recipe paths can keep recipe defaults separately.

---

### 9. Effective serving capacity is ~1 request per replica

After a successful bring-up, logs reported roughly:

- GPU KV cache size: **~2,389 tokens**
- Max concurrency: **~1.17× @ 2048** tokens/request

So for 1000/1000:

- ~**1 in-flight request per DP replica**
- ~**2 cluster-wide** with DP=2

**Bench implication:** Output tok/s rises from C=1 → C=2 (~2×), then **plateaus by C=4** (~85 tok/s). Higher concurrency only adds queue / E2EL. Full C=1…512 sweep (as for TP16/PP2) is not informative and wall-clock explodes; DP2 sweep stopped at **C=1…16**.

---

### 10. Long weight reload vs pod reuse

**Symptom:** Full pod recreate + cold read of ~1.56 TB hostPath weights is multi-hour.

**Mitigation:** Prefer in-pod stop/restart (`stop-inpod-vllm` + re-`run-recipe`) so page cache stays warm while iterating knobs. Tradeoff: leftover GPU processes / stale log lines can confuse “fail fast on OOM” waiters (match on dead process + fresh log).

---

### 11. Git push noreply (harness checkout, not serve)

**Symptom:** `GH007: Your push would publish a private email address` when committer was `mimehta@redhat.com`.

**Fix:** Amend unpushed commit with `GIT_AUTHOR_EMAIL` / `GIT_COMMITTER_EMAIL=30246802+mnmehta@users.noreply.github.com` (do not change global git config).

## Working config (H200 TP8×DP2)

Defaults baked into `deploy-recipe-dp.sh`:

| Knob | Value | Why |
|------|-------|-----|
| `TP_SIZE` / `DP_SIZE` / `DP_SIZE_LOCAL` | 8 / 2 / 1 | Recipe layout: one full replica per node |
| `GPU_MEM_UTIL` | 0.97 | Higher rejected |
| `MAX_MODEL_LEN` | 2048 | Fits 1000/1000 + tiny KV |
| `MAX_NUM_BATCHED_TOKENS` | 256 | Profile / MoE scratch fits |
| `MAX_NUM_SEQS` | 2 | Mamba blocks + KV |
| `KV_CACHE_MEMORY_BYTES` | 160000000 | Explicit ~160 MiB KV |
| `VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE` | 67108864 | 64 MiB vs ~394 MiB default |
| `SKIP_MM_PROFILING` | 1 | Avoid 396 MiB mm profile alloc |
| Rust FE / V2 runner | 0 | Stability / footprint |

## What still isn’t “recipe parity”

- Context and concurrency far below Dynamo / recipe GB300 targets (`max-model-len` 8192, large `max-num-seqs`, large KV).
- DP2 throughput comparison to TP16/PP2 is **KV-limited**, not a fair scale-out comparison on H200.
- Fair head-to-head needs more free VRAM per GPU (GB300 or aggressive memory reductions not attempted here: quantization already MXFP4, weights are the floor).

## Timeline of failed attempts (condensed)

1. Recipe-shaped knobs (`util` 0.97–0.99, len 8192, seqs 256, Rust/V2) → util reject or **396 MiB** OOM.  
2. Pin small KV + skip full `profile_run` → init OK, **MoE workspace** too small at first batch.  
3. Re-enable profile, lower `max-model-len` / pin KV → **max-model-len vs KV GiB** ValueError; then **Mamba blocks vs max-num-seqs**.  
4. Shrink FlashInfer workspace + **`--skip-mm-profiling`** + `max-model-len=2048` + `max-num-seqs=2` → **/health** + completions + C=1…16 sweep.
