# Torch profiler comparison (GPU 0 / replica 0)

Side-by-side look at the four completed minimal-depth profiler sweeps under
[`profiler-results/`](../profiler-results/), focused on **how parallelization
shows up on a single GPU** (`dp0_pp0_tp0_…_rank0` on pod `vllm-recipe-0`).

## Workload (same for all four)

| Knob | Value |
|------|------:|
| Weights | dummy |
| Layers / experts | 4 / 32 (2 routed + 1 shared) |
| Prompt | warmup + 10 in / 5 out |
| Nodes × GPUs | 2 × 8 H200 |
| Capture | vLLM torch profiler → Chrome/Perfetto `.pt.trace.json.gz` |

Traces analyzed:

| Config | Parallelism | Worker name in trace | Trace file |
|--------|-------------|----------------------|------------|
| `tp16` | TP=16 | `VLLM::Worker_TP0` | `profiler-results/tp16/rank-0/dp0_pp0_tp0_…_rank0.*.gz` |
| `tep16` | TP=16 + EP | `VLLM::Worker_TP0_EP0` | `profiler-results/tep16/rank-0/…` |
| `pp2` | PP=2 × TP=8 | `VLLM::Worker_PP0_TP0` | `profiler-results/pp2/rank-0/…` |
| `dp2` | DP=2 × TP=8 | `VLLM::Worker_DP0_TP0` | `profiler-results/dp2/rank-0/…` |

**Caveats.** This is a shallow dummy model and a tiny request, so **communication
dominates** and absolute milliseconds are not production throughput. Numbers
below are **GPU kernel time on rank 0 only** (excluding `execute_context_*`
annotation spans). Prefer reading them as *shape of the timeline*, not as a
leaderboard.

## Kernel-time summary (rank 0)

| Bucket | tp16 | tep16 | pp2 | dp2 |
|--------|-----:|------:|----:|----:|
| **Total kernel time** | 16.6 ms | 19.2 ms | 38.1 ms | 35.0 ms |
| AllReduce / multimem AR | 12.5 ms (75%) | 15.4 ms (80%) | 4.5 ms (12%) | 15.6 ms (45%) |
| AllGather | 0.5 ms | 0.4 ms | — | 6.4 ms (18%) |
| ReduceScatter | — | — | — | 0.8 ms (2%) |
| Broadcast | — | — | **29.8 ms (78%)** | — |
| SendRecv | — | — | **2.1 ms (6%)** | — |
| GEMM (incl. Marlin MoE) | 2.4 ms | 2.0 ms | 1.1 ms | 8.1 ms |
| Attention | 0.4 ms | 0.4 ms | 0.1 ms | 1.4 ms |
| `#kernels` | 784 | 784 | 382 | 3076 |

Notable CPU-side coordination (not GPU kernels):

| Config | CPU coordination signature |
|--------|----------------------------|
| tp16 / tep16 | `vllm::all_reduce` + `vllm::all_gather` (sub-ms) |
| pp2 | **`gloo:send` ~18 ms**, `c10d::send` / `nccl:send 0->1`, `nccl:broadcast` |
| dp2 | **`gloo:all_reduce` ~84 ms** (DP / engine sync), `c10d::allreduce_` |

---

## What differs, by strategy

### 1. TP16 — cross-node tensor-parallel AllReduce wall

Rank 0 is one shard of a **16-way TP group spanning both nodes**. The GPU
timeline is almost entirely **NCCL AllReduce** (TREE_LL for most calls, RING_LL
for a few), with a handful of AllGathers:

- `ncclDevKernel_AllReduce_Sum_bf16_TREE_LL` ≈ 9.9 ms (52 launches)
- `ncclDevKernel_AllReduce_Sum_bf16_RING_LL` ≈ 2.6 ms (8 launches)
- No `multimem_all_reduce` — the TP group is larger than one node, so NVLS
  multimem AR is not used.

MoE still runs **Marlin** on this rank (`marlin_moe_wna16` ≈ 0.41 ms). Experts
are TP-sharded, not expert-parallel, so each rank still does a full-width slice
of the MoE work without an AllToAll dispatch visible here.

**Attribution:** the dominant cost is the classic TP activation sync over the
2-node NCCL clique.

### 2. TEP16 — same collective *shape*, less local MoE compute

On GPU 0 the collective mix is **still AllReduce + AllGather** — nearly the
same launch counts as TP16 (60 AR-class kernels, 5 AllGather). There are **no
`nccl AllToAll` device kernels** in this trace.

What *does* change in a way that matches EP:

| Signal | tp16 | tep16 |
|--------|-----:|------:|
| Worker label | `TP0` | `TP0_EP0` |
| Marlin MoE kernel time | 0.41 ms | **0.06 ms** (~6× lower) |
| `moe_sum_vec_kernel` template | `…, false>` | `…, true>` |
| NCCL AllReduce time | 12.5 ms | **15.4 ms** (higher) |
| Kernel name set | ≈70 shared | only MoE-sum flag differs |

So for this short/shallow capture, expert parallel shows up as **fewer experts
(and less GEMM) on this rank**, not as a textbook token-dispatch AllToAll on
the NCCL timeline. Python frames also touch `use_sequence_parallel_moe` /
cudagraph dispatch helpers; the `moe_sum_vec_kernel<…, true>` bit aligns with
that path.

**Attribution:** EP partitions MoE compute across ranks; TP-style AllReduce
remains the visible interconnect tax on rank 0 for this workload.

### 3. PP2 — pipeline P2P / broadcast, half the layers, intra-node TP

Rank 0 is **pipeline stage 0** (`Worker_PP0_TP0`) with **TP=8 on one node**.
The communication signature flips completely:

1. **`ncclDevKernel_Broadcast_RING_LL`** — ~30 ms, 10 launches (~78% of kernel time)
2. **`ncclDevKernel_SendRecv`** — ~2.1 ms, annotated `nccl:send 0->1` (stage handoff)
3. **TP AllReduce becomes `multimem_all_reduce`** (~4.5 ms, 25 launches) — NVLink
   multimem within the 8-GPU node instead of 16-rank TREE/RING

Compute is roughly **half** of TP16/TEP16 (382 vs 784 kernels; Marlin ~0.14 ms)
because PP0 only owns the first half of the 4-layer stack. CPU shows matching
pipeline control traffic (`gloo:send`, `c10d::send`, `c10d::broadcast_`).

**Attribution:** almost all of the “extra” GPU time vs TP16 is **PP stage
transfer** (broadcast + send/recv to PP1), while TP sync gets cheaper because
the TP group no longer crosses InfiniBand.

### 4. DP2 — independent replica + DP control plane

Rank 0 is **data-parallel replica 0** (`Worker_DP0_TP0`) with TP=8. GPU
collectives look like a **single-node TP worker**, not like a second TP clique:

- **`multimem_all_reduce`** dominates GPU AR (~15.6 ms, 198 launches) — intra-node TP
- **AllGather + ReduceScatter** (54 each) appear as the TP shard exchange pattern
  for this TP=8 path (TP16 preferred fused AllReduce instead)
- **No** cross-replica NCCL AllReduce on the GPU for the model forward

The DP-specific cost is on the **CPU / gloo** side: `gloo:all_reduce` ≈ **84 ms**
across 18 calls (also visible as `gpu_user_annotation`), i.e. engine / DP
synchronization around the step rather than weight-gradient sync.

This trace also has far more kernel launches (3076) and no
`execute_context_*` GPU annotations (unlike the other three), so treat absolute
DP2 kernel totals cautiously — the *kinds* of ops are the reliable signal.

**Attribution:** each DP replica runs a full TP=8 forward locally; parallelism
across replicas shows up as **gloo DP sync**, not as extra model collectives on
GPU 0.

---

## Parallelism → collective fingerprint

```text
tp16   :  NCCL AllReduce (TREE/RING, 16-rank, cross-node)  + AllGather
tep16  :  same as tp16 on GPU0, but much less Marlin MoE (EP shard) 
pp2    :  NCCL Broadcast + SendRecv (PP)  + multimem AllReduce (TP=8)
dp2    :  multimem AllReduce + AllGather/ReduceScatter (TP=8)
          + gloo all_reduce (DP control plane)
```

| If you see… | It usually means… |
|-------------|-------------------|
| Many `AllReduce_*_TREE_LL` / `RING_LL`, worker `TP0` | Multi-node TP |
| Same AR shape + `TP0_EP0` + collapsed Marlin time | TP + EP (experts sharded) |
| `Broadcast` + `SendRecv` / `nccl:send 0->1` + `PP0` | Pipeline stage boundary |
| `multimem_all_reduce` without Broadcast/SendRecv | TP group fits on one node |
| Heavy `gloo:all_reduce` with `DP0` worker | Data-parallel coordination |

## How to re-open in Perfetto

```bash
# pick GPU0 / rank0 of a config
gunzip -k profiler-results/tp16/rank-0/dp0_pp0_tp0_dcp0_ep0_rank0.*.pt.trace.json.gz
# open the .json in https://ui.perfetto.dev
```

Filter / search tips: `nccl`, `multimem`, `Marlin`, `SendRecv`, `Broadcast`,
`gloo`, `execute_context`.

## Gaps

- **`pd` (P/D)** not included — fourth node was missing when the sweep ran.
- TEP AllToAll / DeepEP dispatch may appear on larger batches or other ranks;
  it is **not** in this GPU-0 short-query capture.
- Absolute times are for a 4-layer dummy model; use the **op mix** when comparing
  strategies.
