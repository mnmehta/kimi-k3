# TP16 (multi-node TP) on H200: deployment issues

Post-mortem of bringing up [vLLM multi-node tensor parallel](https://recipes.vllm.ai/moonshotai/Kimi-K3?strategy=multi_node_tp) for `moonshotai/Kimi-K3` on 2×8 H200.

Working entrypoint: [`./scripts/deploy.sh tp16`](../scripts/deploy.sh) → recipe [`configs/recipes/tp16.yaml`](../configs/recipes/tp16.yaml) → [`scripts/run-vllm-kimi-k3-recipe.sh`](../scripts/run-vllm-kimi-k3-recipe.sh).  
Results / report: [`bench-results/conc-sweep-real-1000-1000/`](../bench-results/conc-sweep-real-1000-1000/), [`reports/tp16-vs-pp2-1000-1000.qmd`](../reports/tp16-vs-pp2-1000-1000.qmd).

Siblings: [`tep16-h200-deployment-issues.md`](tep16-h200-deployment-issues.md), [`tp8-pp2-h200-deployment-issues.md`](tp8-pp2-h200-deployment-issues.md), [`tp8-dp2-h200-deployment-issues.md`](tp8-dp2-h200-deployment-issues.md).

## Root constraint

| Item | Value |
|------|--------|
| Cluster | ns `kimi-k3` (`export KUBECONFIG=...`) |
| Hardware | 2×8 H200 (140 GiB); recipe target **≥8× GB300** |
| Image | `vllm/vllm-openai:kimi-k3` |
| STS | `manifests/vllm-recipe.yaml` — `hostNetwork`, `hostIPC`, privileged, 8 GPU + RDMA |
| Weights | hostPath `/mnt/local/kimi-k3/models` → `/models/Kimi-K3` (~1.56 TB MXFP4) |
| Layout | TP=16 across 2 nodes; rank ≠ 0 `--headless` |

TP16 loads ~**129.75 GiB/GPU**. Archived success log: available KV ~**1.7–2.2 GiB** → ~**55k** KV tokens, max concurrency ~**6.76× @ 8192**.

## Issues (in roughly the order hit)

### 1. Storage: PVC auth failure, then wrong hostPath

**Symptom:** `shared-vast` PVC never Bound (`Authentication failure` on StorageClass provisioner). Early hostPath under `/var/lib/...` landed on a **15 G tmpfs**.

**Cause:** No usable VAST CSI for this project; path too small for 1.56 TB.

**Fix:** Per-node CoreWeave NVMe **`/mnt/local/kimi-k3/models`** (see [`manifests/STORAGE.md`](../manifests/STORAGE.md)). Survives pod delete; not guaranteed across node reboot. Early sweeps may still log `OVERLAY` if taken before the durable hostPath cutover.

---

### 2. Gloo / NCCL socket iface and IPv6 link-local

**Symptom:** Cross-node Gloo timeouts when the harness preferred IB ifaces (`ibs*`) that only had `fe80::…`.

**Cause:** Gloo on IPv6 link-local / address-family mismatch with the peer.

**Fix:** Prefer Ethernet `enp*` for `GLOO_SOCKET_IFNAME` / `NCCL_SOCKET_IFNAME`; disable IPv6 under `/proc/sys/net/ipv6/conf/*/disable_ipv6` in the serve script. Working iface in logs: **`enp157s0np0`**.

---

### 3. Latent-MoE tail fusion requires SM100

**Symptom:** Bring-up / kernel path inappropriate for H200 after networking was healthy.

**Cause:** `VLLM_ENABLE_K3_LATENT_MOE_TAIL_FUSION=1` is aimed at **SM100** (Blackwell). H200 is **SM90**.

**Fix:** Default **`VLLM_ENABLE_K3_LATENT_MOE_TAIL_FUSION=0`** in the real-weight recipe script (Ashish/`notes.md` docker snippets often leave it on).

---

### 4. MLA `dcp_world_size` stuck at `-1`

**Symptom:**

```text
cp_world_size must be positive ... Use 1 if CP is not enabled.
```

**Cause:** Kimi MLA calls `forward_mqa` without going through `MLAAttention.forward` lazy-init, so `dcp_world_size` stays `-1`.

**Fix:** In-pod patch in `run-vllm-kimi-k3-recipe.sh` on `…/kimi_k3/nvidia/mla.py` (set DCP size from `get_dcp_group()` or `1`).

---

### 5. ZMQ `EADDRINUSE` after full-weight load

**Symptom:** Workers die with ZMQ **Address already in use** (e.g. colliding on `tcp://…:39555`) after ~130 GiB/GPU is resident — long after “weights loaded”.

**Cause:** TOCTOU in `get_open_port()` under `hostNetwork` with many TP workers.

**Fix:** Harness patch on `shm_broadcast.py` — retry bind up to 64 times on `zmq.EADDRINUSE` (“Kimi-K3 harness: retry ZMQ bind”).

---

### 6. `max-num-seqs` vs Mamba cache blocks

**Symptom:**

```text
max_num_seqs (256) exceeds available Mamba cache blocks (161)
```

**Cause:** Full MXFP4 weights leave a limited Mamba/KDA block budget; recipe-ish 256 seqs cannot finish CUDA graph capture.

**Fix:** **`MAX_NUM_SEQS=128`** for TP16 real ([`configs/layers/base.yaml`](../configs/layers/base.yaml) / `tp16` recipe default). Lowering `max-model-len` alone did not free enough blocks.

---

### 7. Rust frontend off by default

**Symptom:** Extra fragility with Rust FE on this image/path (esp. dummy/`hf_overrides` era); recipe page enables it.

**Fix:** **`VLLM_USE_RUST_FRONTEND=0`** default for real recipe serve; V2 model runner stays on (`1`).

---

### 8. Soft allocator pressure (non-fatal)

**Symptom (in successful TP16 log):**

```text
expandable_segments: memory mapping failed with OOM on device N while trying to map ...
```

Bring-up still completed at `--gpu-memory-utilization 0.97`. Treat as noise unless the engine dies.

---

### 9. NCCL dmabuf registration (preemptive)

**Symptom (recipe note / possible init failure):** `mlx5dv_reg_dmabuf_mr` **errno 524** → `NCCL error: unhandled system error`.

**Fix:** `NCCL_DMABUF_ENABLE=0` (fall back to `nvidia_peermem`).

---

### 10. Headless start order

**Symptom:** Rank-0 API up before workers join → hangs / half-clusters.

**Fix:** Deploy starts **workers (rank ≠ 0, `--headless`) first**, then head. `MASTER_ADDR` = `vllm-recipe-0` pod IP.

## Working config (H200 TP16)

| Knob | Value | Why |
|------|-------|-----|
| `TP_SIZE` / `PP_SIZE` / `NNODES` | 16 / 1 / 2 | Recipe multi-node TP |
| `GPU_MEM_UTIL` | 0.97 | Fits with ~2 GiB KV |
| `MAX_NUM_SEQS` | 128 | Mamba blocks |
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
| Patches | MLA dcp + ZMQ bind retry | Image bugs |

## What still isn’t recipe parity

- H200 vs GB300.
- `max-model-len` **8192** (not 32k / 1M-class recipe contexts).
- `max-num-seqs` **128** (not 256).
- Rust FE off; latent-MoE fusion off; `LOAD_FORMAT` auto (some notes use `fastsafetensors`).
- World size 16 → SymmMem unsupported; TP allreduce via PYNCCL.
- Durable hostPath is harness-specific (recipe docker assumes HF cache volume).

## Timeline (condensed)

1. Unblock storage (VAST → overlay → hostPath NVMe).  
2. Fix Gloo iface (`enp*`) + IPv6 disable.  
3. Disable SM100-only latent fusion; patch MLA DCP.  
4. Survive post-load ZMQ races with bind retry.  
5. Cap `max-num-seqs` at 128 for Mamba; sweep C=1…512.
