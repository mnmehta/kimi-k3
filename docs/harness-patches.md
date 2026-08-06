# In-pod harness patches

Inventory of runtime patches applied on `vllm-recipe-*` before / during
[`scripts/run-vllm-kimi-k3-recipe.sh`](../scripts/run-vllm-kimi-k3-recipe.sh) and
[`scripts/lib/deploy_multi_node.sh`](../scripts/lib/deploy_multi_node.sh).

Patches are **idempotent** where possible (needle / marker grep). They exist because the
cluster image lags upstream merges or needs Fozzie/H200-specific workarounds. Prefer
dropping a patch once the image includes the upstream fix.

| Patch | Where | Upstream / why | Marker / skip condition |
|-------|--------|----------------|-------------------------|
| MLA `dcp_world_size` lazy-init | `run-vllm-kimi-k3-recipe.sh` → `…/kimi_k3/nvidia/mla.py` | Kimi MLA skips `MLAAttention.forward` lazy-init → `cp_world_size must be positive` | Comment contains `Kimi MLA bypasses MLAAttention.forward lazy-init` |
| ZMQ bind retry on `EADDRINUSE` | same → `…/shm_broadcast.py` | `get_open_port()` TOCTOU under `hostNetwork` + many TP workers | `Kimi-K3 harness: retry ZMQ bind` |
| Allow `expandable_segments` with KV connector | same → `…/config/vllm.py` | Stock rejects expandable + Nixl/Offload unless CuMem; CuMem OOMs TP8 weight load on H200 | Only if `KV_TRANSFER_CONFIG` set; marker `Kimi-K3 harness: allow expandable_segments with KV connector` |
| Skip CuMem weight pool | same → `…/v1/worker/gpu_worker.py` | Belt-and-suspenders with expandable path when connector set | Only if `KV_TRANSFER_CONFIG` set; marker `Kimi-K3 harness: skip CuMem pool for weights` |
| Mamba hybrid scalar `idx_mapping` fill | same → `…/model_states/mamba_hybrid.py` | [vllm#50327](https://github.com/vllm-project/vllm/pull/50327) (merged); issue [vllm#50947](https://github.com/vllm-project/vllm/issues/50947). `index_fill_` needs int64; mapping is int32 + PP `-1` sentinels | Skip if `_fill_num_accepted_kernel` already present |
| Humming `MoEActivation.SITU` allowlist | `deploy_multi_node.sh` → [`patch_humming_situ.sh`](../scripts/lib/patch_humming_situ.sh) on `fused_humming_moe.py` | [vllm#50510](https://github.com/vllm-project/vllm/pull/50510); only when `MOE_BACKEND=humming` | Unpatched on stop via `stop-inpod-vllm.sh` |

## Deploy wiring notes (not file patches)

| Issue | Fix location |
|-------|----------------|
| JSON in `KV_TRANSFER_CONFIG` loses quotes inside `bash -c` | `deploy_multi_node.sh`: `kubectl exec -- env KEY=value …` |
| Prefix caching for hybrid OffloadingConnector | Layer [`kv-cpu-offload-200.yaml`](../configs/layers/kv-cpu-offload-200.yaml) + serve `--enable-prefix-caching` |
| Hard-stop leftover serve / workers | [`scripts/stop-inpod-vllm.sh`](../scripts/stop-inpod-vllm.sh) (pid/args matching; no self-`pkill -f`) |

## Related post-mortems

- Shared bring-up (MLA, ZMQ, storage, iface): [`tp16-h200-deployment-issues.md`](tp16-h200-deployment-issues.md)
- PP RoCE GID: [`tp8-pp2-h200-deployment-issues.md`](tp8-pp2-h200-deployment-issues.md)
- NIXL / CuMem / expandable: [`pd-h200-deployment-issues.md`](pd-h200-deployment-issues.md)
- AgentX + OffloadingConnector + #50327: [`pp2-humming-agentx-kv-offload-deployment-issues.md`](pp2-humming-agentx-kv-offload-deployment-issues.md)
