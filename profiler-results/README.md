# Profiler sweep results

Minimal-depth torch.profiler capture across five H200 recipe strategies
(see [`profiler_sweep.md`](../profiler_sweep.md)).

## How to run

```bash
export KUBECONFIG=...
./scripts/run-profiler-sweep.sh
# subset: CONFIGS="tp16 tep16" ./scripts/run-profiler-sweep.sh
```

## Knobs

| Knob | Value | Notes |
|------|------:|-------|
| `NUM_LAYERS` | 4 | Dense + MoE + KDA + MLA (1-indexed layer 3) |
| `NUM_EXPERTS` | **32** | Plan listed 8 (single-pod TP=2). Multi-node **EP16** needs `num_experts % 16 == 0`; 8 failed marlin MXFP4 repack. |
| `NUM_EXPERTS_PER_TOKEN` | 2 | |
| `NUM_SHARED_EXPERTS` | 1 | |
| `MAX_MODEL_LEN` | 1024 | |
| load format | `dummy` | |
| protocol | warmup + 10 in / 5 out | |

## Entrypoints

| Config | Deploy | Capture |
|--------|--------|---------|
| `pd` | `./scripts/deploy.sh pd-dummy-profiler` | `profile-pd-short-query.sh` (dual engine start/stop + router) |
| `tp16` | `./scripts/deploy.sh tp16-dummy-profiler` | `profile-short-query.sh` |
| `tep16` | `./scripts/deploy.sh tep16-dummy-profiler` | `profile-short-query.sh` |
| `pp2` | `./scripts/deploy.sh pp2-dummy-profiler` | `profile-short-query.sh` |
| `dp2` | `./scripts/deploy.sh dp2-dummy-profiler` | `profile-short-query.sh` |

Collect: `CONFIG=… PROFILE_DIR=… NNODES=… ./scripts/copy-profiler-traces.sh`  
(also archives `vllm-recipe-<rank>.log` per pod alongside `rank-<i>/` traces)

## Status

- **2026-08-02 (restart):** Clean sweep with traces **and** `vllm-recipe-*.log`.
  - Done (17 gz + serve logs each): `tp16`, `tep16`, `pp2`, `dp2`.
  - Fixes: PP needs `NCCL_IB_HCA=mlx5_5` / GID 3; dummy DP needs `VLLM_USE_RUST_FRONTEND=0` (nested `hf-overrides`).
  - **P/D blocked**: cluster has only 3 nodes (`g11cd44` missing). Wait timed out after 1h. Re-run with `CONFIGS=pd ./scripts/run-profiler-sweep.sh` when 4 Ready.
- Latent-MoE tail fusion **off** on H200 (SM100-only). `NUM_EXPERTS=32` for EP16.
