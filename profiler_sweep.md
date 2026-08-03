# Minimal-depth torch.profiler sweep (5 recipe strategies)

Plan only — execute when pointed back at this file. Do **not** start until the
in-flight P/D concurrency sweep finishes and its artifacts are copied.

## Goal

Capture Chrome / Perfetto traces for **all five selectable H200 recipe strategies**,
using **`--load-format dummy`** and a **shallow architecture** via `--hf-overrides`
(not the full real-weight stack). Capture protocol stays: warmup + **10 in / 5 out**.

| # | Shorthand | Recipe id | Deploy entrypoint | Nodes |
|---|-----------|-----------|-------------------|-------|
| 1 | TP16 | `multi_node_tp` | `./scripts/deploy-recipe-dummy.sh` (+ profiler env) | 2 |
| 2 | TEP16 | `multi_node_tep` | dummy TP16 deploy + `ENABLE_EXPERT_PARALLEL=1` (add thin wrapper if missing) | 2 |
| 3 | TP8×PP2 | `multi_node_tp_pp` | `./scripts/deploy-recipe-dummy-pp.sh` | 2 |
| 4 | TP8×DP2 | `multi_node_tp_dp` | `./scripts/deploy-recipe-dummy-dp.sh` | 2 |
| 5 | P/D | `pd_cluster` | dummy-capable P/D bring-up (extend `deploy-recipe-pd.sh` or add `deploy-recipe-dummy-pd.sh`) | 4 |

Out of scope: `multi_node_dep` (unsupported on H200 in the recipe UI).

## Architecture override (this sweep)

### Earlier single-pod smoke (`scripts/run-vllm-kimi-k3-dummy.sh`)

That run used:

| Knob | Value |
|------|------:|
| `NUM_LAYERS` (`num_hidden_layers`) | **12** |
| `NUM_EXPERTS` | **8** |
| `NUM_EXPERTS_PER_TOKEN` | **2** |
| `NUM_SHARED_EXPERTS` | **1** |
| `MAX_MODEL_LEN` | 1024 |
| load format | `dummy` |
| TP | 2 (single pod) |

Passed as:

```json
{"text_config": {
  "num_hidden_layers": 12,
  "num_experts": 8,
  "num_experts_per_token": 2,
  "num_shared_experts": 1
}}
```

Rationale in that script: layer 0 = dense + KDA; layers ≥1 = MoE; every 4th
**1-indexed** layer (3, 7, 11, …) = full MLA; `attn_res_block_size=12` → 12 layers
covers one full residual-attn block.

Also set: `VLLM_ENABLE_K3_LATENT_MOE_TAIL_FUSION=1`,
`-cc.pass_config.fuse_allreduce_rms=False`, profiler
`active_iterations=32`, stacks on.

### This sweep (changed)

Same expert shrink and dummy load; **layers cut to 4**:

| Knob | Value |
|------|------:|
| `NUM_LAYERS` | **4** |
| `NUM_EXPERTS` | **8** |
| `NUM_EXPERTS_PER_TOKEN` | **2** |
| `NUM_SHARED_EXPERTS` | **1** |
| `MAX_MODEL_LEN` | 1024 (keep unless a recipe path needs more for bring-up) |

```json
{"text_config": {
  "num_hidden_layers": 4,
  "num_experts": 8,
  "num_experts_per_token": 2,
  "num_shared_experts": 1
}}
```

With 4 layers (1-indexed 1…4), layer **3** is still an MLA layer, so traces should
include dense, MoE, KDA, and one MLA — just not a full `attn_res` block of 12.

**Do not** use real MXFP4 weights with this override; keep `--load-format dummy`
(architecture no longer matches the full checkpoint).

## Capture protocol (same as prior dummy smoke)

Mirror `scripts/profile-short-query.sh` / README “Single-pod torch.profiler smoke”:

1. **Warmup** (outside profiler): one `/v1/completions` with the same shape.
2. `POST /start_profile`
3. **Profiled request:** ~**10 prompt tokens** in, **5** completion tokens out
   (`temperature=0`, prompt via `/tokenize` then trimmed token ids).
4. `POST /stop_profile` (allow a long flush; see RPC timeout below).
5. Collect gzipped traces (`*.pt.trace.json.gz`) from **every rank/role** that
   wrote under `torch_profiler_dir`.

```bash
# dummy serve model id is usually the HF id or whatever the dummy recipe sets
MODEL="${MODEL:-moonshotai/Kimi-K3}"   # or /models/Kimi-K3 if that is what serve registered
PROFILE_DIR=/tmp/vllm_profile/<config> \
  bash /tmp/profile-short-query.sh
```

Confirm served id via `/v1/models` before capture.

## Differences from the old dummy profiler sweep

| | Old (done) | This sweep |
|--|------------|------------|
| Weights | `--load-format dummy` | same |
| Depth | `NUM_LAYERS=12`, experts 8/2/1 | **`NUM_LAYERS=4`**, experts 8/2/1 |
| Topology | Single pod TP=2 | Multi-node recipe layouts (2 or 4 pods) |
| Serve script | `run-vllm-kimi-k3-dummy.sh` (profiler already wired) | `run-vllm-kimi-k3-recipe-dummy.sh` (+ profiler wiring) |
| Trace volume | Small | Still modest vs full real weights |

## Prerequisites (before any profile bring-up)

1. **P/D concurrency sweep done** and copied (`bench-results/conc-sweep-pd-1000-1000/`
   etc.). Confirm `summary.tsv` / `conc-512.json` and `sweep complete`.
2. Cluster quiet: `stop-inpod-vllm.sh` on all ranks before redeploying for profiles.
3. Dummy path does **not** need full weight download for correctness of shapes, but
   pods / image / HF config access must still work as today’s dummy recipe deploys do.
4. Host free space for 5 configs × N ranks of traces (much smaller than full-weight).

## Code changes required (implement when executing)

### A. `scripts/run-vllm-kimi-k3-recipe-dummy.sh`

Already supports optional `NUM_LAYERS` / `NUM_EXPERTS` / … via `--hf-overrides`.
For this sweep always pass:

```bash
NUM_LAYERS=4 NUM_EXPERTS=8 NUM_EXPERTS_PER_TOKEN=2 NUM_SHARED_EXPERTS=1
```

Add opt-in profiler (same as planned for real recipe script):

- `ENABLE_TORCH_PROFILER=0|1` (default `0`).
- When `1`: `PROFILE_DIR`, `VLLM_RPC_TIMEOUT=1800000`, `--profiler-config` with
  `torch` / stacks / `active_iterations=32` (no `warmup_iterations` unless we
  want a schedule).

Also keep / set for parity with single-pod smoke where harmless on multi-node:

- `VLLM_ENABLE_K3_LATENT_MOE_TAIL_FUSION=1` (if not already)
- `-cc.pass_config.fuse_allreduce_rms=False` if the dummy recipe script does not
  already pass it (check at execute time; add if missing).

### B. Dummy deploy scripts

- Forward `ENABLE_TORCH_PROFILER`, `PROFILE_DIR`, `VLLM_RPC_TIMEOUT`,
  `NUM_LAYERS`, `NUM_EXPERTS`, `NUM_EXPERTS_PER_TOKEN`, `NUM_SHARED_EXPERTS`,
  `MAX_MODEL_LEN` through `OPT_ENV` / env into in-pod `run-recipe` dummy script.
- Example:

  ```bash
  ENABLE_TORCH_PROFILER=1 \
  PROFILE_DIR=/tmp/vllm_profile/tp16 \
  NUM_LAYERS=4 NUM_EXPERTS=8 NUM_EXPERTS_PER_TOKEN=2 NUM_SHARED_EXPERTS=1 \
  MAX_MODEL_LEN=1024 \
  ./scripts/deploy-recipe-dummy.sh
  ```

- Add **`deploy-recipe-dummy-tep.sh`** if missing (`ENABLE_EXPERT_PARALLEL=1` on
  top of dummy TP16).
- P/D: either teach `deploy-recipe-pd.sh` a `LOAD_FORMAT=dummy` + hf-overrides
  path, or add `deploy-recipe-dummy-pd.sh`. Profiler must be on all four roles.

### C. `scripts/profile-short-query.sh`

- Keep warmup + 10/5 protocol.
- Resolve `MODEL` from `/v1/models` or require explicit env after dummy bring-up.

### D. `scripts/copy-profiler-traces.sh`

```text
CONFIG=tp16 PROFILE_DIR=/tmp/vllm_profile/tp16 NNODES=2 \
  ./scripts/copy-profiler-traces.sh
→ host: profiler-results/<config>/rank-<i>/...
```

Include `meta.txt`: UTC, git SHA, deploy cmd, hf-overrides, `non-default args`,
prompt 10/5, warmup=1.

## Recommended execution order

1. After P/D **real-weight** bench is archived, tear down / stop serves.
2. Bring up **dummy P/D** with 4-layer overrides + profiler → capture → collect.
3. Scale to 2 nodes; profile TP16 → TEP16 → PP2 → DP2 (each dummy + 4-layer + profiler).

## Per-config procedure (2-node)

For each `CONFIG` in `tp16`, `tep16`, `pp2`, `dp2`:

### 1. Deploy

```bash
export KUBECONFIG=...
NS=kimi-k3
export ENABLE_TORCH_PROFILER=1
export PROFILE_DIR=/tmp/vllm_profile/$CONFIG
export NUM_LAYERS=4 NUM_EXPERTS=8 NUM_EXPERTS_PER_TOKEN=2 NUM_SHARED_EXPERTS=1
export MAX_MODEL_LEN=1024
# matching dummy deploy-* for that strategy
```

Wait for `/health` + `/v1/models` on the API head.

### 2. Capture

```bash
kubectl -n $NS cp scripts/profile-short-query.sh vllm-recipe-0:/tmp/profile-short-query.sh
kubectl -n $NS exec vllm-recipe-0 -- bash -c \
  "MODEL=<served-id> PROFILE_DIR=/tmp/vllm_profile/$CONFIG \
   bash /tmp/profile-short-query.sh"
```

Collect from **all** ranks; unique `PROFILE_DIR` per config.

### 3. Stop before next config

`stop-inpod-vllm.sh` on each rank.

## Per-config procedure (P/D — 4 nodes)

Same dual-engine start/stop caveats as before (router may not proxy
`/start_profile` / `/stop_profile`):

- Start profile on prefill `:8001` and decode `:8002`.
- Warmup + 10/5 via router `:8000`.
- Stop both heads; collect ranks 0–3.

All four serves: dummy + `NUM_LAYERS=4` + expert overrides + profiler.

## Host artifact layout

```text
profiler-results/
  tp16/   meta.txt  rank-0/  rank-1/
  tep16/
  pp2/
  dp2/
  pd/     meta.txt  rank-0/ … rank-3/
```

Do **not** commit large traces unless explicitly requested.

## Knobs / risks checklist

| Risk | Mitigation |
|------|------------|
| Real weights + 4-layer override | Don’t; use `load-format dummy` |
| Missing MLA in 4-layer stack | 1-indexed layer 3 is MLA — keep `NUM_LAYERS≥3` (we use 4) |
| `/stop_profile` hangs | `VLLM_RPC_TIMEOUT=1800000` |
| Wrong model id | Read `/v1/models` after serve |
| Mixing configs | Unique `PROFILE_DIR` |
| Accidental profiler on next real bench | Default `ENABLE_TORCH_PROFILER=0` |
| P/D router vs engines | Dual start/stop until proxy verified |
| Dummy TEP / P/D scripts missing | Add thin wrappers at execute time |

## Success criteria

For each of the five configs:

- [ ] Dummy serve up with hf-overrides `4 / 8 / 2 / 1` + profiler config
- [ ] Warmup + one 10-in / 5-out profiled completion succeeded
- [ ] `/stop_profile` completed
- [ ] Host has gzipped traces for every rank (2 or 4)
- [ ] Rank-0 TP0 (or prefill+decode TP0) opens in Perfetto

## When ready to execute

Point the agent at this file and say to run the profiler sweep. Suggested first
steps:

1. Confirm P/D real bench is archived; cluster state.
2. Wire profiler + forward `NUM_LAYERS=4` (and expert envs) on dummy recipe path;
   add missing dummy-TEP / dummy-P/D entrypoints.
3. Profile P/D (dummy) → TP16 → TEP16 → PP2 → DP2.
4. Note outcomes under `profiler-results/README.md`.
