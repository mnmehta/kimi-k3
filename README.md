# kimi-k3

Evaluation of Kimi K3 using vLLM parallelization recipes from
[recipes.vllm.ai/moonshotai/Kimi-K3](https://recipes.vllm.ai/moonshotai/Kimi-K3).

**Published reports (GitHub Pages):**

| Report | URL |
|--------|-----|
| Hub | https://mnmehta.github.io/kimi-k3/kimireports.html |
| Recipe strategy evaluation (primary / `index.html`) | https://mnmehta.github.io/kimi-k3/ |
| Full harness history (incl. dummy) | https://mnmehta.github.io/kimi-k3/concurrency-sweep-1000-1000.html |
| Sweep reproducibility | https://mnmehta.github.io/kimi-k3/repro-1000-1000.html |
| PP2 Marlin vs Humming | https://mnmehta.github.io/kimi-k3/pp2-marlin-vs-humming-1000-1000.html |
| AgentX MVP (C=1–4) | https://mnmehta.github.io/kimi-k3/agentx-mvp-pp2-humming.html |
| AgentX MVP + CPU KV offload (C=1–5) | https://mnmehta.github.io/kimi-k3/agentx-mvp-pp2-humming-kv-offload.html |

Harnesses for running / profiling `moonshotai/Kimi-K3` in the `kimi-k3`
namespace with the [`vllm/vllm-openai:kimi-k3`](https://recipes.vllm.ai/moonshotai/Kimi-K3)
image. Report sources and data: [`reports/README.md`](reports/README.md).

```bash
export KUBECONFIG=...
NS=kimi-k3
```

## Model weights

`moonshotai/Kimi-K3` is ~1.56 TB. Weights live on **real hostPath**
`/mnt/local/kimi-k3/models` per node (CoreWeave local NVMe). See `manifests/STORAGE.md`.

```bash
./scripts/download-model.sh
./scripts/monitor-model-download.sh
```

Optional HF auth: `kubectl -n kimi-k3 create secret generic hf-token --from-literal=token=HF_...`

## Recipe-shaped multi-node (real weights)

After downloads finish on both ranks. Deploys are **config-driven** — see [`configs/README.md`](configs/README.md).

```bash
# Preferred entrypoint
./scripts/deploy.sh tp16      # multi_node_tp
./scripts/deploy.sh tep16     # multi_node_tep
./scripts/deploy.sh pp2       # multi_node_tp_pp
./scripts/deploy.sh dp2       # multi_node_tp_dp
./scripts/deploy.sh pd        # pd_cluster (4×8)
./scripts/wait-pd-and-sweep.sh

# Compose / inspect
./scripts/deploy.sh --print tp16
./scripts/deploy.sh configs/layers/base.yaml \
  configs/layers/weights-real.yaml \
  configs/layers/strategy-pp2.yaml

kubectl -n $NS exec vllm-recipe-0 -- curl -sS http://127.0.0.1:8000/v1/models
```

Legacy `./scripts/deploy-recipe*.sh` wrappers still work (they call `deploy.sh`).  
In-pod entrypoint: `scripts/run-vllm-kimi-k3-recipe.sh`.  
Manifests: `vllm-recipe.yaml` (hostPath) / `vllm-recipe-overlay.yaml` (ephemeral legacy).

## Recipe-shaped multi-node (dummy weights)

Same STS layout with `--load-format dummy`:

```bash
./scripts/deploy.sh tp16-dummy
./scripts/deploy.sh tep16-dummy
./scripts/deploy.sh pp2-dummy
./scripts/deploy.sh dp2-dummy
./scripts/deploy.sh pd-dummy
kubectl -n $NS exec vllm-recipe-0 -- curl -sS http://127.0.0.1:8000/v1/models
```

Profiler presets: `tp16-dummy-profiler`, `tep16-dummy-profiler`, `pp2-dummy-profiler`, `dp2-dummy-profiler`, `pd-dummy-profiler`.  
In-pod entrypoint: `scripts/run-vllm-kimi-k3-recipe-dummy.sh`.

### Minimal-depth torch.profiler sweep (5 strategies)

See [`profiler_sweep.md`](profiler_sweep.md) / [`profiler-results/README.md`](profiler-results/README.md):

```bash
./scripts/run-profiler-sweep.sh
```

## Single-pod torch.profiler smoke (TP=2)

### 1. Start the pod

```bash
kubectl -n $NS apply -f manifests/vllm-pod.yaml
kubectl -n $NS wait --for=condition=Ready pod/vllm --timeout=30m
```

### 2. Launch vLLM (dummy, TP=2, torch profiler on)

```bash
kubectl -n $NS cp scripts/run-vllm-kimi-k3-dummy.sh vllm:/tmp/run.sh
kubectl -n $NS exec vllm -- bash -c 'nohup bash /tmp/run.sh > /tmp/vllm.log 2>&1 &'
kubectl -n $NS exec vllm -- curl -sf http://127.0.0.1:8000/health
```

### 3. Capture a short trace (10 tokens in, 5 out)

```bash
kubectl -n $NS cp scripts/profile-short-query.sh vllm:/tmp/profile-short-query.sh
kubectl -n $NS exec vllm -- bash /tmp/profile-short-query.sh
kubectl -n $NS cp vllm:/tmp/vllm_profile ./vllm_profile
```

Open `vllm_profile/dp0_pp0_tp0_*.pt.trace.json.gz` in [Perfetto](https://ui.perfetto.dev/).

## Report setup / local render

Published pages (also listed at the top of this README):

- Hub: https://mnmehta.github.io/kimi-k3/kimireports.html
- Primary: https://mnmehta.github.io/kimi-k3/
- History: https://mnmehta.github.io/kimi-k3/concurrency-sweep-1000-1000.html
- Repro: https://mnmehta.github.io/kimi-k3/repro-1000-1000.html
- PP2 Marlin vs Humming: https://mnmehta.github.io/kimi-k3/pp2-marlin-vs-humming-1000-1000.html
- AgentX MVP (C=1–4): https://mnmehta.github.io/kimi-k3/agentx-mvp-pp2-humming.html

Sources: [`reports/README.md`](reports/README.md).

Install the [Quarto CLI](https://quarto.org/docs/get-started/) first (not bundled with this repo). On macOS:

```bash
brew install quarto
```

Other platforms: download from [quarto.org/docs/get-started](https://quarto.org/docs/get-started/), or see the [install docs](https://quarto.org/docs/download/). CI installs Quarto via [`quarto-dev/quarto-actions/setup`](https://github.com/quarto-dev/quarto-actions).

Then set up the Python kernel and render:

```bash
python3 -m venv reports/.venv
reports/.venv/bin/pip install -r reports/requirements.txt
reports/.venv/bin/python -m ipykernel install --user \
  --name=kimi-k3-reports --display-name="Python (kimi-k3 reports)"

export QUARTO_PYTHON="$PWD/reports/.venv/bin/python"
quarto render reports/tp16-vs-pp2-1000-1000.qmd --to html
quarto render reports/concurrency-sweep-1000-1000.qmd --to html
quarto render reports/repro-1000-1000.qmd --to html
quarto render reports/pp2-marlin-vs-humming-1000-1000.qmd --to html
quarto render reports/agentx-mvp-pp2-humming.qmd --to html
```
