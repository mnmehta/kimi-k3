# kimi-k3

Evaluation of Kimi K3 using vLLM parallelization recipes from
[recipes.vllm.ai/moonshotai/Kimi-K3](https://recipes.vllm.ai/moonshotai/Kimi-K3).

**Results report (GitHub Pages):** https://mnmehta.github.io/kimi-k3/

Harnesses for running / profiling `moonshotai/Kimi-K3` on the fozzie `kimi-k3`
namespace with the [`vllm/vllm-openai:kimi-k3`](https://recipes.vllm.ai/moonshotai/Kimi-K3)
image. Report sources and data: [`reports/README.md`](reports/README.md).

```bash
export KUBECONFIG=/Users/mimehta/kubeconfigs/kubeconfig.fozzie
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

After downloads finish on both ranks:

```bash
# TP16 — hostPath /mnt/local/kimi-k3/models (default)
./scripts/deploy-recipe.sh

# TEP16 — TP16 + --enable-expert-parallel
# https://recipes.vllm.ai/moonshotai/Kimi-K3?strategy=multi_node_tep
./scripts/deploy-recipe-tep.sh

# TP8 × PP2
./scripts/deploy-recipe-pp.sh

# TP8 × DP2 (one replica per node)
# https://recipes.vllm.ai/moonshotai/Kimi-K3?strategy=multi_node_tp_dp
./scripts/deploy-recipe-dp.sh

# Prefill/Decode disaggregation (2×8 H200 adaptation of recipe 4-node P/D)
# https://recipes.vllm.ai/moonshotai/Kimi-K3?strategy=pd_cluster
# Recipe 4×8 layout: TEP16 prefill (pods 0–1) + TEP16 decode (pods 2–3) + router :8000
./scripts/deploy-recipe-pd.sh
./scripts/wait-pd-and-sweep.sh

kubectl -n $NS exec vllm-recipe-0 -- curl -sS http://127.0.0.1:8000/v1/models
```

In-pod entrypoint: `scripts/run-vllm-kimi-k3-recipe.sh`.  
Manifests: `vllm-recipe.yaml` (hostPath) / `vllm-recipe-overlay.yaml` (ephemeral legacy).

## Recipe-shaped multi-node (dummy weights)

Same STS layout with `--load-format dummy`:

```bash
./scripts/deploy-recipe-dummy.sh       # TP16
./scripts/deploy-recipe-dummy-tep.sh   # TEP16
./scripts/deploy-recipe-dummy-pp.sh    # TP8 × PP2
./scripts/deploy-recipe-dummy-dp.sh    # TP8 × DP2
./scripts/deploy-recipe-dummy-pd.sh    # P/D 4×8
kubectl -n $NS exec vllm-recipe-0 -- curl -sS http://127.0.0.1:8000/v1/models
```

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

Published report: https://mnmehta.github.io/kimi-k3/ (`index.html`).  
Sources: [`reports/README.md`](reports/README.md).

```bash
python3 -m venv reports/.venv
reports/.venv/bin/pip install -r reports/requirements.txt
reports/.venv/bin/python -m ipykernel install --user \
  --name=kimi-k3-reports --display-name="Python (kimi-k3 reports)"

export PATH="$PWD/reports/bin:$PATH"
export QUARTO_PYTHON="$PWD/reports/.venv/bin/python"
quarto render reports/tp16-vs-pp2-1000-1000.qmd --to html
quarto render reports/concurrency-sweep-1000-1000.qmd --to html
```
