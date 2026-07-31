# kimi-k3

Harnesses for running / profiling `moonshotai/Kimi-K3` on the fozzie `kimi-k3` namespace with the [`vllm/vllm-openai:kimi-k3`](https://recipes.vllm.ai/moonshotai/Kimi-K3) image.

```bash
export KUBECONFIG=/Users/mimehta/kubeconfigs/kubeconfig.fozzie
NS=kimi-k3
```

## Model weights

`moonshotai/Kimi-K3` is ~1.56 TB. Prefer a shared PVC; on fozzie, VAST provisioning
is currently broken so downloads use container-local `/models` (see `manifests/STORAGE.md`).

```bash
./scripts/download-model.sh                 # auto: PVC then fallback
./scripts/monitor-model-download.sh         # watch both ranks
# PVC artifacts:  manifests/model-pvc.yaml, manifests/model-download-job.yaml
# In-pod script:  scripts/download-model-inpod.sh
# Force fallback: STORAGE_BACKEND=hostpath ./scripts/download-model.sh
```

Optional HF auth: `kubectl -n kimi-k3 create secret generic hf-token --from-literal=token=HF_...`

## Recipe-shaped multi-node TP (real weights)

After downloads finish on both ranks:

```bash
./scripts/deploy-recipe.sh
kubectl -n $NS exec vllm-recipe-0 -- curl -sS http://127.0.0.1:8000/v1/models
```

In-pod entrypoint: `scripts/run-vllm-kimi-k3-recipe.sh`.  
Manifest: `manifests/vllm-recipe.yaml` (mounts `/models`).

## Recipe-shaped multi-node TP (dummy weights)

Same STS layout with `--load-format dummy` (no need to wait on weights):

```bash
./scripts/deploy-recipe-dummy.sh
kubectl -n $NS exec vllm-recipe-0 -- curl -sS http://127.0.0.1:8000/v1/models
```

In-pod entrypoint: `scripts/run-vllm-kimi-k3-recipe-dummy.sh`.

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

## Reports

Concurrency sweep Quarto report (real weights vs full/shallow dummy):

- Source: `reports/concurrency-sweep-1000-1000.qmd`
- Data: `bench-results/conc-sweep-real-1000-1000/`, `conc-sweep-full-1000-1000/`, `conc-sweep-1000-1000/`
- GitHub Pages: https://mnmehta.github.io/kimi-k3/

Local render:

```bash
export PATH="$PWD/reports/bin:$PATH"
export QUARTO_PYTHON="$PWD/reports/.venv/bin/python"
quarto render reports/concurrency-sweep-1000-1000.qmd --to html
```
