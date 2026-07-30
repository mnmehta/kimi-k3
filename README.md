# kimi-k3

Harnesses for running / profiling `moonshotai/Kimi-K3` on the fozzie `kimi-k3` namespace with the [`vllm/vllm-openai:kimi-k3`](https://recipes.vllm.ai/moonshotai/Kimi-K3) image.

```bash
export KUBECONFIG=/Users/mimehta/kubeconfigs/kubeconfig.fozzie
NS=kimi-k3
```

## Recipe-shaped multi-node TP (dummy weights)

Mirrors the [vLLM Kimi-K3 recipe](https://recipes.vllm.ai/moonshotai/Kimi-K3) multi-node TP layout (2×8 GPUs, TP=16, FLASHMLA, marlin) but uses `--load-format dummy` and a shallow MoE so it fits for bring-up.

```bash
./scripts/deploy-recipe-dummy.sh
# API on rank-0 hostNetwork IP:8000
kubectl -n $NS exec vllm-recipe-0 -- curl -sS http://127.0.0.1:8000/v1/models
```

In-pod entrypoint: `scripts/run-vllm-kimi-k3-recipe-dummy.sh`. Manifest: `manifests/vllm-recipe.yaml`.

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

Concurrency sweep Quarto report (full HF dummy vs shallow 12L):

- Source: `reports/concurrency-sweep-1000-1000.qmd`
- Data: `bench-results/conc-sweep-full-1000-1000/`, `bench-results/conc-sweep-1000-1000/`
- GitHub Pages: https://mnmehta.github.io/kimi-k3/

Local render:

```bash
export PATH="$PWD/reports/bin:$PATH"
export QUARTO_PYTHON="$PWD/reports/.venv/bin/python"
quarto render reports/concurrency-sweep-1000-1000.qmd --to html
```
