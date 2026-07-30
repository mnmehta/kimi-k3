# kimi-k3

Quick PyTorch traces for `moonshotai/Kimi-K3` via vLLM (dummy weights, shallow MoE, TP=2 so NCCL shows up).

```bash
export KUBECONFIG=/Users/mimehta/kubeconfigs/kubeconfig.fozzie
NS=kimi-k3
```

### 1. Start the pod

```bash
kubectl -n $NS apply -f manifests/vllm-pod.yaml
kubectl -n $NS wait --for=condition=Ready pod/vllm --timeout=30m
```

### 2. Launch vLLM (dummy, TP=1, torch profiler on)

```bash
kubectl -n $NS cp scripts/run-vllm-kimi-k3-dummy.sh vllm:/tmp/run.sh
kubectl -n $NS exec vllm -- bash -c 'nohup bash /tmp/run.sh > /tmp/vllm.log 2>&1 &'
# wait until ready
kubectl -n $NS exec vllm -- curl -sf http://127.0.0.1:8000/health
```

Uses TP=2, 12 layers / 8 experts so dense MLP, MoE, KDA, MLA, attn_res, and TP collectives all run.

### 3. Capture a short trace (10 tokens in, 5 out)

```bash
kubectl -n $NS cp scripts/profile-short-query.sh vllm:/tmp/profile-short-query.sh
kubectl -n $NS exec vllm -- bash /tmp/profile-short-query.sh
kubectl -n $NS cp vllm:/tmp/vllm_profile ./vllm_profile
```

Open `vllm_profile/dp0_pp0_tp0_*.pt.trace.json.gz` in [Perfetto](https://ui.perfetto.dev/) (or `chrome://tracing`).
