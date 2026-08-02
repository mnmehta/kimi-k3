# Deployment lessons

Post-mortems for each measured [vLLM Kimi-K3 recipe](https://recipes.vllm.ai/moonshotai/Kimi-K3) strategy on fozzie (2×8 H200):

| Strategy | Doc |
|----------|-----|
| Multi-Node Tensor Parallel (`multi_node_tp`) | [tp16-h200-deployment-issues.md](tp16-h200-deployment-issues.md) |
| Multi-Node Tensor + Expert Parallel (`multi_node_tep`) | [tep16-h200-deployment-issues.md](tep16-h200-deployment-issues.md) |
| Multi-Node TP + Pipeline Parallel (`multi_node_tp_pp`) | [tp8-pp2-h200-deployment-issues.md](tp8-pp2-h200-deployment-issues.md) |
| Multi-Node TP + Data Parallel (`multi_node_tp_dp`) | [tp8-dp2-h200-deployment-issues.md](tp8-dp2-h200-deployment-issues.md) |
| Prefill/Decode Disaggregation (`pd_cluster`) — **blocked on H200** | [pd-h200-deployment-issues.md](pd-h200-deployment-issues.md) |

Comparative results: [`reports/tp16-vs-pp2-1000-1000.qmd`](../reports/tp16-vs-pp2-1000-1000.qmd).
