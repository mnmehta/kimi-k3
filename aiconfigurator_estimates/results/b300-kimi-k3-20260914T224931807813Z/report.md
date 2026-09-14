# Kimi-K3 B300 comparison

This report compares AIC Kimi K3 B300 estimates to data in the blog at [Kimi K3 Performance Optimizations in vLLM, 2.2–2.8× Higher Throughput](https://16ca0411.vllm-blog-source.pages.dev/2026/09/13/kimi-k3-performance-optimization).

## Candidate summary

MAPE = Mean Absolute Percentage Error.

| Candidate | Points ok | Throughput MAPE | TTFT MAPE | TPOT MAPE |
|---|---:|---:|---:|---:|
| vLLM TP8 MoE TP8 | 3/3 | 22.9% | 62.1% | — |

## Point comparisons

| Point | Candidate | Target C | AIC C | Measured tok/s | AIC tok/s | Tok/s error | Measured TTFT | AIC TTFT | TTFT error |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Blog vLLM 0.27.1 C=1 | vLLM TP8 MoE TP8 | 1 | 1 | 83.3 | 50.6 | -39.2% | 2262.9 | 963.0 | -57.4% |
| Blog vLLM 0.27.1 C=4 | vLLM TP8 MoE TP8 | 4 | 4 | 166.7 | 146.1 | -12.3% | 2314.9 | 1204.2 | -48.0% |
| Blog vLLM 0.27.1 C=16 | vLLM TP8 MoE TP8 | 16 | 16 | 258.3 | 302.7 | +17.2% | 7601.1 | 1446.6 | -81.0% |

## Notes

- The AIC side uses the most current checked-in AIC B300 vLLM data, `0.24.0`; the measured rows are the blog's `0.27.1` numbers.
- This manifest assumes `tp_size=8`, `pp_size=1`, `attention_dp_size=1`, `moe_tp_size=8`, `moe_ep_size=1` for every comparison row.
- Only the measured fields present in the manifest are rendered in the point table.

## Artifacts

- Script: [aiconfigurator_estimates/tools/kimi_k3_b300_compare.py](https://github.com/mnmehta/kimi-k3/blob/main/aiconfigurator_estimates/tools/kimi_k3_b300_compare.py)
- Commands source: [commands.txt](https://github.com/mnmehta/kimi-k3/blob/main/aiconfigurator_estimates/results/b300-kimi-k3-20260914T224931807813Z/commands.txt)
- Results root: [aiconfigurator_estimates/results](https://github.com/mnmehta/kimi-k3/tree/main/aiconfigurator_estimates/results)
