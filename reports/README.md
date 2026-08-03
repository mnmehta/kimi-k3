# Reports

Quarto sources for publishing harness results. Setup and local render (including Quarto CLI install): [repo README — Report setup](../README.md#report-setup--local-render).

## vLLM Kimi-K3 recipe strategy evaluation (primary)

Goal: evaluate strategies from [recipes.vllm.ai/moonshotai/Kimi-K3](https://recipes.vllm.ai/moonshotai/Kimi-K3) on H200. Measured: `multi_node_tp`, `multi_node_tep`, `multi_node_tp_pp`, `multi_node_tp_dp`. Pending: P/D. `multi_node_dep` is unsupported/greyed out on H200 in the recipe UI.

- Source: [`tp16-vs-pp2-1000-1000.qmd`](https://github.com/mnmehta/kimi-k3/blob/main/reports/tp16-vs-pp2-1000-1000.qmd)
- Data: [`conc-sweep-real-1000-1000/`](https://github.com/mnmehta/kimi-k3/tree/main/bench-results/conc-sweep-real-1000-1000) (TP16, C=1…512), [`conc-sweep-tep-1000-1000/`](https://github.com/mnmehta/kimi-k3/tree/main/bench-results/conc-sweep-tep-1000-1000) (TEP16, C=1…512), [`conc-sweep-pp2-1000-1000/`](https://github.com/mnmehta/kimi-k3/tree/main/bench-results/conc-sweep-pp2-1000-1000) (PP2, C=1…512), [`conc-sweep-dp2-1000-1000/`](https://github.com/mnmehta/kimi-k3/tree/main/bench-results/conc-sweep-dp2-1000-1000) (DP2, C=1…16; H200 KV-limited)
- Prior C≤256 archives: [`…-c1-256/`](https://github.com/mnmehta/kimi-k3/tree/main/bench-results/conc-sweep-real-1000-1000-c1-256), [`…-pp2-…-c1-256/`](https://github.com/mnmehta/kimi-k3/tree/main/bench-results/conc-sweep-pp2-1000-1000-c1-256)
- Pages: [mnmehta.github.io/kimi-k3](https://mnmehta.github.io/kimi-k3/) (this report is `index.html`)

## Full harness history (incl. dummy)

- Source: [`concurrency-sweep-1000-1000.qmd`](https://github.com/mnmehta/kimi-k3/blob/main/reports/concurrency-sweep-1000-1000.qmd)
- Also published at [/concurrency-sweep-1000-1000.html](https://mnmehta.github.io/kimi-k3/concurrency-sweep-1000-1000.html)
