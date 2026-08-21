# vLLM 0.27.1 vs older IX / `nnodes` multi-node DP launch

This note tracks the BenchFlow RHAIIS distributed `raw-vllm` switch from the
kimi-k3 / InferenceX aggregated DP launch (`--nnodes` / `--node-rank` /
`--master-addr`) to stock **vLLM 0.27.1** external DP CLI.

**Canonical write-up:** BenchFlow
`docs/vllm-0.27.1-vs-older-ix-agg-dp.md` (branch `cks/kimi-k3`).

## Short version

| | Older IX / kimi-k3 recipe | Stock vLLM 0.27.1 |
|---|---|---|
| Membership | `--nnodes` + `--node-rank` + `--master-addr`/`--master-port` | `--data-parallel-size` + `--data-parallel-size-local` + `--data-parallel-start-rank` + `--data-parallel-address` + `--data-parallel-rpc-port` |
| LB mode | Often internal / recipe-managed | **`--data-parallel-external-lb` required** for multi-node |
| Failure if mixed | Recipe may accept hybrid argv | Workers: `AssertionError: … internal DPLB, which is incompatible` |

Related local history: [tp8-dp2-h200-deployment-issues.md](tp8-dp2-h200-deployment-issues.md) (same DP-vs-`nnodes` lesson on the kimi-k3 recipe path).
