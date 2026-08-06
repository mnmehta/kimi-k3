# Deploy configs

Hierarchical YAML configs for Kimi-K3 recipe deploys. Later files / keys override earlier ones.

## Layout

| Path | Role |
|------|------|
| `layers/` | Reusable fragments (base, weights, strategy, profiler) |
| `recipes/` | Named compositions ready to deploy |

## Usage

```bash
export KUBECONFIG=...
./scripts/deploy.sh tp16                 # → configs/recipes/tp16.yaml
./scripts/deploy.sh tep16
./scripts/deploy.sh pp2
./scripts/deploy.sh dp2
./scripts/deploy.sh pd
./scripts/deploy.sh pp2-humming-agentx   # PP2+Humming, 256k ctx (AgentX)
./scripts/deploy.sh pp2-humming-agentx-kv-offload  # + OffloadingConnector CPU KV 200 GiB
# Dummy / profiler
./scripts/deploy.sh tp16-dummy
./scripts/deploy.sh tp16-dummy-profiler

# Explicit composition (order = merge order)
./scripts/deploy.sh configs/layers/base.yaml \
  configs/layers/weights-real.yaml \
  configs/layers/strategy-pp2.yaml

# Dump merged config without deploying
./scripts/deploy.sh --print tp16
```

Legacy `./scripts/deploy-recipe*.sh` wrappers call `deploy.sh` with the matching recipe.

## Schema

```yaml
include:                    # optional list of relative YAML paths
  - layers/base.yaml

name: tp16                  # short id
description: "…"
deploy_mode: multi_node     # multi_node | pd
serve_script: scripts/run-vllm-kimi-k3-recipe.sh
verify_weights: true        # skip for dummy
storage_backend: hostpath   # hostpath | overlay
health_port: 8000
health_timeout_polls: 360   # ×10s

env:                        # exported into the serve process
  TP_SIZE: 16
  # use "" for empty string, null / omit to leave unset
```

For P/D, `env` holds shared knobs; `pd:` holds role-specific settings (see `layers/strategy-pd.yaml`).

Deployment issues / harness patches: [`docs/README.md`](../docs/README.md), especially
[`docs/pp2-humming-agentx-kv-offload-deployment-issues.md`](../docs/pp2-humming-agentx-kv-offload-deployment-issues.md)
and [`docs/harness-patches.md`](../docs/harness-patches.md).
