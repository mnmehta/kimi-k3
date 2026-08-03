# Model storage

Target layout inside each recipe container:

| Path | Purpose |
|------|---------|
| `/models/Kimi-K3` | `moonshotai/Kimi-K3` weights (~1.56 TB) |
| `/models/Kimi-K3/.download-complete` | written after shard/index verification |
| `/models/hf` | `HF_HOME` / hub cache |

## Default: real hostPath (CoreWeave local NVMe)

Node path: **`/mnt/local/kimi-k3/models`** → container `/models`.

| Survives pod delete (same node)? | Survives node reboot? | Shared across nodes? |
|----------------------------------|-----------------------|----------------------|
| Yes | No (crypto-shreded) | No — one copy per node |

```bash
./scripts/download-model.sh                 # fill /models on both recipe ranks
./scripts/monitor-model-download.sh
STORAGE_BACKEND=hostpath ./scripts/deploy.sh tp16
# or set storage_backend: hostpath in configs (default in recipes)
```

Manifest: `manifests/vllm-recipe.yaml`.

Stay under `/mnt/local/`. Do **not** use `/var/lib/...` (can be tiny/ram0).

## Legacy: container overlay

`STORAGE_BACKEND=overlay` → `manifests/vllm-recipe-overlay.yaml`.  
Weights live in the container writable layer and are **deleted with the pod**.
