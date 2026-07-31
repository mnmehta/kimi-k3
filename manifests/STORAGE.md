# Model storage

Target layout inside each recipe container:

| Path | Purpose |
|------|---------|
| `/models/Kimi-K3` | `hf download --local-dir` of `moonshotai/Kimi-K3` (~1.56 TB) |
| `/models/Kimi-K3/.download-complete` | written after shard/index verification |
| `/models/hf` | `HF_HOME` / hub cache |

## Preferred: PVC (`shared-vast`, RWX)

Artifacts:

- `manifests/model-pvc.yaml` — `kimi-k3-model`, 4Ti, RWX
- `manifests/model-download-job.yaml` — one-shot Job
- `scripts/download-model-inpod.sh` — download + verify
- `STORAGE_BACKEND=pvc ./scripts/download-model.sh`

### fozzie status (2026-07-30)

Dynamic provisioning failed:

```text
failed to provision volume with StorageClass "shared-vast":
Authentication failure, user: k8s-6787d4-...
```

Namespace `cw-vast-csi` only runs the node DaemonSet; no CSI controller
Deployment is present. Retry PVC later with `STORAGE_BACKEND=pvc`.

## Current: container-local `/models` (node overlay ~28T)

`vllm-recipe` pods write weights into the container filesystem (overlay on the
node's ~28T disk). No volume mount.

| Pros | Cons |
|------|------|
| Enough space for full weights | Lost if the pod is recreated |
| Works without VAST | Duplicated per rank (~1.56 TB × N) |

```bash
STORAGE_BACKEND=hostpath ./scripts/download-model.sh   # name is historical; uses recipe pods
./scripts/monitor-model-download.sh
./scripts/deploy-recipe.sh
```

### Why not hostPath `/var/lib/...`?

On fozzie, `hostPath: /var/lib/kimi-k3/models` resolved to `/dev/ram0` (15G tmpfs),
which is far too small. Do not re-enable that path without verifying:

```bash
kubectl -n kimi-k3 exec vllm-recipe-0 -- findmnt -T /models
```

## Switching back to PVC later

1. Confirm `kubectl -n kimi-k3 get pvc kimi-k3-model` → Bound  
2. Mount the claim at `/models` on the StatefulSet (see comments in `vllm-recipe.yaml`)  
3. `STORAGE_BACKEND=pvc REPLACE_JOB=1 ./scripts/download-model.sh`  
4. `./scripts/deploy-recipe.sh`
