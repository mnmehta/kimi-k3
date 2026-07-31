#!/usr/bin/env bash
# Runs inside the model-download Job. Downloads moonshotai/Kimi-K3 onto the
# PVC mounted at MODEL_ROOT (default /models).
set -euo pipefail

MODEL_ID="${MODEL_ID:-moonshotai/Kimi-K3}"
MODEL_ROOT="${MODEL_ROOT:-/models}"
LOCAL_DIR="${LOCAL_DIR:-${MODEL_ROOT}/Kimi-K3}"
MARKER="${MARKER:-${LOCAL_DIR}/.download-complete}"
EXPECTED_SHARDS="${EXPECTED_SHARDS:-96}"

export HF_HOME="${HF_HOME:-${MODEL_ROOT}/hf}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
# Newer huggingface_hub prefers Xet high-performance over hf_transfer.
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export PYTHONUNBUFFERED=1

mkdir -p "$LOCAL_DIR" "$HUGGINGFACE_HUB_CACHE"

if [[ -f "$MARKER" ]]; then
  echo "Marker present ($MARKER); verifying existing download..."
else
  echo "==> Ensuring hf_transfer is available"
  python3 - <<'PY'
import importlib.util
import subprocess
import sys

if importlib.util.find_spec("hf_transfer") is None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "hf_transfer"])
print("hf_transfer ok")
PY

  echo "==> Downloading $MODEL_ID -> $LOCAL_DIR"
  echo "    HF_HOME=$HF_HOME HF_HUB_ENABLE_HF_TRANSFER=$HF_HUB_ENABLE_HF_TRANSFER"
  # Prefer `hf` (newer) then huggingface-cli.
  if command -v hf >/dev/null 2>&1; then
    hf download "$MODEL_ID" --local-dir "$LOCAL_DIR"
  else
    huggingface-cli download "$MODEL_ID" --local-dir "$LOCAL_DIR"
  fi
fi

echo "==> Verifying download"
python3 - <<PY
import json
import sys
from pathlib import Path

local = Path("${LOCAL_DIR}")
index = local / "model.safetensors.index.json"
if not index.is_file():
    # Some repos use a differently named index; accept any *.safetensors.index.json.
    matches = sorted(local.glob("*.safetensors.index.json"))
    if not matches:
        raise SystemExit(f"missing safetensors index under {local}")
    index = matches[0]

meta = json.loads(index.read_text())
weight_map = meta.get("weight_map") or {}
shards = sorted(set(weight_map.values()))
expected = int("${EXPECTED_SHARDS}")
print(f"index={index.name} tensors={len(weight_map)} shards_in_index={len(shards)}")

missing = [s for s in shards if not (local / s).is_file()]
if missing:
    raise SystemExit(f"missing {len(missing)} shard files (e.g. {missing[:3]})")

on_disk = sorted(p.name for p in local.glob("*.safetensors"))
print(f"safetensors_on_disk={len(on_disk)}")
if expected and len(shards) != expected:
    print(f"WARNING: expected {expected} shards in index, found {len(shards)}", file=sys.stderr)

# Config / tokenizer must exist for vLLM serve.
for name in ("config.json",):
    if not (local / name).is_file():
        raise SystemExit(f"missing required file: {name}")

print("verification ok")
PY

date -u +"downloaded_at=%Y-%m-%dT%H:%M:%SZ" > "$MARKER"
echo "MODEL_ID=$MODEL_ID" >> "$MARKER"
echo "LOCAL_DIR=$LOCAL_DIR" >> "$MARKER"
echo "==> Done. Marker written to $MARKER"
df -h "$MODEL_ROOT"
du -sh "$LOCAL_DIR" "$HF_HOME" 2>/dev/null || true
ls -lh "$LOCAL_DIR" | head -40
