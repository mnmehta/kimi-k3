#!/usr/bin/env bash
# Patch / unpatch fused_humming_moe.py to allow MoEActivation.SITU (Kimi-K3).
# Mirrors https://github.com/vllm-project/vllm/pull/50510 (allowlist only).
#
# Usage (in-pod or via kubectl exec):
#   bash patch_humming_situ.sh patch
#   bash patch_humming_situ.sh unpatch
#   bash patch_humming_situ.sh status
set -euo pipefail

ACTION="${1:-status}"
TARGET="${HUMMING_MOE_PY:-/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/experts/fused_humming_moe.py}"
BACKUP="${TARGET}.bak-pre-situ"

if [[ ! -f "$TARGET" ]]; then
  echo "humming experts file not found: $TARGET" >&2
  exit 1
fi

has_situ() {
  grep -q 'MoEActivation\.SITU' "$TARGET"
}

clear_pyc() {
  local d
  d="$(dirname "$TARGET")/__pycache__"
  rm -f "$d"/fused_humming_moe*.pyc 2>/dev/null || true
}

case "$ACTION" in
  status)
    if has_situ; then
      echo "patched: MoEActivation.SITU present in $TARGET"
    else
      echo "unpatched: MoEActivation.SITU absent in $TARGET"
    fi
    [[ -f "$BACKUP" ]] && echo "backup: $BACKUP" || echo "backup: (none)"
    ;;
  patch)
    if has_situ; then
      echo "already patched (SITU present)"
      exit 0
    fi
    if [[ ! -f "$BACKUP" ]]; then
      cp -a "$TARGET" "$BACKUP"
      echo "backed up -> $BACKUP"
    fi
    # Insert SITU after SWIGLUOAI (same as PR #50510).
    if ! grep -q 'MoEActivation\.SWIGLUOAI' "$TARGET"; then
      echo "unexpected file shape: SWIGLUOAI not found; aborting" >&2
      exit 1
    fi
    python3 - "$TARGET" <<'PY'
import pathlib, sys
path = pathlib.Path(sys.argv[1])
text = path.read_text()
needle = "            MoEActivation.SWIGLUOAI,\n"
insert = needle + "            MoEActivation.SITU,\n"
if "MoEActivation.SITU" in text:
    print("SITU already present")
elif needle not in text:
    raise SystemExit("SWIGLUOAI allowlist line not found")
else:
    path.write_text(text.replace(needle, insert, 1))
    print("inserted MoEActivation.SITU")
PY
    clear_pyc
    has_situ || { echo "patch failed" >&2; exit 1; }
    echo "patched OK"
    ;;
  unpatch)
    if [[ -f "$BACKUP" ]]; then
      cp -a "$BACKUP" "$TARGET"
      clear_pyc
      echo "restored from $BACKUP"
    elif has_situ; then
      python3 - "$TARGET" <<'PY'
import pathlib, sys
path = pathlib.Path(sys.argv[1])
text = path.read_text()
text2 = text.replace("            MoEActivation.SITU,\n", "", 1)
if text2 == text:
    raise SystemExit("could not remove SITU line")
path.write_text(text2)
print("removed MoEActivation.SITU line")
PY
      clear_pyc
    else
      echo "already unpatched"
    fi
    ;;
  *)
    echo "usage: $0 {patch|unpatch|status}" >&2
    exit 2
    ;;
esac
