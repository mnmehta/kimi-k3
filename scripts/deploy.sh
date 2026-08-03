#!/usr/bin/env bash
# Config-driven deploy entrypoint.
#
# Usage:
#   export KUBECONFIG=...
#   ./scripts/deploy.sh tp16
#   ./scripts/deploy.sh tep16 pp2   # invalid — one recipe; use composition instead:
#   ./scripts/deploy.sh configs/layers/base.yaml configs/layers/weights-real.yaml configs/layers/strategy-pp2.yaml
#   ./scripts/deploy.sh --print tp16
#   ./scripts/deploy.sh --print-json dp2
#
# See configs/README.md.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PRINT_ONLY=0
PRINT_JSON=0
ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --print|-n)
      PRINT_ONLY=1
      shift
      ;;
    --print-json)
      PRINT_JSON=1
      shift
      ;;
    -h|--help)
      sed -n '2,20p' "$0"
      echo
      echo "Available recipes:"
      ls "$ROOT/configs/recipes"/*.yaml 2>/dev/null | xargs -n1 basename | sed 's/\.yaml$/  /' | tr '\n' ' '
      echo
      exit 0
      ;;
    *)
      ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ ${#ARGS[@]} -eq 0 ]]; then
  echo "usage: $0 [--print|--print-json] <recipe|config.yaml> [more.yaml...]" >&2
  echo "try: $0 --help" >&2
  exit 2
fi

pick_python() {
  local c
  for c in \
    "${CFG_PYTHON:-}" \
    "$ROOT/reports/.venv/bin/python" \
    python3
  do
    [[ -z "$c" ]] && continue
    if "$c" -c 'import yaml' 2>/dev/null; then
      echo "$c"
      return 0
    fi
  done
  return 1
}

if ! PYTHON="$(pick_python)"; then
  echo "PyYAML not found. Install with:" >&2
  echo "  python3 -m pip install pyyaml" >&2
  echo "  # or use reports/.venv after: pip install -r reports/requirements.txt" >&2
  exit 1
fi

MERGE=("$PYTHON" "$ROOT/scripts/lib/merge_config.py")

if [[ "$PRINT_JSON" == "1" ]]; then
  "${MERGE[@]}" --print-json "${ARGS[@]}"
  exit 0
fi

if [[ "$PRINT_ONLY" == "1" ]]; then
  "${MERGE[@]}" --print-exports "${ARGS[@]}"
  exit 0
fi

# shellcheck disable=SC1091
eval "$("${MERGE[@]}" --print-exports "${ARGS[@]}")"

echo "==> Merged recipe: ${DEPLOY_NAME:-?}  mode=${DEPLOY_MODE:-?}  serve=${DEPLOY_SERVE_SCRIPT:-?}"

case "${DEPLOY_MODE}" in
  multi_node)
    exec bash "$ROOT/scripts/lib/deploy_multi_node.sh"
    ;;
  pd)
    if [[ "${DEPLOY_VERIFY_WEIGHTS}" == "1" ]]; then
      exec bash "$ROOT/scripts/lib/deploy_pd.sh"
    else
      exec bash "$ROOT/scripts/lib/deploy_pd_dummy.sh"
    fi
    ;;
  *)
    echo "unknown deploy_mode=${DEPLOY_MODE}" >&2
    exit 1
    ;;
esac
