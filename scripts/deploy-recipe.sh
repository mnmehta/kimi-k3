#!/usr/bin/env bash
# Backward-compatible wrapper → config-driven deploy.
# Prefer: ./scripts/deploy.sh tp16
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$ROOT/scripts/deploy.sh" tp16 "$@"
