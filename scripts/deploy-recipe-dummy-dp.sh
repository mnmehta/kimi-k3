#!/usr/bin/env bash
# Backward-compatible wrapper → config-driven deploy.
# Prefer: ./scripts/deploy.sh dp2-dummy
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$ROOT/scripts/deploy.sh" dp2-dummy "$@"
