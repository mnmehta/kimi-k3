#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/redhat-performance/aiconfigurator.git}"
TARGET_DIR="${1:-aiconfigurator}"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"

if ! command -v git >/dev/null 2>&1; then
    echo "git is required but was not found on PATH" >&2
    exit 1
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "Expected ${PYTHON_BIN} on PATH. Set PYTHON_BIN to a compatible Python 3.11-3.13 interpreter." >&2
    exit 1
fi

if [[ -e "${TARGET_DIR}" ]] && [[ -n "$(ls -A "${TARGET_DIR}" 2>/dev/null)" ]]; then
    echo "Target directory ${TARGET_DIR} already exists and is not empty" >&2
    exit 1
fi

git clone "${REPO_URL}" "${TARGET_DIR}"
cd "${TARGET_DIR}"

"${PYTHON_BIN}" -m venv .venv

if command -v rustup >/dev/null 2>&1; then
    rustup update stable
else
    echo "rustup not found; using the existing Rust toolchain" >&2
fi

mkdir -p .matplotlib
export MPLCONFIGDIR="${PWD}/.matplotlib"

.venv/bin/python -m pip install ./aic-core
.venv/bin/python -m pip install .

.venv/bin/python -c "import aiconfigurator, aiconfigurator_core; print(aiconfigurator.__file__); print(aiconfigurator_core.__file__)"
.venv/bin/aiconfigurator --help
