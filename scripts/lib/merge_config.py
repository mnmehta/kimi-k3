#!/usr/bin/env python3
"""Merge hierarchical deploy YAML configs and emit shell-exportable env.

Usage:
  merge_config.py [--print-json|--print-exports|--print-meta KEY] CONFIG [CONFIG ...]

Each CONFIG is a path or a recipe name under configs/recipes/<name>.yaml.
Includes are resolved relative to the file that declares them.
Later files / keys override earlier ones. env: null deletes a key.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as e:  # pragma: no cover
    sys.stderr.write(
        "PyYAML required. Try: reports/.venv/bin/python or "
        "pip install pyyaml\n"
    )
    raise SystemExit(1) from e

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "configs"
RECIPES = CONFIGS / "recipes"

META_KEYS = (
    "name",
    "description",
    "deploy_mode",
    "serve_script",
    "verify_weights",
    "storage_backend",
    "health_port",
    "health_timeout_polls",
    "pd",
)


def deep_merge(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    out = dict(a)
    for k, v in b.items():
        if k == "include":
            continue
        if v is None and k in out.get("env", {}):
            # Handled in env merge below.
            pass
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            if k == "env":
                env = dict(out.get("env") or {})
                for ek, ev in v.items():
                    # Keep None so print_exports can emit `unset`.
                    env[ek] = ev
                out["env"] = env
            else:
                out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_yaml(path: Path, stack: list[Path] | None = None) -> dict[str, Any]:
    stack = stack or []
    path = path.resolve()
    if path in stack:
        cycle = " -> ".join(str(p) for p in stack + [path])
        raise SystemExit(f"include cycle: {cycle}")
    if not path.is_file():
        raise SystemExit(f"config not found: {path}")
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"config root must be a mapping: {path}")

    includes = data.get("include") or []
    if isinstance(includes, str):
        includes = [includes]
    merged: dict[str, Any] = {}
    for inc in includes:
        inc_path = (path.parent / inc).resolve()
        merged = deep_merge(merged, load_yaml(inc_path, stack + [path]))
    # Apply this file on top (excluding include list).
    body = {k: v for k, v in data.items() if k != "include"}
    return deep_merge(merged, body)


def resolve_config_arg(arg: str) -> Path:
    p = Path(arg)
    if p.is_file():
        return p.resolve()
    # recipe name
    cand = RECIPES / f"{arg}.yaml"
    if cand.is_file():
        return cand.resolve()
    cand2 = CONFIGS / arg
    if cand2.is_file():
        return cand2.resolve()
    raise SystemExit(f"unknown config/recipe: {arg} (expected file or configs/recipes/{arg}.yaml)")


def shell_value(v: Any) -> str:
    if isinstance(v, bool):
        return "1" if v else "0"
    if v is None:
        return ""
    return str(v)


def print_exports(cfg: dict[str, Any]) -> None:
    env = cfg.get("env") or {}
    # Meta as DEPLOY_* for the bash driver
    meta = {
        "DEPLOY_NAME": cfg.get("name", ""),
        "DEPLOY_MODE": cfg.get("deploy_mode", "multi_node"),
        "DEPLOY_SERVE_SCRIPT": cfg.get("serve_script", "scripts/run-vllm-kimi-k3-recipe.sh"),
        "DEPLOY_VERIFY_WEIGHTS": "1" if cfg.get("verify_weights", True) else "0",
        "DEPLOY_STORAGE_BACKEND": cfg.get("storage_backend", "hostpath"),
        "DEPLOY_HEALTH_PORT": str(cfg.get("health_port", 8000)),
        "DEPLOY_HEALTH_TIMEOUT_POLLS": str(cfg.get("health_timeout_polls", 360)),
    }
    pd = cfg.get("pd") or {}
    if pd:
        meta["DEPLOY_PD_START_ROUTER"] = "1" if pd.get("start_router") else "0"
        if "prefill_pods" in pd:
            meta["DEPLOY_PD_PREFILL_PODS"] = " ".join(str(x) for x in pd["prefill_pods"])
        if "decode_pods" in pd:
            meta["DEPLOY_PD_DECODE_PODS"] = " ".join(str(x) for x in pd["decode_pods"])

    for k, v in meta.items():
        print(f"export {k}={shlex.quote(shell_value(v))}")

    for k, v in sorted(env.items()):
        if v is None:
            print(f"unset {k} || true")
        else:
            print(f"export {k}={shlex.quote(shell_value(v))}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("configs", nargs="+", help="recipe names and/or YAML paths (merge order)")
    ap.add_argument("--print-json", action="store_true", help="print merged JSON")
    ap.add_argument("--print-exports", action="store_true", help="print bash exports (default)")
    ap.add_argument("--print-meta", metavar="KEY", help="print a single top-level meta key")
    args = ap.parse_args()

    merged: dict[str, Any] = {}
    for arg in args.configs:
        path = resolve_config_arg(arg)
        merged = deep_merge(merged, load_yaml(path))

    if args.print_meta:
        v = merged.get(args.print_meta, "")
        if isinstance(v, (dict, list)):
            print(json.dumps(v))
        else:
            print(shell_value(v))
        return

    if args.print_json:
        print(json.dumps(merged, indent=2, default=str))
        return

    print_exports(merged)


if __name__ == "__main__":
    main()
