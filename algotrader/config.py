"""Configuration loading. Merges config/default.yaml with an optional
config/local.yaml override, and exposes a plain dict + dotted access helper.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "config"


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load default config, then merge config/local.yaml if present, then an
    explicit path if given."""
    with open(CONFIG_DIR / "default.yaml") as f:
        cfg = yaml.safe_load(f)

    local = CONFIG_DIR / "local.yaml"
    if local.exists():
        with open(local) as f:
            cfg = _deep_merge(cfg, yaml.safe_load(f) or {})

    if path is not None:
        with open(path) as f:
            cfg = _deep_merge(cfg, yaml.safe_load(f) or {})

    return cfg


def get(cfg: dict, dotted_key: str, default: Any = None) -> Any:
    """Fetch a nested value with a dotted key, e.g. get(cfg, 'backtest.fee_pct')."""
    node: Any = cfg
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def abspath(rel: str | Path) -> Path:
    """Resolve a repo-relative path to an absolute path."""
    p = Path(rel)
    return p if p.is_absolute() else REPO_ROOT / p


def load_dotenv(path: str | Path | None = None) -> None:
    """Minimal .env loader (no dependency). Sets os.environ for KEY=VALUE lines."""
    env_path = Path(path) if path else REPO_ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip())
