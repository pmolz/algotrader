"""On-demand equity curves — the one place the dashboard executes strategy code.

Plotting an equity curve means *running* an LLM-written strategy, so this goes
through the exact same Docker jail the nightly loop uses: no network, read-only
root, dropped capabilities, non-root user, resource caps, hard timeout. The
dashboard process never execs generated code itself.

Results are cached on disk keyed by a hash of (code, data, costs). A curve is a
pure function of those three, so a repeat view is free, and a browser refresh
cannot be used to spam container launches.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from ..agent.sandbox import (
    SandboxError,
    SandboxLimits,
    docker_available,
    image_exists,
    run_gauntlet_sandboxed,
)
from ..config import get
from ..data import fetch
from ..data.loader import load_cached


class EquityUnavailable(RuntimeError):
    """Raised with a human-readable reason the curve can't be produced."""


# Bump when the payload shape or the numbers it carries change, so a cached
# result computed by older code is never served. Without this, a fix to the
# backtest engine leaves stale (wrong) curves on disk indefinitely.
CACHE_VERSION = 2


def json_safe(obj):
    """Replace non-finite floats with None, recursively.

    Python's json module happily writes `NaN` / `Infinity`, which are **not
    valid JSON** — the browser's JSON.parse rejects the whole document. And
    these values occur naturally here: Calmar is inf when max drawdown is zero,
    a flat equity curve gives a 0/0 Sharpe. So this is not defensive padding;
    without it a legitimate strategy breaks the endpoint.
    """
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return obj


def _cache_key(code: str, symbol: str, source: str, timeframe: str, cfg: dict) -> str:
    bt = cfg.get("backtest", {})
    blob = json.dumps(
        {
            "v": CACHE_VERSION,
            "code": code,
            "symbol": symbol,
            "source": source,
            "timeframe": timeframe,
            # costs change the curve, so they belong in the key
            "fee": bt.get("fee_pct"),
            "slip": bt.get("slippage_pct"),
            "cash": bt.get("initial_cash"),
            "short": bt.get("allow_short"),
        },
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def cached_curve(exp: dict, cfg: dict) -> dict | None:
    """Return an already-computed curve, or None. Never launches a container.

    Lets the page render a known curve server-side instead of making the browser
    fetch it, while guaranteeing that merely loading a URL can never start a
    container.
    """
    code = exp.get("code")
    if not code or not exp.get("symbol"):
        return None
    cache_dir = Path(get(cfg, "dashboard.cache_dir", "experiments/equity_cache"))
    key = _cache_key(code, exp["symbol"], exp.get("source"), exp.get("timeframe"), cfg)
    f = cache_dir / f"{key}.json"
    if not f.exists():
        return None
    try:
        out = json.loads(f.read_text())
    except (OSError, ValueError):
        return None
    out["cached"] = True
    return out


def equity_curve(exp: dict, cfg: dict, *, use_cache: bool = True) -> dict:
    """Compute (or read from cache) the equity curve for one experiment.

    Raises EquityUnavailable with an explanation the UI can show directly.
    """
    code = exp.get("code")
    if not code:
        raise EquityUnavailable(
            "This experiment has no stored code — it failed before any was produced."
        )

    symbol, source = exp.get("symbol"), exp.get("source")
    timeframe = exp.get("timeframe")
    if not symbol:
        raise EquityUnavailable("This experiment has no symbol recorded.")

    cache_dir = Path(get(cfg, "dashboard.cache_dir", "experiments/equity_cache"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _cache_key(code, symbol, source, timeframe, cfg)
    cache_file = cache_dir / f"{key}.json"
    if use_cache and cache_file.exists():
        out = json.loads(cache_file.read_text())
        out["cached"] = True
        return out

    if not docker_available():
        raise EquityUnavailable(
            "Docker isn't available, and strategy code is never run outside the "
            "sandbox. Start Docker and try again."
        )
    image = get(cfg, "sandbox.image", "algotrader-sandbox:latest")
    if not image_exists(image):
        raise EquityUnavailable(
            f"Sandbox image {image!r} is missing. Build it: bash sandbox/build.sh"
        )

    # Prefer cache: this is a chart, not a trading decision, and the dashboard
    # should never block on a slow exchange API.
    df = load_cached(get(cfg, "data.cache_dir", "data_cache"), source, symbol, timeframe)
    if df is None or not len(df):
        try:
            df = fetch(
                symbol, source=source, timeframe=timeframe,
                cache_dir=get(cfg, "data.cache_dir", "data_cache"),
                exchange=get(cfg, "data.ccxt_exchange", "bitstamp"),
            )
        except Exception as e:  # noqa: BLE001
            raise EquityUnavailable(
                f"No cached data for {symbol} and the download failed: "
                f"{type(e).__name__}: {e}"
            ) from e

    limits = SandboxLimits(
        image=image,
        memory=get(cfg, "sandbox.memory", "1g"),
        cpus=str(get(cfg, "sandbox.cpus", "2")),
        pids=int(get(cfg, "sandbox.pids", 128)),
        timeout_s=int(get(cfg, "dashboard.equity_timeout_s", 60)),
    )
    try:
        out = run_gauntlet_sandboxed(df, code, cfg, n_trials=1, limits=limits,
                                     mode="equity")
    except SandboxError as e:
        raise EquityUnavailable(f"Sandbox failed: {e}") from e

    if not out.get("ok"):
        raise EquityUnavailable(
            "The strategy crashed when re-run: "
            + (out.get("error") or "").strip().splitlines()[-1][:200]
        )

    out = json_safe(_downsample(out))
    # allow_nan=False turns "we wrote invalid JSON" into a loud failure here
    # rather than a JSON.parse error in the browser.
    cache_file.write_text(json.dumps(out, allow_nan=False))
    out["cached"] = False
    return out


def _downsample(out: dict, max_points: int = 1200) -> dict:
    """Keep the payload small for the browser without distorting the shape.

    Takes every Nth point but always keeps the last one, so the final equity
    value on the chart matches the reported metrics.
    """
    n = len(out.get("dates") or [])
    if n <= max_points:
        return out
    step = n // max_points + 1
    idx = list(range(0, n, step))
    if idx[-1] != n - 1:
        idx.append(n - 1)
    for field in ("dates", "equity", "benchmark", "positions"):
        series = out.get(field)
        if series:
            out[field] = [series[i] for i in idx]
    out["downsampled_from"] = n
    return out
