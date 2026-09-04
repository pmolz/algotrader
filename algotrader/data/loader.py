"""Market data fetching + local Parquet cache.

Design rules:
  * Cache aggressively. Never hit an API twice for the same slice.
  * Always return a canonical OHLCV DataFrame indexed by UTC timestamp with
    lowercase columns: open, high, low, close, volume.
  * Keep sources pluggable; a strategy/backtest never knows where data came from.

Free sources supported here:
  * ccxt      -> crypto exchange OHLCV (real data, 24/7, no key needed for public)
  * yfinance  -> equities/ETFs/indices (delayed, free)
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

CANONICAL_COLS = ["open", "high", "low", "close", "volume"]


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_").lower()


def cache_path(cache_dir: str | Path, source: str, symbol: str, timeframe: str) -> Path:
    fname = f"{_slug(source)}__{_slug(symbol)}__{_slug(timeframe)}.parquet"
    return Path(cache_dir) / fname


def load_cached(cache_dir, source, symbol, timeframe) -> pd.DataFrame | None:
    p = cache_path(cache_dir, source, symbol, timeframe)
    if p.exists():
        return pd.read_parquet(p)
    return None


def _validate(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in CANONICAL_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Data missing canonical columns: {missing}")
    df = df[~df.index.duplicated(keep="last")].sort_index()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index.name = "timestamp"
    return df[CANONICAL_COLS]


def _fetch_ccxt(symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    import ccxt  # lazy import so the package works without it installed

    ex = ccxt.binance({"enableRateLimit": True})
    raw = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df.set_index("timestamp")


def _fetch_yfinance(symbol: str, timeframe: str, period: str) -> pd.DataFrame:
    import yfinance as yf

    # yfinance uses '1d','1h','1wk' style intervals already
    df = yf.download(symbol, period=period, interval=timeframe, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)
    df.index = pd.to_datetime(df.index, utc=True)
    return df


def fetch(
    symbol: str,
    source: str = "ccxt",
    timeframe: str = "1d",
    cache_dir: str | Path = "data_cache",
    limit: int = 1000,
    period: str = "5y",
    refresh: bool = False,
) -> pd.DataFrame:
    """Fetch OHLCV, using cache unless refresh=True.

    Args:
        symbol:    e.g. 'BTC/USDT' (ccxt) or 'SPY' (yfinance)
        source:    'ccxt' | 'yfinance'
        timeframe: e.g. '1d', '1h', '1wk'
        limit:     max bars (ccxt)
        period:    lookback window (yfinance), e.g. '5y', '10y', 'max'
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if not refresh:
        cached = load_cached(cache_dir, source, symbol, timeframe)
        if cached is not None and len(cached):
            return cached

    if source == "ccxt":
        df = _fetch_ccxt(symbol, timeframe, limit)
    elif source == "yfinance":
        df = _fetch_yfinance(symbol, timeframe, period)
    else:
        raise ValueError(f"Unknown source: {source!r}")

    df = _validate(df)
    df.to_parquet(cache_path(cache_dir, source, symbol, timeframe))
    return df
