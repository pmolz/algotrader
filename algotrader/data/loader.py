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

# Which ccxt venue to pull public OHLCV from. Not all exchanges serve every
# region — Binance, for instance, refuses requests from some countries — so this
# is configurable via `data.ccxt_exchange`. Bitstamp serves ~1000 daily bars of
# public history without a key, which is enough for the validation gauntlet.
DEFAULT_EXCHANGE = "bitstamp"


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


# Bar size in milliseconds, for paging backwards through history.
_TIMEFRAME_MS = {
    "1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000, "30m": 1_800_000,
    "1h": 3_600_000, "2h": 7_200_000, "4h": 14_400_000, "6h": 21_600_000,
    "12h": 43_200_000, "1d": 86_400_000, "3d": 259_200_000, "1w": 604_800_000,
}


def timeframe_ms(timeframe: str) -> int:
    if timeframe not in _TIMEFRAME_MS:
        raise ValueError(f"Unsupported timeframe {timeframe!r}. "
                         f"Known: {', '.join(sorted(_TIMEFRAME_MS))}")
    return _TIMEFRAME_MS[timeframe]


def _fetch_ccxt(
    symbol: str, timeframe: str, limit: int, exchange: str = DEFAULT_EXCHANGE,
    bars: int | None = None,
) -> pd.DataFrame:
    """Fetch OHLCV, paging backwards when `bars` exceeds one request.

    Exchanges cap a single fetch_ohlcv at ~1000 bars, which is 41 days of hourly
    data and only 10 days of 15m — not enough to walk-forward across, let alone
    hold out a fifth of it. So page: ask for a window, advance `since` past what
    came back, and stop when the exchange repeats itself or reaches now.

    Pages are deduplicated by timestamp rather than trusting page boundaries.
    Venues return partial pages (bitstamp hands back 999 of a requested 1000)
    and overlap them, so `len(batch) < limit` is not a reliable end-of-data
    signal — treating it as one silently truncates history to a single page.
    """
    import ccxt  # lazy import so the package works without it installed

    if not hasattr(ccxt, exchange):
        raise ValueError(f"Unknown ccxt exchange: {exchange!r}")
    ex = getattr(ccxt, exchange)({"enableRateLimit": True})

    target = int(bars or limit)
    step = timeframe_ms(timeframe)
    page = min(int(limit), 1000)

    if target <= page:
        raw = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=page)
    else:
        now = ex.milliseconds()
        since = now - target * step
        seen: set[int] = set()
        raw = []
        while len(raw) < target:
            batch = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=page, since=since)
            new = [b for b in batch if b[0] not in seen]
            if not new:
                break                      # exchange has nothing older/newer to give
            seen.update(b[0] for b in new)
            raw.extend(new)
            since = max(b[0] for b in new) + step
            if since > now:
                break
        raw.sort(key=lambda b: b[0])

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
    exchange: str = DEFAULT_EXCHANGE,
    bars: int | None = None,
) -> pd.DataFrame:
    """Fetch OHLCV, using cache unless refresh=True.

    Args:
        symbol:    e.g. 'BTC/USDT' (ccxt) or 'SPY' (yfinance)
        source:    'ccxt' | 'yfinance'
        timeframe: e.g. '1d', '1h', '1wk'
        limit:     bars per request (ccxt); venues cap this around 1000
        bars:      total bars wanted (ccxt); pages backwards to reach it
        period:    lookback window (yfinance), e.g. '5y', '10y', 'max'
        exchange:  ccxt venue to query (some are geo-restricted)
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if not refresh:
        cached = load_cached(cache_dir, source, symbol, timeframe)
        if cached is not None and len(cached):
            return cached

    if source == "ccxt":
        df = _fetch_ccxt(symbol, timeframe, limit, exchange=exchange, bars=bars)
    elif source == "yfinance":
        df = _fetch_yfinance(symbol, timeframe, period)
    else:
        raise ValueError(f"Unknown source: {source!r}")

    df = _validate(df)
    df.to_parquet(cache_path(cache_dir, source, symbol, timeframe))
    return df
