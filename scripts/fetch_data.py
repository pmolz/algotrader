#!/usr/bin/env python3
"""Fetch and cache market data.

Examples:
    python scripts/fetch_data.py --symbol BTC/USDT --source ccxt --timeframe 1d
    python scripts/fetch_data.py --symbol SPY --source yfinance --timeframe 1d --period 10y
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from algotrader.config import get, load_config  # noqa: E402
from algotrader.data import fetch  # noqa: E402


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--source", default=get(cfg, "data.default_source"))
    ap.add_argument("--timeframe", default=get(cfg, "data.default_timeframe"))
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--period", default="5y")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--exchange", default=get(cfg, "data.ccxt_exchange", "bitstamp"),
                    help="ccxt venue (some, e.g. binance, are geo-restricted)")
    args = ap.parse_args()

    df = fetch(
        args.symbol, source=args.source, timeframe=args.timeframe,
        cache_dir=get(cfg, "data.cache_dir"), limit=args.limit,
        period=args.period, refresh=args.refresh, exchange=args.exchange,
    )
    print(f"Fetched {len(df)} bars for {args.symbol} [{args.source} {args.timeframe}]")
    print(df.tail())


if __name__ == "__main__":
    main()
