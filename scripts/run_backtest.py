#!/usr/bin/env python3
"""Backtest a named strategy and optionally run it through the validation gauntlet.

Examples:
    python scripts/run_backtest.py --strategy sma_crossover --symbol BTC/USDT
    python scripts/run_backtest.py --strategy rsi_mean_reversion --symbol BTC/USDT --gauntlet
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from algotrader.backtest import backtest  # noqa: E402
from algotrader.config import get, load_config  # noqa: E402
from algotrader.data import fetch  # noqa: E402
from algotrader.strategies import get_strategy, list_strategies  # noqa: E402
from algotrader.validation import run_gauntlet  # noqa: E402


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", required=True, help=f"one of: {list_strategies()}")
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--source", default=get(cfg, "data.default_source"))
    ap.add_argument("--timeframe", default=get(cfg, "data.default_timeframe"))
    ap.add_argument("--gauntlet", action="store_true", help="run the full validation gauntlet")
    args = ap.parse_args()

    df = fetch(args.symbol, source=args.source, timeframe=args.timeframe,
               cache_dir=get(cfg, "data.cache_dir"))
    cls = get_strategy(args.strategy)
    strat = cls()

    bt = cfg["backtest"]
    res = backtest(df, strat, initial_cash=bt["initial_cash"], fee_pct=bt["fee_pct"],
                   slippage_pct=bt["slippage_pct"], allow_short=bt["allow_short"])
    print(f"\n== Backtest: {args.strategy} on {args.symbol} ==")
    print(res.summary())

    if args.gauntlet:
        report = run_gauntlet(df, strat, cfg, n_trials=1)
        print("\n" + report.pretty())


if __name__ == "__main__":
    main()
