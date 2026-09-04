"""Vectorized backtest engine.

Key correctness choices:
  * Signals generated at close[t] are ACTED ON at t+1 (positions are shifted by 1).
    This is the single most important line for avoiding lookahead bias.
  * Costs (fees + slippage) are charged on TURNOVER = |position change| at each bar.
  * Returns compound on close-to-close, scaled by the (lagged) target position.

This is intentionally simple and honest rather than feature-rich. If you later
need intrabar fills, partial fills, or margin, graduate to NautilusTrader — but
keep this as the fast first-pass filter for the agent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..strategies.base import Strategy, coerce_positions
from .metrics import compute_metrics


@dataclass
class BacktestResult:
    equity_curve: pd.Series        # portfolio value over time
    returns: pd.Series             # per-bar strategy returns (after costs)
    positions: pd.Series           # actually-held position (lagged), per bar
    trades: int                    # number of position changes
    metrics: dict                  # summary stats from compute_metrics
    meta: dict

    def summary(self) -> str:
        m = self.metrics
        return (
            f"trades={self.trades} "
            f"CAGR={m['cagr']:.2%} Sharpe={m['sharpe']:.2f} "
            f"Sortino={m['sortino']:.2f} MaxDD={m['max_drawdown']:.2%} "
            f"WinRate={m['win_rate']:.2%}"
        )


def backtest(
    df: pd.DataFrame,
    strategy: Strategy,
    initial_cash: float = 10_000.0,
    fee_pct: float = 0.001,
    slippage_pct: float = 0.0005,
    allow_short: bool = False,
    periods_per_year: int | None = None,
) -> BacktestResult:
    """Run a strategy over OHLCV data and return performance."""
    sig = strategy.generate_signals(df)
    target = coerce_positions(sig.positions, df.index)

    if not allow_short:
        target = target.clip(lower=0.0)
    target = target.clip(-1.0, 1.0)

    # Act on the signal at the NEXT bar. This prevents using close[t] to trade at t.
    held = target.shift(1).fillna(0.0)

    close = df["close"]
    asset_ret = close.pct_change().fillna(0.0)

    # Turnover = change in position from one bar to the next -> transaction costs.
    turnover = held.diff().abs().fillna(held.abs())
    cost_rate = fee_pct + slippage_pct
    costs = turnover * cost_rate

    strat_ret = held * asset_ret - costs
    equity = initial_cash * (1.0 + strat_ret).cumprod()

    trades = int((held.diff().fillna(held).abs() > 1e-9).sum())

    if periods_per_year is None:
        periods_per_year = _infer_periods_per_year(df.index)

    metrics = compute_metrics(strat_ret, equity, held, periods_per_year)

    return BacktestResult(
        equity_curve=equity,
        returns=strat_ret,
        positions=held,
        trades=trades,
        metrics=metrics,
        meta={
            "strategy": strategy.describe(),
            "fee_pct": fee_pct,
            "slippage_pct": slippage_pct,
            "allow_short": allow_short,
            "periods_per_year": periods_per_year,
            "n_bars": len(df),
        },
    )


def _infer_periods_per_year(index: pd.DatetimeIndex) -> int:
    if len(index) < 3:
        return 252
    median_delta = np.median(np.diff(index.view("int64"))) / 1e9  # seconds
    seconds_per_year = 365.25 * 24 * 3600
    if median_delta <= 0:
        return 252
    est = seconds_per_year / median_delta
    # snap to common conventions
    for canonical in (252, 365, 8760, 12, 52):
        if abs(est - canonical) / canonical < 0.25:
            return canonical
    return int(round(est))
