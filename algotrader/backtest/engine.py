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
from .pathsim import simulate


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
    cost_rate = fee_pct + slippage_pct
    risk_meta: dict = {}

    has_risk = (getattr(sig, "stop_loss_pct", None) is not None
                or getattr(sig, "take_profit_pct", None) is not None)
    if has_risk and not allow_short:
        # A stop fires inside a bar, which close-to-close arithmetic cannot see.
        # Walk the path instead — see pathsim for the fill rules.
        path = simulate(
            df, held,
            stop_loss_pct=sig.stop_loss_pct,
            take_profit_pct=sig.take_profit_pct,
            fee_pct=fee_pct, slippage_pct=slippage_pct,
        )
        strat_ret = path.returns
        held = path.positions
        trades = len(path.trades)
        risk_meta = {
            "risk_managed": True,
            "exit_counts": path.exit_counts,
            # How often "stop wins ties" actually decided the outcome. If this is
            # a large share of exits the result rests on an assumption, not on
            # anything the data can confirm.
            "ambiguous_bars": path.ambiguous_bars,
            "avg_bars_held": (
                float(np.mean([t["bars_held"] for t in path.trades])) if path.trades else 0.0
            ),
        }
    else:
        asset_ret = close.pct_change().fillna(0.0)
        # Turnover = change in position from one bar to the next -> costs.
        turnover = held.diff().abs().fillna(held.abs())
        strat_ret = held * asset_ret - turnover * cost_rate
        trades = int((held.diff().fillna(held).abs() > 1e-9).sum())
        risk_meta = {"risk_managed": False}

    equity = initial_cash * (1.0 + strat_ret).cumprod()

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
            # Cost drag is the whole game intraday: a few trades a day at 15m is
            # ~1000 round trips a year, and each one pays the spread twice.
            "cost_drag_annual": float(
                trades * cost_rate * periods_per_year / max(len(df), 1)
            ),
            "trades_per_day": float(
                trades / max((len(df) / (periods_per_year / 365.25)), 1e-9)
            ),
            **risk_meta,
        },
    )


def _infer_periods_per_year(index: pd.DatetimeIndex) -> int:
    if len(index) < 3:
        return 252
    # Measure the bar spacing in seconds *unit-independently*. A DatetimeIndex
    # can be backed by datetime64[ns], [us], [ms] or [s] — a Parquet round-trip
    # commonly yields [us] — so reading the raw int64 and assuming nanoseconds
    # silently scales the answer by 1000x per unit step. That inflates
    # periods_per_year, and Sharpe with it by its square root.
    deltas = pd.Series(index).diff().dt.total_seconds().dropna()
    if deltas.empty:
        return 252
    median_delta = float(deltas.median())
    seconds_per_year = 365.25 * 24 * 3600
    if median_delta <= 0:
        return 252
    est = seconds_per_year / median_delta
    # snap to common conventions
    for canonical in (252, 365, 8760, 12, 52):
        if abs(est - canonical) / canonical < 0.25:
            return canonical
    return int(round(est))
