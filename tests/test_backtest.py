"""Tests for the backtest engine — these protect every decision the system makes."""

import numpy as np
import pandas as pd

from algotrader.backtest import backtest
from algotrader.strategies.baselines import BuyAndHold, SmaCrossover


def test_buy_and_hold_matches_asset_return_minus_one_entry_cost(trending_ohlcv):
    df = trending_ohlcv
    res = backtest(df, BuyAndHold(), fee_pct=0.0, slippage_pct=0.0)
    # with zero costs, buy&hold equity should track close (after 1-bar entry lag)
    asset_total = df["close"].iloc[-1] / df["close"].iloc[1] - 1
    strat_total = res.metrics["total_return"]
    assert abs(strat_total - asset_total) < 0.05


def test_costs_reduce_returns(trending_ohlcv):
    df = trending_ohlcv
    cheap = backtest(df, SmaCrossover(), fee_pct=0.0, slippage_pct=0.0)
    pricey = backtest(df, SmaCrossover(), fee_pct=0.01, slippage_pct=0.01)
    assert pricey.metrics["total_return"] < cheap.metrics["total_return"]


def test_positions_are_lagged_no_lookahead(trending_ohlcv):
    """Held position at bar t must equal the target signal at t-1."""
    df = trending_ohlcv
    strat = SmaCrossover()
    target = strat.generate_signals(df).positions.fillna(0.0)
    res = backtest(df, strat, fee_pct=0.0, slippage_pct=0.0)
    expected = target.shift(1).fillna(0.0)
    assert np.allclose(res.positions.to_numpy(), expected.to_numpy())


def test_flat_strategy_has_zero_return(trending_ohlcv):
    from algotrader.strategies.base import Strategy, StrategyResult

    class Flat(Strategy):
        name = "flat_test"
        def generate_signals(self, df):
            return StrategyResult(positions=pd.Series(0.0, index=df.index))

    res = backtest(trending_ohlcv, Flat())
    assert abs(res.metrics["total_return"]) < 1e-9
    assert res.trades == 0
