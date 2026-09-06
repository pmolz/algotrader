"""Fill rules for stop loss / take profit.

Every test here is a hand-checkable number, because this file is where a
backtest either tells the truth or quietly manufactures an edge. The three rules
that matter, in the order they bite:

  * a bar containing both levels resolves to the STOP, since OHLC cannot say
    which came first and guessing the target is free money;
  * a bar that opens through a level fills at the OPEN, not the level;
  * after a stop the position stays flat until the signal goes flat and asks
    again, or the stop is just a fee generator.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from algotrader.backtest.engine import backtest
from algotrader.backtest.pathsim import simulate
from algotrader.strategies.base import Strategy, StrategyResult

COST = dict(fee_pct=0.0, slippage_pct=0.0)   # isolate fill logic from costs


def frame(rows) -> pd.DataFrame:
    """rows: list of (open, high, low, close)."""
    idx = pd.date_range("2026-01-01", periods=len(rows), freq="15min", tz="UTC")
    o, h, low, c = zip(*rows)
    return pd.DataFrame(
        {"open": o, "high": h, "low": low, "close": c, "volume": [1.0] * len(rows)},
        index=idx,
    )


def held(*vals) -> pd.Series:
    return pd.Series(list(vals), dtype=float)


def run(df, target, **kw):
    t = pd.Series(list(target), index=df.index, dtype=float)
    return simulate(df, t, **{**COST, **kw})


# -- basic exits --------------------------------------------------------------
def test_stop_fills_at_the_stop_price():
    # enter at close=100 on bar 0; bar 1 trades down through 98 but not to the open
    df = frame([(100, 100, 100, 100), (100, 101, 97, 99)])
    r = run(df, [1, 1], stop_loss_pct=0.02)          # stop at 98
    assert len(r.trades) == 1
    t = r.trades[0]
    assert t["reason"] == "stop"
    assert t["exit_px"] == pytest.approx(98.0)
    assert t["ret"] == pytest.approx(-0.02)


def test_target_fills_at_the_target_price():
    df = frame([(100, 100, 100, 100), (100, 104, 99, 101)])
    r = run(df, [1, 1], take_profit_pct=0.03)        # target at 103
    assert r.trades[0]["reason"] == "target"
    assert r.trades[0]["exit_px"] == pytest.approx(103.0)
    assert r.trades[0]["ret"] == pytest.approx(0.03)


def test_a_bar_holding_both_levels_resolves_to_the_stop():
    """The whole point. OHLC cannot order the two touches, so take the loss."""
    df = frame([(100, 100, 100, 100), (100, 104, 96, 101)])
    r = run(df, [1, 1], stop_loss_pct=0.02, take_profit_pct=0.03)
    assert r.trades[0]["reason"] == "stop"
    assert r.ambiguous_bars == 1, "the assumption must be counted, not hidden"


def test_unambiguous_bars_are_not_counted_as_ambiguous():
    df = frame([(100, 100, 100, 100), (100, 104, 99, 101)])
    r = run(df, [1, 1], stop_loss_pct=0.02, take_profit_pct=0.03)
    assert r.trades[0]["reason"] == "target" and r.ambiguous_bars == 0


# -- gaps ---------------------------------------------------------------------
def test_a_gap_through_the_stop_fills_at_the_open_not_the_stop():
    """Filling at the stop on a gap is the quiet way to invent edge."""
    df = frame([(100, 100, 100, 100), (95, 96, 94, 95)])   # opens below the 98 stop
    r = run(df, [1, 1], stop_loss_pct=0.02)
    assert r.trades[0]["exit_px"] == pytest.approx(95.0)
    assert r.trades[0]["ret"] == pytest.approx(-0.05)      # worse than the -2% stop


def test_a_gap_through_the_target_fills_at_the_open():
    df = frame([(100, 100, 100, 100), (106, 107, 105, 106)])   # opens above the 103 target
    r = run(df, [1, 1], take_profit_pct=0.03)
    assert r.trades[0]["exit_px"] == pytest.approx(106.0)
    assert r.trades[0]["ret"] == pytest.approx(0.06)


# -- re-entry policy ----------------------------------------------------------
def test_no_re_entry_while_the_signal_stays_long():
    """A constant target=1 with a stop must not re-enter on the next bar; that
    turns the stop into a fee pump."""
    df = frame([(100, 100, 100, 100), (100, 101, 97, 99),
                (99, 100, 98, 99), (99, 100, 98, 99)])
    r = run(df, [1, 1, 1, 1], stop_loss_pct=0.02)
    assert len(r.trades) == 1, f"re-entered while still signalled long: {r.trades}"


def test_re_entry_allowed_after_the_signal_goes_flat():
    df = frame([(100, 100, 100, 100), (100, 101, 97, 99),
                (99, 100, 98, 99), (99, 100, 98, 99), (99, 100, 98, 99)])
    r = run(df, [1, 1, 0, 1, 1], stop_loss_pct=0.02)
    assert len(r.trades) == 1                  # second entry opened, not yet closed
    assert r.positions.iloc[4] == pytest.approx(1.0)


def test_signal_exit_is_labelled_and_marks_to_close():
    df = frame([(100, 100, 100, 100), (100, 102, 99, 101)])
    r = run(df, [1, 0], stop_loss_pct=0.10)
    assert r.trades[0]["reason"] == "signal"
    assert r.trades[0]["exit_px"] == pytest.approx(101.0)


# -- costs and accounting -----------------------------------------------------
def test_costs_are_charged_on_entry_and_exit():
    df = frame([(100, 100, 100, 100), (100, 101, 97, 99)])
    r = run(df, [1, 1], stop_loss_pct=0.02, fee_pct=0.001, slippage_pct=0.0005)
    # entry bar pays one fill, exit bar pays the -2% move plus one fill
    assert r.returns.iloc[0] == pytest.approx(-0.0015)
    assert r.returns.iloc[1] == pytest.approx(-0.02 - 0.0015)


def test_an_open_position_at_the_end_is_not_force_closed():
    """Liquidating on the last bar invents a trade the strategy never made."""
    df = frame([(100, 100, 100, 100), (100, 102, 99, 101)])
    r = run(df, [1, 1], stop_loss_pct=0.50)
    assert r.trades == []
    assert r.positions.iloc[1] == pytest.approx(1.0)


def test_a_stop_of_100pct_or_more_is_treated_as_no_stop():
    df = frame([(100, 100, 100, 100), (100, 101, 1, 99)])
    r = run(df, [1, 1], stop_loss_pct=1.0)
    assert r.trades == []


# -- integration with the engine ----------------------------------------------
def strat(fn, **risk) -> Strategy:
    return type("S", (Strategy,), {
        "name": "s",
        "generate_signals": lambda self, d, _f=fn: StrategyResult(positions=_f(d), **risk),
    })()


def price_frame(n=300, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-01-01", periods=n, freq="15min", tz="UTC")
    c = 100 * np.exp(rng.normal(0, 0.004, n).cumsum())
    return pd.DataFrame({"open": c, "high": c * 1.003, "low": c * 0.997,
                         "close": c, "volume": rng.uniform(1, 5, n)}, index=idx)


def test_engine_without_a_risk_overlay_is_unchanged():
    """Declaring no stop must keep the fast vectorized path, bit for bit."""
    df = price_frame()
    fn = lambda d: (d.close > d.close.rolling(20).mean()).astype(float)  # noqa: E731
    res = backtest(df, strat(fn), fee_pct=0.001, slippage_pct=0.0005)
    assert res.meta["risk_managed"] is False


def test_engine_switches_to_path_simulation_when_a_stop_is_declared():
    df = price_frame()
    fn = lambda d: (d.close > d.close.rolling(20).mean()).astype(float)  # noqa: E731
    res = backtest(df, strat(fn, stop_loss_pct=0.01, take_profit_pct=0.02),
                   fee_pct=0.001, slippage_pct=0.0005)
    assert res.meta["risk_managed"] is True
    assert set(res.meta["exit_counts"]) <= {"stop", "target", "signal"}
    assert res.meta["avg_bars_held"] > 0
    assert len(res.equity_curve) == len(df)


def test_engine_reports_cost_drag_and_trade_frequency():
    """Intraday, these two numbers decide whether anything can work at all."""
    df = price_frame()
    fn = lambda d: (d.close > d.close.rolling(5).mean()).astype(float)  # noqa: E731
    res = backtest(df, strat(fn), fee_pct=0.001, slippage_pct=0.0005)
    assert res.meta["cost_drag_annual"] > 0
    assert res.meta["trades_per_day"] > 0


def test_a_per_bar_stop_series_is_accepted():
    """ATR-style stops size the level per bar rather than using a constant."""
    df = price_frame()
    def fn(d):
        return (d.close > d.close.rolling(20).mean()).astype(float)
    atr = (df.high - df.low).rolling(14).mean() / df.close
    res = backtest(df, strat(fn, stop_loss_pct=atr * 2), fee_pct=0.0, slippage_pct=0.0)
    assert res.meta["risk_managed"] is True
