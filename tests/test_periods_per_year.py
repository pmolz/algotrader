"""Regression tests for annualization.

The bug these exist to prevent: `_infer_periods_per_year` read the index's raw
int64 and assumed nanoseconds. A Parquet round-trip hands back datetime64[ms],
so daily bars were read as 0.0864-second bars and periods_per_year came out at
365,250,000 instead of 365 — inflating every Sharpe by ~1000x (sqrt of 1e6).

That is not a cosmetic error: it silently defeats the entire gauntlet. With
Sharpe inflated 1000x, `min_sharpe: 1.0` is cleared by anything with a positive
mean return, so walk-forward stops rejecting and every downstream number lies.
"""

import numpy as np
import pandas as pd
import pytest

from algotrader.backtest.engine import _infer_periods_per_year, backtest
from algotrader.strategies.baselines import BuyAndHold

UNITS = ["ns", "us", "ms", "s"]


@pytest.mark.parametrize("unit", UNITS)
@pytest.mark.parametrize(
    "freq,expected", [("D", 365), ("h", 8760), ("W", 52)]
)
def test_inference_is_independent_of_the_index_backing_unit(unit, freq, expected):
    idx = pd.date_range("2020-01-01", periods=400, freq=freq, tz="UTC").as_unit(unit)
    assert _infer_periods_per_year(idx) == expected


@pytest.mark.parametrize("unit", UNITS)
def test_sharpe_is_identical_across_backing_units(unit, trending_ohlcv):
    df = trending_ohlcv.copy()
    df.index = df.index.as_unit(unit)
    res = backtest(df, BuyAndHold())
    ns = backtest(trending_ohlcv, BuyAndHold())
    assert res.metrics["sharpe"] == pytest.approx(ns.metrics["sharpe"], rel=1e-9)


def test_sharpe_matches_the_textbook_formula(trending_ohlcv):
    """The number the gauntlet judges must be the number the formula gives."""
    res = backtest(trending_ohlcv, BuyAndHold())
    r = res.returns.fillna(0.0).to_numpy()
    expected = r.mean() / r.std(ddof=1) * np.sqrt(365)
    assert res.meta["periods_per_year"] == 365
    assert res.metrics["sharpe"] == pytest.approx(expected, rel=1e-9)


def test_daily_sharpe_stays_in_a_sane_range(trending_ohlcv):
    """A guard against the failure mode's *shape*, not just its cause.

    No daily-bar strategy has a Sharpe in the hundreds. If this ever trips again,
    annualization is broken somewhere regardless of the mechanism.
    """
    res = backtest(trending_ohlcv, BuyAndHold())
    assert abs(res.metrics["sharpe"]) < 20, (
        f"Sharpe {res.metrics['sharpe']:.1f} is not physically plausible on daily "
        "bars — annualization is wrong."
    )


def test_cagr_does_not_overflow(trending_ohlcv):
    """The same bug made `years` ~1e-6, so the CAGR exponent overflowed."""
    res = backtest(trending_ohlcv, BuyAndHold())
    assert np.isfinite(res.metrics["cagr"])
    assert -1.0 < res.metrics["cagr"] < 10.0
