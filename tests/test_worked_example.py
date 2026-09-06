"""The worked example in the prompt is executable code that gets imitated.

A small model copies its structure far more literally than it follows prose, so
a broken example is worse than none: every proposal inherits the break. This
happened once during development — an edit removed the `entries = ...` line and
the example raised NameError, which would have poisoned every candidate until
someone noticed.

It is also the only place the "hold once entered" pattern is demonstrated. A
bare `entries.astype(float)` returns to 0 on the next bar and the engine exits
on the signal before the stop or target can fire; measured on real data that put
71 of 90 exits in the "signal" bucket with a 2-bar average hold, making the risk
overlay decorative. These assertions pin that behaviour.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

from algotrader.agent.codegen import load_strategy_class
from algotrader.agent.prompts import WORKED_EXAMPLE
from algotrader.backtest.engine import backtest
from algotrader.strategies.base import Strategy
from algotrader.validation.lookahead import lookahead_check


@pytest.fixture(scope="module")
def example() -> Strategy:
    code = re.search(r"```python\s*(.*?)```", WORKED_EXAMPLE, re.DOTALL).group(1)
    return load_strategy_class(code)()


@pytest.fixture(scope="module")
def bars() -> pd.DataFrame:
    """Synthetic 15m crypto-ish series: enough bars for the 2000-bar quantile."""
    n = 12000
    rng = np.random.default_rng(7)
    idx = pd.date_range("2025-01-01", periods=n, freq="15min", tz="UTC")
    ret = rng.normal(0, 0.002, n)
    c = 100 * np.exp(ret.cumsum())
    wiggle = np.abs(rng.normal(0, 0.003, n))
    return pd.DataFrame(
        {"open": c, "high": c * (1 + wiggle), "low": c * (1 - wiggle),
         "close": c, "volume": rng.uniform(1, 5, n)}, index=idx)


def test_the_example_loads_under_the_real_import_hook(example):
    """It must survive the same restricted namespace generated code runs in."""
    assert isinstance(example, Strategy)
    assert example.name


def test_the_example_runs_and_returns_a_usable_series(example, bars):
    res = example.generate_signals(bars)
    assert len(res.positions) == len(bars)
    assert res.positions.between(0, 1).all()


def test_the_example_declares_a_risk_overlay(example, bars):
    res = example.generate_signals(bars)
    assert res.stop_loss_pct is not None and res.take_profit_pct is not None


def test_the_example_is_lookahead_free(example, bars):
    """It teaches the rolling-quantile pattern; a whole-sample quantile there
    would be lookahead and the agent would copy it."""
    report = lookahead_check(bars, example)
    assert report["passed"], report["violations"]


def test_the_example_holds_positions_so_the_stop_can_actually_fire(example, bars):
    res = backtest(bars, example, fee_pct=0.001, slippage_pct=0.0005)
    assert res.meta["risk_managed"] is True
    assert res.meta["avg_bars_held"] >= 5, "positions collapse before the levels matter"
    exits = res.meta["exit_counts"]
    level_exits = exits.get("stop", 0) + exits.get("target", 0)
    assert level_exits > exits.get("signal", 0), \
        f"most exits should come from the risk levels, got {exits}"


def test_the_example_lands_inside_the_configured_trade_band(example, bars):
    """If the demonstration itself would be rejected for overtrading, it is
    teaching the wrong calibration."""
    from algotrader.config import get, load_config
    cfg = load_config()
    res = backtest(bars, example, fee_pct=0.001, slippage_pct=0.0005)
    assert res.meta["trades_per_day"] <= get(cfg, "validation.max_trades_per_day")


def test_the_example_uses_a_rolling_quantile_not_a_whole_sample_one():
    code = re.search(r"```python\s*(.*?)```", WORKED_EXAMPLE, re.DOTALL).group(1)
    assert ".rolling(" in code and ".quantile(" in code
    assert not re.search(r"(?<!\))\.quantile\(", code.replace(".rolling(self.rank_len, min_periods=500).quantile(", "OK(")), \
        "a bare .quantile() over the whole series is lookahead"


def test_every_parameter_has_a_default(example):
    """The framework instantiates with no arguments; a required parameter is a
    guaranteed FAIL codegen, and the example is what gets copied."""
    import inspect
    sig = inspect.signature(type(example).__init__)
    required = [n for n, p in sig.parameters.items()
                if n not in ("self", "kwargs") and p.default is inspect.Parameter.empty
                and p.kind is not inspect.Parameter.VAR_KEYWORD]
    assert not required, f"parameters without defaults: {required}"
