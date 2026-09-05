"""Tests for the validation gauntlet — especially the lookahead detector, which
is the single most valuable safety mechanism in the repo.
"""

import pandas as pd

from algotrader.strategies.base import Strategy, StrategyResult
from algotrader.strategies.baselines import SmaCrossover
from algotrader.validation import (
    deflated_sharpe_ratio,
    lookahead_check,
    train_oos_split,
    walk_forward_splits,
)


class CheatingStrategy(Strategy):
    """Deliberately peeks at the future — the gauntlet MUST catch this."""

    name = "cheater"

    def generate_signals(self, df):
        future_ret = df["close"].shift(-1) / df["close"] - 1  # uses t+1 — illegal
        pos = (future_ret > 0).astype(float)
        return StrategyResult(positions=pos)


def test_lookahead_detects_cheater(trending_ohlcv):
    result = lookahead_check(trending_ohlcv, CheatingStrategy())
    assert result["passed"] is False
    assert len(result["violations"]) > 0


def test_lookahead_passes_honest_strategy(trending_ohlcv):
    result = lookahead_check(trending_ohlcv, SmaCrossover())
    assert result["passed"] is True


def test_train_oos_split_is_chronological(trending_ohlcv):
    dev, oos = train_oos_split(trending_ohlcv, oos_frac=0.2)
    assert len(oos) == int(len(trending_ohlcv) * 0.2)
    assert dev.index.max() < oos.index.min()  # no time overlap/leak


def test_walk_forward_folds_are_forward_only(trending_ohlcv):
    for train, test in walk_forward_splits(trending_ohlcv, n_splits=4):
        assert train.index.max() < test.index.min()


def test_deflated_sharpe_penalizes_many_trials(trending_ohlcv):
    ret = trending_ohlcv["close"].pct_change().dropna()
    one = deflated_sharpe_ratio(ret, n_trials=1)["dsr"]
    many = deflated_sharpe_ratio(ret, n_trials=1000)["dsr"]
    assert many <= one  # more trials -> harder to believe the edge is real


# -- trading intensity (intraday) ---------------------------------------------
# At 15m bars a round trip pays the spread twice, so trade frequency decides
# whether an idea can exist at all. These failures also become `lessons_context`,
# so "never traded" and "lost money" must not collapse into one message.
def _intraday_frame(n=3000, seed=0):
    import numpy as np
    import pandas as pd
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-01-01", periods=n, freq="15min", tz="UTC")
    c = 100 * np.exp(rng.normal(0, 0.003, n).cumsum())
    return pd.DataFrame({"open": c, "high": c * 1.004, "low": c * 0.996,
                         "close": c, "volume": rng.uniform(1, 5, n)}, index=idx)


def _strat(fn, **risk):
    from algotrader.strategies.base import Strategy, StrategyResult
    return type("S", (Strategy,), {
        "name": "s",
        "generate_signals": lambda self, d, _f=fn: StrategyResult(positions=_f(d), **risk),
    })()


def _cfg():
    from algotrader.config import load_config
    return load_config()


def test_a_strategy_that_never_trades_says_so_rather_than_reporting_a_sharpe():
    import pandas as pd
    from algotrader.validation.gauntlet import run_gauntlet
    rep = run_gauntlet(_intraday_frame(), _strat(lambda d: pd.Series(0.0, index=d.index)),
                       _cfg(), n_trials=1)
    assert not rep.promoted
    assert any("never traded" in r for r in rep.reasons), rep.reasons
    assert not any("walk-forward" in r for r in rep.reasons), \
        "a no-trade candidate must not be labelled a losing one"


def test_overtrading_is_rejected_with_the_cost_arithmetic():
    import pandas as pd
    from algotrader.validation.gauntlet import run_gauntlet
    # alternate every bar -> ~48 round trips a day
    def churn(d):
        return pd.Series([float(i % 2) for i in range(len(d))], index=d.index)
    rep = run_gauntlet(_intraday_frame(), _strat(churn), _cfg(), n_trials=1)
    assert not rep.promoted
    assert any("overtrading" in r for r in rep.reasons), rep.reasons
    assert rep.checks["trading_intensity"]["trades_per_day"] > 4.0


def test_trading_intensity_is_recorded_even_when_the_candidate_passes_it():
    from algotrader.validation.gauntlet import run_gauntlet
    def sometimes(d):
        return (d.close > d.close.rolling(200).mean()).astype(float)
    rep = run_gauntlet(_intraday_frame(), _strat(sometimes), _cfg(), n_trials=1)
    ti = rep.checks["trading_intensity"]
    assert "trades_per_day" in ti and "cost_drag_annual" in ti
    assert ti["cost_per_round_trip"] > 0
