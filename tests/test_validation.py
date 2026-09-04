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
