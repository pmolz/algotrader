"""Adversarial tests for the lookahead detector.

This is the gate that decides whether a "discovered edge" is real, so it is
tested against strategies that actively cheat rather than only against honest
ones. Every cheat below is a pattern LLM-generated strategies actually produce.

The two properties that matter, in order:
  * NO FALSE NEGATIVES — a peeking strategy must never pass. A missed cheat
    becomes a promoted strategy with a fake edge.
  * NO FALSE POSITIVES — an honest strategy must never be flagged. Those cost
    iterations, and worse, they poison `lessons_context` with bad advice.

The sparse-signal cases exist because of a real regression: an earlier
truncation-only implementation caught a dense `shift(-1)` cheat 99% of the time
and a sparse one 4% of the time, because truncation only perturbs the last k
bars of the overlap. Keep these.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from algotrader.strategies.base import Strategy, StrategyResult
from algotrader.validation.lookahead import lookahead_check


def make_df(seed: int = 0, n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
    close = 100 * np.exp(np.linspace(0, 0.5, n) + rng.normal(0, 0.02, n).cumsum())
    return pd.DataFrame(
        {"open": close, "high": close * 1.01, "low": close * 0.99,
         "close": close, "volume": rng.uniform(1e3, 5e3, n)},
        index=idx,
    )


def strat(fn) -> Strategy:
    """Wrap a signal function into a Strategy."""
    return type("S", (Strategy,), {
        "name": "s",
        "generate_signals": lambda self, d, _f=fn: StrategyResult(positions=_f(d)),
    })()


# -- honest strategies: must never be flagged --------------------------------
HONEST = {
    "sma_crossover":
        lambda d: (d.close.rolling(10).mean() > d.close.rolling(50).mean()).astype(float),
    "expanding_mean":
        lambda d: (d.close > d.close.expanding().mean()).astype(float),
    "momentum":
        lambda d: (d.close.pct_change().rolling(14).mean() > 0).astype(float),
    "always_long":
        lambda d: pd.Series(1.0, index=d.index),
    "always_flat":
        lambda d: pd.Series(0.0, index=d.index),
    "trailing_breakout":
        lambda d: (d.close >= d.close.rolling(20).max()).astype(float),
    "vol_scaled":
        lambda d: (1 / (1 + d.close.pct_change().rolling(20).std())).fillna(0),
    "uses_volume":
        lambda d: (d.volume > d.volume.rolling(20).mean()).astype(float),
    "intrabar_range":
        lambda d: ((d.high - d.low) / d.close > 0.015).astype(float),
}


@pytest.mark.parametrize("name", sorted(HONEST))
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_honest_strategies_are_not_flagged(name, seed):
    report = lookahead_check(make_df(seed), strat(HONEST[name]))
    assert report["passed"], f"false positive on {name}: {report['violations']}"


# -- cheats: must always be caught -------------------------------------------
CHEATS = {
    # bounded-horizon peeking — the most common LLM mistake
    "shift_minus_1_dense":
        lambda d: (d.close.shift(-1) > d.close).astype(float),
    "shift_minus_1_threshold_2pct":
        lambda d: ((d.close.shift(-1) / d.close - 1) > 0.025).astype(float),
    "shift_minus_1_threshold_5pct":     # ~1% of bars; the case that used to slip
        lambda d: ((d.close.shift(-1) / d.close - 1) > 0.05).astype(float),
    "shift_minus_10":
        lambda d: ((d.close.shift(-10) / d.close - 1) > 0.05).astype(float),
    "future_high":
        lambda d: (d.high.shift(-1) > d.close * 1.01).astype(float),
    # whole-sample statistics
    "global_max_normalisation":
        lambda d: (d.close / d.close.max() > 0.8).astype(float),
    "full_sample_zscore":
        lambda d: ((d.close - d.close.mean()) / d.close.std() > 0.5).astype(float),
    "global_rank":
        lambda d: (d.close.rank(pct=True) > 0.7).astype(float),
    "centered_rolling":
        lambda d: (d.close > d.close.rolling(21, center=True).mean()).astype(float),
    "reverse_cummax":                   # "did it ever rise 10% from here?"
        lambda d: (d.close.iloc[::-1].cummax().iloc[::-1] / d.close > 1.1).astype(float),
}


@pytest.mark.parametrize("name", sorted(CHEATS))
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_cheating_strategies_are_caught(name, seed):
    report = lookahead_check(make_df(seed), strat(CHEATS[name]))
    assert not report["passed"], f"MISSED lookahead in {name}"
    assert report["n_violations"] >= 1


def test_a_sparse_cheat_is_caught_on_every_seed():
    """Detection must not depend on how often the strategy trades.

    The old implementation's hit rate was ~= 1 - (1 - signal_density)^n_probes,
    so a high-conviction strategy that acts on 1% of bars sailed through.
    """
    fn = CHEATS["shift_minus_1_threshold_5pct"]
    caught = sum(not lookahead_check(make_df(s), strat(fn))["passed"] for s in range(20))
    assert caught == 20, f"only caught {caught}/20"


def test_violations_name_the_probe_that_fired():
    report = lookahead_check(make_df(0), strat(CHEATS["shift_minus_1_dense"]))
    probes = {v["probe"] for v in report["violations"]}
    assert probes & {"future_x10", "future_x0.1", "future_frozen", "truncate"}


def test_nondeterministic_strategy_is_reported_as_such_not_as_peeking():
    """Randomness would otherwise look identical to lookahead. Say which it is."""
    rng = np.random.default_rng()
    noisy = strat(lambda d: pd.Series(rng.random(len(d)), index=d.index))
    report = lookahead_check(make_df(0), noisy)
    assert not report["passed"]
    assert report["violations"][0]["probe"] == "determinism"


def test_report_is_bounded_for_the_experiment_log():
    """Violations get inlined into the reasons string stored in SQLite."""
    report = lookahead_check(make_df(0), strat(CHEATS["global_rank"]))
    assert len(report["violations"]) <= 5
    assert report["n_violations"] >= len(report["violations"])


def test_short_series_is_skipped_rather_than_crashing():
    report = lookahead_check(make_df(0, n=5), strat(HONEST["always_long"]))
    assert report["passed"] and report["checked"] == 0


# -- regressions found by re-checking against real stored strategies ----------
def test_integer_volume_column_does_not_crash_the_probes():
    """yfinance ships int64 volume. Writing perturbed floats into an int column
    raises, and the gauntlet then mislabels the crash as 'FAIL strategy runtime'
    — blaming the strategy for a bug in the detector."""
    df = make_df(0)
    df["volume"] = df["volume"].astype("int64")
    report = lookahead_check(df, strat(HONEST["uses_volume"]))
    assert report["passed"] and report["checked"] > 0


def test_integer_volume_still_catches_a_cheat():
    df = make_df(0)
    df["volume"] = df["volume"].astype("int64")
    cheat = strat(lambda d: (d.volume.shift(-1) > d.volume).astype(float))
    assert not lookahead_check(df, cheat)["passed"]


def test_strategy_that_mutates_its_input_is_not_falsely_flagged():
    """Generated strategies often assign working columns into the frame they are
    handed. If probes share that frame, one call contaminates the next and an
    honest strategy looks like it is peeking."""
    def messy(d):
        d["tmp_ma"] = d.close.rolling(10).mean()      # writes into the caller's frame
        d["tmp_flag"] = d.close > d["tmp_ma"]         # ...and a bool column
        return d["tmp_flag"].astype(float)

    report = lookahead_check(make_df(0), strat(messy))
    assert report["passed"], f"false positive: {report['violations']}"


def test_mutating_strategy_that_also_cheats_is_still_caught():
    def messy_cheat(d):
        d["tmp"] = d.close.shift(-1)
        return (d["tmp"] > d.close).astype(float)

    assert not lookahead_check(make_df(0), strat(messy_cheat))["passed"]
