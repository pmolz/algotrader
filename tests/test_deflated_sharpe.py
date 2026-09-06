"""The multiple-testing gate, and the units bug that made it unpassable.

`deflated_sharpe_ratio` works in PER-BAR Sharpes, but the benchmark it compared
against was built with `sr_variance=1.0` — the variance of trial Sharpes if a
typical trial scattered by a whole unit of Sharpe *per bar*. At n_trials=747
that put the bar at 3.17 per bar, an annualised Sharpe near 600, so DSR was 0
for every input and no candidate could ever clear stage 5.

It went unnoticed because it is latent: exactly 1 of 747 logged experiments ever
reached this stage — walk-forward and cost stress kill everything before it — so
the gate had almost no opportunity to be wrong out loud. These tests pin the
units, because a promotion gate that always fails looks exactly like a search
that has not found anything yet.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from algotrader.validation.deflated_sharpe import (
    deflated_sharpe_ratio,
    expected_max_sharpe,
)

BARS_PER_YEAR = 35040          # 15m crypto bars
N = 28032                      # a dev set after a 20% holdout on two years


def returns_with_annual_sharpe(sharpe: float, n: int = N, seed: int = 0) -> pd.Series:
    """A return series with an exactly known annualised Sharpe."""
    rng = np.random.default_rng(seed)
    r = pd.Series(rng.standard_normal(n))
    r = (r - r.mean()) / r.std(ddof=1) * 0.001
    return r + 0.001 * sharpe / np.sqrt(BARS_PER_YEAR)


def annualised_bar(n_trials: int, n: int = N) -> float:
    """The annualised Sharpe the benchmark corresponds to, for readability."""
    return expected_max_sharpe(n_trials, 1.0 / n) * np.sqrt(BARS_PER_YEAR)


# -- the units -------------------------------------------------------------------

def test_benchmark_is_a_reachable_annualised_sharpe():
    """The regression. Before the fix this was ~594, so nothing could pass."""
    assert 3.0 < annualised_bar(747) < 5.0


@pytest.mark.parametrize("n_trials", [40, 747, 5600])
def test_a_strong_edge_passes_and_a_weak_one_does_not(n_trials):
    bar = annualised_bar(n_trials)
    assert deflated_sharpe_ratio(returns_with_annual_sharpe(bar * 2), n_trials=n_trials)["dsr"] > 0.5
    assert deflated_sharpe_ratio(returns_with_annual_sharpe(bar / 2), n_trials=n_trials)["dsr"] < 0.5


def test_pure_noise_is_rejected_however_many_bars():
    """The gate's actual job: a track record with no edge must not clear it."""
    for n in (5_000, N, 70_080):
        r = returns_with_annual_sharpe(0.0, n=n, seed=n)
        assert deflated_sharpe_ratio(r, n_trials=747)["dsr"] < 0.05


def test_the_bar_rises_with_trials_but_not_explosively():
    """More trials must cost you something, and not so much that the search is
    pointless: 7x the trials should not double what a candidate has to clear."""
    few, many = annualised_bar(800), annualised_bar(5600)
    assert few < many < 2 * few


def test_variance_defaults_to_the_sampling_variance_of_the_estimator():
    d = deflated_sharpe_ratio(returns_with_annual_sharpe(1.0), n_trials=100)
    assert d["sr_variance"] == pytest.approx(1.0 / d["n"])


def test_an_explicit_variance_overrides_the_default():
    """The default is the LOWER bound on trial scatter. A caller with a real
    estimate of the null spread must be able to make the gate stricter."""
    r = returns_with_annual_sharpe(annualised_bar(747) * 1.5)
    assert deflated_sharpe_ratio(r, n_trials=747)["dsr"] > 0.5
    strict = deflated_sharpe_ratio(r, n_trials=747, sr_variance=100.0 / N)
    assert strict["dsr"] < 0.5


def test_an_explicit_benchmark_still_wins():
    d = deflated_sharpe_ratio(returns_with_annual_sharpe(1.0), n_trials=747,
                              sr_benchmark=0.0)
    assert d["sr_benchmark"] == 0.0
    assert d["dsr"] > 0.5      # no correction applied, so a real Sharpe passes


# -- longer track records are worth more -----------------------------------------

def test_more_bars_make_the_same_sharpe_more_believable():
    """Sharpe held fixed, only the length of the record changes. This is the
    part `sr_variance=1.0` destroyed: with a constant benchmark, track-record
    length only entered through `se` and the trade-off was incoherent."""
    sharpe = annualised_bar(747) * 1.1
    short = deflated_sharpe_ratio(returns_with_annual_sharpe(sharpe, n=7_000, seed=1),
                                  n_trials=747)["dsr"]
    long_ = deflated_sharpe_ratio(returns_with_annual_sharpe(sharpe, n=70_080, seed=1),
                                  n_trials=747)["dsr"]
    assert long_ > short
