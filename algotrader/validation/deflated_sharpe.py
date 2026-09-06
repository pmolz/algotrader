"""Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014).

When you test many strategies, the best one looks great by luck alone. The
Deflated Sharpe Ratio (DSR) corrects the observed Sharpe for:
  * the number of independent trials you ran (multiple testing),
  * non-normal returns (skew and kurtosis),
  * the length of the track record.

DSR is the probability that the TRUE Sharpe is > 0 given all of the above.
Rule of thumb: require DSR > 0.95 before believing an edge is real. This single
number is your best defence against the agent fooling itself.

WHERE V[SR] COMES FROM, AND WHY IT IS NOT THE OBSERVED SPREAD
Bailey & Lopez de Prado scale the benchmark by sqrt(V[SR]), the variance of the
Sharpes ACROSS the trials. That is the right quantity when the trials are draws
from a null with no edge. This project's trial population is not that: it is
dominated by transaction-cost drag, so the Sharpes are centred far below zero
with a long left tail from overtrading candidates. Measured on a 747-experiment
log: the 504 candidates with a dev Sharpe had mean -8.0 and sd 15.9 annualised
(per-bar sd 0.085); even the 104 that reached walk-forward had mean -13.1.
Those are not null draws, they are arithmetically doomed strategies, and using
their spread puts the benchmark at an annualised Sharpe near 50 -- unreachable,
which is what made this gate unpassable.

So V[SR] defaults to the SAMPLING variance of the per-bar Sharpe estimator under
the null, 1/n over an n-bar track record. The comparison then reduces to a clean
question: does this candidate's Sharpe t-statistic exceed the expected maximum of
n_trials standard normals? Be clear about the direction of the assumption -- 1/n
is the LOWER bound on how much trial Sharpes can scatter, because real trials
differ by more than sampling noise (holding period, trade rate, leverage). This
default is therefore the least conservative honest choice, and a PASS here is
"not obviously luck", not "definitely an edge". Pass an explicit `sr_variance`
if you can estimate the null spread of a comparable trial population.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def norm_cdf(x: float) -> float:
    """Standard normal CDF via erf; avoids a scipy dependency."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _sharpe_stats(returns: np.ndarray):
    r = returns[~np.isnan(returns)]
    n = len(r)
    if n < 3 or r.std(ddof=1) == 0:
        return 0.0, 0.0, 0.0, n
    sr = r.mean() / r.std(ddof=1)         # per-period (non-annualized) Sharpe
    skew = float(((r - r.mean()) ** 3).mean() / r.std(ddof=0) ** 3)
    kurt = float(((r - r.mean()) ** 4).mean() / r.std(ddof=0) ** 4)  # non-excess
    return sr, skew, kurt, n


def expected_max_sharpe(n_trials: int, sr_variance: float = 1.0) -> float:
    """Expected maximum Sharpe from n_trials independent strategies with zero
    true edge — the benchmark the observed Sharpe must beat.

    `sr_variance` must be in the same units as the Sharpe it will be compared
    against. `deflated_sharpe_ratio` works in per-bar Sharpes, so passing the
    default 1.0 there compares a per-bar number against a benchmark scaled for
    an annualised one and rejects everything; see the module docstring.
    """
    if n_trials < 2:
        return 0.0
    e = 0.5772156649  # Euler-Mascheroni
    z1 = _inv_norm(1 - 1.0 / n_trials)
    z2 = _inv_norm(1 - 1.0 / (n_trials * np.e))
    return float(np.sqrt(sr_variance) * ((1 - e) * z1 + e * z2))


def deflated_sharpe_ratio(
    returns: pd.Series,
    n_trials: int = 1,
    sr_benchmark: float | None = None,
    sr_variance: float | None = None,
) -> dict:
    """Compute DSR for a return series.

    Args:
        returns:      per-bar strategy returns.
        n_trials:     number of strategy configurations tried to find this one.
        sr_benchmark: expected max PER-BAR Sharpe under the null. If None it is
                      derived from n_trials and sr_variance.
        sr_variance:  variance of per-bar trial Sharpes under the null. If None,
                      the sampling variance 1/n of the Sharpe estimator over this
                      track record — see the module docstring for why the
                      observed cross-trial spread is not used here.
    Returns dict with observed per-period Sharpe, benchmark, and DSR probability.
    """
    r = returns.to_numpy() if isinstance(returns, pd.Series) else np.asarray(returns)
    sr, skew, kurt, n = _sharpe_stats(r)
    if n < 3:
        return {"dsr": 0.0, "sharpe_periodic": 0.0, "sr_benchmark": 0.0, "n": n}

    if sr_benchmark is None:
        # 1/n, not 1.0: `sr` below is a per-bar Sharpe, and a benchmark built for
        # unit-variance trial Sharpes sits ~3.2 per bar — an annualised Sharpe in
        # the hundreds, which no candidate can reach and nothing ever passed.
        if sr_variance is None:
            sr_variance = 1.0 / n
        sr_benchmark = expected_max_sharpe(n_trials, sr_variance)

    # standard error of the Sharpe estimator with higher-moment adjustment
    denom = np.sqrt(max(1e-12, 1 - skew * sr + (kurt - 1) / 4 * sr**2))
    se = denom / np.sqrt(n - 1)
    dsr = norm_cdf((sr - sr_benchmark) / se) if se > 0 else 0.0

    return {
        "dsr": float(dsr),
        "sharpe_periodic": float(sr),
        "sr_benchmark": float(sr_benchmark),
        "skew": skew,
        "kurtosis": kurt,
        "n": n,
        "n_trials": n_trials,
        "sr_variance": float(sr_variance) if sr_variance is not None else None,
    }


# --- minimal normal CDF / inverse without a scipy hard dependency ---------------
def _inv_norm(p: float) -> float:
    """Acklam's rational approximation to the inverse normal CDF."""
    if p <= 0:
        return -np.inf
    if p >= 1:
        return np.inf
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = np.sqrt(-2 * np.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = np.sqrt(-2 * np.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
