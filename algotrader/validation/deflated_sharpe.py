"""Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014).

When you test many strategies, the best one looks great by luck alone. The
Deflated Sharpe Ratio (DSR) corrects the observed Sharpe for:
  * the number of independent trials you ran (multiple testing),
  * non-normal returns (skew and kurtosis),
  * the length of the track record.

DSR is the probability that the TRUE Sharpe is > 0 given all of the above.
Rule of thumb: require DSR > 0.95 before believing an edge is real. This single
number is your best defence against the agent fooling itself.
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
    true edge — the benchmark the observed Sharpe must beat."""
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
) -> dict:
    """Compute DSR for a return series.

    Args:
        returns:      per-bar strategy returns.
        n_trials:     number of strategy configurations tried to find this one.
        sr_benchmark: expected max Sharpe under the null; if None, estimated from
                      n_trials assuming unit variance of trial Sharpes.
    Returns dict with observed per-period Sharpe, benchmark, and DSR probability.
    """
    r = returns.to_numpy() if isinstance(returns, pd.Series) else np.asarray(returns)
    sr, skew, kurt, n = _sharpe_stats(r)
    if n < 3:
        return {"dsr": 0.0, "sharpe_periodic": 0.0, "sr_benchmark": 0.0, "n": n}

    if sr_benchmark is None:
        sr_benchmark = expected_max_sharpe(n_trials)

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
