"""Performance metrics. Keep these standard and well-tested — the whole system's
decisions rest on them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_metrics(
    returns: pd.Series,
    equity: pd.Series,
    positions: pd.Series,
    periods_per_year: int,
) -> dict:
    r = returns.fillna(0.0).to_numpy()
    n = len(r)
    if n == 0:
        return _empty()

    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    years = n / periods_per_year
    cagr = float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1.0) if years > 0 else 0.0

    mean = r.mean()
    std = r.std(ddof=1) if n > 1 else 0.0
    sharpe = float(mean / std * np.sqrt(periods_per_year)) if std > 0 else 0.0

    downside = r[r < 0]
    dstd = downside.std(ddof=1) if len(downside) > 1 else 0.0
    sortino = float(mean / dstd * np.sqrt(periods_per_year)) if dstd > 0 else 0.0

    max_dd = _max_drawdown(equity)
    calmar = float(cagr / abs(max_dd)) if max_dd < 0 else 0.0

    # trade-level win rate approximated on bars with active exposure
    active = r[positions.to_numpy() != 0.0] if len(positions) == len(r) else r
    win_rate = float((active > 0).mean()) if len(active) else 0.0

    exposure = float((positions.to_numpy() != 0.0).mean()) if len(positions) else 0.0

    return {
        "total_return": total_return,
        "cagr": cagr,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "max_drawdown": float(max_dd),
        "win_rate": win_rate,
        "volatility_annual": float(std * np.sqrt(periods_per_year)),
        "exposure": exposure,
        "n_bars": n,
    }


def _max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min()) if len(dd) else 0.0


def _empty() -> dict:
    return {
        "total_return": 0.0, "cagr": 0.0, "sharpe": 0.0, "sortino": 0.0,
        "calmar": 0.0, "max_drawdown": 0.0, "win_rate": 0.0,
        "volatility_annual": 0.0, "exposure": 0.0, "n_bars": 0,
    }
