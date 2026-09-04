"""Hand-written baseline strategies.

These exist to (a) exercise the whole pipeline and (b) give the agent honest
reference points. Do not expect them to be profitable after costs — that's the
point. If your gauntlet says a naive SMA crossover is a money printer, the
gauntlet is broken.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Strategy, StrategyResult
from .registry import register


@register
class SmaCrossover(Strategy):
    """Long when fast SMA > slow SMA, flat otherwise. The canonical baseline."""

    name = "sma_crossover"

    def __init__(self, fast: int = 20, slow: int = 50, **kw):
        super().__init__(fast=fast, slow=slow, **kw)

    def generate_signals(self, df: pd.DataFrame) -> StrategyResult:
        close = df["close"]
        fast = close.rolling(self.fast).mean()
        slow = close.rolling(self.slow).mean()
        pos = (fast > slow).astype(float)      # 1.0 long / 0.0 flat, computed at close[t]
        pos[slow.isna()] = 0.0                 # no position until slow SMA is defined
        return StrategyResult(
            positions=pos,
            indicators={"sma_fast": fast, "sma_slow": slow},
        )


@register
class RsiMeanReversion(Strategy):
    """Buy oversold, exit on recovery. Classic mean-reversion baseline."""

    name = "rsi_mean_reversion"

    def __init__(self, period: int = 14, oversold: float = 30, exit_level: float = 50, **kw):
        super().__init__(period=period, oversold=oversold, exit_level=exit_level, **kw)

    def _rsi(self, close: pd.Series) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(self.period).mean()
        loss = (-delta.clip(upper=0)).rolling(self.period).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - 100 / (1 + rs)

    def generate_signals(self, df: pd.DataFrame) -> StrategyResult:
        rsi = self._rsi(df["close"])
        pos = pd.Series(0.0, index=df.index)
        holding = False
        # stateful: enter when oversold, exit when recovered. Uses only past/current data.
        vals = rsi.to_numpy()
        out = np.zeros(len(vals))
        for i, r in enumerate(vals):
            if np.isnan(r):
                out[i] = 0.0
                continue
            if not holding and r < self.oversold:
                holding = True
            elif holding and r > self.exit_level:
                holding = False
            out[i] = 1.0 if holding else 0.0
        pos = pd.Series(out, index=df.index)
        return StrategyResult(positions=pos, indicators={"rsi": rsi})


@register
class BuyAndHold(Strategy):
    """The benchmark every strategy must beat on a risk-adjusted basis."""

    name = "buy_and_hold"

    def generate_signals(self, df: pd.DataFrame) -> StrategyResult:
        return StrategyResult(positions=pd.Series(1.0, index=df.index))
