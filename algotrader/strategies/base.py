"""The Strategy contract.

Every strategy — hand-written or LLM-generated — implements `generate_signals`,
which returns a *target position* series aligned to the input bars.

    position[t] in [-1, 1]   (fraction of capital; negative = short)

CRITICAL — avoiding lookahead bias:
  The position for bar t must be computable using data up to and INCLUDING bar t.
  The backtester enters that position at the NEXT bar's open (t+1), so signals may
  use close[t]. Never use any information from t+1 or later. The validation layer
  runs an automated shift-test to catch violations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class StrategyResult:
    """A strategy's output: target positions plus optional debug/indicator data."""

    positions: pd.Series               # index-aligned to input, values in [-1, 1]
    indicators: dict[str, pd.Series] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)


class Strategy:
    """Base class. Subclass and implement generate_signals().

    Params are passed as kwargs and stored on self for introspection/logging.
    """

    name: str = "base"

    def __init__(self, **params: Any):
        self.params = params
        for k, v in params.items():
            setattr(self, k, v)

    def generate_signals(self, df: pd.DataFrame) -> StrategyResult:
        """Return target positions for each bar. Must be lookahead-free.

        Args:
            df: OHLCV DataFrame (columns: open, high, low, close, volume).
        """
        raise NotImplementedError

    def describe(self) -> dict:
        return {"name": self.name, "params": self.params}


def coerce_positions(positions, index: pd.Index) -> pd.Series:
    """Normalize whatever a strategy returned into a float Series on `index`.

    Generated strategies sometimes return a numpy array, a list, or a Series with
    a different index. This makes the rest of the pipeline robust to that.
    """
    if isinstance(positions, pd.Series):
        s = positions.reindex(index)
    else:
        arr = pd.array(positions) if not hasattr(positions, "__len__") else positions
        if len(arr) != len(index):
            raise ValueError(
                f"positions length {len(arr)} != number of bars {len(index)}"
            )
        s = pd.Series(list(arr), index=index)
    return s.astype(float).fillna(0.0)
