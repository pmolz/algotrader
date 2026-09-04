"""Automated lookahead-bias detector.

Idea: a lookahead-free strategy's signal at bar t must not depend on data after t.
We test this by truncating the data at various points and checking that the signal
values for the earlier bars are IDENTICAL whether or not future bars were present.

If appending future bars changes past signals, the strategy is peeking. This is the
#1 way LLM-generated strategies produce fake edges, so we run it on every candidate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..strategies.base import Strategy


def lookahead_check(
    df: pd.DataFrame,
    strategy: Strategy,
    probe_points: int = 5,
    atol: float = 1e-8,
) -> dict:
    """Return {'passed': bool, 'violations': [...], 'checked': int}.

    A violation means the signal at some past bar changed when future data was
    revealed — i.e. the strategy used information it shouldn't have.
    """
    full = strategy.generate_signals(df).positions.reindex(df.index).fillna(0.0).to_numpy()
    n = len(df)
    violations = []
    checked = 0

    # probe at several truncation points across the second half of the series
    starts = np.linspace(int(n * 0.5), n - 1, probe_points, dtype=int)
    for cut in sorted(set(int(c) for c in starts)):
        if cut < 2:
            continue
        truncated = df.iloc[:cut]
        trunc_sig = (
            strategy.generate_signals(truncated)
            .positions.reindex(truncated.index)
            .fillna(0.0)
            .to_numpy()
        )
        checked += 1
        # compare overlapping region [0, cut)
        diff = np.abs(trunc_sig - full[:cut])
        bad = np.where(diff > atol)[0]
        if len(bad):
            violations.append(
                {"cut": int(cut), "n_changed_bars": int(len(bad)), "first_bar": int(bad[0])}
            )

    return {"passed": len(violations) == 0, "violations": violations, "checked": checked}
