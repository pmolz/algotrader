"""Automated lookahead-bias detector.

A lookahead-free strategy's signal at bar t depends only on data up to and
including t. We test that with two complementary probes at each of several cut
points, because they fail in different ways:

1. TRUNCATION.  Cut the data at `cut` and re-run. Signals on the overlapping
   region must be unchanged. This catches whole-sample statistics — `close.mean()`,
   `.max()`, `.rank()`, `.std()`, centered rolling windows — because dropping the
   tail moves the statistic and shifts signals everywhere.

2. FUTURE PERTURBATION.  Keep the length, but replace every bar from `cut`
   onward with something wildly different (x10, x0.1, frozen). Signals *before*
   `cut` must be bit-identical. This catches bounded-horizon peeking — `.shift(-k)`
   and friends.

Why both. Truncation alone is close to useless against `.shift(-k)`: for a k-bar
lookahead, dropping the tail only changes bars in [cut-k, cut) — everything
earlier is already fully determined by data inside the window. So each truncation
probe effectively inspects k bars, and since `coerce_positions` fills the
resulting NaN with 0.0, even those only differ when the true signal there is
non-zero. Measured on random series, truncation-only detection tracked
`1 - (1-signal_density)^n_probes`: it caught a dense `shift(-1)` cheat 99% of the
time and a sparse one 4% of the time. It was measuring how often the strategy
traded, not whether it peeked.

Perturbation has no such blind spot. If a strategy is lookahead-free then mutating
the future cannot change the past — that is a guarantee, not a heuristic — so any
difference is proof of peeking and honest strategies cannot be flagged. The one
way to get a spurious violation is a `generate_signals` that isn't a pure function
of its input, so we check determinism first and report that separately.

This is the #1 way LLM-generated strategies produce fake edges, so it runs on
every candidate before anything else in the gauntlet.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from ..strategies.base import Strategy, coerce_positions

# Mutations applied to the future half of the frame. Scaling every column by the
# same factor keeps OHLC internally consistent (high >= low still holds), so a
# violation means peeking rather than the strategy choking on impossible bars.
_MUTATIONS: dict[str, Callable] = {
    "future_x10": lambda block: block * 10.0,
    "future_x0.1": lambda block: block * 0.1,
    # Flat future: kills any variation a peeking strategy might key on, and
    # catches cheats that survive a monotone rescale (ratios, ranks, z-scores).
    "future_frozen": lambda block: block.iloc[[0]].reindex(block.index, method="ffill"),
}

_MAX_REPORTED = 5


def _signals(strategy: Strategy, df: pd.DataFrame) -> np.ndarray:
    # Hand every call its own copy. Generated strategies not infrequently assign
    # working columns into the frame they are given; without this, one probe
    # mutates the input for the next and the difference reads as a violation.
    return coerce_positions(
        strategy.generate_signals(df.copy()).positions, df.index
    ).to_numpy()


def lookahead_check(
    df: pd.DataFrame,
    strategy: Strategy,
    probe_points: int = 12,
    atol: float = 1e-8,
) -> dict:
    """Return {'passed', 'violations', 'checked', 'n_violations'}.

    A violation means the signal at some bar changed in response to data at a
    later bar — i.e. the strategy used information it could not have had.
    """
    n = len(df)
    if n < 8:
        return {"passed": True, "violations": [], "checked": 0, "n_violations": 0,
                "skipped": "series too short to probe"}

    # Columns to perturb, decided before any strategy call so a strategy that
    # adds columns can't widen the set. Bools are excluded: scaling them is
    # meaningless. Widen to float once, up front, so the mutated values can
    # actually be written back — yfinance ships integer volume, and assigning
    # floats into an int64 column raises.
    numeric = [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])
    ]
    df = df.copy()
    if numeric:
        df[numeric] = df[numeric].astype(float)

    full = _signals(strategy, df)

    # A strategy that isn't a pure function of its input would produce
    # differences that look exactly like peeking. Say so instead of guessing.
    if np.abs(_signals(strategy, df) - full).max() > atol:
        return {
            "passed": False, "checked": 0, "n_violations": 1,
            "violations": [{"probe": "determinism",
                            "reason": "generate_signals is not deterministic"}],
        }

    cuts = sorted({int(c) for c in np.linspace(max(2, int(n * 0.2)), n - 2, probe_points)})

    violations: list[dict] = []
    checked = 0

    for cut in cuts:
        if cut < 2 or cut >= n:
            continue

        # -- probe 1: truncation invariance ------------------------------------
        truncated = df.iloc[:cut]
        checked += 1
        bad = np.where(np.abs(_signals(strategy, truncated) - full[:cut]) > atol)[0]
        if len(bad):
            violations.append({"cut": cut, "probe": "truncate",
                               "n_changed_bars": len(bad),
                               "first_bar": int(bad[0])})

        # -- probe 2: the future cannot change the past ------------------------
        for label, mutate in _MUTATIONS.items():
            mutated = df.copy()
            block = mutated.iloc[cut:][numeric]
            mutated.iloc[cut:, [mutated.columns.get_loc(c) for c in numeric]] = (
                mutate(block).to_numpy()
            )
            checked += 1
            bad = np.where(np.abs(_signals(strategy, mutated)[:cut] - full[:cut]) > atol)[0]
            if len(bad):
                violations.append({"cut": cut, "probe": label,
                                   "n_changed_bars": len(bad),
                                   "first_bar": int(bad[0])})

    return {
        "passed": len(violations) == 0,
        # The full list can run to dozens of entries and ends up inline in the
        # experiment log; the first few identify the cause just as well.
        "violations": violations[:_MAX_REPORTED],
        "n_violations": len(violations),
        "checked": checked,
    }
