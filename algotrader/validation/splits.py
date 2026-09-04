"""Time-series data splitting. NEVER shuffle time-series — that leaks the future."""

from __future__ import annotations

import pandas as pd


def train_oos_split(df: pd.DataFrame, oos_frac: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into (development, out-of-sample holdout).

    The holdout is the most recent `oos_frac` of the data and must be touched
    ONLY once, at the very end, to sanity-check a strategy the agent developed
    without ever seeing it.
    """
    n = len(df)
    cut = int(n * (1 - oos_frac))
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


def walk_forward_splits(
    df: pd.DataFrame,
    train_frac: float = 0.6,
    n_splits: int = 5,
):
    """Yield (train, test) folds rolling forward through time (anchored expanding
    window). Each test fold is strictly after its train fold.

    Yields:
        (train_df, test_df) tuples.
    """
    n = len(df)
    if n_splits < 1:
        raise ValueError("n_splits must be >= 1")

    initial_train = int(n * train_frac)
    remaining = n - initial_train
    if remaining < n_splits:
        # not enough data to make requested folds; degrade gracefully
        n_splits = max(1, remaining)
    fold = remaining // n_splits

    for i in range(n_splits):
        train_end = initial_train + i * fold
        test_end = train_end + fold if i < n_splits - 1 else n
        train = df.iloc[:train_end].copy()
        test = df.iloc[train_end:test_end].copy()
        if len(test) == 0:
            continue
        yield train, test
