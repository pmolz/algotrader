import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def trending_ohlcv():
    """Deterministic upward-trending series with noise — enough bars for splits."""
    n = 800
    rng = np.random.default_rng(42)
    idx = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
    drift = np.linspace(0, 0.8, n)
    noise = rng.normal(0, 0.02, n).cumsum()
    close = 100 * np.exp(drift + noise)
    high = close * (1 + rng.uniform(0, 0.01, n))
    low = close * (1 - rng.uniform(0, 0.01, n))
    open_ = close * (1 + rng.normal(0, 0.005, n))
    vol = rng.uniform(1000, 5000, n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol},
        index=idx,
    )
