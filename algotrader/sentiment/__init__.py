"""Free market-sentiment sources. Treat sentiment as a FEATURE/FILTER, never a
standalone signal. All functions return tidy DataFrames you can merge onto OHLCV.
"""

from .sources import fear_greed_crypto, reddit_mentions

__all__ = ["fear_greed_crypto", "reddit_mentions"]
