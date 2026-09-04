"""Free sentiment sources.

Implemented:
  * fear_greed_crypto : Alternative.me Crypto Fear & Greed Index (free, no key).

Stubbed (wire up when you're ready — keys go in .env):
  * reddit_mentions   : PRAW-based subreddit ticker mention counts + sentiment.
  * GDELT / NewsAPI   : news-flow sentiment. (add under this module)

Sentiment is a filter, not a signal. E.g. "only take longs when Fear & Greed is
in 'extreme fear'" — a contrarian regime filter — rather than trading it directly.
"""

from __future__ import annotations

import os

import pandas as pd


def fear_greed_crypto(limit: int = 0) -> pd.DataFrame:
    """Crypto Fear & Greed Index. Returns DataFrame indexed by UTC date with
    columns: value (0-100), classification.

    limit=0 fetches the full available history.
    """
    import urllib.request
    import json

    url = f"https://api.alternative.me/fng/?limit={limit}&format=json"
    with urllib.request.urlopen(url, timeout=15) as resp:
        data = json.loads(resp.read())
    rows = data.get("data", [])
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="s", utc=True)
    df["value"] = df["value"].astype(int)
    df = df.rename(columns={"value_classification": "classification"})
    return df.set_index("timestamp")[["value", "classification"]].sort_index()


def reddit_mentions(subreddit: str, ticker: str, limit: int = 200) -> pd.DataFrame:
    """Count + score recent mentions of `ticker` in `subreddit` via PRAW.

    Requires REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET / REDDIT_USER_AGENT in env.
    Returns a DataFrame with columns: created_utc, title, score, sentiment(optional).
    """
    if not os.environ.get("REDDIT_CLIENT_ID"):
        raise RuntimeError("Set REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET / REDDIT_USER_AGENT")
    import praw  # lazy import

    reddit = praw.Reddit(
        client_id=os.environ["REDDIT_CLIENT_ID"],
        client_secret=os.environ["REDDIT_CLIENT_SECRET"],
        user_agent=os.environ.get("REDDIT_USER_AGENT", "algotrader/0.1"),
    )
    rows = []
    for post in reddit.subreddit(subreddit).search(ticker, limit=limit):
        rows.append(
            {
                "created_utc": pd.to_datetime(post.created_utc, unit="s", utc=True),
                "title": post.title,
                "score": post.score,
                "num_comments": post.num_comments,
            }
        )
    df = pd.DataFrame(rows)
    return df.sort_values("created_utc") if not df.empty else df
