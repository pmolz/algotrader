---
id: session-handoff-drift
title: Session handoff drift
family: seasonality
sources: []
---

MECHANISM
Crypto trades 24/7 but the people trading it do not. Desk risk limits, CME
futures hours and the US equity session mean the marginal risk-taker changes
hands at fixed clock times. Two handoffs matter: the London open (~07:00-08:00
UTC) and the US cash open (~13:30-14:30 UTC). Positioning built up in the thin
Asian book gets repriced when a deeper book arrives, and the repricing has
direction — it continues the move the thin session started, because the arriving
desk is reacting to the same information with more size.

This is a WHEN, not a WHAT. The seasonal window is a filter; it needs a
directional score inside it.

WHY IT CAN CLEAR COSTS
At most one setup per day by construction, held through the handoff for hours.
The move being harvested is a session repricing, typically 0.5-2%.

FALSIFIER
The edge must be concentrated in the named hours. Compute the same score on a
shifted window (say 03:00 UTC) — if that pays just as well, there is no session
effect and you have found ordinary momentum with a random gate on it.

SIGNAL
  hour = df.index.hour                                   # UTC, from the index
  in_window = pd.Series(np.isin(hour, list(open_hours)), index=df.index)
  # Directional score: how much the thin overnight session already moved,
  # normalised by volatility, so only decisive overnight moves qualify.
  overnight = close.pct_change(lookback)                 # lookback ~ 24 bars
  atr_frac  = ATR(atr_len) / close
  score     = (overnight / atr_frac).where(in_window, 0.0)

ENTRY
  thresh  = score.rolling(rank_len, min_periods=500).quantile(entry_q)
  entries = score > thresh
Start at entry_q = 0.98. The window filter already removes ~90% of bars, so the
quantile is doing less work than usual — expect to LOWER entry_q, not raise it.

RISK
stop = 1.5 * atr_frac, target = 2.5 * atr_frac, max_hold = 24 bars (6 hours),
so a trade cannot survive into the next day's window and stack on itself.

PARAMS
open_hours=(7, 8, 13, 14), lookback=24, atr_len=48, rank_len=2000,
entry_q=0.98, max_hold=24, atr_stop=1.5, atr_target=2.5

PITFALLS
- The index is UTC. Do not convert it, and do not assume local time.
- LOOKAHEAD TRAP: do not compute a mean return per hour over the sample and
  trade the best hours. That reads the end of the backtest to trade its start
  and will be rejected. The hours are fixed constants chosen from the mechanism,
  not fitted from the data.
- `df.index.hour` is a numpy array, not a Series — wrap it before `.where`.
