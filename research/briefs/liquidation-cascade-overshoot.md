---
id: liquidation-cascade-overshoot
title: Liquidation cascade overshoot
family: mean-reversion
sources: []
---

MECHANISM
Crypto perps carry high leverage against a thin spot book. When price breaches
maintenance margin, liquidation engines sell at market regardless of price —
sellers who are not choosing to sell. That forced flow overshoots, and once the
margin queue drains the book refills and price recovers part of the move. The
edge is not "dips bounce"; it is that one identifiable kind of dip has a
non-discretionary seller behind it. Signature in OHLCV: a bar with unusually
large downward range, unusually large volume, and a close well off the low.

WHY IT CAN CLEAR COSTS
Cascades overshoot by whole percent, not basis points, so the recovery leg is
1-3% — well above the 0.30% round trip. It is rare by construction: a joint
volume-and-range extreme is a few bars a month.

FALSIFIER
Returns should concentrate in the 4-24 bars after entry and be stronger the
larger the volume surge. If profit is flat across volume terciles this is just
dip-buying and the liquidation story is decoration.

SIGNAL (build a single ranked score, higher = stronger setup)
  ret      = close.pct_change()
  atr_frac = ATR(atr_len) / close                      # as in the example
  drop     = (-ret / atr_frac).clip(lower=0)           # down-move in ATR units
  vol_z    = (volume - volume.rolling(vol_len).mean())
             / volume.rolling(vol_len).std()
  wick     = ((close - low) / (high - low).replace(0, np.nan)).fillna(0.5)
  score    = drop * vol_z.clip(lower=0) * wick

All three terms must be large together — that conjunction is what separates a
cascade from an ordinary red bar. Multiply them, do not AND fixed thresholds.

ENTRY
  thresh  = score.rolling(rank_len, min_periods=500).quantile(entry_q)
  entries = score > thresh
Start at entry_q = 0.995. This fires rarely on purpose; if the trade count comes
back too low, step down to 0.99 before touching anything else.

RISK
stop  = 1.2 * atr_frac  (tight — a cascade that keeps cascading is not your trade)
target = 3.0 * atr_frac
max_hold = 32 bars (8 hours). The recovery is fast or it is not happening.

PARAMS
atr_len=48, vol_len=192, rank_len=2000, entry_q=0.995, max_hold=32,
atr_stop=1.2, atr_target=3.0

PITFALLS
- volume.rolling().std() can be 0 on a dead stretch; replace 0 with NaN.
- Do not add a trend filter. Cascades happen in downtrends; filtering to
  uptrends removes the population the mechanism is about.
