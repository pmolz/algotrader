---
id: round-number-stop-cascade
title: Round-number breach and the stop cascade above it
family: microstructure
sources:
  - 'Osler (2003), "Currency Orders and Exchange Rate Dynamics", J. Finance - take-profit orders cluster ON round numbers, stop-loss orders cluster JUST BEYOND them'
  - 'Osler (2005), "Stop-loss orders and price cascades in currency markets", JIMF'
---

MECHANISM
Resting orders are not spread evenly over price. Osler's order data shows
take-profit orders clustering ON round numbers and stop-loss orders clustering
JUST BEYOND them. So a round level is a wall of profit-taking supply on the
approach and a pool of stop-triggered buying immediately above it. When price
finally trades through, those stops fire as market buys - people buying because
their risk limit says so, not because they like the price - and the move
accelerates. The edge is the BREACH, not the approach: buying just under the
level buys into the take-profit wall. Crypto inherits this; perp shorts stack
stops above the same round figures every screen quotes.

WHY IT CAN CLEAR COSTS
A grid step here is 1-2% of price, so a resolved breach travels most of a cell.
It fires only on a fresh crossing: a few a day before ranking, under one after.

FALSIFIER
The edge must attach to round numbers specifically. Recompute the identical
score on a grid shifted by half a cell (levels at L + grid/2). If the offset
grid pays the same, there is no order clustering here and this is ordinary
breakout momentum wearing a story.

SIGNAL
  grid   = (10 ** np.floor(np.log10(close))) / grid_div   # 1000 near 60k
  cross  = (np.floor(close / grid) >
            np.floor(close.shift(1) / grid)).astype(float)   # fresh breach up
  atr_frac = ATR(atr_len) / close
  push   = (close.pct_change() / atr_frac).clip(lower=0)
  vol_r  = (volume / volume.rolling(vol_len).mean()).clip(0, 10)
  score  = (cross * push * vol_r).fillna(0.0)

ENTRY
  thresh  = score.rolling(rank_len, min_periods=500).quantile(entry_q)
  entries = score > thresh
Start at entry_q = 0.99.

RISK
stop = 1.0 * atr_frac (a breach that fails, fails immediately),
target = 2.5 * atr_frac, max_hold = 24 bars (6 hours).

PARAMS
grid_div=10, atr_len=48, vol_len=192, rank_len=2000, entry_q=0.99, max_hold=24,
atr_stop=1.0, atr_target=2.5

PITFALLS
- Keep entry_q above ~0.98. Crossings are only a few percent of bars, so a lower
  quantile puts the threshold at 0.0 and every crossing fires.
- Do not hard-code a dollar grid. Price spans an order of magnitude across the
  sample; the log10 form keeps the cell a fixed fraction of price.
- np.floor of a Series is a Series. Do not call .values.
- Crowded: round numbers are among the most published patterns in markets.
  Expect a thin edge or none.
