---
id: absorption-low-impact-volume
title: Absorption - heavy volume that fails to move price
family: volume/flow
sources: []
---

MECHANISM
Price impact per unit of volume (Kyle's lambda) is normally stable. A bar where
a lot of volume trades and price barely moves has unusually low impact: someone
is standing there with resting size, filling every aggressive seller without
stepping back. That is a large passive buyer working an order, and they are not
done - a buyer who was done would stop bidding. The impatient sellers exhaust
first, and price then travels toward the absorber.

Mirror image of a liquidation bar: same volume extreme, NARROW range instead of
wide. Signature: volume far above normal, range small for that volume, close in
the upper part of the bar (the absorber was on the bid).

WHY IT CAN CLEAR COSTS
Working a large order takes hours, so what follows is the rest of their fill,
typically over 1%. A volume-high / range-low extreme is a few bars a month.

FALSIFIER
If bars picked on high volume alone pay the same, absorption is decoration. And
it must predict DIRECTION: if these bars are followed by big moves either way it
is a variance signal, useless long-only.

SIGNAL (one ranked score, all three terms large together)
  rng      = ((high - low) / close).replace(0.0, np.nan)
  atr_frac = ATR(atr_len) / close
  vol_r    = (volume / volume.rolling(vol_len).mean()).clip(0, 20)
  narrow   = (atr_frac / rng).clip(upper=10)     # >1 = tighter than usual
  close_loc = ((close - low) / (high - low).replace(0.0, np.nan)).fillna(0.5)
  score    = (vol_r * narrow * close_loc ** 2).fillna(0.0)

ENTRY
  thresh  = score.rolling(rank_len, min_periods=500).quantile(entry_q)
  entries = score > thresh
Start at entry_q = 0.99; step to 0.98 if the trade count comes back too low.

RISK
stop = 1.2 * atr_frac, target = 3.0 * atr_frac, max_hold = 48 bars (12 hours).

PARAMS
atr_len=48, vol_len=192, rank_len=2000, entry_q=0.99, max_hold=48,
atr_stop=1.2, atr_target=3.0

PITFALLS
- (high - low) is 0 on a flat bar. Replace 0 with NaN before dividing, then
  fillna(0.0) the finished score.
- close_loc is SQUARED, not thresholded. An AND on close_loc > 0.8 deletes most
  of the population and the trade count comes back at zero.
- No trend filter. Absorption happens into weakness.
- Volume alone is the liquidation setup (WIDE range) and points the other way.
  The narrowness term is the whole idea.
- No paper behind this: the lambda framing is standard, the absorption reading
  is practitioner folklore. Treat it as untested.
