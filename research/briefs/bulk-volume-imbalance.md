---
id: bulk-volume-imbalance
title: Bulk-volume order imbalance (informed accumulation)
family: microstructure
sources:
  - 'Easley, Lopez de Prado & O''Hara (2012), "Flow Toxicity and Liquidity in a High Frequency World" — bulk volume classification'
  - 'Kyle (1985), "Continuous Auctions and Insider Trading" — informed traders split orders over time'
---

MECHANISM
An informed trader does not buy in one clip; they split the order over hours to
hide it (Kyle). So informed buying shows up as many consecutive bars where more
volume traded on the up-tick side than the down-tick side, and price adjusts
gradually rather than in one jump. Persistent signed order imbalance therefore
leads price, and the adjustment is still incomplete when the imbalance is
visible.

You have no trade-level tape, but bulk volume classification recovers a usable
estimate from bars: split each bar's volume into buy and sell parts in
proportion to how large its return was relative to recent return volatility.

WHY IT CAN CLEAR COSTS
The signal is a multi-hour accumulation, so the position is held for hours and
the move harvested is the full adjustment, not a tick. Requiring the imbalance
to be an extreme of its own recent history keeps it rare.

FALSIFIER
Imbalance should lead returns, not lag them. If the same score computed on
FUTURE bars (in a scratch check, never in the strategy) is no more predictive
than the past version, this is momentum with extra arithmetic. Also: the edge
should weaken as you extend the holding period, since the adjustment completes.

SIGNAL
  ret   = close.pct_change()
  sd    = ret.rolling(vol_len).std().replace(0.0, np.nan)
  z     = (ret / sd).clip(-4, 4)
  # Fraction of the bar's volume that was buyer-initiated. The normal CDF used
  # in the paper is unavailable here (no scipy); tanh is a close-enough
  # substitute and is vectorised.
  buy_frac = 0.5 * (1.0 + np.tanh(z / 1.6))
  signed   = volume * (2.0 * buy_frac - 1.0)      # +ve = net buying
  imb      = signed.rolling(imb_len).sum() / volume.rolling(imb_len).sum()
  score    = imb                                   # in [-1, 1]

ENTRY
  thresh  = score.rolling(rank_len, min_periods=500).quantile(entry_q)
  entries = score > thresh
Start at entry_q = 0.97.

RISK
stop = 1.5 * atr_frac, target = 3.0 * atr_frac, max_hold = 48 bars (12 hours).

PARAMS
vol_len=96, imb_len=32, atr_len=48, rank_len=2000, entry_q=0.97, max_hold=48,
atr_stop=1.5, atr_target=3.0

PITFALLS
- np.tanh on a Series returns a Series; keep it as one, do not call .values.
- Divide the rolling sums, do not average the per-bar ratio — a big bar must
  count more than a small one, which is the whole point of volume weighting.
- Do not take abs(imb). That is VPIN, which measures toxicity and predicts
  volatility, not direction, and is useless to a long-only strategy.
