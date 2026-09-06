---
id: schedule-adjusted-volume-shock
title: Volume shock measured against the clock, not the average
family: seasonality
sources:
  - 'Hansen, Kim & Kimbrough (2021), "Periodicity in Cryptocurrency Volatility and Liquidity", arXiv:2109.12142 - systematic hour-of-day and day-of-week patterns in BTC/ETH volume, strengthening over time, tied to algorithmic execution and perp funding schedules'
  - 'Andersen & Bollerslev (1997), "Intraday periodicity and volatility persistence in financial markets" - deseasonalise before reading high-frequency activity'
---

MECHANISM
Most crypto volume arrives on a clock. Hansen, Kim & Kimbrough find systematic
hour-of-day and day-of-week patterns in BTC/ETH volume that have grown STRONGER
over the years, and tie them to algorithmic execution and perp funding
settlements. That flow is scheduled, not informed: a TWAP slice at 14:00 knows
nothing. So a plain volume z-score mostly measures what time it is, and a
strategy built on one spends its selectivity re-discovering the working day.

Subtract the clock and the residual is discretionary arrival - someone chose to
trade size in a slot where size does not normally trade. That is the flow with
information behind it. Pair it with the move it arrived on for direction.

WHY IT CAN CLEAR COSTS
Informed size takes hours to complete, so the continuation is a multi-hour
repricing. Requiring a residual extreme keeps it under a trade a day.

FALSIFIER
The correction must be what pays. Recompute with base as a plain
volume.rolling(sea_len * 96).mean(), no slot grouping. If that scores the same,
periodicity is irrelevant here and this is volume-weighted momentum.

SIGNAL
  slot = df.index.hour * 4 + df.index.minute // 15     # 96 slots per day
  # Baseline for this slot from PREVIOUS days only; the .shift(1) inside the
  # group is what keeps the bar out of its own baseline.
  base = volume.groupby(slot).transform(
      lambda s: s.shift(1).rolling(sea_len, min_periods=5).mean())
  surprise = (volume / base.replace(0.0, np.nan)).clip(0.01, 20)
  atr_frac = ATR(atr_len) / close
  push     = (close.pct_change(mom_len) / atr_frac).clip(lower=0)
  score    = (push * np.log(surprise).clip(lower=0)).fillna(0.0)

ENTRY
  thresh  = score.rolling(rank_len, min_periods=500).quantile(entry_q)
  entries = score > thresh
Start at entry_q = 0.97.

RISK
stop = 1.5 * atr_frac, target = 3.0 * atr_frac, max_hold = 32 bars (8 hours).

PARAMS
sea_len=20, mom_len=8, atr_len=48, rank_len=2000, entry_q=0.97, max_hold=32,
atr_stop=1.5, atr_target=3.0

PITFALLS
- LOOKAHEAD TRAP: do NOT take a per-slot mean over the whole sample. The
  rolling-inside-groupby above is the only safe form; .shift(1) is not optional.
- df.index.hour and .minute are numpy arrays. Build slot from them and pass the
  array straight to groupby; do not add it as a column of df.
- min_periods=5 leaves early bars NaN per slot; fillna(0.0) the score.
- log, not the raw ratio: a 30x volume bar is not 10x the evidence of a 3x bar.
