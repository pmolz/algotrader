"""Bar-path simulation for strategies that declare a stop loss or take profit.

The vectorized engine multiplies a held position by close-to-close returns. That
is exactly right when a position only changes on a signal, and wrong the moment a
strategy says "get me out if this drops 1.5%", because the exit happens *inside*
a bar and the close never sees it.

So when a strategy declares a risk overlay, the engine walks the bars instead.
The fill rules, which are where backtests usually start lying:

  ENTRY at the close of the bar whose signal asked for it — the same instant the
  vectorized path assumes, so turning a stop on doesn't silently re-time entries.

  STOP WINS TIES. If a bar's [low, high] contains both the stop and the target,
  OHLC cannot tell you which came first, and assuming the target is how a
  backtest invents an edge that isn't there. We take the loss. This understates
  a genuinely good strategy and never flatters a bad one; `ambiguous_bars` in
  the result counts how often the assumption was load-bearing.

  GAPS FILL AT THE OPEN. If a bar opens already through the level, you did not
  get the level — you got the open. Filling at the stop price on a gap is a
  second, quieter way to manufacture edge.

  NO IMMEDIATE RE-ENTRY. After a stop or target exit, the position stays flat
  until the signal goes flat and asks again. Without this a strategy that holds a
  constant target=1 would re-enter on the very next bar and the stop would do
  nothing but bleed fees.

Long/flat only. Shorts would need borrow and perp funding costs that this engine
does not model, and leaving them out would flatter every short strategy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class PathResult:
    returns: pd.Series                  # per-bar strategy return, after costs
    positions: pd.Series                # position actually held during each bar
    trades: list[dict] = field(default_factory=list)
    ambiguous_bars: int = 0             # bars where stop and target both sat inside the range
    exit_counts: dict[str, int] = field(default_factory=dict)


def _as_array(level, index: pd.Index, name: str) -> np.ndarray | None:
    """Accept a scalar or a Series; return a per-bar array of positive fractions."""
    if level is None:
        return None
    if isinstance(level, pd.Series):
        arr = level.reindex(index).astype(float).to_numpy()
    else:
        arr = np.full(len(index), float(level))
    # A non-positive or missing level means "no level on this bar" rather than
    # "exit immediately at the entry price".
    arr = np.where(np.isfinite(arr) & (arr > 0), arr, np.nan)
    if name == "stop_loss_pct":
        arr = np.where(arr >= 1.0, np.nan, arr)   # a 100% stop is not a stop
    return arr


def simulate(
    df: pd.DataFrame,
    target: pd.Series,
    stop_loss_pct=None,
    take_profit_pct=None,
    fee_pct: float = 0.001,
    slippage_pct: float = 0.0005,
) -> PathResult:
    """Walk the bars applying the risk overlay. `target` is the desired position
    per bar, already lookahead-shifted by the caller's convention."""
    idx = df.index
    n = len(idx)
    open_ = df["open"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    want = np.clip(np.nan_to_num(target.to_numpy(dtype=float)), 0.0, 1.0)

    sl = _as_array(stop_loss_pct, idx, "stop_loss_pct")
    tp = _as_array(take_profit_pct, idx, "take_profit_pct")
    cost_rate = fee_pct + slippage_pct

    held = np.zeros(n)          # position held *during* bar i
    rets = np.zeros(n)
    trades: list[dict] = []
    exit_counts: dict[str, int] = {}
    ambiguous = 0

    pos = 0.0                   # current position
    entry_price = np.nan
    entry_i = -1
    stop_px = np.nan
    take_px = np.nan
    blocked = False             # stopped out; wait for the signal to go flat

    for i in range(n):
        # --- carry yesterday's position into this bar -------------------------
        bar_ret = 0.0
        turnover = 0.0

        if pos > 0:
            exit_px = None
            reason = None
            hit_stop = not np.isnan(stop_px) and low[i] <= stop_px
            hit_take = not np.isnan(take_px) and high[i] >= take_px
            if hit_stop and hit_take:
                ambiguous += 1
            if hit_stop:
                # gapped through -> you got the open, not the level
                exit_px = min(open_[i], stop_px)
                reason = "stop"
            elif hit_take:
                exit_px = max(open_[i], take_px)
                reason = "target"

            if exit_px is not None:
                bar_ret = pos * (exit_px / close[i - 1] - 1.0)
                turnover += pos
                held[i] = pos
                trades.append({
                    "entry_i": entry_i, "exit_i": i,
                    "entry_time": idx[entry_i], "exit_time": idx[i],
                    "entry_px": entry_price, "exit_px": exit_px,
                    "bars_held": i - entry_i, "reason": reason,
                    "ret": exit_px / entry_price - 1.0,
                })
                exit_counts[reason] = exit_counts.get(reason, 0) + 1
                pos, entry_price, stop_px, take_px = 0.0, np.nan, np.nan, np.nan
                blocked = True
            else:
                bar_ret = pos * (close[i] / close[i - 1] - 1.0)
                held[i] = pos

        # --- act on the signal at this bar's close ----------------------------
        if blocked and want[i] <= 0:
            blocked = False                      # signal went flat; armed again

        desired = 0.0 if blocked else want[i]
        if pos > 0 and desired <= 0:             # signalled exit
            bar_ret = pos * (close[i] / close[i - 1] - 1.0) if held[i] == 0 else bar_ret
            held[i] = pos
            turnover += pos
            trades.append({
                "entry_i": entry_i, "exit_i": i,
                "entry_time": idx[entry_i], "exit_time": idx[i],
                "entry_px": entry_price, "exit_px": close[i],
                "bars_held": i - entry_i, "reason": "signal",
                "ret": close[i] / entry_price - 1.0,
            })
            exit_counts["signal"] = exit_counts.get("signal", 0) + 1
            pos, entry_price, stop_px, take_px = 0.0, np.nan, np.nan, np.nan
        elif pos == 0.0 and desired > 0:         # entry
            turnover += desired
            pos = desired
            entry_price = close[i]
            entry_i = i
            stop_px = entry_price * (1.0 - sl[i]) if sl is not None and not np.isnan(sl[i]) else np.nan
            take_px = entry_price * (1.0 + tp[i]) if tp is not None and not np.isnan(tp[i]) else np.nan
        elif pos > 0 and desired > 0 and abs(desired - pos) > 1e-9:
            turnover += abs(desired - pos)       # resize, levels stay off the original entry
            pos = desired

        rets[i] = bar_ret - turnover * cost_rate

    # A position still open at the end is marked to the last close, not closed —
    # forcing a liquidation would invent a trade the strategy never made.
    return PathResult(
        returns=pd.Series(rets, index=idx),
        positions=pd.Series(held, index=idx),
        trades=trades,
        ambiguous_bars=ambiguous,
        exit_counts=exit_counts,
    )
