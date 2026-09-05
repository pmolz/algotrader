"""The validation gauntlet — the most important component in this repo.

A candidate strategy must survive ALL of these before it can be trusted:

  1. Lookahead check      — no peeking at future data.
  2. Walk-forward         — consistent out-of-sample performance across time folds.
  3. Cost stress test     — edge survives 1.5x / 2x modelled transaction costs.
  4. Deflated Sharpe      — edge likely survives the multiple-testing you did.
  5. Final OOS holdout    — one shot on data developed-against-blind.
  6. Beats buy & hold     — risk-adjusted, or why bother.

`run_gauntlet` returns a GauntletReport with a single boolean `promoted` plus all
the evidence. The agent and the human read the same report.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..backtest.engine import backtest
from ..strategies.base import Strategy
from ..strategies.baselines import BuyAndHold
from .deflated_sharpe import deflated_sharpe_ratio
from .lookahead import lookahead_check
from .splits import train_oos_split, walk_forward_splits


@dataclass
class GauntletReport:
    promoted: bool
    reasons: list[str] = field(default_factory=list)     # why it failed / passed notes
    checks: dict[str, Any] = field(default_factory=dict)  # detailed evidence

    def pretty(self) -> str:
        status = "PROMOTED ✅" if self.promoted else "REJECTED ❌"
        lines = [f"Gauntlet: {status}"]
        for r in self.reasons:
            lines.append(f"  - {r}")
        return "\n".join(lines)


def run_gauntlet(
    df: pd.DataFrame,
    strategy: Strategy,
    cfg: dict,
    n_trials: int = 1,
) -> GauntletReport:
    """Run the full gauntlet. `cfg` is the loaded config dict.

    Args:
        df:       full OHLCV history (gauntlet does its own train/OOS split).
        strategy: the candidate.
        cfg:      config dict (uses cfg['backtest'] and cfg['validation']).
        n_trials: how many strategy variants were tried to arrive at this one —
                  drives the multiple-testing correction. BE HONEST HERE.
    """
    bt = cfg["backtest"]
    vc = cfg["validation"]
    reasons: list[str] = []
    checks: dict[str, Any] = {}

    def _bt(data, cost_mult=1.0):
        return backtest(
            data, strategy,
            initial_cash=bt["initial_cash"],
            fee_pct=bt["fee_pct"] * cost_mult,
            slippage_pct=bt["slippage_pct"] * cost_mult,
            allow_short=bt["allow_short"],
        )

    # --- 0. reserve the final holdout the strategy is developed blind to --------
    dev, oos = train_oos_split(df, vc["oos_holdout_frac"])

    # --- 1. lookahead check -----------------------------------------------------
    la = lookahead_check(dev, strategy)
    checks["lookahead"] = la
    if not la["passed"]:
        reasons.append(f"FAIL lookahead: signal depends on future data {la['violations']}")
        return GauntletReport(False, reasons, checks)
    reasons.append("PASS lookahead (no future peeking detected)")

    # --- 2. trading intensity ---------------------------------------------------
    # Intraday this is the binding constraint, so it fails before the expensive
    # checks. A round trip costs (fee + slippage) twice, so at 15m bars a
    # strategy trading three times a day pays ~330% of capital a year in
    # friction. Nothing survives that, and a candidate churning at that rate is
    # not a near miss worth walk-forwarding — it is a different kind of answer.
    dev_res = _bt(dev)
    per_day = float(dev_res.meta.get("trades_per_day", 0.0))
    drag = float(dev_res.meta.get("cost_drag_annual", 0.0))
    round_trip = 2.0 * (bt["fee_pct"] + bt["slippage_pct"])
    checks["trading_intensity"] = {
        "trades": dev_res.trades,
        "trades_per_day": per_day,
        "cost_drag_annual": drag,
        "cost_per_round_trip": round_trip,
        "exit_counts": dev_res.meta.get("exit_counts", {}),
        "avg_bars_held": dev_res.meta.get("avg_bars_held"),
    }
    # A strategy that barely trades is not a strategy that lost money, and the
    # difference matters: these reasons are fed back as lessons, and labelling a
    # no-trade candidate "mean OOS Sharpe -1.80" teaches the agent that an idea
    # family fails when it was never actually tested.
    if dev_res.trades < vc["min_trades"]:
        if dev_res.trades == 0:
            reasons.append(
                "FAIL never traded: the entry condition never fired on the "
                f"development set ({len(dev)} bars) — the filter is too tight, "
                "not the idea wrong"
            )
        else:
            reasons.append(
                f"FAIL too few trades: {dev_res.trades} < {vc['min_trades']} "
                f"({per_day:.2f}/day) — too little evidence for a Sharpe to mean "
                "anything, loosen the entry filter"
            )
        return GauntletReport(False, reasons, checks)

    max_per_day = vc.get("max_trades_per_day")
    if max_per_day and per_day > max_per_day:
        reasons.append(
            f"FAIL overtrading: {per_day:.1f} trades/day > {max_per_day} "
            f"({drag:.0%} of capital per year in costs; each round trip must "
            f"clear {round_trip:.2%} before anything is left over)"
        )
        return GauntletReport(False, reasons, checks)
    reasons.append(
        f"PASS trading intensity: {per_day:.2f} trades/day, "
        f"{drag:.0%}/yr cost drag"
    )

    # An exit the data cannot actually adjudicate is an assumption, not a
    # result. A strategy whose P&L rests mostly on stop-wins-ties is untestable
    # on this timeframe rather than merely unprofitable.
    ambiguous = dev_res.meta.get("ambiguous_bars")
    max_amb = vc.get("max_ambiguous_exit_frac")
    if ambiguous and max_amb and dev_res.trades:
        frac = ambiguous / dev_res.trades
        checks["trading_intensity"]["ambiguous_exit_frac"] = frac
        if frac > max_amb:
            reasons.append(
                f"FAIL ambiguous exits: {frac:.0%} of trades ({ambiguous}) hit "
                f"stop and target inside the same bar, so the result rests on "
                f"the stop-wins-ties assumption rather than on the data"
            )
            return GauntletReport(False, reasons, checks)

    # --- 3. walk-forward on the development set ---------------------------------
    wf_sharpes, wf_returns = [], []
    for train, test in walk_forward_splits(
        dev, vc["walk_forward_train_frac"], vc["walk_forward_n_splits"]
    ):
        res = _bt(test)
        wf_sharpes.append(res.metrics["sharpe"])
        wf_returns.append(res.returns)
    checks["walk_forward"] = {
        "fold_sharpes": [float(s) for s in wf_sharpes],
        "mean_sharpe": float(np.mean(wf_sharpes)) if wf_sharpes else 0.0,
        "positive_folds": int(np.sum(np.array(wf_sharpes) > 0)),
        "n_folds": len(wf_sharpes),
    }
    mean_wf = checks["walk_forward"]["mean_sharpe"]
    if mean_wf < vc["min_sharpe"]:
        reasons.append(
            f"FAIL walk-forward: mean OOS Sharpe {mean_wf:.2f} < {vc['min_sharpe']}"
        )
        return GauntletReport(False, reasons, checks)
    reasons.append(
        f"PASS walk-forward: mean OOS Sharpe {mean_wf:.2f} "
        f"({checks['walk_forward']['positive_folds']}/{len(wf_sharpes)} folds positive)"
    )

    # --- 4. cost stress test ----------------------------------------------------
    stress = {}
    for mult in vc["cost_stress_multipliers"]:
        res = _bt(dev, cost_mult=mult)
        stress[str(mult)] = float(res.metrics["sharpe"])
    checks["cost_stress"] = stress
    worst = min(stress.values())
    if worst < 0:
        reasons.append(f"FAIL cost stress: Sharpe goes negative under stress {stress}")
        return GauntletReport(False, reasons, checks)
    reasons.append(f"PASS cost stress: worst-case Sharpe {worst:.2f} across {stress}")

    # --- 5. deflated Sharpe (multiple-testing correction) -----------------------
    dsr = deflated_sharpe_ratio(dev_res.returns, n_trials=n_trials)
    checks["deflated_sharpe"] = dsr
    if dsr["dsr"] <= 0.5:  # below coin-flip confidence the edge is real
        reasons.append(
            f"FAIL deflated Sharpe: DSR={dsr['dsr']:.2f} after {n_trials} trials"
        )
        return GauntletReport(False, reasons, checks)
    reasons.append(f"PASS deflated Sharpe: DSR={dsr['dsr']:.2f} (n_trials={n_trials})")

    # --- 6. drawdown limit ------------------------------------------------------
    if dev_res.metrics["max_drawdown"] < -abs(vc["max_drawdown_pct"]):
        reasons.append(
            f"FAIL drawdown: {dev_res.metrics['max_drawdown']:.2%} exceeds "
            f"{vc['max_drawdown_pct']:.0%} limit"
        )
        return GauntletReport(False, reasons, checks)

    # --- 7. beat buy & hold on the dev set (risk-adjusted) ----------------------
    bh = backtest(
        dev, BuyAndHold(),
        initial_cash=bt["initial_cash"], fee_pct=bt["fee_pct"],
        slippage_pct=bt["slippage_pct"], allow_short=bt["allow_short"],
    )
    checks["buy_hold_sharpe"] = float(bh.metrics["sharpe"])
    if dev_res.metrics["sharpe"] <= bh.metrics["sharpe"]:
        reasons.append(
            f"FAIL vs benchmark: Sharpe {dev_res.metrics['sharpe']:.2f} "
            f"<= buy&hold {bh.metrics['sharpe']:.2f}"
        )
        return GauntletReport(False, reasons, checks)
    reasons.append(
        f"PASS vs benchmark: Sharpe {dev_res.metrics['sharpe']:.2f} "
        f"> buy&hold {bh.metrics['sharpe']:.2f}"
    )

    # --- 8. FINAL out-of-sample holdout (the moment of truth) -------------------
    oos_res = _bt(oos)
    checks["oos_holdout"] = oos_res.metrics
    if oos_res.metrics["sharpe"] < vc["min_sharpe"]:
        reasons.append(
            f"FAIL final OOS holdout: Sharpe {oos_res.metrics['sharpe']:.2f} "
            f"< {vc['min_sharpe']} (likely overfit to dev set)"
        )
        return GauntletReport(False, reasons, checks)
    reasons.append(
        f"PASS final OOS holdout: Sharpe {oos_res.metrics['sharpe']:.2f}"
    )

    reasons.append("ALL CHECKS PASSED — candidate promoted to paper trading.")
    return GauntletReport(True, reasons, checks)
