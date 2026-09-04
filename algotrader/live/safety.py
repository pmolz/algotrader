"""Hard guardrails for any real-money path. These are non-negotiable.

Even in Phase 4, EVERY order must pass through a SafetyGate that enforces:
  * max position size per symbol,
  * max total gross exposure,
  * a daily loss kill-switch,
  * a global human-approved "armed" flag (default: DISARMED).

Design intent: it should be *impossible* to send a live order without explicitly
arming the gate and staying inside limits. Fail closed, always.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RiskLimits:
    max_position_pct: float = 0.10        # per-symbol fraction of equity
    max_gross_exposure_pct: float = 0.50  # sum of |positions|
    daily_loss_kill_pct: float = 0.03     # halt trading if daily loss exceeds this
    max_order_notional: float = 500.0     # absolute cap per order (start tiny!)


class SafetyGate:
    """Fail-closed gate. Real orders require arm() to have been called by a human."""

    def __init__(self, limits: RiskLimits, live: bool = False):
        self.limits = limits
        self.live = live          # True = real money; must be explicit
        self._armed = False       # human approval flag; default disarmed
        self._halted = False      # kill-switch tripped
        self._day_start_equity: float | None = None

    def arm(self, confirm: str) -> None:
        """A human must pass the exact confirmation phrase to enable live orders."""
        if confirm != "I ACCEPT THE RISK":
            raise PermissionError("Refused to arm: confirmation phrase mismatch.")
        self._armed = True

    def kill(self) -> None:
        self._halted = True

    def mark_day_start(self, equity: float) -> None:
        self._day_start_equity = equity

    def check_order(self, *, symbol: str, notional: float, equity: float,
                    current_gross: float) -> None:
        """Raise if an order violates any limit. Returns None if allowed."""
        if self._halted:
            raise PermissionError("Trading halted (kill-switch active).")
        if self.live and not self._armed:
            raise PermissionError("Live trading not armed by a human.")
        if abs(notional) > self.limits.max_order_notional:
            raise PermissionError(
                f"Order notional {notional} exceeds cap {self.limits.max_order_notional}."
            )
        if equity > 0 and abs(notional) / equity > self.limits.max_position_pct:
            raise PermissionError("Order exceeds max per-symbol position limit.")
        gross_after = current_gross + abs(notional)
        if equity > 0 and gross_after / equity > self.limits.max_gross_exposure_pct:
            raise PermissionError("Order exceeds max gross exposure limit.")
        if self._day_start_equity:
            loss = (self._day_start_equity - equity) / self._day_start_equity
            if loss >= self.limits.daily_loss_kill_pct:
                self.kill()
                raise PermissionError("Daily loss limit hit — kill-switch tripped.")
