"""Live / paper trading adapters. Phase 3+. Nothing here touches real money
without the guardrails in safety.py.
"""

from .safety import RiskLimits, SafetyGate

__all__ = ["RiskLimits", "SafetyGate"]
