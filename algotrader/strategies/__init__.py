from .base import Strategy, StrategyResult
from .registry import register, get_strategy, list_strategies
from . import baselines  # noqa: F401  (populates the registry)

__all__ = [
    "Strategy",
    "StrategyResult",
    "register",
    "get_strategy",
    "list_strategies",
]
