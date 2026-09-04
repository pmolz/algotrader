"""A tiny registry so strategies can be looked up by name (CLI, agent, etc.)."""

from __future__ import annotations

from typing import Type

from .base import Strategy

_REGISTRY: dict[str, Type[Strategy]] = {}


def register(cls: Type[Strategy]) -> Type[Strategy]:
    """Class decorator: register a Strategy subclass under its `name`."""
    key = cls.name
    if key in _REGISTRY and _REGISTRY[key] is not cls:
        raise ValueError(f"Strategy name already registered: {key!r}")
    _REGISTRY[key] = cls
    return cls


def get_strategy(name: str) -> Type[Strategy]:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown strategy {name!r}. Known: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def list_strategies() -> list[str]:
    return sorted(_REGISTRY)
