"""Parse and (carefully) load LLM-generated strategy code.

SECURITY WARNING
----------------
Executing code produced by an LLM is running untrusted code. The `exec` path here
is fine for LOCAL, PHASE-1 experimentation where you review what runs. For the
automated Phase-2 loop you MUST run generation inside a locked-down Docker sandbox
(no network, no filesystem, resource limits). See ROADMAP Phase 2 and
algotrader/live/safety.py. Do not point this at an unattended machine with secrets.
"""

from __future__ import annotations

import re
import types
from typing import Type

import numpy as np
import pandas as pd

from ..strategies.base import Strategy, StrategyResult

CODE_BLOCK_RE = re.compile(r"```python\s*(.*?)```", re.DOTALL)
HYPOTHESIS_RE = re.compile(r"HYPOTHESIS:\s*(.*)", re.IGNORECASE)

# crude denylist to catch obviously dangerous generated code before exec
FORBIDDEN = ["import os", "import sys", "subprocess", "open(", "__import__",
             "eval(", "exec(", "socket", "requests", "urllib", "shutil",
             "import importlib", "globals(", "getattr(__"]


def parse_response(text: str) -> tuple[str, str]:
    """Extract (hypothesis, code) from the LLM response."""
    m = CODE_BLOCK_RE.search(text)
    if not m:
        raise ValueError("No ```python code block found in LLM response.")
    code = m.group(1).strip()
    hm = HYPOTHESIS_RE.search(text)
    hypothesis = hm.group(1).strip() if hm else "(no hypothesis provided)"
    return hypothesis, code


def _screen(code: str) -> None:
    lowered = code.lower()
    for bad in FORBIDDEN:
        if bad in lowered:
            raise ValueError(f"Refusing to exec generated code containing {bad!r}")


# Generated code may only import from this whitelist (models add imports despite
# instructions). Everything else raises ImportError.
_ALLOWED_ROOTS = {"pandas", "numpy", "math"}

# Names the framework injects into the namespace. Models habitually import them
# anyway — `from strategy import Strategy` and friends were 31 of 41 codegen
# failures in the experiment log, a quarter of all iterations thrown away over a
# redundant import line. The names being asked for are exactly the objects
# already in scope, so satisfy the request instead of rejecting the candidate.
_FRAMEWORK_EXPORTS = {"Strategy": Strategy, "StrategyResult": StrategyResult}


def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    root = name.split(".")[0]
    if root in _ALLOWED_ROOTS:
        return __import__(name, globals, locals, fromlist, level)
    # `from <anything> import Strategy [, StrategyResult]` -> hand back the real
    # classes. Nothing is actually imported and no other name resolves this way,
    # so the sandbox surface is unchanged: a module that isn't whitelisted still
    # cannot be reached, it just can't be used to smuggle a name in either.
    if fromlist and set(fromlist) <= set(_FRAMEWORK_EXPORTS):
        return types.SimpleNamespace(**_FRAMEWORK_EXPORTS)
    raise ImportError(f"Import of {name!r} not allowed in generated strategy.")


def load_strategy_class(code: str) -> Type[Strategy]:
    """Exec the code in a restricted namespace and return the Strategy subclass.

    Only pd, np, Strategy, StrategyResult are exposed, plus a whitelisted import
    hook allowing pandas/numpy/math. Builtins are limited.
    """
    _screen(code)

    safe_builtins = {
        "range": range, "len": len, "min": min, "max": max, "abs": abs,
        "sum": sum, "float": float, "int": int, "bool": bool, "list": list,
        "dict": dict, "enumerate": enumerate, "zip": zip, "round": round,
        "print": print, "isinstance": isinstance, "super": super,
        "__build_class__": __build_class__, "__name__": "generated_strategy",
        "__import__": _safe_import,
    }
    ns: dict = {
        "pd": pd, "np": np,
        "Strategy": Strategy, "StrategyResult": StrategyResult,
        "__builtins__": safe_builtins,
    }
    exec(compile(code, "<generated_strategy>", "exec"), ns)

    classes = [
        v for v in ns.values()
        if isinstance(v, type) and issubclass(v, Strategy) and v is not Strategy
    ]
    if not classes:
        raise ValueError("No Strategy subclass defined in generated code.")
    return classes[-1]
