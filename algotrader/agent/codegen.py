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


def load_strategy_class(code: str) -> Type[Strategy]:
    """Exec the code in a restricted namespace and return the Strategy subclass.

    Only pd, np, Strategy, StrategyResult are exposed. Builtins are limited.
    """
    _screen(code)

    safe_builtins = {
        "range": range, "len": len, "min": min, "max": max, "abs": abs,
        "sum": sum, "float": float, "int": int, "bool": bool, "list": list,
        "dict": dict, "enumerate": enumerate, "zip": zip, "round": round,
        "print": print, "isinstance": isinstance, "super": super,
        "__build_class__": __build_class__, "__name__": "generated_strategy",
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
