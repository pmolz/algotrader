"""What the codegen layer must accept and must refuse.

The accept-side cases come from the experiment log: 31 of 41 recorded codegen
failures were a redundant `from strategy import Strategy` line, i.e. a quarter of
all iterations discarded over an import of names that were already in scope.
Tolerating that is worth more than any prompt wording.

The refuse-side cases are the reason this file is paranoid: generated code is
untrusted, and the import hook is one of the walls holding it.
"""

from __future__ import annotations

import pytest

from algotrader.agent.codegen import load_strategy_class
from algotrader.strategies.base import Strategy

BODY = '''
class Trivial(Strategy):
    name = "trivial"
    def generate_signals(self, df):
        return StrategyResult(positions=df["close"] * 0)
'''


@pytest.mark.parametrize("line", [
    "from strategy import Strategy",
    "from strategy import Strategy, StrategyResult",
    "from Strategy import Strategy",
    "from trading_framework.strategy import Strategy",
    "from algotrader.strategies.base import Strategy, StrategyResult",
    "from base import StrategyResult",
])
def test_redundant_framework_imports_are_tolerated(line):
    cls = load_strategy_class(f"{line}\n{BODY}")
    assert issubclass(cls, Strategy)


def test_the_injected_names_are_the_real_classes():
    """Tolerating the import must not hand back a lookalike — the gauntlet does
    issubclass checks against the real Strategy."""
    code = "from strategy import Strategy, StrategyResult\n" + BODY
    assert issubclass(load_strategy_class(code), Strategy)


@pytest.mark.parametrize("line", [
    "import pandas as pd",
    "import numpy as np",
    "import math",
    "from math import sqrt",
])
def test_whitelisted_imports_still_work(line):
    assert load_strategy_class(f"{line}\n{BODY}")


@pytest.mark.parametrize("line", [
    "import sklearn",
    "import ta",
    "from scipy import stats",
    "from pathlib import Path",
])
def test_non_whitelisted_modules_are_still_refused(line):
    with pytest.raises(ImportError):
        load_strategy_class(f"{line}\n{BODY}")


def test_the_framework_carve_out_cannot_smuggle_other_names():
    """`from os import Strategy, system` must not resolve — the carve-out is for
    the framework's own names only, and it imports nothing."""
    with pytest.raises(ImportError):
        load_strategy_class(f"from os import Strategy, system\n{BODY}")


def test_a_bare_import_of_an_unknown_module_is_refused():
    with pytest.raises(ImportError):
        load_strategy_class(f"import strategy\n{BODY}")


def test_denylist_still_screens_dangerous_source():
    with pytest.raises(ValueError):
        load_strategy_class(f"import os\n{BODY}")
