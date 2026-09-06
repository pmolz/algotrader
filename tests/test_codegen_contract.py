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


# -- hypothesis extraction ----------------------------------------------------
# The hypothesis feeds lessons_context, so dropping one costs the loop its memory
# of why an idea was tried. Models emit the label inconsistently.
from algotrader.agent.codegen import parse_response  # noqa: E402

CODE = 'class T(Strategy):\n    name = "t"\n'


def test_labelled_hypothesis_is_extracted():
    h, c = parse_response(f"HYPOTHESIS: liquidations overshoot\n```python\n{CODE}```")
    assert h == "liquidations overshoot" and "class T" in c


def test_a_multi_line_hypothesis_is_kept_whole():
    """The prompt asks for a mechanism AND a falsification test — two lines."""
    text = ("HYPOTHESIS: liquidations force selling that overshoots.\n"
            "Falsified if reversion does not exceed costs.\n"
            f"```python\n{CODE}```")
    h, _ = parse_response(text)
    assert "Falsified if" in h, "only the first line survived"


def test_unlabelled_prose_before_the_code_is_used_as_the_hypothesis():
    text = f"**Mean reversion after a volume spike.**\n\n```python\n{CODE}```"
    h, _ = parse_response(text)
    assert "Mean reversion after a volume spike" in h


def test_markdown_headings_are_stripped_from_the_fallback():
    h, _ = parse_response(f"## Volume Spike Reversion\nBuys the dip.\n```python\n{CODE}```")
    assert h.startswith("Volume Spike Reversion")


def test_a_bare_fence_without_a_language_tag_still_parses():
    h, c = parse_response(f"HYPOTHESIS: x\n```\n{CODE}```")
    assert "class T" in c


def test_no_code_block_is_still_an_error():
    with pytest.raises(ValueError):
        parse_response("HYPOTHESIS: I forgot the code")


def test_nothing_at_all_falls_back_to_a_placeholder():
    h, _ = parse_response(f"```python\n{CODE}```")
    assert h == "(no hypothesis provided)"
