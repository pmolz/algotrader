"""The self-improvement loop: propose -> code -> validate -> record -> reflect.

Flow per iteration:
  1. Build a "lessons learned" context from the experiment DB.
  2. Ask the LLM for a new strategy (hypothesis + code). If no LLM is available,
     fall back to randomly mutating a baseline so the pipeline still runs offline.
  3. Load the strategy safely, retrying once with a fix prompt on failure.
  4. Run it through the FULL validation gauntlet with honest n_trials.
  5. Record everything to the DB. Promoted strategies get their code saved.

The loop never touches money. Promotion just means "eligible for paper trading".
"""

from __future__ import annotations

import random
import traceback
from pathlib import Path

import pandas as pd

from ..config import get
from ..strategies.base import Strategy
from ..validation.gauntlet import run_gauntlet
from .codegen import load_strategy_class, parse_response
from .llm import AnthropicClient
from .memory import ExperimentDB
from .prompts import FIX_TEMPLATE, PROPOSE_TEMPLATE, SYSTEM_PROMPT


class AgentLoop:
    def __init__(self, df: pd.DataFrame, cfg: dict, *, symbol: str, source: str, timeframe: str):
        self.df = df
        self.cfg = cfg
        self.symbol = symbol
        self.source = source
        self.timeframe = timeframe
        self.db = ExperimentDB(get(cfg, "experiments.db_path"))
        self.generated_dir = Path(get(cfg, "experiments.generated_code_dir"))
        self.generated_dir.mkdir(parents=True, exist_ok=True)
        self.llm = AnthropicClient(get(cfg, "agent.model"))

    # -- generation ---------------------------------------------------------------
    def _propose(self) -> tuple[str, str, str]:
        """Return (hypothesis, code, strategy_source_label)."""
        lessons = self.db.lessons_context(
            n_recent=get(self.cfg, "agent.memory_context_n", 8)
        )
        if self.llm.available():
            user = PROPOSE_TEMPLATE.format(
                symbol=self.symbol, timeframe=self.timeframe,
                source=self.source, lessons=lessons,
            )
            text = self.llm.complete(SYSTEM_PROMPT, user)
            hypothesis, code = parse_response(text)
            return hypothesis, code, "llm"
        # offline fallback: mutate a baseline so the plumbing is testable
        return self._offline_proposal()

    def _offline_proposal(self) -> tuple[str, str, str]:
        fast = random.choice([5, 10, 15, 20])
        slow = random.choice([30, 50, 100, 200])
        code = f'''
class GeneratedSma(Strategy):
    name = "gen_sma_{fast}_{slow}"
    def __init__(self, fast={fast}, slow={slow}, **kw):
        super().__init__(fast=fast, slow=slow, **kw)
    def generate_signals(self, df):
        c = df["close"]
        f = c.rolling(self.fast).mean()
        s = c.rolling(self.slow).mean()
        pos = (f > s).astype(float)
        pos[s.isna()] = 0.0
        return StrategyResult(positions=pos)
'''.strip()
        hyp = f"[offline] SMA crossover {fast}/{slow} — plumbing test, no LLM available."
        return hyp, code, "offline"

    def _load(self, code: str) -> Strategy:
        cls = load_strategy_class(code)
        return cls()

    # -- one iteration ------------------------------------------------------------
    def step(self) -> dict:
        hypothesis, code, gen_src = self._propose()

        strategy = None
        error = None
        for attempt in range(2):
            try:
                strategy = self._load(code)
                break
            except Exception as e:  # noqa: BLE001
                error = f"{type(e).__name__}: {e}"
                if self.llm.available() and attempt == 0:
                    text = self.llm.complete(
                        SYSTEM_PROMPT, FIX_TEMPLATE.format(error=error, code=code)
                    )
                    try:
                        hypothesis, code = parse_response(text)
                    except Exception:  # noqa: BLE001
                        break
                else:
                    break

        if strategy is None:
            exp_id = self.db.record(
                symbol=self.symbol, source=self.source, timeframe=self.timeframe,
                strategy_name=f"({gen_src})", hypothesis=hypothesis, params={},
                code=code, promoted=False, metrics=None, gauntlet_checks=None,
                reasons=["FAIL codegen"], error=error,
            )
            return {"id": exp_id, "promoted": False, "error": error}

        # honest multiple-testing count = every experiment ever run on this symbol
        n_trials = max(1, self.db.total_trials(self.symbol) + 1)

        try:
            report = run_gauntlet(self.df, strategy, self.cfg, n_trials=n_trials)
        except Exception:  # noqa: BLE001
            err = traceback.format_exc(limit=3)
            exp_id = self.db.record(
                symbol=self.symbol, source=self.source, timeframe=self.timeframe,
                strategy_name=strategy.name, hypothesis=hypothesis,
                params=strategy.params, code=code, promoted=False,
                metrics=None, gauntlet_checks=None, reasons=["FAIL gauntlet crash"],
                error=err,
            )
            return {"id": exp_id, "promoted": False, "error": err}

        metrics = report.checks.get("oos_holdout") or report.checks.get(
            "deflated_sharpe", {}
        )
        exp_id = self.db.record(
            symbol=self.symbol, source=self.source, timeframe=self.timeframe,
            strategy_name=strategy.name, hypothesis=hypothesis,
            params=strategy.params, code=code, promoted=report.promoted,
            metrics=metrics, gauntlet_checks=report.checks, reasons=report.reasons,
        )

        if report.promoted:
            (self.generated_dir / f"{exp_id:05d}_{strategy.name}.py").write_text(code)

        return {
            "id": exp_id,
            "promoted": report.promoted,
            "strategy": strategy.name,
            "hypothesis": hypothesis,
            "report": report,
        }

    def run(self, iterations: int) -> list[dict]:
        results = []
        for i in range(iterations):
            print(f"\n=== Iteration {i + 1}/{iterations} ===")
            r = self.step()
            if "report" in r:
                print(f"[{r['strategy']}] {r['hypothesis']}")
                print(r["report"].pretty())
            else:
                print(f"codegen/gauntlet error: {r.get('error')}")
            results.append(r)
        return results
