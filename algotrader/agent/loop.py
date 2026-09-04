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
from .llm import make_client
from .memory import ExperimentDB
from .prompts import FIX_TEMPLATE, PROPOSE_TEMPLATE, SYSTEM_PROMPT
from .sandbox import (
    SandboxError,
    SandboxLimits,
    docker_available,
    run_gauntlet_sandboxed,
)


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
        self.llm = make_client(get(cfg, "agent.provider"), get(cfg, "agent.model"))

        self.use_sandbox = bool(get(cfg, "agent.use_sandbox", False))
        self.sandbox_limits = SandboxLimits(
            image=get(cfg, "sandbox.image", "algotrader-sandbox:latest"),
            memory=get(cfg, "sandbox.memory", "1g"),
            cpus=str(get(cfg, "sandbox.cpus", "2")),
            pids=int(get(cfg, "sandbox.pids", 128)),
            timeout_s=int(get(cfg, "sandbox.timeout_s", 120)),
        )
        if self.use_sandbox and not docker_available():
            raise RuntimeError(
                "agent.use_sandbox is true but Docker is not available. "
                "Start Docker or set use_sandbox: false."
            )

    def _llm_ready(self) -> bool:
        return self.llm is not None and self.llm.available()

    def _evaluate(self, code: str, n_trials: int) -> dict:
        """Load + gauntlet a candidate. Returns a normalized report dict:
        {ok, promoted, reasons, checks, error, strategy_name?, params?}.

        Sandbox mode NEVER execs the untrusted code in this process.
        """
        if self.use_sandbox:
            try:
                return run_gauntlet_sandboxed(
                    self.df, code, self.cfg, n_trials=n_trials, limits=self.sandbox_limits
                )
            except SandboxError as e:
                return {"ok": False, "promoted": False, "reasons": ["FAIL sandbox"],
                        "checks": {}, "error": str(e)}
        # in-process path (supervised local use only)
        try:
            strategy = self._load(code)
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "promoted": False, "reasons": ["FAIL codegen"],
                    "checks": {}, "error": f"{type(e).__name__}: {e}"}
        try:
            report = run_gauntlet(self.df, strategy, self.cfg, n_trials=n_trials)
        except Exception:  # noqa: BLE001
            return {"ok": False, "promoted": False, "reasons": ["FAIL gauntlet crash"],
                    "checks": {}, "error": traceback.format_exc(limit=3)}
        return {"ok": True, "promoted": report.promoted, "reasons": report.reasons,
                "checks": report.checks, "error": None,
                "strategy_name": strategy.name, "params": strategy.params}

    # -- generation ---------------------------------------------------------------
    def _propose(self) -> tuple[str, str, str]:
        """Return (hypothesis, code, strategy_source_label)."""
        lessons = self.db.lessons_context(
            n_recent=get(self.cfg, "agent.memory_context_n", 8)
        )
        if self._llm_ready():
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
        try:
            hypothesis, code, gen_src = self._propose()
        except Exception as e:  # noqa: BLE001
            err = f"proposal failed: {type(e).__name__}: {e}"
            exp_id = self.db.record(
                symbol=self.symbol, source=self.source, timeframe=self.timeframe,
                strategy_name="(proposal)", hypothesis="", params={},
                code=None, promoted=False, metrics=None, gauntlet_checks=None,
                reasons=["FAIL proposal"], error=err,
            )
            return {"id": exp_id, "promoted": False, "error": err}

        # honest multiple-testing count = every experiment ever run on this symbol
        n_trials = max(1, self.db.total_trials(self.symbol) + 1)

        # Evaluate, with one LLM-assisted fix retry on a code/parse error.
        rep = self._evaluate(code, n_trials)
        if not rep["ok"] and self._llm_ready() and rep["reasons"] == ["FAIL codegen"]:
            try:
                text = self.llm.complete(
                    SYSTEM_PROMPT, FIX_TEMPLATE.format(error=rep["error"], code=code)
                )
                hypothesis, code = parse_response(text)
                rep = self._evaluate(code, n_trials)
            except Exception:  # noqa: BLE001
                pass

        name = rep.get("strategy_name") or f"({gen_src})"
        params = rep.get("params") or {}
        checks = rep.get("checks") or {}
        metrics = checks.get("oos_holdout") or checks.get("deflated_sharpe", {})

        exp_id = self.db.record(
            symbol=self.symbol, source=self.source, timeframe=self.timeframe,
            strategy_name=name, hypothesis=hypothesis, params=params,
            code=code, promoted=rep["promoted"], metrics=metrics,
            gauntlet_checks=checks, reasons=rep["reasons"], error=rep["error"],
        )

        if rep["promoted"]:
            (self.generated_dir / f"{exp_id:05d}_{name}.py").write_text(code)

        return {
            "id": exp_id,
            "promoted": rep["promoted"],
            "strategy": name,
            "hypothesis": hypothesis,
            "reasons": rep["reasons"],
            "error": rep["error"],
        }

    def run(self, iterations: int) -> list[dict]:
        results = []
        for i in range(iterations):
            print(f"\n=== Iteration {i + 1}/{iterations} ===")
            r = self.step()
            print(f"[{r.get('strategy')}] {r.get('hypothesis', '')}")
            status = "PROMOTED ✅" if r.get("promoted") else "REJECTED ❌"
            print(f"Gauntlet: {status}")
            for reason in r.get("reasons") or []:
                print(f"  - {reason}")
            if r.get("error"):
                print(f"  error: {r['error'].splitlines()[-1] if r['error'] else ''}")
            results.append(r)
        return results
