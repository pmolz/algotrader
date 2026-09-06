"""The self-improvement loop: propose -> code -> validate -> record -> reflect.

Flow per iteration:
  1. Build a "lessons learned" context from the experiment DB.
  2. Ask the LLM for a new strategy (hypothesis + code) — implementing a research
     brief if one is available, otherwise inventing an idea. If no LLM is
     available, fall back to randomly mutating a baseline so the pipeline still
     runs offline.
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
from .prompts import (
    BRIEF_TEMPLATE,
    FIX_TEMPLATE,
    PROPOSE_TEMPLATE,
    SYSTEM_PROMPT,
    WORKED_EXAMPLE,
)
from .research import BriefLibrary
from .sandbox import (
    SandboxError,
    SandboxLimits,
    docker_available,
    run_gauntlet_sandboxed,
)


# Failures worth spending another LLM call on. Two kinds qualify: the code did
# not run, and the strategy is untestable because of how often it trades.
#
# The second is a calibration problem, not a result. "Fired once in 28,000 bars"
# and "traded 6.7 times a day" both mean the idea was never actually evaluated,
# and the fix — loosen or tighten the entry filter — is one the model can make
# from the number in the message without ever seeing a P&L.
#
# Verdicts about PERFORMANCE are deliberately absent. Retrying a candidate
# because its Sharpe was too low is asking the model to search until the dev set
# says yes, which is the definition of overfitting and the thing this whole
# gauntlet exists to prevent. Those get recorded and the loop moves on.
_REPAIRABLE_PREFIXES = (
    "FAIL codegen",
    "FAIL strategy runtime",
    "FAIL never traded",
    "FAIL too few trades",
    "FAIL overtrading",
)


def _is_repairable(reasons: list[str] | None) -> bool:
    if not reasons:
        return False
    return any(str(reasons[-1]).startswith(p) for p in _REPAIRABLE_PREFIXES)


class AgentLoop:
    def __init__(
        self,
        df: pd.DataFrame,
        cfg: dict,
        *,
        symbol: str,
        source: str,
        timeframe: str,
        db: ExperimentDB | None = None,
        run_id: int | None = None,
        llm=None,
    ):
        self.df = df
        self.cfg = cfg
        self.symbol = symbol
        self.source = source
        self.timeframe = timeframe
        # Advice earned on daily equities does not transfer to 15m crypto.
        self.regime = f"{source}:{timeframe}"
        # An unattended session shares one DB handle and tags every experiment
        # with its run id, so the morning report can scope to that session.
        self.db = db or ExperimentDB(get(cfg, "experiments.db_path"))
        self.run_id = run_id
        self.generated_dir = Path(get(cfg, "experiments.generated_code_dir"))
        self.generated_dir.mkdir(parents=True, exist_ok=True)
        # An unattended session resolves one endpoint up front and injects the
        # shared client, so every iteration of a night reports the same host.
        self.llm = llm if llm is not None else make_client(cfg)

        # Externally-researched ideas for the model to code, if any have been
        # written. An empty or absent directory is the normal, supported state:
        # the loop then invents its own ideas exactly as it did before.
        self.briefs: BriefLibrary | None = None
        if bool(get(cfg, "research.enabled", True)):
            self.briefs = BriefLibrary(
                get(cfg, "research.briefs_dir", "research/briefs"),
                max_attempts=int(get(cfg, "research.max_attempts_per_brief", 3)),
            )
            for problem in self.briefs.problems:
                print(f"[research] IGNORED malformed brief — {problem}")

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
            # same label the sandbox runner uses, so the report histograms agree
            return {"ok": False, "promoted": False, "reasons": ["FAIL strategy runtime"],
                    "checks": {}, "error": traceback.format_exc(limit=3)}
        return {"ok": True, "promoted": report.promoted, "reasons": report.reasons,
                "checks": report.checks, "error": None,
                "strategy_name": strategy.name, "params": strategy.params}

    # -- generation ---------------------------------------------------------------
    def _select_brief(self):
        """The research brief to implement this iteration, or None to invent one.

        Exhausting the library is not a failure — it means every researched idea
        has had its attempts, and the loop goes back to generating its own until
        the next batch of briefs lands.
        """
        if self.briefs is None:
            return None
        return self.briefs.select(self.db.brief_tally(self.symbol))

    def _propose(self) -> tuple[str, str, str, str | None]:
        """Return (hypothesis, code, strategy_source_label, brief_id)."""
        lessons = self.db.lessons_context(
            n_recent=get(self.cfg, "agent.memory_context_n", 8),
            symbol=self.symbol, regime=self.regime,
        )
        if self._llm_ready():
            brief = self._select_brief()
            if brief is not None:
                user = BRIEF_TEMPLATE.format(
                    symbol=self.symbol, timeframe=self.timeframe,
                    source=self.source, lessons=lessons, example=WORKED_EXAMPLE,
                    title=brief.title,
                    brief=brief.render(int(get(self.cfg, "research.max_chars", 2500))),
                )
            else:
                user = PROPOSE_TEMPLATE.format(
                    symbol=self.symbol, timeframe=self.timeframe,
                    source=self.source, lessons=lessons, example=WORKED_EXAMPLE,
                )
            text = self.llm.complete(SYSTEM_PROMPT, user)
            hypothesis, code = parse_response(text)
            return hypothesis, code, "llm", (brief.id if brief else None)
        # offline fallback: mutate a baseline so the plumbing is testable
        return self._offline_proposal()

    def _offline_proposal(self) -> tuple[str, str, str, str | None]:
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
        return hyp, code, "offline", None

    def _load(self, code: str) -> Strategy:
        cls = load_strategy_class(code)
        return cls()

    # -- one iteration ------------------------------------------------------------
    def step(self) -> dict:
        try:
            hypothesis, code, gen_src, brief_id = self._propose()
        except Exception as e:  # noqa: BLE001
            err = f"proposal failed: {type(e).__name__}: {e}"
            exp_id = self.db.record(
                symbol=self.symbol, source=self.source, timeframe=self.timeframe,
                strategy_name="(proposal)", hypothesis="", params={},
                code=None, promoted=False, metrics=None, gauntlet_checks=None,
                reasons=["FAIL proposal"], error=err, run_id=self.run_id,
                researched_from=None,
            )
            return {"id": exp_id, "promoted": False, "error": err}

        # honest multiple-testing count = every experiment ever run on this symbol
        n_trials = max(1, self.db.total_trials(self.symbol) + 1)

        # Evaluate, with LLM-assisted fix retries.
        #
        # Runtime failures used to get zero retries — the gate only fired on
        # FAIL codegen — even though they were 21% of all experiments and are
        # exactly the kind of thing a model fixes when shown the traceback
        # (wrong dtype, a column that doesn't exist, a bad quantile argument).
        # FIX_TEMPLATE was already written for both cases.
        rep = self._evaluate(code, n_trials)
        max_attempts = int(get(self.cfg, "agent.max_fix_attempts", 2))
        attempts = 0
        while (not rep["ok"] and self._llm_ready() and attempts < max_attempts
               and _is_repairable(rep["reasons"])):
            attempts += 1
            try:
                text = self.llm.complete(
                    SYSTEM_PROMPT, FIX_TEMPLATE.format(error=rep["error"], code=code)
                )
                # Keep the ORIGINAL hypothesis. A repair changes the code, not
                # the idea, and the model's reply here explains the fix — letting
                # that overwrite the hypothesis files "the error was caused by a
                # missing column" in the experiment log as the reason the
                # strategy was tried, which then feeds lessons_context.
                _discarded, code = parse_response(text)
                rep = self._evaluate(code, n_trials)
            except Exception:  # noqa: BLE001
                break

        name = rep.get("strategy_name") or f"({gen_src})"
        params = rep.get("params") or {}
        checks = rep.get("checks") or {}
        # Prefer the honest out-of-sample numbers; fall back to the dev-set ones
        # so a rejected candidate is still a data point rather than a blank row.
        metrics = checks.get("oos_holdout") or checks.get("dev_metrics") or {}

        exp_id = self.db.record(
            symbol=self.symbol, source=self.source, timeframe=self.timeframe,
            strategy_name=name, hypothesis=hypothesis, params=params,
            code=code, promoted=rep["promoted"], metrics=metrics,
            gauntlet_checks=checks, reasons=rep["reasons"], error=rep["error"],
            run_id=self.run_id, researched_from=brief_id,
        )

        if rep["promoted"]:
            (self.generated_dir / f"{exp_id:05d}_{name}.py").write_text(code)

        return {
            "id": exp_id,
            "symbol": self.symbol,
            "brief": brief_id,
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
            if r.get("brief"):
                print(f"Brief: {r['brief']}")
            print(f"[{r.get('strategy')}] {r.get('hypothesis', '')}")
            status = "PROMOTED ✅" if r.get("promoted") else "REJECTED ❌"
            print(f"Gauntlet: {status}")
            for reason in r.get("reasons") or []:
                print(f"  - {reason}")
            if r.get("error"):
                print(f"  error: {r['error'].splitlines()[-1] if r['error'] else ''}")
            results.append(r)
        return results
