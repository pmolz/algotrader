"""Unattended overnight session: propose -> code -> validate -> log -> reflect.

This is the loop cron drives. Everything here exists to make *leaving it alone*
safe:

  * Sandbox required.  Unattended + in-process `exec` of model-written code is
    not a combination worth having. You must pass allow_unsandboxed=True to
    override, and the run row records that you did.
  * One lock.  A session that overruns into the next cron slot is skipped rather
    than run twice against the same DB.
  * Errors are per-iteration.  A crashed candidate costs one iteration, not the
    night. Data that won't load costs one symbol, not the night.
  * Two budgets.  Wall clock and iteration count; whichever binds first stops
    the session. Wall clock is checked with a running estimate of iteration cost
    so a session doesn't start an iteration it can't finish before the deadline.
  * Signals are graceful.  SIGTERM/SIGINT finish the current iteration, then
    reflect and close the run row cleanly.
  * Round-robin symbols.  Budget is spread across markets instead of being spent
    entirely on whichever one happens to be first.

The reflect step is what makes the loop cumulative: the agent summarizes its own
session, that summary is stored, and `ExperimentDB.lessons_context` feeds it back
into tomorrow's proposals.
"""

from __future__ import annotations

import json
import os
import signal
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TextIO

from ..config import get
from ..data import fetch
from ..data.loader import load_cached
from .llm import make_client
from .loop import AgentLoop
from .memory import ExperimentDB
from .prompts import REFLECT_SYSTEM, REFLECT_TEMPLATE
from .report import session_digest_text
from .sandbox import docker_available, image_exists


@dataclass
class SymbolSpec:
    symbol: str
    source: str
    timeframe: str

    @classmethod
    def parse(cls, entry, cfg: dict) -> SymbolSpec:
        """Accept either 'BTC/USDT' or {symbol:, source:, timeframe:}."""
        default_source = get(cfg, "data.default_source", "ccxt")
        default_tf = get(cfg, "data.default_timeframe", "1d")
        if isinstance(entry, str):
            return cls(entry, default_source, default_tf)
        return cls(
            entry["symbol"],
            entry.get("source", default_source),
            entry.get("timeframe", default_tf),
        )


@dataclass
class Budget:
    max_hours: float | None = 8.0
    max_iterations: int | None = None
    reflect_every: int = 10          # 0 disables mid-session reflection

    def as_dict(self) -> dict:
        return {
            "max_hours": self.max_hours,
            "max_iterations": self.max_iterations,
            "reflect_every": self.reflect_every,
        }


class LockBusy(RuntimeError):
    pass


class SessionLock:
    """PID lock file. A lock whose PID is gone is treated as stale and taken."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.acquired = False

    def _holder_alive(self) -> bool:
        try:
            pid = int(self.path.read_text().split()[0])
        except (OSError, ValueError, IndexError):
            return False
        if pid == os.getpid():
            return False
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True     # exists, owned by someone else
        return True

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            if self._holder_alive():
                raise LockBusy(
                    f"another session holds {self.path} "
                    f"({self.path.read_text().strip()}); not starting a second one"
                )
            self.path.unlink(missing_ok=True)   # stale
        self.path.write_text(f"{os.getpid()} {datetime.now().astimezone().isoformat()}\n")
        self.acquired = True

    def release(self) -> None:
        if self.acquired:
            self.path.unlink(missing_ok=True)
            self.acquired = False

    def __enter__(self) -> SessionLock:
        self.acquire()
        return self

    def __exit__(self, *exc) -> None:
        self.release()


@dataclass
class NightlyResult:
    run_id: int
    iterations: int
    promoted: int
    errors: int
    stop_reason: str
    skipped_symbols: list[str] = field(default_factory=list)


class NightlySession:
    def __init__(
        self,
        cfg: dict,
        symbols: list[SymbolSpec],
        *,
        budget: Budget | None = None,
        kind: str = "nightly",
        allow_unsandboxed: bool = False,
        log_file: TextIO | None = None,
        host: str | None = None,
    ):
        self.cfg = cfg
        self.specs = symbols
        self.budget = budget or Budget()
        self.kind = kind
        self.log_file = log_file
        self._stop_reason: str | None = None

        if not self.specs:
            raise ValueError("no symbols configured for the nightly session")

        self.sandbox = bool(get(cfg, "agent.use_sandbox", False))
        if not self.sandbox and not allow_unsandboxed:
            raise RuntimeError(
                "Refusing to run unattended with the sandbox off. Set "
                "agent.use_sandbox: true (and run `bash sandbox/build.sh`), or "
                "pass --allow-unsandboxed if you really mean it."
            )
        if self.sandbox:
            image = get(cfg, "sandbox.image", "algotrader-sandbox:latest")
            if not docker_available():
                raise RuntimeError("sandbox enabled but Docker is not available")
            if not image_exists(image):
                raise RuntimeError(
                    f"sandbox image {image!r} missing — run: bash sandbox/build.sh"
                )

        self.db = ExperimentDB(get(cfg, "experiments.db_path"))
        self.provider = get(cfg, "agent.provider")
        self.model = get(cfg, "agent.model")
        # Resolve the LLM endpoint once for the whole session and share it with
        # every AgentLoop, so the run row records which machine did the work.
        self.llm = make_client(cfg, host=host, logger=self.log)
        self.llm_available = self.llm is not None and self.llm.available()
        # describe() is cosmetic — never let logging take down a session.
        self.host = (getattr(self.llm, "describe", lambda: self.model)()
                     if self.llm is not None else None)
        if not self.llm_available:
            # Worth shouting about: the session will still run, but every
            # candidate will come from the offline baseline mutator, which is a
            # plumbing test rather than research.
            self.log(f"WARNING no LLM endpoint available ({self.host or self.provider}) "
                     f"— the session will fall back to offline baseline mutation")

    # -- plumbing ----------------------------------------------------------------
    def log(self, msg: str = "") -> None:
        line = f"[{datetime.now().astimezone().strftime('%H:%M:%S')}] {msg}" if msg else ""
        print(line, flush=True)
        if self.log_file:
            self.log_file.write(line + "\n")
            self.log_file.flush()

    def _install_signal_handlers(self) -> None:
        def handler(signum, _frame):
            name = signal.Signals(signum).name
            self._stop_reason = "signal"
            self.log(f"received {name} — finishing the current iteration, then stopping")

        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                signal.signal(sig, handler)
            except ValueError:
                pass    # not on the main thread (e.g. under a test runner)

    def _load_data(self) -> tuple[list[tuple[SymbolSpec, object]], list[str]]:
        """Fetch every symbol up front, while the session still has network.

        A refresh failure falls back to whatever is cached — a stale-but-real
        history is far better than losing the night. Only a symbol with no data
        at all is skipped.
        """
        loaded, skipped = [], []
        cache_dir = get(self.cfg, "data.cache_dir", "data_cache")
        exchange = get(self.cfg, "data.ccxt_exchange", "bitstamp")
        for spec in self.specs:
            try:
                df = fetch(spec.symbol, source=spec.source, timeframe=spec.timeframe,
                           cache_dir=cache_dir, refresh=True, exchange=exchange)
                self.log(f"data {spec.symbol} [{spec.timeframe}] {len(df)} bars (fresh)")
            except Exception as e:  # noqa: BLE001
                df = load_cached(cache_dir, spec.source, spec.symbol, spec.timeframe)
                if df is None or not len(df):
                    self.log(f"SKIP {spec.symbol}: no data ({type(e).__name__}: {e})")
                    skipped.append(spec.symbol)
                    continue
                self.log(
                    f"data {spec.symbol} [{spec.timeframe}] {len(df)} bars "
                    f"(CACHED — refresh failed: {type(e).__name__})"
                )
            loaded.append((spec, df))
        return loaded, skipped

    def _budget_exhausted(self, started: float, iterations: int, avg_s: float) -> str | None:
        if self._stop_reason:
            return self._stop_reason
        if self.budget.max_iterations and iterations >= self.budget.max_iterations:
            return "iterations"
        if self.budget.max_hours:
            deadline = started + self.budget.max_hours * 3600
            now = time.time()
            if now >= deadline:
                return "budget"
            # don't start an iteration that probably can't finish in time
            if avg_s and now + avg_s > deadline:
                self.log(
                    f"stopping early: ~{avg_s:.0f}s per iteration won't fit in the "
                    f"{deadline - now:.0f}s left"
                )
                return "budget"
        return None

    # -- reflection --------------------------------------------------------------
    def _reflect(self, run_id: int, since: float) -> None:
        """Summarize the session so far and store it as memory for next time."""
        exps = self.db.experiments_between(since, time.time() + 1)
        exps = [e for e in exps if e.get("run_id") == run_id]
        if not exps:
            return
        summary = session_digest_text(exps)
        llm = self.llm
        text, source = summary, "heuristic"
        if llm is not None and llm.available():
            try:
                out = llm.complete(
                    REFLECT_SYSTEM, REFLECT_TEMPLATE.format(summary=summary)
                ).strip()
                if out:
                    text, source = out, "llm"
            except Exception as e:  # noqa: BLE001
                self.log(f"reflection failed ({type(e).__name__}: {e}); storing digest")
        self.db.record_reflection(text, run_id=run_id, source=source)
        self.log(f"reflection stored ({source}, {len(text)} chars)")

    # -- the session -------------------------------------------------------------
    def run(self) -> NightlyResult:
        started = time.time()
        symbols = [s.symbol for s in self.specs]
        run_id = self.db.start_run(
            kind=self.kind, symbols=symbols, provider=self.provider,
            model=self.model, sandbox=self.sandbox, budget=self.budget.as_dict(),
            host=self.host,
        )
        self.log(f"=== session {run_id} start ===")
        self.log(
            f"provider={self.provider} model={self.model} "
            f"llm={self.host or 'none'} "
            f"sandbox={'on' if self.sandbox else 'OFF'} "
            f"budget={json.dumps(self.budget.as_dict())}"
        )
        self._install_signal_handlers()

        iterations = promoted = errors = 0
        stop_reason = "budget"
        loaded, skipped = [], list(symbols)
        try:
            loaded, skipped = self._load_data()
            if not loaded:
                stop_reason = "fatal"
                self.log("no symbol had usable data — nothing to do")
            else:
                loops = [
                    AgentLoop(df, self.cfg, llm=self.llm,
                              symbol=spec.symbol, source=spec.source,
                              timeframe=spec.timeframe, db=self.db, run_id=run_id)
                    for spec, df in loaded
                ]
                avg_s = 0.0
                last_reflect_at = 0
                while True:
                    reason = self._budget_exhausted(started, iterations, avg_s)
                    if reason:
                        stop_reason = reason
                        break
                    loop = loops[iterations % len(loops)]
                    t0 = time.time()
                    iterations += 1
                    self.log(f"--- iteration {iterations} [{loop.symbol}] ---")
                    try:
                        r = loop.step()
                        if r.get("promoted"):
                            promoted += 1
                        if r.get("error"):
                            errors += 1
                        self._log_iteration(r)
                    except Exception:  # noqa: BLE001
                        errors += 1
                        self.log("iteration crashed outside the harness:")
                        self.log(traceback.format_exc(limit=4))
                    took = time.time() - t0
                    # running mean, so the deadline estimate adapts to reality
                    avg_s = took if avg_s == 0 else 0.7 * avg_s + 0.3 * took
                    self.log(f"iteration took {took:.0f}s (avg {avg_s:.0f}s)")

                    every = self.budget.reflect_every
                    if every and iterations - last_reflect_at >= every:
                        self._reflect(run_id, started)
                        last_reflect_at = iterations
        except Exception:  # noqa: BLE001
            stop_reason = "fatal"
            self.log("session aborted:")
            self.log(traceback.format_exc(limit=6))
        finally:
            try:
                self._reflect(run_id, started)
            except Exception:  # noqa: BLE001
                self.log("final reflection failed; continuing to close the run")
            self.db.finish_run(
                run_id, iterations=iterations, promoted=promoted,
                errors=errors, stop_reason=stop_reason,
            )
            self.log(
                f"=== session {run_id} end: {iterations} iterations, "
                f"{promoted} promoted, {errors} errors, stopped on {stop_reason} "
                f"after {(time.time() - started) / 60:.0f}m ==="
            )

        return NightlyResult(run_id, iterations, promoted, errors, stop_reason, skipped)

    def _log_iteration(self, r: dict) -> None:
        status = "PROMOTED ✅" if r.get("promoted") else "rejected"
        self.log(f"#{r.get('id')} {r.get('strategy')} -> {status}")
        hyp = " ".join((r.get("hypothesis") or "").split())
        if hyp:
            self.log(f"  hypothesis: {hyp[:200]}")
        for reason in r.get("reasons") or []:
            self.log(f"  {reason}")
        if r.get("error"):
            self.log(f"  error: {r['error'].strip().splitlines()[-1]}")


def specs_from_config(cfg: dict) -> list[SymbolSpec]:
    entries = get(cfg, "nightly.symbols", []) or []
    return [SymbolSpec.parse(e, cfg) for e in entries]


def budget_from_config(cfg: dict) -> Budget:
    return Budget(
        max_hours=get(cfg, "nightly.max_hours", 8.0),
        max_iterations=get(cfg, "nightly.max_iterations", None),
        reflect_every=int(get(cfg, "nightly.reflect_every", 10) or 0),
    )
