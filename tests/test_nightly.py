"""Tests for the unattended session: budgets, locking, and the run/reflection
bookkeeping the morning report depends on.

These never call an LLM or Docker — the session is driven with a stub loop so the
control flow is what's under test.
"""

import os
import time

import pytest

from algotrader.agent.memory import ExperimentDB
from algotrader.agent.nightly import (
    Budget,
    LockBusy,
    NightlySession,
    SessionLock,
    SymbolSpec,
    budget_from_config,
    specs_from_config,
)
from algotrader.config import load_config


@pytest.fixture
def cfg(tmp_path):
    c = load_config()
    c["experiments"]["db_path"] = str(tmp_path / "exp.db")
    c["experiments"]["generated_code_dir"] = str(tmp_path / "generated")
    c["agent"]["use_sandbox"] = False
    c["nightly"]["reflect_every"] = 0
    return c


class StubLoop:
    """Stands in for AgentLoop.step(), recording to the real DB."""

    def __init__(self, db, symbol, *, promote=False, raise_on=None, delay=0.0):
        self.db = db
        self.symbol = symbol
        self.promote = promote
        self.raise_on = raise_on
        self.delay = delay
        self.calls = 0

    def step(self):
        self.calls += 1
        if self.delay:
            time.sleep(self.delay)
        if self.raise_on == self.calls:
            raise RuntimeError("boom")
        exp_id = self.db.record(
            symbol=self.symbol, source="test", timeframe="1d",
            strategy_name=f"stub_{self.calls}", hypothesis="a stub idea",
            params={}, code="class X: pass", promoted=self.promote,
            metrics={"sharpe": 0.5}, gauntlet_checks={"walk_forward": {"mean_sharpe": 0.5}},
            reasons=["FAIL walk-forward: mean OOS Sharpe 0.50 < 1.0"],
            run_id=self.run_id,
        )
        return {"id": exp_id, "symbol": self.symbol, "promoted": self.promote,
                "strategy": f"stub_{self.calls}", "hypothesis": "a stub idea",
                "reasons": ["FAIL walk-forward: mean OOS Sharpe 0.50 < 1.0"],
                "error": None}


def _session(cfg, symbols=("AAA",), **budget_kw):
    specs = [SymbolSpec(s, "test", "1d") for s in symbols]
    return NightlySession(
        cfg, specs, budget=Budget(**budget_kw), allow_unsandboxed=True
    )


def _patch(session, monkeypatch, **stub_kw):
    """Replace data loading + AgentLoop with stubs; return the stub loops."""
    stubs = []

    def fake_load():
        return [(s, object()) for s in session.specs], []

    monkeypatch.setattr(session, "_load_data", fake_load)

    def fake_agentloop(df, cfg, *, symbol, source, timeframe, db, run_id, **kw):
        st = StubLoop(db, symbol, **stub_kw)
        st.run_id = run_id
        stubs.append(st)
        return st

    monkeypatch.setattr("algotrader.agent.nightly.AgentLoop", fake_agentloop)
    return stubs


# -- budgets -------------------------------------------------------------------
def test_iteration_budget_stops_the_session(cfg, monkeypatch):
    s = _session(cfg, max_hours=None, max_iterations=3, reflect_every=0)
    _patch(s, monkeypatch)
    r = s.run()
    assert r.iterations == 3
    assert r.stop_reason == "iterations"
    assert r.promoted == 0


def test_wall_clock_budget_stops_the_session(cfg, monkeypatch):
    # 0.4s budget, ~0.15s per iteration: the estimator should stop it in a few.
    s = _session(cfg, max_hours=0.4 / 3600, max_iterations=None, reflect_every=0)
    _patch(s, monkeypatch, delay=0.15)
    r = s.run()
    assert r.stop_reason == "budget"
    assert 1 <= r.iterations <= 4


def test_symbols_are_round_robined(cfg, monkeypatch):
    s = _session(cfg, symbols=("AAA", "BBB"), max_hours=None, max_iterations=4,
                 reflect_every=0)
    stubs = _patch(s, monkeypatch)
    s.run()
    assert [st.calls for st in stubs] == [2, 2]


# -- resilience ----------------------------------------------------------------
def test_iteration_crash_does_not_kill_the_session(cfg, monkeypatch):
    s = _session(cfg, max_hours=None, max_iterations=3, reflect_every=0)
    _patch(s, monkeypatch, raise_on=2)
    r = s.run()
    assert r.iterations == 3
    assert r.errors == 1
    assert r.stop_reason == "iterations"


def test_no_usable_data_is_fatal_but_closes_the_run(cfg, monkeypatch):
    s = _session(cfg, max_hours=None, max_iterations=2)
    monkeypatch.setattr(s, "_load_data", lambda: ([], ["AAA"]))
    r = s.run()
    assert r.stop_reason == "fatal"
    assert r.iterations == 0
    assert s.db.run(r.run_id)["finished_at"] is not None


def test_signal_stops_after_current_iteration(cfg, monkeypatch):
    s = _session(cfg, max_hours=None, max_iterations=50, reflect_every=0)
    stubs = _patch(s, monkeypatch)

    real_log = s.log

    def log_then_stop(msg=""):
        real_log(msg)
        if "iteration 2" in msg:
            s._stop_reason = "signal"

    monkeypatch.setattr(s, "log", log_then_stop)
    r = s.run()
    assert r.stop_reason == "signal"
    assert r.iterations == 2
    assert sum(st.calls for st in stubs) == 2


# -- sandbox gate --------------------------------------------------------------
def test_unattended_run_requires_sandbox(cfg):
    cfg["agent"]["use_sandbox"] = False
    with pytest.raises(RuntimeError, match="sandbox off"):
        NightlySession(cfg, [SymbolSpec("AAA", "test", "1d")])


# -- bookkeeping ---------------------------------------------------------------
def test_run_row_and_experiment_tagging(cfg, monkeypatch):
    s = _session(cfg, max_hours=None, max_iterations=2, reflect_every=0)
    _patch(s, monkeypatch, promote=True)
    r = s.run()

    row = s.db.run(r.run_id)
    assert row["kind"] == "nightly"
    assert row["iterations"] == 2 and row["promoted"] == 2
    assert row["finished_at"] >= row["started_at"]

    exps = s.db.experiments_between(0, time.time() + 1)
    assert len(exps) == 2
    assert all(e["run_id"] == r.run_id for e in exps)


def test_reflection_is_stored_without_an_llm(cfg, monkeypatch):
    s = _session(cfg, max_hours=None, max_iterations=2, reflect_every=0)
    _patch(s, monkeypatch)
    monkeypatch.setattr("algotrader.agent.nightly.make_client", lambda *a, **k: None)
    r = s.run()
    refl = s.db.reflections_for_run(r.run_id)
    assert len(refl) == 1
    assert refl[0]["source"] == "heuristic"
    assert "Experiments run: 2" in refl[0]["text"]


def test_reflection_uses_the_llm_when_available(cfg, monkeypatch):
    class FakeLLM:
        def describe(self):
            return "fake"

        def available(self):
            return True

        def complete(self, system, user, **kw):
            assert "Ideas tried" in user
            return "  Stop testing SMA crossovers; try volume-based ideas.  "

    # The session resolves its endpoint once at construction and shares that
    # client with every iteration, so this has to be in place beforehand.
    monkeypatch.setattr("algotrader.agent.nightly.make_client", lambda *a, **k: FakeLLM())
    s = _session(cfg, max_hours=None, max_iterations=1, reflect_every=0)
    _patch(s, monkeypatch)
    r = s.run()
    refl = s.db.reflections_for_run(r.run_id)
    assert refl[0]["source"] == "llm"
    assert refl[0]["text"].startswith("Stop testing SMA")


def test_reflections_feed_back_into_prompt_context(cfg):
    db = ExperimentDB(cfg["experiments"]["db_path"])
    db.record_reflection("Volume ideas next.", run_id=1, source="llm")
    ctx = db.lessons_context()
    assert "Volume ideas next." in ctx
    assert "conclusions from previous sessions" in ctx


def test_mid_session_reflection_fires_on_cadence(cfg, monkeypatch):
    s = _session(cfg, max_hours=None, max_iterations=4, reflect_every=2)
    _patch(s, monkeypatch)
    monkeypatch.setattr("algotrader.agent.nightly.make_client", lambda *a, **k: None)
    r = s.run()
    # iterations 2 and 4, plus the final one in the finally block
    assert len(s.db.reflections_for_run(r.run_id)) == 3


# -- lock ----------------------------------------------------------------------
def test_lock_blocks_a_second_live_session(tmp_path):
    path = tmp_path / "nightly.lock"
    with SessionLock(path):
        other = SessionLock(path)
        # simulate a different live process holding it
        path.write_text(f"{os.getppid()} now\n")
        with pytest.raises(LockBusy):
            other.acquire()
    assert not path.exists()


def test_stale_lock_is_taken_over(tmp_path):
    path = tmp_path / "nightly.lock"
    path.write_text("999999999 stale\n")   # a pid that cannot exist
    with SessionLock(path) as lock:
        assert lock.acquired
        assert str(os.getpid()) in path.read_text()
    assert not path.exists()


def test_lock_released_on_exception(tmp_path):
    path = tmp_path / "nightly.lock"
    with pytest.raises(ValueError):
        with SessionLock(path):
            raise ValueError("nope")
    assert not path.exists()


# -- config wiring -------------------------------------------------------------
def test_config_provides_symbols_and_budget():
    c = load_config()
    specs = specs_from_config(c)
    assert specs and all(s.symbol and s.source and s.timeframe for s in specs)
    b = budget_from_config(c)
    assert b.max_hours and b.max_hours > 0


def test_symbol_spec_parses_both_forms():
    c = load_config()
    plain = SymbolSpec.parse("BTC/USDT", c)
    assert plain.source == c["data"]["default_source"]
    mapped = SymbolSpec.parse(
        {"symbol": "SPY", "source": "yfinance", "timeframe": "1wk"}, c
    )
    assert (mapped.symbol, mapped.source, mapped.timeframe) == ("SPY", "yfinance", "1wk")
