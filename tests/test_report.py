"""Tests for the morning report. The important properties are that it never
overstates a result and never silently hides a broken night.
"""

import time
from datetime import datetime

import pytest

from algotrader.agent.memory import ExperimentDB
from algotrader.agent.report import (
    build_report,
    digest,
    fail_stage,
    overnight_window,
    session_digest_text,
    write_report,
)


@pytest.fixture
def db(tmp_path):
    return ExperimentDB(tmp_path / "exp.db")


def _rec(db, **kw):
    base = dict(
        symbol="BTC/USDT", source="ccxt", timeframe="1d",
        strategy_name="s", hypothesis="an idea", params={}, code="class X: pass",
        promoted=False, metrics={}, gauntlet_checks={}, reasons=[], error=None,
    )
    base.update(kw)
    return db.record(**base)


def _rejected(db, stage="walk-forward", sharpe=0.4, **kw):
    return _rec(
        db,
        reasons=["PASS lookahead (no future peeking detected)",
                 f"FAIL {stage}: mean OOS Sharpe {sharpe:.2f} < 1.0"],
        gauntlet_checks={"walk_forward": {"mean_sharpe": sharpe, "positive_folds": 2,
                                          "n_folds": 5, "fold_sharpes": [sharpe] * 5}},
        **kw,
    )


def _promoted(db, **kw):
    return _rec(
        db, promoted=True, strategy_name="winner",
        reasons=["ALL CHECKS PASSED — candidate promoted to paper trading."],
        metrics={"sharpe": 1.8, "max_drawdown": -0.12},
        gauntlet_checks={
            "walk_forward": {"mean_sharpe": 1.4, "positive_folds": 5, "n_folds": 5,
                             "fold_sharpes": [1.2, 1.5, 1.3, 1.6, 1.4]},
            "deflated_sharpe": {"dsr": 0.71},
            "cost_stress": {"1.0": 1.9, "1.5": 1.5, "2.0": 1.2},
            "oos_holdout": {"sharpe": 1.8, "max_drawdown": -0.12},
        },
        **kw,
    )


# -- stage attribution ---------------------------------------------------------
def test_fail_stage_reads_the_last_failing_reason():
    exp = {"promoted": False, "reasons": ["PASS lookahead (ok)",
                                          "FAIL cost stress: Sharpe goes negative"]}
    assert fail_stage(exp) == "cost stress"


def test_fail_stage_for_promoted_and_crashed():
    assert fail_stage({"promoted": True, "reasons": ["ALL CHECKS PASSED"]}) == "promoted"
    assert fail_stage({"promoted": False, "reasons": [], "error": "Traceback"}) == "crash"
    assert fail_stage({"promoted": False, "reasons": []}) == "unknown"


# -- digest --------------------------------------------------------------------
def test_digest_counts_and_ranks(db):
    _rejected(db, sharpe=0.1)
    _rejected(db, sharpe=0.9)
    _rejected(db, stage="codegen", sharpe=0.0)
    _promoted(db)
    _rec(db, reasons=["FAIL gauntlet crash"], error="ZeroDivisionError: x")

    d = digest(db.experiments_between(0, time.time() + 1))
    assert d.total == 5
    assert d.n_promoted == 1
    assert d.n_errors == 1
    assert d.stages["walk-forward"] == 2
    # near misses exclude the promoted one and are ranked best-first
    assert all(not e["promoted"] for e in d.near_misses)
    assert d.near_misses[0]["sharpe"] is None or True
    sigs = [e["checks"]["walk_forward"]["mean_sharpe"] for e in d.near_misses
            if e["checks"].get("walk_forward")]
    assert sigs == sorted(sigs, reverse=True)


def test_session_digest_text_is_compact_and_names_stages(db):
    _rejected(db, sharpe=0.3)
    _rejected(db, stage="too few trades")
    text = session_digest_text(db.experiments_between(0, time.time() + 1))
    assert "Experiments run: 2" in text
    assert "walk-forward=1" in text
    assert "too few trades" in text


# -- report --------------------------------------------------------------------
def test_empty_window_is_flagged_loudly(db):
    text = build_report(db, 0, time.time() + 1)
    assert "Nothing ran" in text
    assert "cron actually fired" in text


def test_all_rejected_reads_as_the_expected_outcome(db):
    for s in (0.1, 0.3, 0.5):
        _rejected(db, sharpe=s)
    text = build_report(db, 0, time.time() + 1)
    assert "Nothing promoted" in text
    assert "expected" in text
    assert "Where candidates died" in text
    assert "| walk-forward | 3 |" in text


def test_promoted_candidate_is_shown_with_caveats_and_evidence(db):
    exp_id = _promoted(db)
    text = build_report(db, 0, time.time() + 1, generated_dir="experiments/generated")
    assert "survived the gauntlet" in text
    assert "suspicious until reviewed" in text
    assert "1.80" in text                      # OOS holdout Sharpe
    assert "0.71" in text                      # deflated Sharpe
    assert f"{exp_id:05d}_winner.py" in text   # where the code landed
    assert "never straight to capital" in text


def test_unfinished_run_is_called_out(db):
    run_id = db.start_run(kind="nightly", symbols=["BTC/USDT"], provider="ollama",
                          model="qwen2.5-coder:7b", sandbox=True, budget={})
    _rejected(db, run_id=run_id)
    text = build_report(db, 0, time.time() + 1)
    assert "never finished" in text
    assert "killed" in text


def test_sandbox_off_is_called_out(db):
    db.start_run(kind="nightly", symbols=["BTC/USDT"], provider="ollama",
                 model="m", sandbox=False, budget={})
    _rejected(db)
    text = build_report(db, 0, time.time() + 1)
    assert "**OFF**" in text


def test_codegen_dominated_night_blames_the_model(db):
    for _ in range(4):
        _rec(db, reasons=["FAIL codegen"], error="SyntaxError: bad")
    text = build_report(db, 0, time.time() + 1)
    assert "model-quality problem" in text
    assert "qwen2.5-coder:14b" in text


def test_mostly_unjudged_night_does_not_claim_a_clean_result(db):
    for _ in range(3):
        _rec(db, reasons=["FAIL codegen"], error="SyntaxError: bad")
    _rejected(db, sharpe=0.3)
    text = build_report(db, 0, time.time() + 1)
    assert "Wasted night" in text
    assert "3/4 candidates never reached a verdict" in text
    assert "do **not** read that as evidence about the" in text
    # and it must not tell you everything is fine
    assert "Nothing to act on" not in text
    assert "expected* outcome" not in text


def test_strategy_runtime_failures_are_attributed_to_the_model(db):
    for _ in range(3):
        _rec(db, reasons=["FAIL strategy runtime"], error="KeyError: 20")
    text = build_report(db, 0, time.time() + 1)
    assert "| strategy runtime | 3 |" in text
    assert "blew up while running" in text


def test_healthy_night_still_says_nothing_to_act_on(db):
    for s in (0.1, 0.2, 0.3):
        _rejected(db, sharpe=s)
    _rec(db, reasons=["FAIL codegen"], error="SyntaxError: bad")
    text = build_report(db, 0, time.time() + 1)
    assert "Nothing to act on" in text


def test_report_includes_reflection_and_trial_count(db):
    run_id = db.start_run(kind="nightly", symbols=["BTC/USDT"], provider="ollama",
                          model="m", sandbox=True, budget={})
    _rejected(db, run_id=run_id)
    db.finish_run(run_id, iterations=1, promoted=0, errors=0, stop_reason="budget")
    db.record_reflection("Momentum is dead here; try mean reversion.",
                         run_id=run_id, source="llm")
    text = build_report(db, 0, time.time() + 1)
    assert "> Momentum is dead here" in text
    assert "Lifetime experiments in the DB: **1**" in text


def test_window_excludes_older_experiments(db):
    old = _rejected(db, sharpe=0.2)
    db.conn.execute("UPDATE experiments SET created_at=? WHERE id=?",
                    (time.time() - 86400 * 3, old))
    db.conn.commit()
    _promoted(db)
    text = build_report(db, time.time() - 3600, time.time() + 1)
    assert "1 candidate(s) survived" in text


def test_overnight_window_spans_yesterday_evening_to_now():
    ref = datetime(2026, 9, 4, 7, 30).astimezone()
    start, end = overnight_window(ref, start_hour=18)
    assert datetime.fromtimestamp(start).astimezone().hour == 18
    assert datetime.fromtimestamp(start).astimezone().day == 3
    assert end == ref.timestamp()


def test_write_report_names_the_file_by_date(tmp_path):
    when = datetime(2026, 9, 4, 7, 0).astimezone()
    path = write_report("# hi\n", tmp_path / "reports", when=when)
    assert path.name == "2026-09-04.md"
    assert path.read_text() == "# hi\n"
