"""What the agent is actually told about its own past.

This block is the only mechanism by which the loop is "self-improving", and it
is easy for it to be plumbed correctly and still carry nothing. Measured on a
real 155-experiment log before this was fixed: 0% of rows had a Sharpe, so the
"best so far" section was permanently empty; 31% of the block was raw lookahead
cut-indices; and recent entries listed strategy *names* without the ideas behind
them, so "propose something meaningfully different" was an instruction the model
had no information to follow. It proposed reversion ten times in a row.
"""

from __future__ import annotations

import json

import pytest

from algotrader.agent.memory import ExperimentDB, classify_family, summarize_reason


@pytest.fixture
def db(tmp_path):
    return ExperimentDB(tmp_path / "e.db")


def add(db, name, hypothesis="", reasons=(), metrics=None, promoted=False,
        symbol="BTC/USD"):
    return db.record(
        symbol=symbol, source="ccxt", timeframe="15m", strategy_name=name,
        hypothesis=hypothesis, params={}, code="x", promoted=promoted,
        metrics=metrics, gauntlet_checks={}, reasons=list(reasons), error=None,
    )


PASS_LOOK = "PASS lookahead (no future peeking detected)"
PASS_INT = "PASS trading intensity: 1.20 trades/day, 40%/yr cost drag"
FAIL_LOOK = ("FAIL lookahead: signal depends on future data "
             "[{'cut': 5606, 'probe': 'future_x10', 'n_changed_bars': 4, 'first_bar': 2155}]")


# -- reason summarising -------------------------------------------------------
def test_lookahead_verdicts_lose_their_payload():
    """Five cut-indices teach a model nothing and were 31% of the whole block."""
    out = summarize_reason(FAIL_LOOK)
    assert "cut" not in out and "probe" not in out
    assert "future data" in out


def test_verdicts_that_carry_their_own_numbers_are_kept():
    r = "FAIL overtrading: 5.0 trades/day > 4.0 (272% of capital per year in costs)"
    assert "5.0 trades/day" in summarize_reason(r)


def test_cost_stress_dict_is_dropped_but_the_lesson_kept():
    out = summarize_reason("FAIL cost stress: Sharpe goes negative under stress {'1.0': 1.2}")
    assert "1.2" not in out and "cost" in out.lower()


# -- family classification ----------------------------------------------------
@pytest.mark.parametrize("text,fam", [
    ("leveraged_reversion", "mean-reversion"),
    ("range_breakout_reversion buys the dip", "mean-reversion"),
    ("donchian_channel_breakout", "breakout"),
    ("sma_volume_crossover", "momentum/trend"),
    ("atr_squeeze_play", "volatility"),
    ("obv_flow_divergence", "volume/flow"),
    ("market_microstructure_arbitrage", "microstructure"),
    ("rsi_oscillator_fade", "oscillator"),
    ("time_of_day_effect", "seasonality"),
    ("something_unmappable", "other"),
])
def test_family_classification(text, fam):
    assert classify_family(text) == fam


# -- the assembled block ------------------------------------------------------
def test_family_tally_is_stated_so_diversity_is_actionable(db):
    for i in range(3):
        add(db, f"mean_reversion_{i}", reasons=[PASS_LOOK])
    add(db, "donchian_breakout", reasons=[PASS_LOOK])
    ctx = db.lessons_context()
    assert "mean-reversion: 3 attempt(s)" in ctx
    assert "breakout: 1 attempt(s)" in ctx
    assert "Propose a family that is absent" in ctx


def test_hypotheses_appear_so_ideas_are_visible_not_just_names(db):
    add(db, "liquidation_reversion",
        hypothesis="Leveraged liquidations overshoot and revert.",
        reasons=[PASS_LOOK, "FAIL walk-forward: mean OOS Sharpe -4.90 < 1.0"])
    ctx = db.lessons_context()
    assert "Leveraged liquidations overshoot" in ctx


def test_progress_is_ranked_by_stages_cleared_not_by_sharpe(db):
    """Ranking by dev Sharpe would hand the agent a number to overfit."""
    deep = add(db, "got_far", reasons=[PASS_LOOK, PASS_INT, "PASS walk-forward: x",
                                       "FAIL cost stress: y"], metrics={"sharpe": 0.2})
    shallow = add(db, "lucky_sharpe", reasons=[PASS_LOOK, "FAIL walk-forward: y"],
                  metrics={"sharpe": 9.9})
    top = db.furthest(2)
    assert top[0]["id"] == deep, "the deeper candidate must rank first"
    assert top[0]["stages_passed"] == 3
    assert any(e["id"] == shallow for e in top)


def test_raw_verdict_payloads_never_reach_the_block(db):
    add(db, "peeker", hypothesis="h", reasons=[FAIL_LOOK])
    ctx = db.lessons_context()
    assert "'cut'" not in ctx and "n_changed_bars" not in ctx


def test_memory_can_be_scoped_to_one_symbol(db):
    add(db, "btc_idea", symbol="BTC/USD", reasons=[PASS_LOOK])
    add(db, "eth_idea", symbol="ETH/USD", reasons=[PASS_LOOK])
    ctx = db.lessons_context(symbol="ETH/USD")
    assert "eth_idea" in ctx and "btc_idea" not in ctx


def test_a_candidate_that_died_early_still_records_a_sharpe(db):
    """0 of 155 rows had one before dev metrics were published."""
    add(db, "died_at_walkforward", reasons=[PASS_LOOK, "FAIL walk-forward: x"],
        metrics={"sharpe": -1.8})
    assert db.furthest(1)[0]["sharpe"] == -1.8


# -- reflections don't travel across regimes ----------------------------------
def test_reflections_are_scoped_to_the_regime_they_were_written_about(db):
    """A reflection from daily equities recommended sklearn, which the sandbox
    cannot even import. Confident, specific, and wrong for the current setup."""
    db.record_reflection("Try machine learning models.", run_id=1, source="llm",
                         regime="yfinance:1d")
    db.record_reflection("Volume ideas next.", run_id=2, source="llm",
                         regime="ccxt:15m")
    ctx = db.lessons_context(regime="ccxt:15m")
    assert "Volume ideas next" in ctx
    assert "machine learning" not in ctx


def test_untagged_legacy_reflections_do_not_leak_into_a_regime(db):
    db.record_reflection("Old untagged advice.", run_id=1, source="llm")
    assert "Old untagged advice" not in db.lessons_context(regime="ccxt:15m")
    assert "Old untagged advice" in db.lessons_context()


def test_the_block_stays_within_a_sane_token_budget(db):
    for i in range(40):
        add(db, f"strategy_{i}", hypothesis="x" * 900, reasons=[PASS_LOOK, FAIL_LOOK])
    ctx = db.lessons_context()
    assert len(ctx) < 8000, f"memory block ballooned to {len(ctx)} chars"


def test_an_empty_log_produces_an_empty_block_not_a_crash(db):
    assert db.lessons_context() == ""
