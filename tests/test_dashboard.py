"""Dashboard tests.

Two things matter here beyond "the page renders":
  1. The dashboard is read-only. It must not be able to write the experiment log
     the agent is concurrently appending to.
  2. Its numbers must agree with the morning report's. Both derive a candidate's
     cause of death from `agent.report.fail_stage`, and if they ever diverge the
     dashboard is the one that is wrong.
"""

import json
import sqlite3
import time

import pytest

from algotrader.agent.memory import ExperimentDB
from algotrader.agent.report import build_report
from algotrader.config import load_config
from algotrader.dashboard import create_app, queries


@pytest.fixture
def cfg(tmp_path):
    c = load_config()
    c["experiments"]["db_path"] = str(tmp_path / "exp.db")
    c["nightly"]["report_dir"] = str(tmp_path / "reports")
    c["dashboard"]["cache_dir"] = str(tmp_path / "equity_cache")
    return c


@pytest.fixture
def seeded(cfg):
    db = ExperimentDB(cfg["experiments"]["db_path"])
    run_id = db.start_run(kind="nightly", symbols=["BTC/USD"], provider="ollama",
                          model="qwen2.5-coder:7b", sandbox=True, budget={})
    db.record(
        symbol="BTC/USD", source="ccxt", timeframe="1d", strategy_name="winner",
        hypothesis="a promoted idea", params={"fast": 10},
        code="class W(Strategy):\n    name='winner'\n", promoted=True,
        metrics={"sharpe": 1.8, "max_drawdown": -0.12},
        gauntlet_checks={
            "walk_forward": {"mean_sharpe": 1.4, "positive_folds": 5, "n_folds": 5,
                             "fold_sharpes": [1.2, 1.5, 1.3, 1.6, 1.4]},
            "cost_stress": {"1.0": 1.9, "1.5": 1.5, "2.0": 1.2},
            "deflated_sharpe": {"dsr": 0.71},
            "oos_holdout": {"sharpe": 1.8, "max_drawdown": -0.12},
        },
        reasons=["ALL CHECKS PASSED — candidate promoted to paper trading."],
        run_id=run_id,
    )
    db.record(
        symbol="BTC/USD", source="ccxt", timeframe="1d", strategy_name="loser",
        hypothesis="a rejected idea", params={}, code="class L: pass", promoted=False,
        metrics={}, gauntlet_checks={"walk_forward": {"mean_sharpe": 0.3}},
        reasons=["PASS lookahead (ok)", "FAIL walk-forward: mean OOS Sharpe 0.30 < 1.0"],
        run_id=run_id,
    )
    db.record(
        symbol="ETH/USD", source="ccxt", timeframe="1d", strategy_name="(llm)",
        hypothesis="never compiled", params={}, code="def broken(", promoted=False,
        metrics={}, gauntlet_checks={}, reasons=["FAIL codegen"],
        error="SyntaxError: invalid syntax", run_id=run_id,
    )
    db.finish_run(run_id, iterations=3, promoted=1, errors=1, stop_reason="budget")
    db.record_reflection("Momentum is dead here.", run_id=run_id, source="llm")
    db.close()
    return cfg


@pytest.fixture
def client(seeded):
    return create_app(seeded).test_client()


# -- pages render ----------------------------------------------------------------
@pytest.mark.parametrize(
    "path", ["/", "/experiments", "/runs", "/reports", "/healthz", "/experiment/1"]
)
def test_pages_render(client, path):
    r = client.get(path)
    assert r.status_code == 200


def test_unknown_experiment_is_404(client):
    assert client.get("/experiment/9999").status_code == 404


# -- read-only guarantee ----------------------------------------------------------
def test_dashboard_connection_cannot_write(seeded):
    """The agent may be appending to this DB while you browse it."""
    from algotrader.dashboard.app import _connect

    conn = _connect(seeded)
    with pytest.raises(sqlite3.OperationalError, match="readonly|read-only"):
        conn.execute("DELETE FROM experiments")
    conn.close()


def test_no_route_mutates_the_db(client, seeded):
    before = ExperimentDB(seeded["experiments"]["db_path"]).total_trials()
    for path in ["/", "/experiments", "/runs", "/reports", "/experiment/1"]:
        client.get(path)
    after = ExperimentDB(seeded["experiments"]["db_path"]).total_trials()
    assert before == after


# -- the numbers agree with the report ---------------------------------------------
def test_stats_agree_with_the_morning_report(seeded):
    db = ExperimentDB(seeded["experiments"]["db_path"])
    conn = sqlite3.connect(seeded["experiments"]["db_path"])
    conn.row_factory = sqlite3.Row
    f = {"since": None, "until": None}

    s = queries.stats(conn, f)
    report = build_report(db, 0, time.time() + 1)

    assert s["total"] == 3
    assert s["promoted"] == 1
    assert s["unjudged"] == 1                     # the codegen failure
    assert "1 candidate(s) survived the gauntlet" in report
    assert f"Lifetime experiments in the DB: **{s['lifetime_trials']}**" in report


def test_stage_attribution_matches_the_report_definition(seeded):
    conn = sqlite3.connect(seeded["experiments"]["db_path"])
    conn.row_factory = sqlite3.Row
    stages = {s["stage"]: s["count"] for s in queries.stage_histogram(conn, {})}
    assert stages == {"promoted": 1, "walk-forward": 1, "codegen": 1}


# -- honest framing ----------------------------------------------------------------
def test_zero_promoted_is_framed_as_expected_not_as_failure(cfg):
    db = ExperimentDB(cfg["experiments"]["db_path"])
    for _ in range(4):
        db.record(symbol="BTC/USD", source="ccxt", timeframe="1d",
                  strategy_name="s", hypothesis="h", params={}, code="c",
                  promoted=False, metrics={},
                  gauntlet_checks={"walk_forward": {"mean_sharpe": 0.2}},
                  reasons=["FAIL walk-forward: mean OOS Sharpe 0.20 < 1.0"])
    db.close()
    body = create_app(cfg).test_client().get("/?range=all").data.decode()
    assert "expected" in body
    assert "gauntlet is doing its job" in body


def test_mostly_unjudged_slice_is_flagged_as_wasted(cfg):
    db = ExperimentDB(cfg["experiments"]["db_path"])
    for _ in range(3):
        db.record(symbol="BTC/USD", source="ccxt", timeframe="1d",
                  strategy_name="(llm)", hypothesis="h", params={}, code=None,
                  promoted=False, metrics={}, gauntlet_checks={},
                  reasons=["FAIL codegen"], error="SyntaxError")
    db.close()
    body = create_app(cfg).test_client().get("/?range=all").data.decode()
    assert "never reached a verdict" in body
    assert "not a market signal" in body


def test_promoted_candidate_carries_a_review_warning(client):
    body = client.get("/experiment/1").data.decode()
    assert "suspicious until reviewed" in body
    assert "Never straight to capital" in body


def test_closest_rejections_are_not_framed_as_a_shortlist(client):
    body = client.get("/?range=all").data.decode()
    assert "not" in body and "shortlist" in body


# -- filters scope everything -------------------------------------------------------
def test_symbol_filter_scopes_the_stats(client):
    body = client.get("/experiments?range=all&symbol=ETH/USD").data.decode()
    assert "never compiled" in body
    assert "a promoted idea" not in body


def test_promoted_only_filter(client):
    body = client.get("/experiments?range=all&promoted=1").data.decode()
    assert "winner" in body
    assert "loser" not in body


# -- untrusted content ---------------------------------------------------------------
def test_model_written_text_is_escaped(cfg):
    """Hypotheses and strategy names come from an LLM — treat them as hostile."""
    db = ExperimentDB(cfg["experiments"]["db_path"])
    db.record(symbol="BTC/USD", source="ccxt", timeframe="1d",
              strategy_name="<script>alert(1)</script>",
              hypothesis="<img src=x onerror=alert(2)>", params={}, code="x=1",
              promoted=False, metrics={}, gauntlet_checks={}, reasons=["FAIL codegen"])
    db.close()
    body = create_app(cfg).test_client().get("/experiments?range=all").data.decode()
    # The angle brackets are what make it executable; escaped text is inert, so
    # assert on the tags rather than on the payload substring.
    assert "<script>alert(1)</script>" not in body
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in body
    assert "<img src=x" not in body
    assert "&lt;img src=x onerror=alert(2)&gt;" in body


def test_report_path_traversal_is_blocked(client):
    r = client.get("/reports?f=../../../../etc/passwd")
    assert r.status_code == 200
    assert "root:" not in r.data.decode()


# -- equity endpoint ------------------------------------------------------------------
def test_api_emits_spec_valid_json(seeded, monkeypatch):
    """NaN / Infinity are valid Python json but NOT valid JSON.

    The browser's JSON.parse rejects the whole document, so one infinite Calmar
    ratio breaks the entire response. Python's own json.loads accepts these
    tokens, which is exactly why this needs `parse_constant` to fail — a plain
    `get_json()` would have happily parsed the broken payload.
    """
    from algotrader.dashboard import equity as equity_mod

    def boom(_):
        raise AssertionError("payload contained a non-finite JSON constant")

    def fake_run(df, code, cfg, n_trials=1, limits=None, mode="gauntlet"):
        return {"ok": True, "dates": ["2020-01-01T00:00:00"], "equity": [1.0],
                "benchmark": [float("nan")], "positions": [0.0], "trades": 0,
                # calmar is legitimately infinite when max drawdown is zero
                "metrics": {"sharpe": 1.0, "calmar": float("inf")},
                "benchmark_metrics": {"cagr": float("inf")}, "holdout_start": None}

    monkeypatch.setattr(equity_mod, "run_gauntlet_sandboxed", fake_run)
    monkeypatch.setattr(equity_mod, "docker_available", lambda: True)
    monkeypatch.setattr(equity_mod, "image_exists", lambda img: True)
    monkeypatch.setattr(equity_mod, "load_cached",
                        lambda *a, **k: __import__("pandas").DataFrame({"close": [1.0, 2.0]}))

    r = create_app(seeded).test_client().get("/api/equity/1")
    assert r.status_code == 200
    body = r.data.decode()
    assert "NaN" not in body and "Infinity" not in body
    parsed = json.loads(body, parse_constant=boom)      # strict: rejects NaN/Infinity
    assert parsed["metrics"]["calmar"] is None
    assert parsed["benchmark"][0] is None


def test_json_safe_replaces_non_finite_floats():
    from algotrader.dashboard.equity import json_safe

    out = json_safe({"a": float("inf"), "b": [float("nan"), 1.5], "c": {"d": -float("inf")}})
    assert out == {"a": None, "b": [None, 1.5], "c": {"d": None}}


def test_equity_refuses_when_no_code_was_stored(cfg):
    db = ExperimentDB(cfg["experiments"]["db_path"])
    db.record(symbol="BTC/USD", source="ccxt", timeframe="1d", strategy_name="(llm)",
              hypothesis="h", params={}, code=None, promoted=False, metrics={},
              gauntlet_checks={}, reasons=["FAIL proposal"])
    db.close()
    r = create_app(cfg).test_client().get("/api/equity/1")
    assert r.status_code == 409
    assert "no stored code" in r.get_json()["error"]


def test_equity_uses_the_cache_on_a_second_request(seeded, monkeypatch, tmp_path):
    """A browser refresh must not be able to spam container launches."""
    from algotrader.dashboard import equity as equity_mod

    calls = []

    def fake_run(df, code, cfg, n_trials=1, limits=None, mode="gauntlet"):
        calls.append(mode)
        return {"ok": True, "dates": ["2020-01-01T00:00:00"], "equity": [1.0],
                "benchmark": [1.0], "positions": [0.0], "trades": 0,
                "metrics": {"sharpe": 1.0}, "benchmark_metrics": {},
                "holdout_start": None}

    monkeypatch.setattr(equity_mod, "run_gauntlet_sandboxed", fake_run)
    monkeypatch.setattr(equity_mod, "docker_available", lambda: True)
    monkeypatch.setattr(equity_mod, "image_exists", lambda img: True)
    monkeypatch.setattr(equity_mod, "load_cached",
                        lambda *a, **k: __import__("pandas").DataFrame(
                            {"close": [1.0, 2.0, 3.0]}))

    client = create_app(seeded).test_client()
    first = client.get("/api/equity/1")
    second = client.get("/api/equity/1")
    assert first.status_code == 200 and second.status_code == 200
    assert calls == ["equity"], "second request should have been served from cache"
    assert second.get_json()["cached"] is True
