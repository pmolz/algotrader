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


# -- model host card ----------------------------------------------------------
def test_api_hosts_is_served_and_shaped(monkeypatch, tmp_path):
    from algotrader.dashboard import hoststatus
    from algotrader.dashboard.app import create_app
    from algotrader.config import load_config

    cfg = load_config()
    cfg["experiments"]["db_path"] = str(tmp_path / "e.db")
    ExperimentDB(cfg["experiments"]["db_path"])
    monkeypatch.setattr(hoststatus, "collect", lambda c, force=False: {
        "hosts": [{"name": "laptop", "state": "up", "gpu": {"temp_c": 71.0}}],
        "selected": "laptop", "policy": "auto", "error": None, "checked_at": 0,
    })
    r = create_app(cfg).test_client().get("/api/hosts")
    assert r.status_code == 200
    body = r.get_json()
    assert body["enabled"] is True and body["selected"] == "laptop"


def test_host_monitor_can_be_switched_off(tmp_path):
    """It makes outbound SSH connections; anyone binding off loopback needs an
    off switch."""
    from algotrader.dashboard.app import create_app
    from algotrader.config import load_config

    cfg = load_config()
    cfg["experiments"]["db_path"] = str(tmp_path / "e.db")
    ExperimentDB(cfg["experiments"]["db_path"])
    cfg["dashboard"]["host_monitor"] = False
    client = create_app(cfg).test_client()

    assert client.get("/api/hosts").get_json() == {"enabled": False, "hosts": []}
    assert 'id="hosts-card"' not in client.get("/").get_data(as_text=True)


# -- briefs page -----------------------------------------------------------------
#
# The page's job is to answer "is the research pipeline actually feeding the
# loop", and every failure mode it has to surface is a silent one: a malformed
# brief the loop skips without complaint, a library that has been used up, a
# brief whose candidates all die before reaching a verdict. Rendering is the
# easy half; these tests are about the page not being quietly reassuring.

BRIEF = """---
id: {id}
title: {title}
family: microstructure
sources: [a paper]
---

MECHANISM
Forced sellers overshoot. Rank with a rolling quantile, entry_q 0.97.
"""


@pytest.fixture
def with_briefs(cfg, tmp_path):
    d = tmp_path / "briefs"
    d.mkdir()
    (d / "a.md").write_text(BRIEF.format(id="alpha-idea", title="Alpha Idea"))
    (d / "b.md").write_text(BRIEF.format(id="beta-idea", title="Beta Idea"))
    cfg["research"]["briefs_dir"] = str(d)
    cfg["research"]["max_attempts_per_brief"] = 2
    return cfg, d


def test_briefs_page_lists_the_library(with_briefs, seeded):
    cfg, _ = with_briefs
    client = create_app(cfg).test_client()
    body = client.get("/briefs").data.decode()
    assert "Alpha Idea" in body and "Beta Idea" in body
    assert "Forced sellers overshoot" in body       # the body the model is shown
    assert "not tried yet" in body                  # no candidates coded yet


def test_an_unparseable_brief_is_reported_loudly(with_briefs, seeded):
    """The failure this page exists for. The loop skips a malformed brief in
    silence, so a broken library is indistinguishable from a quiet week."""
    cfg, d = with_briefs
    (d / "broken.md").write_text("this is not a brief")
    body = create_app(cfg).test_client().get("/briefs").data.decode()
    assert "not reaching the model" in body
    assert "broken.md" in body


def test_outcomes_come_from_the_log(with_briefs, cfg):
    cfg, _ = with_briefs
    db = ExperimentDB(cfg["experiments"]["db_path"])
    for reasons, promoted in ((["FAIL codegen"], False),
                              (["FAIL walk-forward: mean OOS Sharpe 0.30 < 1.0"], False)):
        db.record(symbol="BTC/USD", source="ccxt", timeframe="1d",
                  strategy_name="s", hypothesis="h", params={}, code="x",
                  promoted=promoted, metrics={}, gauntlet_checks={},
                  reasons=reasons, researched_from="alpha-idea")
    db.close()

    client = create_app(cfg).test_client()
    body = client.get("/briefs?b=alpha-idea").data.decode()

    # The stage breakdown is per-brief, so it is what proves the outcomes are
    # this brief's and not the log's in aggregate.
    assert "Where its candidates died" in body
    assert "codegen" in body and "walk-forward" in body
    # One of the two never reached a verdict, and that is the number worth
    # acting on: a brief whose candidates cannot compile is a brief problem,
    # not an idea problem.
    assert "Never judged" in body

    conn = sqlite3.connect(cfg["experiments"]["db_path"])
    conn.row_factory = sqlite3.Row
    o = queries.brief_outcomes(conn)["alpha-idea"]
    conn.close()
    assert o["attempts"] == 2 and o["unjudged"] == 1 and o["promoted"] == 0


def test_a_used_up_brief_is_not_shown_as_in_rotation(with_briefs, cfg):
    cfg, _ = with_briefs                       # max_attempts_per_brief = 2
    db = ExperimentDB(cfg["experiments"]["db_path"])
    for _ in range(2):
        db.record(symbol="BTC/USD", source="ccxt", timeframe="1d", strategy_name="s",
                  hypothesis="h", params={}, code="x", promoted=False, metrics={},
                  gauntlet_checks={}, reasons=["FAIL codegen"],
                  researched_from="alpha-idea")
    db.close()
    body = create_app(cfg).test_client().get("/briefs").data.decode()
    assert "used up" in body


def test_retired_briefs_stay_visible(with_briefs, seeded):
    """The rotation drops them; the page must not, or a brief taken out of
    service just vanishes with no record that it existed."""
    cfg, d = with_briefs
    (d / "a.md").write_text(
        BRIEF.format(id="alpha-idea", title="Alpha Idea").replace(
            "sources: [a paper]", "sources: [a paper]\nretired: true")
    )
    body = create_app(cfg).test_client().get("/briefs").data.decode()
    assert "Alpha Idea" in body
    assert "retired" in body


def test_disabled_research_says_so(with_briefs, seeded):
    cfg, _ = with_briefs
    cfg["research"]["enabled"] = False
    body = create_app(cfg).test_client().get("/briefs").data.decode()
    assert "switched off" in body


def test_empty_library_is_not_an_error(cfg, tmp_path, seeded):
    cfg["research"]["briefs_dir"] = str(tmp_path / "does-not-exist")
    r = create_app(cfg).test_client().get("/briefs")
    assert r.status_code == 200
    assert "No briefs in" in r.data.decode()


def test_a_log_without_the_provenance_column_still_renders(with_briefs, seeded):
    """The dashboard opens the log read-only and cannot migrate it. Pointing it
    at a DB written before briefs existed must render, not 500."""
    cfg, _ = with_briefs
    conn = sqlite3.connect(cfg["experiments"]["db_path"])
    conn.execute("ALTER TABLE experiments DROP COLUMN researched_from")
    conn.commit()
    conn.close()

    client = create_app(cfg).test_client()
    assert client.get("/briefs").status_code == 200
    assert client.get("/experiments").status_code == 200


def test_a_candidate_links_back_to_its_brief(with_briefs, cfg):
    cfg, _ = with_briefs
    db = ExperimentDB(cfg["experiments"]["db_path"])
    exp_id = db.record(symbol="BTC/USD", source="ccxt", timeframe="1d",
                       strategy_name="s", hypothesis="h", params={}, code="x",
                       promoted=False, metrics={}, gauntlet_checks={},
                       reasons=["FAIL codegen"], researched_from="alpha-idea")
    db.close()
    body = create_app(cfg).test_client().get(f"/experiment/{exp_id}").data.decode()
    assert "/briefs?b=alpha-idea" in body
