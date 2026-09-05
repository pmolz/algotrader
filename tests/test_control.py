"""Tests for session control — the only part of the dashboard with side effects.

Two concerns, in order of how much damage getting them wrong would do:
  1. The guards. This app has no authentication, so a page you happen to visit
     must not be able to start a research session on your machine.
  2. Budget validation. A mistyped duration should be rejected visibly, not
     silently clamped into something that runs for a fortnight.

Nothing here actually launches systemd; the subprocess boundary is stubbed.
"""

import pytest

from algotrader.agent.memory import ExperimentDB
from algotrader.config import load_config
from algotrader.dashboard import control, create_app

GOOD = {"Content-Type": "application/json", "X-Algotrader": "1"}


@pytest.fixture
def cfg(tmp_path):
    c = load_config()
    c["experiments"]["db_path"] = str(tmp_path / "exp.db")
    c["nightly"]["lock_file"] = str(tmp_path / "nightly.lock")
    c["dashboard"]["cache_dir"] = str(tmp_path / "equity_cache")
    ExperimentDB(c["experiments"]["db_path"]).close()
    return c


@pytest.fixture
def client(cfg, monkeypatch):
    monkeypatch.setattr(control, "systemd_available", lambda: True)
    monkeypatch.setattr(control, "unit_active", lambda: False)
    monkeypatch.setattr(control, "timer_enabled", lambda: False)
    return create_app(cfg).test_client()


# -- budget validation -------------------------------------------------------------
@pytest.mark.parametrize(
    "hours,iters,why",
    [
        (None, None, "neither limit given"),
        (0, None, "below the floor"),
        (999, None, "above the ceiling"),
        (-3, None, "negative"),
        ("abc", None, "not a number"),
        (None, 0, "zero iterations"),
        (None, 99999999, "absurd iteration count"),
        (None, "seven", "not a whole number"),
    ],
)
def test_bad_budgets_are_rejected(hours, iters, why):
    with pytest.raises(control.ControlError):
        control.Budget.parse(hours, iters)


@pytest.mark.parametrize(
    "hours,iters,expected",
    [
        (8, None, ["--hours", "8.0"]),
        (None, 20, ["--max-iterations", "20"]),
        (0.25, 5, ["--hours", "0.25", "--max-iterations", "5"]),
        ("2", "", ["--hours", "2.0"]),
    ],
)
def test_good_budgets_become_cli_args(hours, iters, expected):
    assert control.Budget.parse(hours, iters).args() == expected


# -- the guards --------------------------------------------------------------------
@pytest.mark.parametrize("path", ["/api/session/start", "/api/session/stop",
                                  "/api/session/schedule"])
def test_mutating_endpoints_reject_a_missing_custom_header(client, path):
    """A cross-origin HTML form can POST here but cannot set a custom header."""
    r = client.post(path, json={"hours": 1})
    assert r.status_code == 403
    assert "X-Algotrader" in r.get_json()["error"]


@pytest.mark.parametrize("path", ["/api/session/start", "/api/session/stop",
                                  "/api/session/schedule"])
def test_mutating_endpoints_reject_a_foreign_origin(client, path):
    r = client.post(path, json={"hours": 1},
                    headers={**GOOD, "Origin": "https://evil.example"})
    assert r.status_code == 403
    assert "Cross-origin" in r.get_json()["error"]


def test_mutating_endpoints_reject_a_rebound_host(client):
    """DNS rebinding: attacker.example resolving to 127.0.0.1 is same-origin to
    the browser, but the Host header still carries their name."""
    r = client.post("/api/session/start", json={"hours": 1},
                    headers={**GOOD, "Host": "attacker.example"})
    assert r.status_code == 403
    assert "loopback-only" in r.get_json()["error"]


def test_same_origin_request_passes_the_guards(client, monkeypatch):
    monkeypatch.setattr(control, "start", lambda cfg, b: {"started": True})
    r = client.post("/api/session/start", json={"hours": 1},
                    headers={**GOOD, "Origin": "http://127.0.0.1:8765"})
    assert r.status_code == 200


@pytest.mark.parametrize("path", ["/api/session/start", "/api/session/stop",
                                  "/api/session/schedule"])
def test_mutating_endpoints_are_post_only(client, path):
    assert client.get(path).status_code == 405


def test_status_is_readable_without_the_guard(client):
    """Status has no side effects, so it stays a plain GET."""
    assert client.get("/api/session/status").status_code == 200


def test_control_can_be_disabled_in_config(cfg, monkeypatch):
    monkeypatch.setattr(control, "systemd_available", lambda: True)
    monkeypatch.setattr(control, "unit_active", lambda: False)
    cfg["dashboard"]["allow_control"] = False
    c = create_app(cfg).test_client()
    r = c.post("/api/session/start", json={"hours": 1}, headers=GOOD)
    assert r.status_code == 403
    assert "disabled" in r.get_json()["error"]
    assert c.get("/api/session/status").get_json()["enabled"] is False
    # and the UI is not rendered at all
    assert "control-card" not in c.get("/").data.decode()


# -- start / stop behaviour ---------------------------------------------------------
def test_start_builds_a_safe_argument_list(cfg, monkeypatch):
    """No shell, and the budget arrives as separate argv entries."""
    seen = {}

    class R:
        returncode = 0
        stdout = stderr = ""

    def fake_run(cmd, **kw):
        seen["cmd"] = cmd
        seen["shell"] = kw.get("shell", False)
        return R()

    monkeypatch.setattr(control.subprocess, "run", fake_run)
    monkeypatch.setattr(control, "systemd_available", lambda: True)
    monkeypatch.setattr(control, "unit_active", lambda: False)

    control.start(cfg, control.Budget.parse(2, 10))

    assert seen["shell"] is False
    assert isinstance(seen["cmd"], list)
    assert seen["cmd"][0] == "systemd-run"
    assert "--user" in seen["cmd"] and "--collect" in seen["cmd"]
    assert "--property=TimeoutStartSec=infinity" in seen["cmd"]
    assert "--property=KillSignal=SIGTERM" in seen["cmd"]   # graceful, not SIGKILL
    assert seen["cmd"][-4:] == ["--hours", "2.0", "--max-iterations", "10"]
    assert seen["cmd"][-5].endswith("run_nightly.py")


def test_start_refuses_when_one_is_already_running(cfg, monkeypatch):
    monkeypatch.setattr(control, "systemd_available", lambda: True)
    monkeypatch.setattr(control, "unit_active", lambda: True)
    with pytest.raises(control.ControlError, match="already running"):
        control.start(cfg, control.Budget.parse(1, None))


def test_start_refuses_when_a_lock_is_held(cfg, monkeypatch, tmp_path):
    from pathlib import Path

    monkeypatch.setattr(control, "systemd_available", lambda: True)
    monkeypatch.setattr(control, "unit_active", lambda: False)
    Path(cfg["nightly"]["lock_file"]).write_text("4242 now\n")
    with pytest.raises(control.ControlError, match="lock is held"):
        control.start(cfg, control.Budget.parse(1, None))


def test_start_explains_itself_without_systemd(cfg, monkeypatch):
    monkeypatch.setattr(control, "systemd_available", lambda: False)
    with pytest.raises(control.ControlError, match="run_nightly.py"):
        control.start(cfg, control.Budget.parse(1, None))


def test_stop_is_a_no_op_when_nothing_runs(monkeypatch):
    monkeypatch.setattr(control, "unit_active", lambda: False)
    with pytest.raises(control.ControlError, match="No session is running"):
        control.stop()


def test_stop_sends_a_graceful_systemctl_stop(monkeypatch):
    seen = {}

    class R:
        returncode = 0
        stdout = stderr = ""

    monkeypatch.setattr(control, "unit_active", lambda: True)
    monkeypatch.setattr(control.subprocess, "run",
                        lambda cmd, **kw: (seen.update(cmd=cmd), R())[1])
    control.stop()
    assert seen["cmd"] == ["systemctl", "--user", "stop", "algotrader-session.service"]


# -- status ------------------------------------------------------------------------
def test_status_reports_live_progress(cfg, monkeypatch):
    import sqlite3
    import time

    db = ExperimentDB(cfg["experiments"]["db_path"])
    run_id = db.start_run(kind="nightly", symbols=["BTC/USD"], provider="ollama",
                          model="qwen2.5-coder:7b", sandbox=True,
                          budget={"max_hours": 2.0})
    for promoted in (False, True):
        db.record(symbol="BTC/USD", source="ccxt", timeframe="1d",
                  strategy_name="s", hypothesis="h", params={}, code="c",
                  promoted=promoted, metrics={}, gauntlet_checks={}, reasons=[],
                  run_id=run_id)
    db.close()

    monkeypatch.setattr(control, "unit_active", lambda: True)
    monkeypatch.setattr(control, "systemd_available", lambda: True)
    monkeypatch.setattr(control, "timer_enabled", lambda: False)

    conn = sqlite3.connect(cfg["experiments"]["db_path"])
    conn.row_factory = sqlite3.Row
    s = control.status(conn, cfg)

    assert s["active"] is True
    assert s["run"]["id"] == run_id
    assert s["run"]["iterations"] == 2
    assert s["run"]["promoted"] == 1
    assert s["run"]["orphaned"] is False
    assert 0.0 <= s["run"]["progress"] < 0.01     # just started, 2h budget
    assert s["run"]["elapsed_s"] < time.time()


def test_status_flags_an_orphaned_run(cfg, monkeypatch):
    """Run row open, no live unit — the process died. Say so."""
    import sqlite3

    db = ExperimentDB(cfg["experiments"]["db_path"])
    db.start_run(kind="nightly", symbols=["BTC/USD"], provider="ollama",
                 model="m", sandbox=True, budget={})
    db.close()

    monkeypatch.setattr(control, "unit_active", lambda: False)
    monkeypatch.setattr(control, "systemd_available", lambda: True)
    monkeypatch.setattr(control, "timer_enabled", lambda: False)

    conn = sqlite3.connect(cfg["experiments"]["db_path"])
    conn.row_factory = sqlite3.Row
    s = control.status(conn, cfg)
    assert s["active"] is False
    assert s["run"]["orphaned"] is True


def test_schedule_toggle_calls_systemctl(client, monkeypatch):
    seen = {}

    class R:
        returncode = 0
        stdout = stderr = ""

    monkeypatch.setattr(control.subprocess, "run",
                        lambda cmd, **kw: (seen.update(cmd=cmd), R())[1])
    r = client.post("/api/session/schedule", json={"enabled": True}, headers=GOOD)
    assert r.status_code == 200
    assert seen["cmd"][:4] == ["systemctl", "--user", "enable", "--now"]

    client.post("/api/session/schedule", json={"enabled": False}, headers=GOOD)
    assert seen["cmd"][:4] == ["systemctl", "--user", "disable", "--now"]
