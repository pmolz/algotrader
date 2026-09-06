"""Local dashboard for browsing the experiment DB.

The DB is opened read-only, so the dashboard can never write the log the agent
is appending to. Three endpoints do have side effects, and they are the whole
of the app's blast radius:

  * `/api/equity/<id>`      re-runs one strategy inside the Docker jail
  * `/api/session/start`    launches a session as a transient systemd unit
  * `/api/session/stop`     SIGTERMs it (a graceful stop, not a kill)
  * `/api/session/schedule` enables/disables the nightly timer

Nothing here can promote a candidate, edit an experiment, or place an order.
The mutating endpoints are POST-only and wear `@local_only` — see `guard.py`
for why "it's just localhost" is not a defence on its own.

Binds to 127.0.0.1 by default. There is no authentication; if you ever change
the host, add auth first, and consider `dashboard.allow_control: false`.
"""

from __future__ import annotations

import json
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path

from flask import Flask, abort, jsonify, render_template, request
from flask.json.provider import DefaultJSONProvider

from ..agent.research import BriefLibrary
from ..config import REPO_ROOT, get, load_config
from . import control, hoststatus, queries
from .equity import EquityUnavailable, cached_curve, equity_curve, json_safe
from .guard import local_only

# Presets for the date-range control. Order matters: shown as rows, in this order.
RANGES = {
    "24h": ("Last 24 hours", 1),
    "7d": ("Last 7 days", 7),
    "30d": ("Last 30 days", 30),
    "90d": ("Last 90 days", 90),
    "all": ("All time", None),
}
DEFAULT_RANGE = "30d"


def _connect(cfg: dict) -> sqlite3.Connection:
    db_path = Path(get(cfg, "experiments.db_path", "experiments/experiments.db"))
    if not db_path.is_absolute():
        db_path = REPO_ROOT / db_path
    # read-only URI: the dashboard cannot corrupt the log the agent is writing
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _filters(args) -> dict:
    """Parse the shared filter row. One set of filters scopes every view."""
    rng = args.get("range", DEFAULT_RANGE)
    if rng not in RANGES:
        rng = DEFAULT_RANGE
    days = RANGES[rng][1]
    f = {
        "range": rng,
        "since": (
            (datetime.now().astimezone() - timedelta(days=days)).timestamp()
            if days else None
        ),
        "symbol": args.get("symbol") or None,
        "run_id": args.get("run") or None,
        "promoted_only": args.get("promoted") == "1",
    }
    return f


class StrictJSONProvider(DefaultJSONProvider):
    """Emit spec-valid JSON only.

    Flask's default provider inherits Python's `allow_nan=True`, which writes
    bare `NaN` / `Infinity` tokens. Those are not JSON, and the browser's
    JSON.parse rejects the entire document — so one infinite Calmar ratio breaks
    a whole response. Non-finite floats become null, which every client can read.
    """

    def dumps(self, obj, **kwargs):
        kwargs.setdefault("allow_nan", False)
        return super().dumps(json_safe(obj), **kwargs)


def create_app(cfg: dict | None = None) -> Flask:
    cfg = cfg or load_config()
    app = Flask(__name__)
    app.json = StrictJSONProvider(app)
    app.config["ALGOTRADER_CFG"] = cfg

    def db():
        # A fresh connection per request; sqlite read-only handles are cheap and
        # this keeps the dashboard safe against the agent writing concurrently.
        return _connect(cfg)

    @app.context_processor
    def _globals():
        conn = db()
        try:
            return {
                "all_symbols": queries.symbols(conn),
                "ranges": RANGES,
                "cur": _filters(request.args),
                "qs": request.query_string.decode(),
                "control_enabled": bool(get(cfg, "dashboard.allow_control", True)),
                "host_monitor": bool(get(cfg, "dashboard.host_monitor", True)),
            }
        finally:
            conn.close()

    # -- pages -------------------------------------------------------------------
    @app.route("/")
    def index():
        conn = db()
        try:
            f = _filters(request.args)
            return render_template(
                "index.html",
                stats=queries.stats(conn, f),
                stages=queries.stage_histogram(conn, f),
                activity=queries.activity(conn, f),
                near=queries.leaderboard(conn, f, n=8),
                recent=queries.experiments(conn, f, limit=12),
            )
        finally:
            conn.close()

    @app.route("/experiments")
    def experiments():
        conn = db()
        try:
            f = _filters(request.args)
            return render_template(
                "experiments.html",
                rows=queries.experiments(conn, f, limit=1000),
                stats=queries.stats(conn, f),
            )
        finally:
            conn.close()

    @app.route("/experiment/<int:exp_id>")
    def experiment(exp_id: int):
        conn = db()
        try:
            exp = queries.experiment(conn, exp_id)
            if exp is None:
                abort(404)
            return render_template(
                "experiment.html",
                exp=exp,
                folds=queries.fold_sharpes(exp),
                stress=queries.cost_stress(exp),
                min_sharpe=get(cfg, "validation.min_sharpe", 1.0),
                # Render a known curve immediately. Cache-only, so loading a URL
                # can never launch a container — that needs the button.
                preloaded=cached_curve(exp, cfg),
            )
        finally:
            conn.close()

    @app.route("/runs")
    def runs():
        conn = db()
        try:
            return render_template(
                "runs.html",
                runs=queries.runs(conn),
                reflections=queries.reflections(conn),
            )
        finally:
            conn.close()

    @app.route("/briefs")
    def briefs():
        """The research library, and whether it is actually feeding the loop.

        Deliberately shows the rotation state next to the outcomes. A brief that
        looks compelling and has produced eight candidates that all died at
        codegen is not a good brief, and that is only visible with both halves
        on the same row.
        """
        d = get(cfg, "research.briefs_dir", "research/briefs")
        d = Path(d) if Path(d).is_absolute() else REPO_ROOT / d
        max_attempts = int(get(cfg, "research.max_attempts_per_brief", 3))
        lib = BriefLibrary(d, max_attempts=max_attempts)

        conn = db()
        try:
            outcomes = queries.brief_outcomes(conn)
            symbols = queries.symbols(conn)
        finally:
            conn.close()

        # Attempts are counted per symbol, so a brief is only out of the
        # rotation once every symbol has used it up. Showing the max would call
        # a brief retired while it is still queued for the other market.
        rows = []
        for b in lib.all_briefs:
            o = outcomes.get(b.id)
            per_symbol = (o or {}).get("by_symbol", {})
            remaining = {s: max(0, max_attempts - per_symbol.get(s, 0))
                         for s in symbols} if symbols else {}
            rows.append({
                "brief": b,
                "body": b.render(int(get(cfg, "research.max_chars", 2500))),
                "outcome": o,
                "remaining": remaining,
                "in_rotation": (not b.retired) and (any(remaining.values())
                                                    if remaining else True),
            })

        which = request.args.get("b")
        if which not in {r["brief"].id for r in rows}:
            which = rows[0]["brief"].id if rows else None

        return render_template(
            "briefs.html",
            rows=rows,
            which=which,
            problems=lib.problems,
            enabled=bool(get(cfg, "research.enabled", True)),
            briefs_dir=str(d),
            max_attempts=max_attempts,
            symbols=symbols,
        )

    @app.route("/reports")
    def reports():
        d = Path(get(cfg, "nightly.report_dir", "experiments/reports"))
        if not d.is_absolute():
            d = REPO_ROOT / d
        files = sorted(d.glob("*.md"), reverse=True) if d.exists() else []
        which = request.args.get("f") or (files[0].name if files else None)
        body = ""
        if which:
            # basename-only: never let a query string walk out of the directory
            target = d / Path(which).name
            if target.exists() and target.parent.resolve() == d.resolve():
                body = target.read_text()
        return render_template(
            "reports.html", files=[f.name for f in files], which=which, body=body
        )

    # -- json ---------------------------------------------------------------------
    @app.route("/api/equity/<int:exp_id>")
    def api_equity(exp_id: int):
        """Re-run one strategy in the sandbox and return its equity curve."""
        conn = db()
        try:
            exp = queries.experiment(conn, exp_id)
        finally:
            conn.close()
        if exp is None:
            return jsonify({"error": "no such experiment"}), 404
        t0 = time.time()
        try:
            out = equity_curve(exp, cfg)
        except EquityUnavailable as e:
            return jsonify({"error": str(e)}), 409
        out["elapsed_s"] = round(time.time() - t0, 1)
        return jsonify(out)

    # -- session control (the only endpoints with side effects) --------------------
    def _control_enabled():
        return bool(get(cfg, "dashboard.allow_control", True))

    @app.route("/api/session/status")
    def api_session_status():
        conn = db()
        try:
            s = control.status(conn, cfg)
        finally:
            conn.close()
        s["enabled"] = _control_enabled()
        s["next_run"] = control.next_run()
        return jsonify(s)

    @app.route("/api/session/start", methods=["POST"])
    @local_only
    def api_session_start():
        if not _control_enabled():
            return jsonify({"error": "Session control is disabled in config."}), 403
        payload = request.get_json(silent=True) or {}
        try:
            budget = control.Budget.parse(payload.get("hours"), payload.get("iterations"))
            return jsonify(control.start(cfg, budget))
        except control.ControlError as e:
            return jsonify({"error": str(e)}), 409

    @app.route("/api/session/stop", methods=["POST"])
    @local_only
    def api_session_stop():
        if not _control_enabled():
            return jsonify({"error": "Session control is disabled in config."}), 403
        try:
            return jsonify(control.stop())
        except control.ControlError as e:
            return jsonify({"error": str(e)}), 409

    @app.route("/api/session/schedule", methods=["POST"])
    @local_only
    def api_session_schedule():
        if not _control_enabled():
            return jsonify({"error": "Session control is disabled in config."}), 403
        payload = request.get_json(silent=True) or {}
        try:
            return jsonify(control.set_timer(bool(payload.get("enabled"))))
        except control.ControlError as e:
            return jsonify({"error": str(e)}), 409

    @app.route("/api/hosts")
    def api_hosts():
        if not bool(get(cfg, "dashboard.host_monitor", True)):
            return jsonify({"enabled": False, "hosts": []})
        payload = hoststatus.collect(cfg)
        payload["enabled"] = True
        return jsonify(payload)

    @app.route("/api/stats")
    def api_stats():
        conn = db()
        try:
            return jsonify(queries.stats(conn, _filters(request.args)))
        finally:
            conn.close()

    @app.route("/healthz")
    def healthz():
        conn = db()
        try:
            conn.execute("SELECT 1").fetchone()
            return jsonify({"ok": True})
        except Exception as e:  # noqa: BLE001
            return jsonify({"ok": False, "error": str(e)}), 500
        finally:
            conn.close()

    # -- template helpers ---------------------------------------------------------
    @app.template_filter("ts")
    def _ts(v):
        return datetime.fromtimestamp(v).astimezone().strftime("%Y-%m-%d %H:%M") if v else "—"

    @app.template_filter("dur")
    def _dur(v):
        if v is None:
            return "—"
        v = int(v)
        h, rem = divmod(v, 3600)
        m, s = divmod(rem, 60)
        return f"{h}h {m}m" if h else (f"{m}m {s}s" if m else f"{s}s")

    @app.template_filter("num")
    def _num(v, nd=2):
        try:
            return f"{float(v):.{nd}f}"
        except (TypeError, ValueError):
            return "—"

    @app.template_filter("pct")
    def _pct(v, nd=0):
        try:
            return f"{float(v) * 100:.{nd}f}%"
        except (TypeError, ValueError):
            return "—"

    @app.template_filter("tojson_safe")
    def _tojson_safe(v):
        return json.dumps(v)

    return app
