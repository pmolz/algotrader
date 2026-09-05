"""Local dashboard for browsing the experiment DB.

Read-only by design. The single exception is the equity-curve endpoint, which
re-runs a strategy inside the Docker jail — see `equity.py`. Nothing here can
promote a candidate, edit the DB, or place an order.

Binds to 127.0.0.1 by default. There is no authentication, because there is no
network exposure; if you ever change the host, add auth first.
"""

from __future__ import annotations

import json
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path

from flask import Flask, abort, jsonify, render_template, request
from flask.json.provider import DefaultJSONProvider

from ..config import REPO_ROOT, get, load_config
from . import queries
from .equity import EquityUnavailable, cached_curve, equity_curve, json_safe

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
