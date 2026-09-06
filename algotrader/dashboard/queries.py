"""Read-only views over the experiment DB for the dashboard.

Every function here takes a connection and returns plain dicts/lists ready to be
JSON-encoded. Nothing in this module writes, and nothing executes strategy code —
that stays behind the sandbox in `equity.py`.

The numbers here are deliberately the same ones the morning report prints. If the
dashboard and the report ever disagree, the dashboard is wrong: both read
`agent.report.fail_stage` so a candidate's cause of death has exactly one
definition in the codebase.
"""

from __future__ import annotations

import json
import time
from typing import Any

from ..agent.report import fail_stage

# Stages where the candidate never reached a verdict. Kept in one place because
# both the "unjudged" stat and the wasted-night warning depend on the same set.
UNJUDGED_STAGES = ("codegen", "strategy runtime", "crash", "sandbox setup", "unknown")


def _has_column(conn, table: str, column: str) -> bool:
    return any(r["name"] == column
               for r in conn.execute(f"PRAGMA table_info({table})"))


def _row(r) -> dict[str, Any]:
    # .keys() is required: iterating a sqlite3.Row yields values, not column names
    return {k: r[k] for k in r.keys()}  # noqa: SIM118


def _experiment(r) -> dict[str, Any]:
    """One experiment, decoded. `code` is omitted — it is large and only the
    detail view needs it."""
    metrics = json.loads(r["metrics_json"] or "{}")
    checks = json.loads(r["gauntlet_json"] or "{}")
    reasons = json.loads(r["reasons"] or "[]")
    exp = {
        "id": r["id"],
        "created_at": r["created_at"],
        "symbol": r["symbol"],
        "source": r["source"],
        "timeframe": r["timeframe"],
        "strategy_name": r["strategy_name"],
        "hypothesis": r["hypothesis"],
        "params": json.loads(r["params_json"] or "{}"),
        "promoted": bool(r["promoted"]),
        "metrics": metrics,
        "checks": checks,
        "reasons": reasons,
        "error": r["error"],
        # likewise: `in r` would search the row's *values*, not its column names
        "run_id": r["run_id"] if "run_id" in r.keys() else None,  # noqa: SIM118
        # Absent on a log written before briefs existed; the dashboard is
        # read-only and cannot add the column, so it renders as "self-generated".
        "researched_from": (
            r["researched_from"] if "researched_from" in r.keys() else None  # noqa: SIM118
        ),
    }
    exp["stage"] = fail_stage(exp)
    exp["sharpe"] = _sharpe_signal(exp)
    return exp


def _sharpe_signal(exp: dict[str, Any]) -> float | None:
    """Most out-of-sample Sharpe available, for ranking. Mirrors the report."""
    checks = exp.get("checks") or {}
    oos = checks.get("oos_holdout") or {}
    if isinstance(oos, dict) and oos.get("sharpe") is not None:
        return float(oos["sharpe"])
    wf = checks.get("walk_forward") or {}
    if isinstance(wf, dict) and wf.get("mean_sharpe") is not None:
        return float(wf["mean_sharpe"])
    m = exp.get("metrics") or {}
    return float(m["sharpe"]) if m.get("sharpe") is not None else None


# -- filtering -------------------------------------------------------------------
def _where(filters: dict) -> tuple[str, list]:
    """Build the shared WHERE clause. Every view uses this, so a filter applied
    at the top of the page scopes the stats, the charts, and the table alike."""
    clauses, params = [], []
    if filters.get("since"):
        clauses.append("created_at >= ?")
        params.append(float(filters["since"]))
    if filters.get("until"):
        clauses.append("created_at < ?")
        params.append(float(filters["until"]))
    if filters.get("symbol"):
        clauses.append("symbol = ?")
        params.append(filters["symbol"])
    if filters.get("run_id"):
        clauses.append("run_id = ?")
        params.append(int(filters["run_id"]))
    if filters.get("promoted_only"):
        clauses.append("promoted = 1")
    return (" WHERE " + " AND ".join(clauses)) if clauses else "", params


def experiments(conn, filters: dict, limit: int | None = None) -> list[dict]:
    where, params = _where(filters)
    sql = f"SELECT * FROM experiments{where} ORDER BY id DESC"
    if limit:
        sql += " LIMIT ?"
        params = [*params, int(limit)]
    return [_experiment(r) for r in conn.execute(sql, params)]


def experiment(conn, exp_id: int) -> dict | None:
    r = conn.execute("SELECT * FROM experiments WHERE id=?", (exp_id,)).fetchone()
    if r is None:
        return None
    exp = _experiment(r)
    exp["code"] = r["code"]          # detail view only
    return exp


def symbols(conn) -> list[str]:
    return [
        r["symbol"]
        for r in conn.execute(
            "SELECT DISTINCT symbol FROM experiments WHERE symbol IS NOT NULL"
            " ORDER BY symbol"
        )
    ]


# -- aggregates ------------------------------------------------------------------
def stats(conn, filters: dict) -> dict[str, Any]:
    """The KPI row. `lifetime_trials` deliberately ignores the filters: it is the
    multiple-testing bar, which every past experiment raises regardless of what
    slice you happen to be looking at."""
    exps = experiments(conn, filters)
    total = len(exps)
    promoted = sum(1 for e in exps if e["promoted"])
    unjudged = sum(1 for e in exps if e["stage"] in UNJUDGED_STAGES)
    judged = total - unjudged
    sharpes = [e["sharpe"] for e in exps if e["sharpe"] is not None and not e["promoted"]]

    where, params = _where(filters)
    runs_n = conn.execute(
        f"SELECT COUNT(*) c FROM runs{where.replace('created_at', 'started_at')}",
        params,
    ).fetchone()["c"] if not filters.get("symbol") and not filters.get("run_id") else None

    return {
        "total": total,
        "promoted": promoted,
        "rejected": total - promoted,
        "unjudged": unjudged,
        "judged": judged,
        "judged_pct": (judged / total) if total else 0.0,
        "promotion_rate": (promoted / total) if total else 0.0,
        "best_rejected_sharpe": max(sharpes) if sharpes else None,
        "sessions": runs_n,
        "lifetime_trials": conn.execute(
            "SELECT COUNT(*) c FROM experiments"
        ).fetchone()["c"],
        "wasted": total > 0 and promoted == 0 and unjudged > total / 2,
    }


def stage_histogram(conn, filters: dict) -> list[dict]:
    """Where candidates died, most common first."""
    counts: dict[str, int] = {}
    for e in experiments(conn, filters):
        counts[e["stage"]] = counts.get(e["stage"], 0) + 1
    total = sum(counts.values()) or 1
    return [
        {"stage": s, "count": n, "share": n / total,
         "unjudged": s in UNJUDGED_STAGES, "promoted": s == "promoted"}
        for s, n in sorted(counts.items(), key=lambda kv: -kv[1])
    ]


def activity(conn, filters: dict) -> list[dict]:
    """Experiments per local day, split by verdict class. Three classes keeps the
    stack inside the comfortable categorical range and each one actionable."""
    buckets: dict[str, dict] = {}
    for e in experiments(conn, filters):
        day = time.strftime("%Y-%m-%d", time.localtime(e["created_at"]))
        b = buckets.setdefault(day, {"day": day, "promoted": 0, "rejected": 0, "unjudged": 0})
        if e["promoted"]:
            b["promoted"] += 1
        elif e["stage"] in UNJUDGED_STAGES:
            b["unjudged"] += 1
        else:
            b["rejected"] += 1
    return [buckets[d] for d in sorted(buckets)]


def leaderboard(conn, filters: dict, n: int = 10) -> list[dict]:
    """Best out-of-sample Sharpe among *rejected* candidates.

    Framed as "closest rejections" everywhere it is shown, never as a shortlist:
    re-testing the top of this list is precisely how you overfit.
    """
    scored = [e for e in experiments(conn, filters)
              if e["sharpe"] is not None and not e["promoted"]]
    scored.sort(key=lambda e: e["sharpe"], reverse=True)
    return scored[:n]


def runs(conn, limit: int = 50) -> list[dict]:
    out = []
    for r in conn.execute("SELECT * FROM runs ORDER BY id DESC LIMIT ?", (limit,)):
        d = _row(r)
        d["symbols"] = json.loads(d.get("symbols") or "[]")
        d["budget"] = json.loads(d.get("budget_json") or "{}")
        d["unfinished"] = d.get("finished_at") is None
        d["duration_s"] = (
            (d["finished_at"] - d["started_at"]) if d.get("finished_at") else None
        )
        out.append(d)
    return out


def reflections(conn, limit: int = 50) -> list[dict]:
    return [
        _row(r)
        for r in conn.execute(
            "SELECT * FROM reflections ORDER BY id DESC LIMIT ?", (limit,)
        )
    ]


def fold_sharpes(exp: dict) -> list[dict]:
    wf = (exp.get("checks") or {}).get("walk_forward") or {}
    return [
        {"fold": i + 1, "sharpe": float(s)}
        for i, s in enumerate(wf.get("fold_sharpes") or [])
    ]


def cost_stress(exp: dict) -> list[dict]:
    cs = (exp.get("checks") or {}).get("cost_stress") or {}
    try:
        items = sorted(cs.items(), key=lambda kv: float(kv[0]))
    except ValueError:
        items = sorted(cs.items())
    return [{"multiplier": k, "sharpe": float(v)} for k, v in items]


def brief_outcomes(conn) -> dict[str, dict[str, Any]]:
    """Per research brief: what the candidates it produced actually did.

    This is the question the briefs page exists to answer. A brief is a claim
    that an idea is worth the loop's time, and the only honest way to judge one
    is by how far its candidates got — not by how well it reads.

    Keyed by brief id. Briefs that have never been coded are absent, which the
    caller renders as "not tried yet" rather than as a zero.
    """
    out: dict[str, dict[str, Any]] = {}
    # The dashboard opens the log read-only and so cannot run the migration that
    # adds this column. A log written before briefs existed is a normal thing to
    # be pointed at, and it should render an empty briefs page rather than a 500.
    if not _has_column(conn, "experiments", "researched_from"):
        return out
    rows = conn.execute(
        "SELECT * FROM experiments WHERE researched_from IS NOT NULL ORDER BY id"
    ).fetchall()
    for r in rows:
        e = _experiment(r)
        bid = r["researched_from"]
        o = out.setdefault(bid, {
            "attempts": 0, "promoted": 0, "by_symbol": {}, "stages": {},
            "best_sharpe": None, "last_id": None, "last_at": None,
        })
        o["attempts"] += 1
        o["promoted"] += int(e["promoted"])
        o["by_symbol"][e["symbol"]] = o["by_symbol"].get(e["symbol"], 0) + 1
        o["stages"][e["stage"]] = o["stages"].get(e["stage"], 0) + 1
        o["last_id"], o["last_at"] = e["id"], e["created_at"]
        if e["sharpe"] is not None and (o["best_sharpe"] is None
                                        or e["sharpe"] > o["best_sharpe"]):
            o["best_sharpe"] = e["sharpe"]
    for o in out.values():
        # Most common cause of death, which is the actionable summary: an idea
        # that keeps dying at codegen needs a clearer brief, one that keeps
        # overtrading needs a different entry_q, and one that reaches
        # walk-forward and loses is the only kind that was actually tested.
        o["top_stage"] = max(o["stages"].items(), key=lambda kv: kv[1])[0]
        o["unjudged"] = sum(n for s, n in o["stages"].items() if s in UNJUDGED_STAGES)
    return out
