"""Experiment log = the system's memory.

Every strategy the agent tries is recorded here with its hypothesis, generated
code, full metrics, and the gauntlet verdict. This is what makes the system
"self-improving": before proposing a new idea, the agent reads what already
failed and why, so it stops repeating mistakes.

Also the bookkeeping for multiple-testing: total_trials() feeds the deflated
Sharpe correction so we don't fool ourselves after thousands of attempts.

Three tables:
  experiments  -- one row per candidate strategy tried
  runs         -- one row per unattended session (the overnight cron loop)
  reflections  -- the agent's own end-of-session summary of what it learned
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at REAL NOT NULL,
    symbol TEXT,
    source TEXT,
    timeframe TEXT,
    strategy_name TEXT,
    hypothesis TEXT,
    params_json TEXT,
    code TEXT,
    promoted INTEGER,          -- 1 if it passed the gauntlet
    metrics_json TEXT,         -- dev-set metrics
    gauntlet_json TEXT,        -- full GauntletReport checks
    reasons TEXT,              -- human-readable pass/fail notes
    error TEXT,                -- populated if the run crashed
    run_id INTEGER             -- FK to runs.id; NULL for ad-hoc runs
);

CREATE INDEX IF NOT EXISTS idx_experiments_created ON experiments(created_at);
CREATE INDEX IF NOT EXISTS idx_experiments_run ON experiments(run_id);

CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at REAL NOT NULL,
    finished_at REAL,          -- NULL while in flight (or if the box died mid-run)
    kind TEXT,                 -- 'nightly' | 'manual'
    symbols TEXT,              -- JSON list of symbols worked on
    provider TEXT,
    model TEXT,
    host TEXT,                 -- which LLM endpoint served this session
    sandbox INTEGER,
    budget_json TEXT,          -- the limits this session was given
    iterations INTEGER,        -- completed iterations
    promoted INTEGER,
    errors INTEGER,            -- iterations that ended in an error
    stop_reason TEXT           -- 'budget' | 'iterations' | 'signal' | 'fatal'
);

CREATE TABLE IF NOT EXISTS reflections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at REAL NOT NULL,
    run_id INTEGER,
    text TEXT NOT NULL,
    source TEXT                -- 'llm' | 'heuristic'
);
"""


class ExperimentDB:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self._migrate()
        self.conn.commit()

    def _migrate(self) -> None:
        """Add columns introduced after a DB was first created. CREATE TABLE IF
        NOT EXISTS won't backfill them, so do it explicitly and idempotently."""
        have = {r["name"] for r in self.conn.execute("PRAGMA table_info(experiments)")}
        if "run_id" not in have:
            self.conn.execute("ALTER TABLE experiments ADD COLUMN run_id INTEGER")
        have = {r["name"] for r in self.conn.execute("PRAGMA table_info(runs)")}
        if "host" not in have:
            self.conn.execute("ALTER TABLE runs ADD COLUMN host TEXT")

    def record(
        self,
        *,
        symbol: str,
        source: str,
        timeframe: str,
        strategy_name: str,
        hypothesis: str,
        params: dict,
        code: str | None,
        promoted: bool,
        metrics: dict | None,
        gauntlet_checks: dict | None,
        reasons: list[str] | None,
        error: str | None = None,
        run_id: int | None = None,
    ) -> int:
        cur = self.conn.execute(
            """INSERT INTO experiments
               (created_at, symbol, source, timeframe, strategy_name, hypothesis,
                params_json, code, promoted, metrics_json, gauntlet_json, reasons,
                error, run_id)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                time.time(), symbol, source, timeframe, strategy_name, hypothesis,
                json.dumps(params), code, int(promoted),
                json.dumps(metrics or {}), json.dumps(gauntlet_checks or {}),
                json.dumps(reasons or []), error, run_id,
            ),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def total_trials(self, symbol: str | None = None) -> int:
        """Number of experiments run — the honest n_trials for deflated Sharpe."""
        if symbol:
            row = self.conn.execute(
                "SELECT COUNT(*) c FROM experiments WHERE symbol=?", (symbol,)
            ).fetchone()
        else:
            row = self.conn.execute("SELECT COUNT(*) c FROM experiments").fetchone()
        return int(row["c"])

    def recent(self, n: int = 8) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM experiments ORDER BY id DESC LIMIT ?", (n,)
        ).fetchall()
        return [self._row_to_lesson(r) for r in rows]

    def best(self, n: int = 4) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """SELECT * FROM experiments
               WHERE json_extract(metrics_json,'$.sharpe') IS NOT NULL
               ORDER BY json_extract(metrics_json,'$.sharpe') DESC LIMIT ?""",
            (n,),
        ).fetchall()
        return [self._row_to_lesson(r) for r in rows]

    def promoted(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM experiments WHERE promoted=1 ORDER BY id DESC"
        ).fetchall()
        return [self._row_to_lesson(r) for r in rows]

    @staticmethod
    def _row_to_lesson(r: sqlite3.Row) -> dict[str, Any]:
        metrics = json.loads(r["metrics_json"] or "{}")
        return {
            "id": r["id"],
            "strategy_name": r["strategy_name"],
            "hypothesis": r["hypothesis"],
            "params": json.loads(r["params_json"] or "{}"),
            "promoted": bool(r["promoted"]),
            "sharpe": metrics.get("sharpe"),
            "max_drawdown": metrics.get("max_drawdown"),
            "reasons": json.loads(r["reasons"] or "[]"),
            "error": r["error"],
        }

    # -- run bookkeeping (unattended sessions) ------------------------------------
    def start_run(
        self,
        *,
        kind: str,
        symbols: list[str],
        provider: str | None,
        model: str | None,
        sandbox: bool,
        budget: dict,
        host: str | None = None,
    ) -> int:
        """Open a run row. Written up-front so a killed session still leaves a
        trace with finished_at NULL — the morning report flags that."""
        cur = self.conn.execute(
            """INSERT INTO runs
               (started_at, kind, symbols, provider, model, host, sandbox,
                budget_json, iterations, promoted, errors)
               VALUES (?,?,?,?,?,?,?,?,0,0,0)""",
            (time.time(), kind, json.dumps(symbols), provider, model, host,
             int(sandbox), json.dumps(budget)),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def finish_run(
        self,
        run_id: int,
        *,
        iterations: int,
        promoted: int,
        errors: int,
        stop_reason: str,
    ) -> None:
        self.conn.execute(
            """UPDATE runs SET finished_at=?, iterations=?, promoted=?, errors=?,
                               stop_reason=? WHERE id=?""",
            (time.time(), iterations, promoted, errors, stop_reason, run_id),
        )
        self.conn.commit()

    def run(self, run_id: int) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        return dict(row) if row else None

    def runs_between(self, start: float, end: float) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM runs WHERE started_at>=? AND started_at<? ORDER BY id",
            (start, end),
        ).fetchall()
        return [dict(r) for r in rows]

    def record_reflection(self, text: str, *, run_id: int | None, source: str) -> int:
        cur = self.conn.execute(
            "INSERT INTO reflections (created_at, run_id, text, source) VALUES (?,?,?,?)",
            (time.time(), run_id, text, source),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def latest_reflections(self, n: int = 2) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM reflections ORDER BY id DESC LIMIT ?", (n,)
        ).fetchall()
        return [dict(r) for r in rows]

    def reflections_for_run(self, run_id: int) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM reflections WHERE run_id=? ORDER BY id", (run_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    # -- windowed queries (what the morning report reads) -------------------------
    def experiments_between(self, start: float, end: float) -> list[dict[str, Any]]:
        """Every experiment created in [start, end), full detail, oldest first."""
        rows = self.conn.execute(
            "SELECT * FROM experiments WHERE created_at>=? AND created_at<? ORDER BY id",
            (start, end),
        ).fetchall()
        return [self._row_to_detail(r) for r in rows]

    @staticmethod
    def _row_to_detail(r: sqlite3.Row) -> dict[str, Any]:
        d = ExperimentDB._row_to_lesson(r)
        d.update(
            created_at=r["created_at"],
            symbol=r["symbol"],
            source=r["source"],
            timeframe=r["timeframe"],
            metrics=json.loads(r["metrics_json"] or "{}"),
            checks=json.loads(r["gauntlet_json"] or "{}"),
            run_id=r["run_id"] if "run_id" in r.keys() else None,
        )
        return d

    def lessons_context(self, n_recent: int = 8, n_best: int = 4) -> str:
        """Compact text block summarizing past results for the LLM prompt."""
        lines = ["Past experiments (learn from these — do not repeat failures):"]
        for e in self.best(n_best):
            lines.append(
                f"  [BEST id={e['id']}] {e['strategy_name']} sharpe={e['sharpe']} "
                f"promoted={e['promoted']} :: {e['hypothesis']}"
            )
        for e in self.recent(n_recent):
            verdict = "PROMOTED" if e["promoted"] else "rejected"
            note = e["reasons"][-1] if e["reasons"] else (e["error"] or "")
            lines.append(
                f"  [id={e['id']} {verdict}] {e['strategy_name']} "
                f"sharpe={e['sharpe']} :: {note}"
            )
        refl = self.latest_reflections(2)
        if refl:
            lines.append("")
            lines.append("Your own conclusions from previous sessions:")
            for r in reversed(refl):
                lines.append(f"  {r['text'].strip()}")
        return "\n".join(lines)

    def close(self):
        self.conn.close()
