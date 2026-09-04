"""Experiment log = the system's memory.

Every strategy the agent tries is recorded here with its hypothesis, generated
code, full metrics, and the gauntlet verdict. This is what makes the system
"self-improving": before proposing a new idea, the agent reads what already
failed and why, so it stops repeating mistakes.

Also the bookkeeping for multiple-testing: total_trials() feeds the deflated
Sharpe correction so we don't fool ourselves after thousands of attempts.
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
    error TEXT                 -- populated if the run crashed
);
"""


class ExperimentDB:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

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
    ) -> int:
        cur = self.conn.execute(
            """INSERT INTO experiments
               (created_at, symbol, source, timeframe, strategy_name, hypothesis,
                params_json, code, promoted, metrics_json, gauntlet_json, reasons, error)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                time.time(), symbol, source, timeframe, strategy_name, hypothesis,
                json.dumps(params), code, int(promoted),
                json.dumps(metrics or {}), json.dumps(gauntlet_checks or {}),
                json.dumps(reasons or []), error,
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
        return "\n".join(lines)

    def close(self):
        self.conn.close()
