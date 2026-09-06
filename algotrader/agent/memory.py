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
import re
import sqlite3
import time
from pathlib import Path
from typing import Any

# Mechanism families, matched against the strategy name and hypothesis. Crude
# on purpose: the point is to tell the model "you have tried reversion eight
# times" so that "propose something different" is an instruction it can act on,
# not to build a taxonomy.
_FAMILIES: list[tuple[str, tuple[str, ...]]] = [
    ("mean-reversion", ("revers", "mean_rev", "pullback", "oversold", "bounce", "dip")),
    ("breakout", ("breakout", "range_break", "channel", "donchian")),
    ("momentum/trend", ("momentum", "trend", "macd", "crossover", "cross", "sma", "ema", "vwap")),
    ("volatility", ("volatil", "atr", "bollinger", "squeeze", "garch")),
    ("volume/flow", ("volume", "obv", "flow", "liquidity", "vpin")),
    ("microstructure", ("arbitrage", "microstructure", "spread", "orderbook", "hft")),
    ("oscillator", ("rsi", "stochastic", "cci", "oscillator")),
    ("seasonality", ("seasonal", "hour", "time_of_day", "weekday", "session")),
]


def _clip(text: str, n: int) -> str:
    t = " ".join((text or "").split())
    return t if len(t) <= n else t[: n - 1] + "\u2026"


def classify_family(text: str) -> str:
    t = (text or "").lower()
    for label, keys in _FAMILIES:
        if any(k in t for k in keys):
            return label
    return "other"


# Raw failure detail is for the human reading the log. A model given five
# lookahead cut-indices learns nothing it can act on, and in practice that noise
# was 31% of the whole memory block — crowding out the part that carries signal.
_NOISE_RE = re.compile(r"\s*[\[\{].*", re.DOTALL)
_REASON_GIST = [
    ("FAIL lookahead", "FAIL lookahead — used future data (no .shift(-k), no whole-sample stats)"),
    ("FAIL overtrading", None),          # already carries its own numbers
    ("FAIL never traded", "FAIL never traded — the entry filter never fired"),
    ("FAIL too few trades", None),
    ("FAIL cost stress", "FAIL cost stress — the edge disappears at higher costs"),
    ("FAIL ambiguous exits", "FAIL ambiguous exits — stop and target sat in the same bar too often"),
    ("FAIL codegen", "FAIL codegen — the code did not load"),
    ("FAIL strategy runtime", "FAIL runtime — the strategy raised while running"),
]


def summarize_reason(reason: str | None) -> str:
    """Collapse a verdict to the part a model can act on."""
    if not reason:
        return "no verdict recorded"
    for prefix, gist in _REASON_GIST:
        if reason.startswith(prefix):
            return gist or _NOISE_RE.sub("", reason).strip()
    return _NOISE_RE.sub("", reason).strip()[:160]


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
    run_id INTEGER,            -- FK to runs.id; NULL for ad-hoc runs
    researched_from TEXT       -- research brief id, or NULL if self-generated
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
    source TEXT,               -- 'llm' | 'heuristic'
    regime TEXT                -- e.g. 'ccxt:15m'; advice does not travel across these
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
        if "researched_from" not in have:
            self.conn.execute("ALTER TABLE experiments ADD COLUMN researched_from TEXT")
        have = {r["name"] for r in self.conn.execute("PRAGMA table_info(runs)")}
        if "host" not in have:
            self.conn.execute("ALTER TABLE runs ADD COLUMN host TEXT")
        have = {r["name"] for r in self.conn.execute("PRAGMA table_info(reflections)")}
        if "regime" not in have:
            self.conn.execute("ALTER TABLE reflections ADD COLUMN regime TEXT")

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
        researched_from: str | None = None,
    ) -> int:
        cur = self.conn.execute(
            """INSERT INTO experiments
               (created_at, symbol, source, timeframe, strategy_name, hypothesis,
                params_json, code, promoted, metrics_json, gauntlet_json, reasons,
                error, run_id, researched_from)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                time.time(), symbol, source, timeframe, strategy_name, hypothesis,
                json.dumps(params), code, int(promoted),
                json.dumps(metrics or {}), json.dumps(gauntlet_checks or {}),
                json.dumps(reasons or []), error, run_id, researched_from,
            ),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def brief_tally(self, symbol: str | None = None) -> dict[str, int]:
        """How many times each research brief has been coded, for this symbol.

        Feeds `research.BriefLibrary.select`, which rotates through the library
        least-attempted-first and retires a brief once it has had its attempts.
        Scoped by symbol for the same reason `recent` is: a brief that died on
        BTC has not yet been tried on ETH.
        """
        clause, args = "", []
        if symbol:
            clause, args = "WHERE symbol=?", [symbol]
        rows = self.conn.execute(
            f"""SELECT researched_from AS b, COUNT(*) AS c FROM experiments
                {clause} {"AND" if clause else "WHERE"} researched_from IS NOT NULL
                GROUP BY researched_from""",
            args,
        ).fetchall()
        return {r["b"]: int(r["c"]) for r in rows}

    def total_trials(self, symbol: str | None = None) -> int:
        """Number of experiments run — the honest n_trials for deflated Sharpe."""
        if symbol:
            row = self.conn.execute(
                "SELECT COUNT(*) c FROM experiments WHERE symbol=?", (symbol,)
            ).fetchone()
        else:
            row = self.conn.execute("SELECT COUNT(*) c FROM experiments").fetchone()
        return int(row["c"])

    def recent(self, n: int = 8, symbol: str | None = None) -> list[dict[str, Any]]:
        # Scoped the same way total_trials() is: the project already treats each
        # symbol as its own experiment stream for the multiple-testing count, so
        # its memory should not silently mix them.
        if symbol:
            rows = self.conn.execute(
                "SELECT * FROM experiments WHERE symbol=? ORDER BY id DESC LIMIT ?",
                (symbol, n),
            ).fetchall()
        else:
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

    def furthest(self, n: int = 4, symbol: str | None = None) -> list[dict[str, Any]]:
        """Candidates that got deepest into the gauntlet.

        Deliberately not "highest Sharpe". Ranking the log by dev-set Sharpe
        would hand the agent a leaderboard to overfit against, which is the exact
        failure this project exists to avoid. How many stages a candidate cleared
        is progress that cannot be gamed by curve-fitting a single number.
        """
        clause, args = "", []
        if symbol:
            clause, args = "WHERE symbol=?", [symbol]
        rows = self.conn.execute(
            f"SELECT * FROM experiments {clause} ORDER BY id DESC LIMIT 400", args
        ).fetchall()
        lessons = [self._row_to_lesson(r) for r in rows]
        lessons = [x for x in lessons if x["stages_passed"] > 0]
        lessons.sort(key=lambda x: (x["stages_passed"], x["sharpe"] or -99), reverse=True)
        return lessons[:n]

    def family_tally(self, limit: int = 60, symbol: str | None = None) -> list[tuple]:
        """(family, tried, furthest_stage) for the mechanism families tried.

        The prompt asks for an idea "meaningfully different from anything above",
        which the model cannot honour from a list of names it has to infer
        families from. This states the tally outright.
        """
        clause, args = "", []
        if symbol:
            clause, args = "WHERE symbol=?", [symbol]
        rows = self.conn.execute(
            f"SELECT * FROM experiments {clause} ORDER BY id DESC LIMIT ?",
            [*args, limit],
        ).fetchall()
        agg: dict[str, list[int]] = {}
        for r in rows:
            lesson = self._row_to_lesson(r)
            fam = classify_family(f"{r['strategy_name']} {r['hypothesis'] or ''}")
            cur = agg.setdefault(fam, [0, 0])
            cur[0] += 1
            cur[1] = max(cur[1], lesson["stages_passed"])
        return sorted(((k, v[0], v[1]) for k, v in agg.items()),
                      key=lambda t: t[1], reverse=True)

    def promoted(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM experiments WHERE promoted=1 ORDER BY id DESC"
        ).fetchall()
        return [self._row_to_lesson(r) for r in rows]

    @staticmethod
    def _row_to_lesson(r: sqlite3.Row) -> dict[str, Any]:
        metrics = json.loads(r["metrics_json"] or "{}")
        reasons = json.loads(r["reasons"] or "[]")
        return {
            "id": r["id"],
            "strategy_name": r["strategy_name"],
            "hypothesis": r["hypothesis"],
            "params": json.loads(r["params_json"] or "{}"),
            "promoted": bool(r["promoted"]),
            "sharpe": metrics.get("sharpe"),
            "max_drawdown": metrics.get("max_drawdown"),
            "reasons": reasons,
            # How deep into the gauntlet it got. Each stage appends one PASS.
            "stages_passed": sum(1 for x in reasons if str(x).startswith("PASS")),
            "died_of": next((x for x in reasons if str(x).startswith("FAIL")), None),
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

    def record_reflection(self, text: str, *, run_id: int | None, source: str,
                          regime: str | None = None) -> int:
        cur = self.conn.execute(
            "INSERT INTO reflections (created_at, run_id, text, source, regime) "
            "VALUES (?,?,?,?,?)",
            (time.time(), run_id, text, source, regime),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def latest_reflections(self, n: int = 2,
                           regime: str | None = None) -> list[dict[str, Any]]:
        """Reflections written about a different market or timeframe are worse
        than useless — they are confident, specific advice about a setup that no
        longer exists. When a regime is given, only that regime's are returned.
        """
        if regime:
            rows = self.conn.execute(
                "SELECT * FROM reflections WHERE regime=? ORDER BY id DESC LIMIT ?",
                (regime, n),
            ).fetchall()
        else:
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
            researched_from=(
                r["researched_from"] if "researched_from" in r.keys() else None
            ),
        )
        return d

    def lessons_context(self, n_recent: int = 8, n_best: int = 4,
                        symbol: str | None = None, regime: str | None = None) -> str:
        """Compact text block summarizing past results for the LLM prompt.

        Written for a model with a limited budget of attention, so every line has
        to earn its tokens: what was tried, how far it got, and why it stopped.
        Raw verdict payloads are summarised rather than dumped, and the family
        tally is stated outright because "propose something different" is not
        actionable against a list of names.
        """
        lines: list[str] = []

        tally = self.family_tally(symbol=symbol)
        if tally:
            lines.append("Mechanism families already tried (count, furthest gauntlet stage):")
            for fam, count, stage in tally:
                lines.append(f"  {fam}: {count} attempt(s), best reached stage {stage}")
            lines.append("  -> Propose a family that is absent or barely explored.")
            lines.append("")

        furthest = self.furthest(n_best, symbol=symbol)
        if furthest:
            lines.append("Closest any idea has come (ranked by stages cleared, not by Sharpe):")
            for e in furthest:
                sharpe = f"{e['sharpe']:.2f}" if e["sharpe"] is not None else "n/a"
                lines.append(
                    f"  [id={e['id']}] {e['strategy_name']} — cleared {e['stages_passed']} "
                    f"stage(s), dev Sharpe {sharpe}"
                )
                if e["hypothesis"]:
                    lines.append(f"      idea: {_clip(e['hypothesis'], 200)}")
                lines.append(f"      died: {summarize_reason(e['died_of'])}")
            lines.append("")

        recent = self.recent(n_recent, symbol=symbol)
        if recent:
            lines.append("Most recent attempts (do not repeat these):")
            for e in recent:
                verdict = "PROMOTED" if e["promoted"] else summarize_reason(e["died_of"])
                lines.append(f"  [id={e['id']}] {e['strategy_name']}: {verdict}")
                if e["hypothesis"] and e["hypothesis"] != "(no hypothesis provided)":
                    lines.append(f"      idea: {_clip(e['hypothesis'], 160)}")
            lines.append("")

        refl = self.latest_reflections(2, regime=regime)
        if refl:
            lines.append("Your own conclusions from previous sessions:")
            for r in reversed(refl):
                lines.append(f"  {_clip(' '.join(r['text'].split()), 700)}")
        return "\n".join(lines).rstrip()

    def close(self):
        self.conn.close()
