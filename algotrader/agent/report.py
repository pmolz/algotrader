"""Morning report — what the agent did while you were asleep.

The point of this file is *honesty at a glance*. After an unattended session you
should be able to read one page and know whether anything real happened, without
digging through logs or being flattered by cherry-picked numbers. So the report
leads with the promotion count (usually zero — that is the healthy outcome),
shows the failure-stage histogram so you can see *where* ideas die, and prints
the rising multiple-testing bar that makes future promotions harder.

`build_report` reads only the experiment DB, so you can regenerate any past
morning's report at any time.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
from datetime import time as dtime
from pathlib import Path
from typing import Any

from .memory import ExperimentDB

# "FAIL walk-forward: mean OOS Sharpe ..." -> "walk-forward"
_FAIL_RE = re.compile(r"^FAIL\s+([^:]+?)(?::|$)")


def fail_stage(exp: dict[str, Any]) -> str:
    """Which gauntlet stage killed this candidate. 'promoted' if none did."""
    if exp.get("promoted"):
        return "promoted"
    for reason in reversed(exp.get("reasons") or []):
        m = _FAIL_RE.match(reason.strip())
        if m:
            return m.group(1).strip()
    if exp.get("error"):
        return "crash"
    return "unknown"


def _sharpe_signal(exp: dict[str, Any]) -> float | None:
    """Best available "how close was it" number, for ranking near-misses.

    Prefers the honest out-of-sample figures over dev-set ones: final holdout
    Sharpe if it got that far, else mean walk-forward Sharpe, else dev Sharpe.
    """
    checks = exp.get("checks") or {}
    oos = checks.get("oos_holdout") or {}
    if isinstance(oos, dict) and oos.get("sharpe") is not None:
        return float(oos["sharpe"])
    wf = checks.get("walk_forward") or {}
    if isinstance(wf, dict) and wf.get("mean_sharpe") is not None:
        return float(wf["mean_sharpe"])
    if exp.get("sharpe") is not None:
        return float(exp["sharpe"])
    return None


def _fmt_ts(ts: float | None) -> str:
    if not ts:
        return "—"
    return datetime.fromtimestamp(ts).astimezone().strftime("%Y-%m-%d %H:%M %Z")


def _fmt_dur(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h {m}m" if h else (f"{m}m {s}s" if m else f"{s}s")


def _fmt_num(v: Any, nd: int = 2) -> str:
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return "—"


@dataclass
class Digest:
    """Aggregate view of a set of experiments."""

    total: int
    promoted: list[dict[str, Any]]
    errors: list[dict[str, Any]]
    stages: Counter
    symbols: Counter
    near_misses: list[dict[str, Any]]

    @property
    def n_promoted(self) -> int:
        return len(self.promoted)

    @property
    def n_errors(self) -> int:
        return len(self.errors)


def digest(experiments: list[dict[str, Any]], n_near: int = 5) -> Digest:
    promoted = [e for e in experiments if e.get("promoted")]
    errors = [e for e in experiments if e.get("error")]
    stages = Counter(fail_stage(e) for e in experiments)
    symbols = Counter(e.get("symbol") or "?" for e in experiments)

    scored = [
        (s, e)
        for e in experiments
        if not e.get("promoted") and (s := _sharpe_signal(e)) is not None
    ]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    near = [e for _, e in scored[:n_near]]
    return Digest(len(experiments), promoted, errors, stages, symbols, near)


def session_digest_text(experiments: list[dict[str, Any]], limit: int = 25) -> str:
    """Compact plain-text digest used as input to the agent's own reflection.

    Deliberately terse: the local model has a small context budget, and what it
    needs is the hypothesis plus the stage that killed it.
    """
    d = digest(experiments)
    lines = [
        f"Experiments run: {d.total} | promoted: {d.n_promoted} | errors: {d.n_errors}",
        "Failure stages: "
        + (", ".join(f"{k}={v}" for k, v in d.stages.most_common()) or "none"),
        "",
        "Ideas tried and how they died:",
    ]
    for e in experiments[:limit]:
        hyp = " ".join((e.get("hypothesis") or "").split())[:180]
        lines.append(f"- [{fail_stage(e)}] {e.get('strategy_name')}: {hyp}")
    if len(experiments) > limit:
        lines.append(f"- ... and {len(experiments) - limit} more")
    return "\n".join(lines)


def overnight_window(
    ref: datetime | None = None, start_hour: int = 18
) -> tuple[float, float]:
    """The window a morning report defaults to: from `start_hour` yesterday
    (local) up to now. Covers a session launched last evening whether it ran
    past midnight or not."""
    now = ref or datetime.now().astimezone()
    start = datetime.combine(now.date() - timedelta(days=1), dtime(hour=start_hour))
    start = start.replace(tzinfo=now.tzinfo)
    return start.timestamp(), now.timestamp()


def build_report(
    db: ExperimentDB,
    start: float,
    end: float,
    *,
    cfg: dict | None = None,
    generated_dir: str | Path = "experiments/generated",
) -> str:
    """Render the morning report for [start, end) as Markdown."""
    exps = db.experiments_between(start, end)
    runs = db.runs_between(start, end)
    d = digest(exps)
    gen_dir = Path(generated_dir)
    # Candidates that never reached a verdict. If this dominates, the night says
    # nothing about the markets — only about the model writing the code.
    unjudged = sum(
        d.stages[s] for s in ("codegen", "strategy runtime", "crash", "sandbox setup")
    )

    L: list[str] = []
    L.append(f"# Morning report — {_fmt_ts(end)}")
    L.append("")
    L.append(f"Window: {_fmt_ts(start)} → {_fmt_ts(end)}")
    L.append("")

    # -- headline ---------------------------------------------------------------
    if d.total == 0:
        L.append("## ⚠️ Nothing ran")
        L.append("")
        L.append(
            "No experiments were recorded in this window. Check the session log "
            "(`experiments/logs/`) and that cron actually fired."
        )
    elif d.n_promoted == 0 and unjudged > d.total / 2:
        L.append(
            f"## ⚠️ Wasted night — {unjudged}/{d.total} candidates never reached a verdict"
        )
        L.append("")
        L.append(
            "Nothing was promoted, but do **not** read that as evidence about the "
            "markets: most candidates failed to compile or crashed while running. "
            "That is a code-generation problem. Fix it before interpreting results."
        )
    elif d.n_promoted == 0:
        L.append(f"## Nothing promoted — {d.total} candidates tried, all rejected")
        L.append("")
        L.append(
            "This is the *expected* outcome. The gauntlet is doing its job; no "
            "action needed. See the failure histogram below for where ideas died."
        )
    else:
        L.append(f"## 🚩 {d.n_promoted} candidate(s) survived the gauntlet")
        L.append("")
        L.append(
            "Treat this as *suspicious until reviewed*, not as good news — a "
            "survivor after many trials is most often a multiple-testing artifact. "
            "Read the code and the checks before letting it near paper trading."
        )
    L.append("")

    # -- sessions ---------------------------------------------------------------
    L.append("## Sessions")
    L.append("")
    if not runs:
        L.append("_No session rows in this window (experiments may be ad-hoc runs)._")
    else:
        L.append("| Run | Kind | Started | Duration | Iters | Promoted | Errors | Stopped because | Sandbox |")
        L.append("|----:|------|---------|---------:|------:|---------:|-------:|-----------------|---------|")
        for r in runs:
            dur = (r["finished_at"] - r["started_at"]) if r["finished_at"] else None
            stop = r["stop_reason"] or "**never finished**"
            L.append(
                f"| {r['id']} | {r['kind'] or '—'} | {_fmt_ts(r['started_at'])} | "
                f"{_fmt_dur(dur)} | {r['iterations'] or 0} | {r['promoted'] or 0} | "
                f"{r['errors'] or 0} | {stop} | "
                f"{'on' if r['sandbox'] else '**OFF**'} |"
            )
        for r in runs:
            if r["finished_at"] is None:
                L.append("")
                L.append(
                    f"> ⚠️ Run {r['id']} has no finish time — the process was killed "
                    "or the machine slept. Its experiment count may be short."
                )
                break
        models = {(r["provider"], r["model"]) for r in runs}
        L.append("")
        L.append(
            "Model(s): "
            + ", ".join(f"`{p or '?'}/{m or '?'}`" for p, m in sorted(models, key=str))
        )
    L.append("")

    if d.total == 0:
        return "\n".join(L) + "\n"

    # -- promoted ---------------------------------------------------------------
    if d.promoted:
        L.append("## Promoted candidates (review these)")
        L.append("")
        for e in d.promoted:
            m = e.get("metrics") or {}
            L.append(f"### #{e['id']} `{e['strategy_name']}` — {e.get('symbol')}")
            L.append("")
            L.append(f"**Hypothesis:** {' '.join((e.get('hypothesis') or '').split())}")
            L.append("")
            L.append(
                f"- Final OOS holdout Sharpe: **{_fmt_num(m.get('sharpe'))}**, "
                f"max drawdown {_fmt_num(m.get('max_drawdown'), 3)}"
            )
            wf = (e.get("checks") or {}).get("walk_forward") or {}
            if wf:
                L.append(
                    f"- Walk-forward: mean Sharpe {_fmt_num(wf.get('mean_sharpe'))} "
                    f"({wf.get('positive_folds')}/{wf.get('n_folds')} folds positive), "
                    f"folds {[round(float(s), 2) for s in wf.get('fold_sharpes', [])]}"
                )
            dsr = (e.get("checks") or {}).get("deflated_sharpe") or {}
            if dsr:
                L.append(f"- Deflated Sharpe: {_fmt_num(dsr.get('dsr'))}")
            stress = (e.get("checks") or {}).get("cost_stress") or {}
            if stress:
                L.append(
                    "- Cost stress: "
                    + ", ".join(f"{k}x → {_fmt_num(v)}" for k, v in stress.items())
                )
            L.append(f"- Params: `{json.dumps(e.get('params') or {})}`")
            path = gen_dir / f"{e['id']:05d}_{e['strategy_name']}.py"
            L.append(f"- Code: `{path}`")
            L.append("")

    # -- where ideas died -------------------------------------------------------
    L.append("## Where candidates died")
    L.append("")
    L.append("| Stage | Count | Share |")
    L.append("|-------|------:|------:|")
    for stage, n in d.stages.most_common():
        L.append(f"| {stage} | {n} | {n / d.total:.0%} |")
    L.append("")
    top_stage = next((s for s, _ in d.stages.most_common() if s != "promoted"), None)
    if top_stage == "codegen":
        L.append(
            "> Most candidates never compiled. That is a **model-quality problem, "
            "not a market one** — use a bigger coder-tuned model before reading "
            "anything into tonight's results."
        )
    elif top_stage == "walk-forward":
        L.append(
            "> Ideas are dying at the first out-of-sample hurdle, which is the "
            "normal, healthy shape for this histogram."
        )
    elif top_stage == "strategy runtime":
        L.append(
            "> Most candidates compiled but blew up while running (bad indexing, "
            "NaNs, shape mismatches). Also a model-quality signal, not a market one."
        )
    elif top_stage in ("crash", "sandbox setup"):
        L.append(
            "> Most iterations failed before being judged. Check the errors below — "
            "the harness, not the strategies, is likely at fault."
        )
    L.append("")

    # -- near misses ------------------------------------------------------------
    if d.near_misses:
        L.append("## Closest rejections")
        L.append("")
        L.append(
            "_Ranked by the most out-of-sample Sharpe available. These are **not** "
            "shortlist candidates — re-testing them is how overfitting happens._"
        )
        L.append("")
        L.append("| # | Symbol | Strategy | Sharpe signal | Died at |")
        L.append("|--:|--------|----------|--------------:|---------|")
        for e in d.near_misses:
            L.append(
                f"| {e['id']} | {e.get('symbol')} | `{e['strategy_name']}` | "
                f"{_fmt_num(_sharpe_signal(e))} | {fail_stage(e)} |"
            )
        L.append("")

    # -- errors -----------------------------------------------------------------
    if d.errors:
        kinds = Counter(
            (e["error"] or "").strip().splitlines()[-1][:120] for e in d.errors
        )
        L.append(f"## Errors ({d.n_errors})")
        L.append("")
        for msg, n in kinds.most_common(8):
            L.append(f"- ×{n} `{msg}`")
        L.append("")

    # -- coverage ---------------------------------------------------------------
    L.append("## Coverage")
    L.append("")
    for sym, n in d.symbols.most_common():
        L.append(f"- {sym}: {n} experiments")
    L.append("")

    # -- multiple testing -------------------------------------------------------
    L.append("## Multiple-testing bar")
    L.append("")
    L.append(
        f"Lifetime experiments in the DB: **{db.total_trials()}**. Every one of "
        "them raises the deflated-Sharpe bar a future candidate must clear. This "
        "number only goes up — that is intentional."
    )
    L.append("")

    # -- reflection -------------------------------------------------------------
    refl = [r for r in db.latest_reflections(10) if start <= r["created_at"] < end]
    if refl:
        L.append("## The agent's own reflection")
        L.append("")
        for r in refl:
            L.append(f"_({r['source']}, run {r['run_id']})_")
            L.append("")
            L.append("> " + r["text"].strip().replace("\n", "\n> "))
            L.append("")

    # -- next actions -----------------------------------------------------------
    L.append("## Suggested next actions")
    L.append("")
    if d.promoted:
        L.append("1. Read the promoted code above line by line, looking for lookahead the automated check could miss.")
        L.append("2. If it holds up, forward-test it on paper — never straight to capital.")
    if any(r["finished_at"] is None for r in runs):
        L.append("- Investigate the unfinished session (sleep/suspend settings, OOM, cron timeout).")
    if top_stage in ("codegen", "strategy runtime"):
        L.append(
            f"- **{d.stages[top_stage]}/{d.total} candidates never got judged** "
            f"({top_stage}). Pull a larger coder model "
            "(`ollama pull qwen2.5-coder:14b`) before drawing conclusions."
        )
    unjudged = sum(
        d.stages[s] for s in ("codegen", "strategy runtime", "crash", "sandbox setup")
    )
    if not d.promoted and d.total and unjudged <= d.total / 2:
        L.append("- Nothing to act on. Let it keep running.")
    L.append("")

    return "\n".join(L) + "\n"


def write_report(text: str, out_dir: str | Path, when: datetime | None = None) -> Path:
    """Write the report to `out_dir/YYYY-MM-DD.md` and return the path."""
    when = when or datetime.now().astimezone()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{when.strftime('%Y-%m-%d')}.md"
    path.write_text(text)
    return path
