"""Start and stop agent sessions from the dashboard.

This is the one part of the dashboard with side effects, so it is deliberately
narrow: it can launch exactly one command line, with validated arguments, as a
transient systemd unit — and it can stop that unit. It cannot run arbitrary
input, and it still never writes the experiment DB itself (the session process
does that, as it always has).

Why systemd rather than a bare subprocess: a session outlives the dashboard.
Restarting or crashing the web app must not kill a four-hour research run, and
`systemctl stop` gives a clean SIGTERM that `NightlySession` already handles by
finishing the current iteration, writing its reflection, and closing the run row.

Arguments are passed as a list, never through a shell.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from ..config import REPO_ROOT, get

UNIT = "algotrader-session"          # transient; --collect reaps it when it exits
UNIT_SERVICE = f"{UNIT}.service"

# Budget limits. A session is a long-running job on the user's own machine, but
# accepting an unbounded number from a form is how you end up with a typo
# pinning the CPU for a fortnight.
MIN_HOURS, MAX_HOURS = 0.01, 24.0
MAX_ITERATIONS = 10_000


class ControlError(RuntimeError):
    """Raised with a message intended to be shown directly in the UI."""


@dataclass
class Budget:
    hours: float | None
    max_iterations: int | None

    @classmethod
    def parse(cls, hours, iterations) -> Budget:
        """Validate what came off the form. Rejects rather than clamps, so a
        mistyped value is visible instead of silently becoming something else."""
        h = None
        if hours not in (None, "", "none"):
            try:
                h = float(hours)
            except (TypeError, ValueError) as e:
                raise ControlError(f"Not a number: {hours!r}") from e
            if not (MIN_HOURS <= h <= MAX_HOURS):
                raise ControlError(
                    f"Duration must be between {MIN_HOURS} and {MAX_HOURS} hours."
                )
        n = None
        if iterations not in (None, "", "none"):
            try:
                n = int(iterations)
            except (TypeError, ValueError) as e:
                raise ControlError(f"Not a whole number: {iterations!r}") from e
            if not (1 <= n <= MAX_ITERATIONS):
                raise ControlError(f"Iterations must be between 1 and {MAX_ITERATIONS}.")
        if h is None and n is None:
            raise ControlError("Set a duration or an iteration count — not neither.")
        return cls(h, n)

    def args(self) -> list[str]:
        out = []
        if self.hours is not None:
            out += ["--hours", str(self.hours)]
        if self.max_iterations is not None:
            out += ["--max-iterations", str(self.max_iterations)]
        return out


def systemd_available() -> bool:
    if not (shutil.which("systemd-run") and shutil.which("systemctl")):
        return False
    try:
        r = subprocess.run(
            ["systemctl", "--user", "is-system-running"],
            capture_output=True, timeout=5, text=True,
        )
        # "degraded"/"starting" are still usable; only a missing manager is not
        return r.returncode == 0 or "running" in r.stdout or "degraded" in r.stdout
    except (OSError, subprocess.SubprocessError):
        return False


def unit_active() -> bool:
    try:
        r = subprocess.run(
            ["systemctl", "--user", "is-active", UNIT_SERVICE],
            capture_output=True, timeout=5, text=True,
        )
        return r.stdout.strip() == "active"
    except (OSError, subprocess.SubprocessError):
        return False


def _python(cfg: dict) -> str:
    venv = REPO_ROOT / ".venv" / "bin" / "python"
    return str(venv) if venv.exists() else "python3"


def start(cfg: dict, budget: Budget) -> dict:
    """Launch a session as a transient systemd unit."""
    if not systemd_available():
        raise ControlError(
            "systemd --user isn't available here, so the dashboard can't manage "
            "sessions. Run `python scripts/run_nightly.py` in a terminal instead."
        )
    if unit_active():
        raise ControlError("A session is already running. Stop it first.")

    # The session's own PID lock is the real guard against two concurrent runs;
    # this just turns it into a clearer message than an exit code.
    lock = Path(get(cfg, "nightly.lock_file", "experiments/nightly.lock"))
    if not lock.is_absolute():
        lock = REPO_ROOT / lock
    if lock.exists():
        raise ControlError(
            f"A session lock is held ({lock.read_text().strip()}). If that process "
            "is gone, the next start will take the stale lock automatically."
        )

    cmd = [
        "systemd-run", "--user", "--collect",
        f"--unit={UNIT}",
        "--description=algotrader session (started from the dashboard)",
        f"--working-directory={REPO_ROOT}",
        "--property=TimeoutStartSec=infinity",
        # give a graceful SIGTERM room to finish the current iteration
        "--property=TimeoutStopSec=600",
        "--property=KillSignal=SIGTERM",
        "--setenv=PYTHONUNBUFFERED=1",
        _python(cfg), str(REPO_ROOT / "scripts" / "run_nightly.py"),
        *budget.args(),
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=20, text=True)
    except subprocess.SubprocessError as e:
        raise ControlError(f"Could not launch: {e}") from e
    if r.returncode != 0:
        raise ControlError(f"systemd-run failed: {(r.stderr or r.stdout).strip()[:300]}")
    return {"started": True, "unit": UNIT_SERVICE, "budget": budget.args()}


def stop() -> dict:
    """Ask the session to stop. SIGTERM, not SIGKILL: it finishes the current
    iteration, writes its reflection, and closes the run row."""
    if not unit_active():
        raise ControlError("No session is running.")
    try:
        r = subprocess.run(
            ["systemctl", "--user", "stop", UNIT_SERVICE],
            capture_output=True, timeout=30, text=True,
        )
    except subprocess.SubprocessError as e:
        raise ControlError(f"Could not stop: {e}") from e
    if r.returncode != 0:
        raise ControlError(f"systemctl stop failed: {(r.stderr or '').strip()[:300]}")
    return {"stopping": True}


def status(conn, cfg: dict) -> dict:
    """Everything the UI needs to render the control card.

    Combines the unit's state with the in-flight run row, so the page can show
    real progress (iterations done, promoted so far, elapsed vs budget) rather
    than just "running".
    """
    active = unit_active()
    row = conn.execute(
        "SELECT * FROM runs WHERE finished_at IS NULL ORDER BY id DESC LIMIT 1"
    ).fetchone()

    out = {
        "active": active,
        "available": systemd_available(),
        "run": None,
        "scheduled": timer_enabled(),
    }
    if row is None:
        return out

    import json as _json

    budget = _json.loads(row["budget_json"] or "{}")
    started = row["started_at"]
    elapsed = time.time() - started
    done = conn.execute(
        "SELECT COUNT(*) c, COALESCE(SUM(promoted),0) p FROM experiments WHERE run_id=?",
        (row["id"],),
    ).fetchone()

    out["run"] = {
        "id": row["id"],
        "started_at": started,
        "elapsed_s": elapsed,
        "budget": budget,
        "iterations": done["c"],
        "promoted": done["p"],
        "model": row["model"],
        "symbols": _json.loads(row["symbols"] or "[]"),
        # A run row with no finish time and no live unit means the process died.
        "orphaned": not active,
        "progress": (
            min(1.0, elapsed / (budget["max_hours"] * 3600))
            if budget.get("max_hours") else None
        ),
    }
    return out


# -- the schedule ------------------------------------------------------------------
TIMER = "algotrader-nightly.timer"


def timer_enabled() -> bool:
    try:
        r = subprocess.run(
            ["systemctl", "--user", "is-enabled", TIMER],
            capture_output=True, timeout=5, text=True,
        )
        return r.stdout.strip() == "enabled"
    except (OSError, subprocess.SubprocessError):
        return False


def set_timer(enabled: bool) -> dict:
    """Turn the nightly schedule on or off."""
    if not systemd_available():
        raise ControlError("systemd --user isn't available here.")
    verb = ["enable", "--now"] if enabled else ["disable", "--now"]
    try:
        r = subprocess.run(
            ["systemctl", "--user", *verb, TIMER],
            capture_output=True, timeout=20, text=True,
        )
    except subprocess.SubprocessError as e:
        raise ControlError(f"Could not change the schedule: {e}") from e
    if r.returncode != 0:
        raise ControlError(
            f"systemctl {verb[0]} failed: {(r.stderr or '').strip()[:300]}"
        )
    return {"scheduled": enabled}


def next_run() -> str | None:
    """When the timer will next fire, if it is enabled."""
    if not timer_enabled():
        return None
    try:
        r = subprocess.run(
            ["systemctl", "--user", "show", TIMER, "-p", "NextElapseUSecRealtime"],
            capture_output=True, timeout=5, text=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    _, _, v = r.stdout.strip().partition("=")
    return v or None
