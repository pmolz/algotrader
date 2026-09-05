"""Host-side Docker sandbox launcher.

Runs the validation gauntlet on an untrusted, LLM-generated strategy INSIDE a
locked-down container, so nothing the generated code does can touch your machine,
network, or secrets. This is the Phase-2 replacement for in-process `exec`.

Security posture (all enforced on every run):
  * --network none            no network access at all
  * --read-only               immutable root filesystem
  * --tmpfs /tmp              small writable scratch, wiped on exit
  * --cap-drop ALL            no Linux capabilities
  * --security-opt no-new-privileges
  * --pids-limit / --memory / --cpus   resource caps (fork bombs, OOM, CPU)
  * --user <host uid:gid>     non-root; keeps the mounted /work host-owned
  * subprocess timeout        hard wall-clock kill

The only writable, shared surface is a per-run temp dir bind-mounted at /work,
which holds the code, data, params, and receives report.json. It is deleted after.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


class SandboxError(RuntimeError):
    pass


@dataclass
class SandboxLimits:
    image: str = "algotrader-sandbox:latest"
    memory: str = "1g"
    cpus: str = "2"
    pids: int = 128
    timeout_s: int = 120


def docker_available() -> bool:
    try:
        r = subprocess.run(
            ["docker", "info"], capture_output=True, timeout=10
        )
        return r.returncode == 0
    except Exception:
        return False


def image_exists(image: str) -> bool:
    try:
        r = subprocess.run(
            ["docker", "image", "inspect", image],
            capture_output=True, timeout=10,
        )
        return r.returncode == 0
    except Exception:
        return False


def run_gauntlet_sandboxed(
    df: pd.DataFrame,
    code: str,
    cfg: dict,
    n_trials: int = 1,
    limits: SandboxLimits | None = None,
    mode: str = "gauntlet",
) -> dict:
    """Run `code` inside Docker. Returns the report dict:
        {ok, promoted, reasons, checks, error, strategy_name?, params?}
    or, with mode="equity", the curve dict:
        {ok, dates, equity, benchmark, metrics, holdout_start, ...}
    Raises SandboxError on infrastructure failures (image missing, timeout, etc.).
    """
    limits = limits or SandboxLimits()
    if not image_exists(limits.image):
        raise SandboxError(
            f"Sandbox image {limits.image!r} not found. Build it: bash sandbox/build.sh"
        )

    with tempfile.TemporaryDirectory(prefix="algotrader_sbx_") as tmp:
        work = Path(tmp)
        (work / "strategy.py").write_text(code)
        df.to_parquet(work / "data.parquet")
        (work / "params.json").write_text(
            json.dumps({"config": cfg, "n_trials": n_trials, "mode": mode})
        )

        uid, gid = os.getuid(), os.getgid()
        cmd = [
            "docker", "run", "--rm",
            "--network", "none",
            "--read-only",
            "--tmpfs", "/tmp:size=64m",
            "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges",
            "--pids-limit", str(limits.pids),
            "--memory", limits.memory,
            "--memory-swap", limits.memory,   # disallow swap growth
            "--cpus", limits.cpus,
            "--user", f"{uid}:{gid}",
            "-v", f"{work}:/work",
            limits.image,
        ]
        try:
            proc = subprocess.run(
                cmd, capture_output=True, timeout=limits.timeout_s, text=True
            )
        except subprocess.TimeoutExpired as e:
            raise SandboxError(f"Sandbox timed out after {limits.timeout_s}s") from e

        report_path = work / "report.json"
        if not report_path.exists():
            raise SandboxError(
                "Sandbox produced no report. "
                f"exit={proc.returncode} stderr={proc.stderr[-500:]!r}"
            )
        return json.loads(report_path.read_text())
