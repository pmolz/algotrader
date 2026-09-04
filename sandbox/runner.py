"""In-container entrypoint. Runs the validation gauntlet on an untrusted strategy.

Reads from the bind-mounted /work directory:
    /work/strategy.py   - the LLM-generated strategy code (untrusted)
    /work/data.parquet  - OHLCV data
    /work/params.json   - {"config": {...}, "n_trials": int}
Writes:
    /work/report.json   - {"ok": bool, "promoted": bool, "reasons": [...],
                           "checks": {...}, "error": str|None}

Failures are attributed to a phase rather than lumped together, because the
caller acts on the difference: a `FAIL codegen` earns an LLM fix-retry, while a
`FAIL gauntlet crash` is a harness bug. The morning report histograms the same
labels, so "the model can't write valid code" never gets mistaken for "the ideas
don't survive validation".

This process has no network, a read-only root fs, dropped capabilities, and
resource limits (set by the host launcher). Even if the strategy code is
malicious, it is contained here.
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path

WORK = Path("/work")


def main() -> int:
    out = {"ok": False, "promoted": False, "reasons": [], "checks": {}, "error": None}

    def fail(stage: str) -> None:
        out["reasons"] = [f"FAIL {stage}"]
        out["error"] = traceback.format_exc(limit=4)

    try:
        import pandas as pd

        from algotrader.agent.codegen import load_strategy_class
        from algotrader.validation.gauntlet import run_gauntlet

        code = (WORK / "strategy.py").read_text()
        df = pd.read_parquet(WORK / "data.parquet")
        params = json.loads((WORK / "params.json").read_text())
        cfg = params["config"]
        n_trials = int(params.get("n_trials", 1))
    except Exception:  # noqa: BLE001
        fail("sandbox setup")
        (WORK / "report.json").write_text(json.dumps(out, default=str))
        return 0

    # Phase 1: compile + instantiate. Both are generated code, so both are the
    # model's fault and both are worth one fix-retry.
    try:
        strategy = load_strategy_class(code)()
    except Exception:  # noqa: BLE001
        fail("codegen")
        (WORK / "report.json").write_text(json.dumps(out, default=str))
        return 0

    # Phase 2: validation. A crash in here is usually still the strategy's fault
    # (bad indexing, NaNs), so keep its name for the report.
    try:
        report = run_gauntlet(df, strategy, cfg, n_trials=n_trials)
        out.update(
            ok=True,
            promoted=bool(report.promoted),
            reasons=report.reasons,
            checks=report.checks,
            strategy_name=strategy.name,
            params=strategy.params,
        )
    except Exception:  # noqa: BLE001
        fail("strategy runtime")
        out["strategy_name"] = getattr(strategy, "name", None)
        out["params"] = getattr(strategy, "params", {})

    (WORK / "report.json").write_text(json.dumps(out, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
