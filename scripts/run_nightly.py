#!/usr/bin/env python3
"""Run one unattended session. This is what cron calls.

Examples:
    # exactly what the cron entry does
    python scripts/run_nightly.py

    # short supervised trial before trusting it overnight
    python scripts/run_nightly.py --hours 0.1 --max-iterations 2 --symbol BTC/USDT

Exit codes: 0 session completed (promoting nothing is a success), 1 fatal error,
2 another session already holds the lock.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from algotrader.agent.nightly import (  # noqa: E402
    Budget,
    LockBusy,
    NightlySession,
    SessionLock,
    SymbolSpec,
    budget_from_config,
    specs_from_config,
)
from algotrader.config import get, load_config, load_dotenv  # noqa: E402


def main() -> int:
    load_dotenv()
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", action="append", dest="symbols",
                    help="override configured symbols (repeatable)")
    ap.add_argument("--source", default=get(cfg, "data.default_source"))
    ap.add_argument("--timeframe", default=get(cfg, "data.default_timeframe"))
    ap.add_argument("--hours", type=float, default=None, help="wall-clock budget")
    ap.add_argument("--max-iterations", type=int, default=None)
    ap.add_argument("--reflect-every", type=int, default=None)
    ap.add_argument("--no-sandbox", action="store_true",
                    help=argparse.SUPPRESS)
    ap.add_argument("--allow-unsandboxed", action="store_true",
                    help="run model-written code in-process (NOT for unattended use)")
    ap.add_argument("--log-dir", default=get(cfg, "nightly.log_dir", "experiments/logs"))
    args = ap.parse_args()

    if args.no_sandbox or args.allow_unsandboxed:
        cfg["agent"]["use_sandbox"] = False
    else:
        cfg["agent"]["use_sandbox"] = True

    if args.symbols:
        specs = [SymbolSpec(s, args.source, args.timeframe) for s in args.symbols]
    else:
        specs = specs_from_config(cfg)
    if not specs:
        print("No symbols: set nightly.symbols in config or pass --symbol.", file=sys.stderr)
        return 1

    budget = budget_from_config(cfg)
    budget = Budget(
        max_hours=args.hours if args.hours is not None else budget.max_hours,
        max_iterations=(args.max_iterations if args.max_iterations is not None
                        else budget.max_iterations),
        reflect_every=(args.reflect_every if args.reflect_every is not None
                       else budget.reflect_every),
    )

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"nightly-{datetime.now().astimezone():%Y-%m-%d}.log"

    try:
        with SessionLock(get(cfg, "nightly.lock_file", "experiments/nightly.lock")):
            with open(log_path, "a") as fh:
                fh.write(f"\n\n===== launched {datetime.now().astimezone().isoformat()} =====\n")
                session = NightlySession(
                    cfg, specs, budget=budget,
                    allow_unsandboxed=args.allow_unsandboxed or args.no_sandbox,
                    log_file=fh,
                )
                result = session.run()
    except LockBusy as e:
        print(f"skipping: {e}", file=sys.stderr)
        return 2
    except Exception as e:  # noqa: BLE001
        print(f"fatal: {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    print(f"\nlog: {log_path}")
    if result.skipped_symbols:
        print(f"skipped (no data): {', '.join(result.skipped_symbols)}")
    return 0 if result.stop_reason != "fatal" else 1


if __name__ == "__main__":
    sys.exit(main())
