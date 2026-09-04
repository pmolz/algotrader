#!/usr/bin/env python3
"""Generate the morning report from the experiment DB.

Reads nothing but the DB, so any past window can be re-rendered at any time.

Examples:
    python scripts/morning_report.py                 # last night -> now
    python scripts/morning_report.py --days 7        # the past week
    python scripts/morning_report.py --run 12        # one specific session
    python scripts/morning_report.py --stdout        # don't write a file
"""

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from algotrader.agent.memory import ExperimentDB  # noqa: E402
from algotrader.agent.report import (  # noqa: E402
    build_report,
    overnight_window,
    write_report,
)
from algotrader.config import get, load_config  # noqa: E402


def main() -> int:
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=float, default=None,
                    help="window = the last N days instead of last night")
    ap.add_argument("--run", type=int, default=None,
                    help="window = the span of a single run id")
    ap.add_argument("--out-dir", default=get(cfg, "nightly.report_dir", "experiments/reports"))
    ap.add_argument("--stdout", action="store_true", help="print only, write no file")
    args = ap.parse_args()

    db = ExperimentDB(get(cfg, "experiments.db_path"))

    if args.run is not None:
        row = db.run(args.run)
        if row is None:
            print(f"no run with id {args.run}", file=sys.stderr)
            return 1
        # a run with no finish time (killed) is bounded by now
        now = datetime.now().astimezone()
        start = row["started_at"] - 1
        end = (row["finished_at"] or now.timestamp()) + 1
    elif args.days is not None:
        now = datetime.now().astimezone()
        end = now.timestamp()
        start = (now - timedelta(days=args.days)).timestamp()
    else:
        start, end = overnight_window(
            start_hour=int(get(cfg, "nightly.report_window_start_hour", 18))
        )

    text = build_report(
        db, start, end, cfg=cfg,
        generated_dir=get(cfg, "experiments.generated_code_dir", "experiments/generated"),
    )
    print(text)
    if not args.stdout:
        path = write_report(text, args.out_dir,
                            when=datetime.fromtimestamp(end).astimezone())
        print(f"written: {path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
