#!/usr/bin/env python3
"""Re-run the lookahead detector over strategies already in the experiment log.

Why this exists: the original detector was truncation-only, which is close to
blind against `.shift(-k)` peeking when the signal is sparse (measured: 4%
detection on a strategy acting on ~1% of bars). Every `PASS lookahead` verdict
recorded before the fix was produced by that detector, and those verdicts feed
`lessons_context` — so a cheat that slipped through isn't just a bad row, it is
bad advice being handed to every future proposal.

    python scripts/recheck_lookahead.py              # report only
    python scripts/recheck_lookahead.py --limit 10   # quick sample
    python scripts/recheck_lookahead.py --annotate    # write findings back

Generated code is untrusted, so it runs in the same Docker jail the nightly loop
uses. There is no unsandboxed path here on purpose; use --allow-unsandboxed only
on code you have read.

Exit codes: 0 nothing newly failed, 1 at least one strategy now fails, 2 setup
problem (no Docker image, no cached data).
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from algotrader.agent.memory import ExperimentDB  # noqa: E402
from algotrader.agent.sandbox import (  # noqa: E402
    SandboxLimits,
    docker_available,
    image_exists,
    run_gauntlet_sandboxed,
)
from algotrader.config import get, load_config, load_dotenv  # noqa: E402
from algotrader.data.loader import load_cached  # noqa: E402


def main() -> int:
    load_dotenv()
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="only re-check the N most recent candidates")
    ap.add_argument("--annotate", action="store_true",
                    help="append the finding to each affected experiment's reasons")
    ap.add_argument("--allow-unsandboxed", action="store_true",
                    help="exec stored code in-process (only for code you have read)")
    args = ap.parse_args()

    db = ExperimentDB(get(cfg, "experiments.db_path"))
    rows = [
        r for r in db.conn.execute(
            "SELECT id, symbol, source, timeframe, strategy_name, code, reasons, promoted "
            "FROM experiments WHERE code IS NOT NULL "
            "AND reasons LIKE '%PASS lookahead%' ORDER BY id DESC"
        )
    ]
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        print("No stored candidates carry a PASS lookahead verdict — nothing to re-check.")
        return 0

    limits = SandboxLimits(
        image=get(cfg, "sandbox.image", "algotrader-sandbox:latest"),
        memory=get(cfg, "sandbox.memory", "1g"),
        cpus=str(get(cfg, "sandbox.cpus", "2")),
        pids=int(get(cfg, "sandbox.pids", 128)),
        timeout_s=int(get(cfg, "sandbox.timeout_s", 120)),
    )
    if not args.allow_unsandboxed:
        if not docker_available():
            print("Docker is not available. Start it, or pass --allow-unsandboxed "
                  "if you have read the stored code.", file=sys.stderr)
            return 2
        if not image_exists(limits.image):
            print(f"Sandbox image {limits.image!r} missing — run: bash sandbox/build.sh",
                  file=sys.stderr)
            return 2

    cache_dir = get(cfg, "data.cache_dir", "data_cache")
    data_cache: dict[tuple, object] = {}
    print(f"Re-checking {len(rows)} candidates that previously passed.\n")

    now_failing, errors = [], []
    outcomes: Counter = Counter()

    for row in rows:
        key = (row["source"], row["symbol"], row["timeframe"])
        if key not in data_cache:
            data_cache[key] = load_cached(cache_dir, *key)
        df = data_cache[key]
        if df is None or len(df) == 0:
            outcomes["no data"] += 1
            errors.append((row["id"], f"no cached data for {row['symbol']}"))
            continue

        try:
            if args.allow_unsandboxed:
                from algotrader.agent.codegen import load_strategy_class
                from algotrader.validation.lookahead import lookahead_check
                la = lookahead_check(df, load_strategy_class(row["code"])())
            else:
                rep = run_gauntlet_sandboxed(df, row["code"], cfg, n_trials=1, limits=limits)
                la = (rep.get("checks") or {}).get("lookahead")
                if la is None:
                    raise RuntimeError(rep.get("error") or "no lookahead check in report")
        except Exception as e:  # noqa: BLE001
            outcomes["error"] += 1
            errors.append((row["id"], f"{type(e).__name__}: {e}"))
            continue

        if la.get("passed"):
            outcomes["still passes"] += 1
            continue

        outcomes["NOW FAILS"] += 1
        probes = sorted({v.get("probe", "?") for v in la.get("violations", [])})
        now_failing.append((row["id"], row["strategy_name"], probes, la))
        flag = " [WAS PROMOTED]" if row["promoted"] else ""
        print(f"  id={row['id']:<5} {row['strategy_name'][:34]:<34} "
              f"NOW FAILS via {', '.join(probes)}{flag}")

        if args.annotate:
            reasons = json.loads(row["reasons"] or "[]")
            note = ("RECHECK: FAIL lookahead under the perturbation detector "
                    f"(probes: {', '.join(probes)})")
            if note not in reasons:
                reasons.append(note)
                db.conn.execute("UPDATE experiments SET reasons=? WHERE id=?",
                                (json.dumps(reasons), row["id"]))

    if args.annotate:
        db.conn.commit()

    print("\n" + "=" * 62)
    for k, v in outcomes.most_common():
        print(f"  {k:<14} {v}")
    if errors:
        print(f"\n{len(errors)} could not be re-checked:")
        for exp_id, msg in errors[:10]:
            print(f"  id={exp_id}: {msg[:90]}")

    if now_failing:
        print(f"\n{len(now_failing)} strategies passed the old check and fail the new one.")
        print("These were recorded as lookahead-free and were not. Their results, and any")
        print("reflection written off them, should be treated as noise.")
        if args.annotate:
            print("\nAnnotated in the experiment log.")
        else:
            print("\nRe-run with --annotate to record this in the log.")
        return 1

    print("\nNo strategy that passed the old check fails the new one.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
