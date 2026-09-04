#!/usr/bin/env python3
"""Run the self-improvement agent loop.

Examples:
    python scripts/run_agent.py --symbol BTC/USDT --iterations 5
    # Without ANTHROPIC_API_KEY it runs an offline baseline-mutation fallback so
    # you can verify the whole pipeline before spending on LLM calls.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from algotrader.agent import AgentLoop  # noqa: E402
from algotrader.config import get, load_config, load_dotenv  # noqa: E402
from algotrader.data import fetch  # noqa: E402


def main():
    load_dotenv()
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--source", default=get(cfg, "data.default_source"))
    ap.add_argument("--timeframe", default=get(cfg, "data.default_timeframe"))
    ap.add_argument("--iterations", type=int, default=get(cfg, "agent.max_iterations", 5))
    ap.add_argument("--sandbox", action="store_true",
                    help="run generated code inside the Docker jail (recommended)")
    ap.add_argument("--no-sandbox", action="store_true",
                    help="force in-process exec (supervised local use only)")
    args = ap.parse_args()

    if args.sandbox:
        cfg["agent"]["use_sandbox"] = True
    if args.no_sandbox:
        cfg["agent"]["use_sandbox"] = False

    df = fetch(args.symbol, source=args.source, timeframe=args.timeframe,
               cache_dir=get(cfg, "data.cache_dir"))

    loop = AgentLoop(df, cfg, symbol=args.symbol, source=args.source,
                     timeframe=args.timeframe)
    results = loop.run(args.iterations)

    promoted = [r for r in results if r.get("promoted")]
    print(f"\n=== Done. {len(promoted)}/{len(results)} candidates promoted. ===")
    for r in promoted:
        print(f"  id={r['id']} {r.get('strategy')} :: {r.get('hypothesis')}")


if __name__ == "__main__":
    main()
