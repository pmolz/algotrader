#!/usr/bin/env python3
"""Show the configured LLM endpoints and which one a run would use.

Run this before trusting a night to a machine you can't see:

    python scripts/check_llm.py
    python scripts/check_llm.py --ollama-host laptop

Exit codes: 0 an endpoint is usable, 1 nothing usable (a session would fall
back to offline baseline mutation, which is a plumbing test, not research).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from algotrader.agent.llm import (  # noqa: E402
    OllamaClient,
    parse_endpoints,
    select_candidates,
)
from algotrader.config import get, load_config, load_dotenv  # noqa: E402


def main() -> int:
    load_dotenv()
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--ollama-host", default=None, metavar="NAME|URL",
                    help="override agent.host for this check")
    args = ap.parse_args()

    provider = (get(cfg, "agent.provider") or "").lower()
    print(f"provider: {provider}")
    if provider != "ollama":
        print("(endpoint selection only applies to the ollama provider)")
        return 0

    endpoints = parse_endpoints(cfg)
    try:
        candidates, policy = select_candidates(cfg, endpoints, override=args.ollama_host)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    chosen = {e.url for e in candidates}
    probe_timeout = int(get(cfg, "agent.probe_timeout_s", 3))

    print(f"policy:   {policy}\n")
    rows = []
    for e in endpoints:
        client = OllamaClient(e.model, host=e.url, probe_timeout=probe_timeout,
                              name=e.name)
        # available() is reachable AND has the model — a server missing the
        # model is no more use to us than one that's off.
        ok = client.available()
        rows.append((e, ok, e.url in chosen))

    width = max((len(e.name) for e, _, _ in rows), default=4)
    for e, ok, considered in rows:
        mark = "OK  " if ok else "DOWN"
        note = "" if considered else "   (not a candidate under this policy)"
        print(f"  [{mark}] {e.name:<{width}}  {e.url}  {e.model}{note}")

    live = next((e for e, ok, considered in rows if ok and considered), None)
    print()
    if live is None:
        print("=> no usable endpoint; a run would use offline baseline mutation")
        if policy.startswith("pinned:"):
            print("   (host is pinned — it will not fall back to another machine)")
        return 1
    print(f"=> a run would use: {live.describe()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
