#!/usr/bin/env python3
"""Serve the local dashboard.

    python scripts/dashboard.py                 # http://127.0.0.1:8765
    python scripts/dashboard.py --port 9000

Binds to loopback only. There is no authentication because there is no network
exposure — if you change --host, put a reverse proxy with auth in front of it.
The dashboard opens the experiment DB read-only; the only thing it can execute is
an equity-curve re-run inside the Docker jail.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from algotrader.config import get, load_config, load_dotenv  # noqa: E402
from algotrader.dashboard import create_app  # noqa: E402


def main() -> int:
    load_dotenv()
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=get(cfg, "dashboard.host", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=get(cfg, "dashboard.port", 8765))
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    if args.host not in ("127.0.0.1", "localhost", "::1"):
        print(
            f"WARNING: binding to {args.host} exposes an unauthenticated dashboard "
            "on the network.",
            file=sys.stderr,
        )

    app = create_app(cfg)
    print(f"algotrader dashboard → http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
