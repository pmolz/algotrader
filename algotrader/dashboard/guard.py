"""Request guards for the endpoints that have side effects.

Until now the dashboard only read. Starting and stopping sessions changes that,
and "it's only on localhost" is not by itself a defence:

  * **CSRF.** Any page you visit in the same browser can POST to
    127.0.0.1:8765 with an HTML form — no JavaScript needed, and the browser
    sends it happily. A plain form cannot set a custom header, and a custom
    header on a cross-origin `fetch` triggers a CORS preflight that this app
    never approves. So requiring one is a real barrier, and we check `Origin`
    too, which browsers always attach to cross-origin POSTs.

  * **DNS rebinding.** An attacker's domain can resolve to 127.0.0.1, making
    their page same-origin with this app. Pinning the `Host` header to a
    loopback name defeats that, because the browser sends the attacker's
    hostname, not `localhost`.

Neither guard applies to the read-only pages; they are only wired to the
mutating endpoints, where the cost of being wrong is a session starting without
you asking.
"""

from __future__ import annotations

from functools import wraps
from urllib.parse import urlparse

from flask import jsonify, request

# Hostnames a browser can legitimately use to reach a loopback bind.
ALLOWED_HOSTS = {"127.0.0.1", "localhost", "[::1]", "::1"}
# A cross-origin fetch cannot set this without triggering a preflight we deny.
REQUIRED_HEADER = "X-Algotrader"


def _host_ok() -> bool:
    host = (request.host or "").split(":")[0].strip("[]")
    return host in {h.strip("[]") for h in ALLOWED_HOSTS}


def _origin_ok() -> bool:
    """Same-origin only. A missing Origin is allowed for non-browser clients
    (curl, tests) — those aren't the CSRF threat, since an attacker's leverage
    is precisely the victim's browser, which always sends it on a POST."""
    origin = request.headers.get("Origin")
    if origin is None:
        return True
    parsed = urlparse(origin)
    return parsed.hostname in {h.strip("[]") for h in ALLOWED_HOSTS}


def local_only(fn):
    """Guard a side-effecting endpoint."""

    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not _host_ok():
            return jsonify({"error": "This endpoint is loopback-only."}), 403
        if not _origin_ok():
            return jsonify({"error": "Cross-origin requests are not accepted."}), 403
        if request.headers.get(REQUIRED_HEADER) != "1":
            return jsonify(
                {"error": f"Missing {REQUIRED_HEADER} header — refusing to act."}
            ), 403
        return fn(*args, **kwargs)

    return wrapper
