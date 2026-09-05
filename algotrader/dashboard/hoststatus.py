"""Liveness and GPU telemetry for the machines that serve the model.

The dashboard runs on one box; the model may be on another. Two things are worth
seeing at a glance, and neither is visible from the experiment log:

  * Is the endpoint up, and is the model actually resident? Under `host: auto` a
    laptop that quietly drops off the network doesn't fail anything — the run
    moves to the fallback machine and the night looks completely normal. The
    only symptom is a status dot that went grey.
  * What is the GPU doing? An overnight session on a laptop GPU is a thermal
    question as much as a compute one, and "why is it slow tonight" is usually
    answered by a temperature.

Liveness comes from Ollama's own HTTP API. GPU numbers come from `nvidia-smi`,
run locally or over SSH depending on `agent.hosts.<name>.gpu`:

    hosts:
      local:  {url: "http://127.0.0.1:11434", gpu: local}
      laptop: {url: "http://10.0.0.249:11434", gpu: "ssh:dell"}

SSH targets are read from config, never from the request, and every call is
BatchMode with a hard timeout so a wedged host stalls one card rather than the
page. Results are cached briefly: the page polls, and neither an HTTP probe nor
an SSH round-trip should run once per client per second.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

from ..agent.llm import parse_endpoints
from ..config import get

_NVIDIA_QUERY = (
    "name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw"
)
_CACHE_TTL_S = 5.0
_HTTP_TIMEOUT_S = 2.0
_SSH_TIMEOUT_S = 6.0

_cache: dict = {"at": 0.0, "value": None}
_lock = Lock()


# -- probes ------------------------------------------------------------------
def _get_json(url: str, timeout: float = _HTTP_TIMEOUT_S):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def probe_endpoint(url: str) -> dict:
    """Reachability plus what Ollama currently holds in memory."""
    started = time.perf_counter()
    try:
        tags = _get_json(f"{url}/api/tags")
    except (urllib.error.URLError, OSError, ValueError) as e:
        return {"up": False, "error": f"{type(e).__name__}: {e}", "models": [],
                "resident": None, "latency_ms": None}

    latency_ms = round((time.perf_counter() - started) * 1000)
    out = {"up": True, "error": None, "latency_ms": latency_ms,
           "models": sorted(m.get("name", "") for m in tags.get("models", [])),
           "resident": None}

    # /api/ps is the loaded-into-VRAM set. With OLLAMA_KEEP_ALIVE set for an
    # overnight run this should stay populated; an empty list means every
    # iteration is re-paying the model load.
    try:
        ps = _get_json(f"{url}/api/ps")
        loaded = ps.get("models") or []
        if loaded:
            m = loaded[0]
            out["resident"] = {
                "name": m.get("name"),
                "vram_mb": round((m.get("size_vram") or 0) / 1024 / 1024),
            }
    except Exception:  # noqa: BLE001 - older Ollama builds lack /api/ps
        pass
    return out


def _run_nvidia_smi(gpu: str) -> tuple[str | None, str | None]:
    """Return (stdout, error). `gpu` is 'local' or 'ssh:<target>'."""
    args = ["nvidia-smi", f"--query-gpu={_NVIDIA_QUERY}",
            "--format=csv,noheader,nounits"]
    if gpu == "local":
        if not shutil.which("nvidia-smi"):
            return None, "nvidia-smi not installed"
        cmd, timeout = args, _HTTP_TIMEOUT_S + 2
    elif gpu.startswith("ssh:"):
        target = gpu[4:].strip()
        if not target:
            return None, "empty ssh target"
        # BatchMode: never prompt. A host that needs a password must fail fast
        # rather than hang the request waiting on a tty that isn't there.
        cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=3",
               "-o", "StrictHostKeyChecking=accept-new", target, *args]
        timeout = _SSH_TIMEOUT_S
    else:
        return None, f"unknown gpu source {gpu!r}"

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None, "timed out"
    except FileNotFoundError:
        return None, f"{cmd[0]} not found"
    if proc.returncode != 0:
        return None, (proc.stderr or "").strip().splitlines()[-1:][0] if proc.stderr else "failed"
    return proc.stdout, None


def gpu_stats(gpu: str | None) -> dict | None:
    """Parse the first GPU's line. None when this host reports no GPU source."""
    if not gpu:
        return None
    out, err = _run_nvidia_smi(gpu)
    if err or not out:
        return {"error": err or "no output"}

    line = out.strip().splitlines()[0]
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 6:
        return {"error": f"unparsed: {line[:80]}"}

    def num(v):
        try:
            return float(v)
        except ValueError:      # "[N/A]" on GPUs that don't report power
            return None

    used, total = num(parts[2]), num(parts[3])
    return {
        "error": None,
        "name": parts[0],
        "util_pct": num(parts[1]),
        "mem_used_mb": used,
        "mem_total_mb": total,
        "mem_pct": round(used / total * 100, 1) if used is not None and total else None,
        "temp_c": num(parts[4]),
        "power_w": num(parts[5]),
    }


# -- assembly ----------------------------------------------------------------
def _status_for(endpoint) -> dict:
    ep = probe_endpoint(endpoint.url)
    has_model = any(
        m == endpoint.model or m.split(":")[0] == endpoint.model.split(":")[0]
        for m in ep["models"]
    )
    return {
        "name": endpoint.name,
        "url": endpoint.url,
        "model": endpoint.model,
        "up": ep["up"],
        # Up but missing the model is its own state: the box is fine and still
        # cannot serve this agent, which reads very differently from "offline".
        "has_model": has_model,
        "state": "up" if ep["up"] and has_model else ("no-model" if ep["up"] else "down"),
        "latency_ms": ep["latency_ms"],
        "resident": ep["resident"],
        "error": ep["error"],
        "gpu": gpu_stats(getattr(endpoint, "gpu", None)) if ep["up"] else None,
    }


def collect(cfg: dict, force: bool = False) -> dict:
    """All configured hosts, with which one a run would currently pick."""
    with _lock:
        fresh = time.time() - _cache["at"] < _CACHE_TTL_S
        if fresh and not force and _cache["value"] is not None:
            return _cache["value"]

    try:
        endpoints = parse_endpoints(cfg)
    except (TypeError, ValueError) as e:
        return {"hosts": [], "error": str(e), "selected": None, "checked_at": time.time()}

    # Probe in parallel: a down host costs its own timeout, not everyone else's.
    with ThreadPoolExecutor(max_workers=max(1, len(endpoints))) as pool:
        hosts = list(pool.map(_status_for, endpoints))

    policy = str(get(cfg, "agent.host", "auto") or "auto")
    if policy.lower() == "auto":
        selected = next((h["name"] for h in hosts if h["state"] == "up"), None)
    else:
        selected = next(
            (h["name"] for h in hosts if h["name"] == policy and h["state"] == "up"), None
        )

    value = {"hosts": hosts, "selected": selected, "policy": policy,
             "error": None, "checked_at": time.time()}
    with _lock:
        _cache.update(at=time.time(), value=value)
    return value
