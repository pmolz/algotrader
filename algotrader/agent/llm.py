"""Thin, provider-agnostic LLM client.

Supported providers (chosen via config `agent.provider`):
  * "ollama"    -> local models (Qwen, Llama, etc.) via the Ollama HTTP API. FREE, private.
  * "anthropic" -> Claude via the anthropic SDK.

Every client exposes the same tiny surface: `.available()` and
`.complete(system, user)`. The agent loop doesn't care which one it's talking to.
If none is available it returns False from available() and the loop falls back to
its offline baseline-mutation mode.

Ollama endpoints
----------------
An Ollama model can live on this box or on another machine on the LAN, and the
useful setup is usually both: a desktop GPU that is sometimes busy, and a laptop
that is always on. `agent.hosts` names the endpoints, `agent.host` picks one:

    agent:
      host: auto                  # or a name from `hosts`, or a bare URL
      host_preference: [laptop, local]
      hosts:
        local:  http://127.0.0.1:11434
        laptop: {url: http://192.168.1.42:11434, model: qwen2.5-coder:7b}

"auto" walks `host_preference` and takes the first endpoint that answers *and*
has the model pulled — a reachable server missing the model is no use to us.
Naming an endpoint pins it: if a pinned endpoint is down the loop drops to its
offline fallback rather than quietly running somewhere else, because "which
machine generated this strategy" is part of the experiment record.

For strategy CODE GENERATION, prefer a coder-tuned model, e.g.:
    ollama pull qwen2.5-coder:7b      # good default
    ollama pull qwen2.5-coder:14b     # better, if you have the VRAM/RAM
Small general models (<=3B) tend to produce broken code — expect lots of retries.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass

from ..config import get

DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"

# Generation can be slow on a partially-offloaded model or over the LAN; a 14B
# spilling to CPU emits single-digit tokens/sec. Better to wait than to lose the
# iteration to a timeout.
DEFAULT_REQUEST_TIMEOUT = 900
DEFAULT_PROBE_TIMEOUT = 3


def normalize_url(url: str) -> str:
    """Accept `host:port` as well as a full URL — $OLLAMA_HOST is conventionally
    written without a scheme, and typing one into config is easy to forget."""
    url = (url or "").strip().rstrip("/")
    if not url:
        return DEFAULT_OLLAMA_URL
    if "://" not in url:
        url = f"http://{url}"
    return url


@dataclass(frozen=True)
class OllamaEndpoint:
    """One Ollama server. `model` overrides `agent.model` for this endpoint —
    a 4GB laptop GPU may only fit a smaller model than the desktop runs."""

    name: str
    url: str
    model: str

    def describe(self) -> str:
        return f"{self.name} ({self.url}, {self.model})"


def parse_endpoints(cfg: dict) -> list[OllamaEndpoint]:
    """Build the endpoint list from `agent.hosts`, ordered by
    `agent.host_preference` (names not present in `hosts` are skipped, so the
    shipped preference list can mention a laptop you haven't configured yet).
    """
    default_model = get(cfg, "agent.model", "qwen2.5-coder:7b")
    raw = get(cfg, "agent.hosts") or {}
    if not isinstance(raw, dict):
        raise TypeError("agent.hosts must be a mapping of name -> url|{url, model}")

    if not raw:
        raw = {"local": DEFAULT_OLLAMA_URL}

    endpoints: dict[str, OllamaEndpoint] = {}
    for name, entry in raw.items():
        if isinstance(entry, str):
            url, model = entry, default_model
        elif isinstance(entry, dict):
            url = entry.get("url") or DEFAULT_OLLAMA_URL
            model = entry.get("model") or default_model
        else:
            raise TypeError(f"agent.hosts.{name} must be a url or a mapping")
        endpoints[name] = OllamaEndpoint(name, normalize_url(url), model)

    preference = get(cfg, "agent.host_preference") or list(endpoints)
    ordered = [endpoints[n] for n in preference if n in endpoints]
    # Anything configured but left out of the preference list still stays
    # reachable by name; it just isn't a candidate for "auto".
    ordered += [e for n, e in endpoints.items() if n not in preference]
    return ordered


def _looks_like_address(choice: str) -> bool:
    """A URL, a host:port, or a dotted/IP hostname — as opposed to a bare name
    that was meant to match `agent.hosts`."""
    return "://" in choice or ":" in choice or "." in choice


def select_candidates(
    cfg: dict, endpoints: list[OllamaEndpoint], override: str | None = None
) -> tuple[list[OllamaEndpoint], str]:
    """Return (candidates, policy_label) for the configured selection policy.

    Precedence: explicit override (CLI) > $OLLAMA_HOST > `agent.host` > auto.
    A pinned choice yields exactly one candidate, so there is no silent
    failover away from the machine you asked for.
    """
    default_model = get(cfg, "agent.model", "qwen2.5-coder:7b")
    choice = override or os.environ.get("OLLAMA_HOST") or get(cfg, "agent.host", "auto")
    choice = (choice or "auto").strip()

    if choice.lower() == "auto":
        return list(endpoints), "auto"

    by_name = {e.name: e for e in endpoints}
    if choice in by_name:
        return [by_name[choice]], f"pinned:{choice}"

    # Not a known name. An address is fine — $OLLAMA_HOST and --ollama-host
    # should accept a machine that isn't in config. A bare word is not: it is
    # a typo or an unconfigured host, and inventing http://<word>:11434 would
    # hand back an endpoint that can never answer.
    if not _looks_like_address(choice):
        known = ", ".join(e.name for e in endpoints) or "(none configured)"
        raise ValueError(
            f"unknown Ollama host {choice!r}. Configured hosts: {known}. "
            f"Pass a name from agent.hosts, a URL, or 'auto'."
        )
    url = normalize_url(choice)
    match = next((e for e in endpoints if e.url == url), None)
    if match is not None:
        return [match], f"pinned:{match.name}"
    return [OllamaEndpoint("adhoc", url, default_model)], "pinned:adhoc"


class OllamaClient:
    """Talks to one Ollama server."""

    def __init__(self, model: str = "qwen2.5-coder:7b", host: str | None = None,
                 timeout: int = DEFAULT_REQUEST_TIMEOUT, think: bool = False,
                 probe_timeout: int = DEFAULT_PROBE_TIMEOUT, name: str = "ollama"):
        self.model = model
        self.host = normalize_url(host or os.environ.get("OLLAMA_HOST") or DEFAULT_OLLAMA_URL)
        self.timeout = timeout
        self.probe_timeout = probe_timeout
        self.name = name
        # "thinking" models (e.g. qwen3.x) burn the token budget reasoning before
        # emitting the answer. We disable it for code-gen so `response` is non-empty.
        self.think = think

    def describe(self) -> str:
        return f"{self.name} ({self.host}, {self.model})"

    def available(self) -> bool:
        try:
            with urllib.request.urlopen(
                f"{self.host}/api/tags", timeout=self.probe_timeout
            ) as r:
                tags = json.loads(r.read())
        except Exception:
            return False
        names = {m.get("name", "") for m in tags.get("models", [])}
        # match with or without an explicit :tag
        base = self.model.split(":")[0]
        return any(n == self.model or n.split(":")[0] == base for n in names)

    def complete(self, system: str, user: str, max_tokens: int = 2048) -> str:
        payload = {
            "model": self.model,
            "system": system,
            "prompt": user,
            "stream": False,
            "think": self.think,
            "options": {"temperature": 0.7, "num_predict": max_tokens},
        }
        req = urllib.request.Request(
            f"{self.host}/api/generate",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            data = json.loads(r.read())
        return data.get("response", "")


class OllamaPool:
    """Selects among configured endpoints and survives one going away.

    Presents the same `available()` / `complete()` surface as a single client.
    An unattended session runs for hours; the laptop serving it can reboot or
    drop off the wifi, and re-selecting on the next iteration is the difference
    between losing one candidate and losing the night. Under a pinned policy
    there is only ever one candidate, so pinning still means pinning.
    """

    def __init__(self, candidates: list[OllamaClient], policy: str = "auto",
                 logger: Callable[[str], None] | None = None):
        self.candidates = candidates
        self.policy = policy
        self.logger = logger
        self.current: OllamaClient | None = None

    # -- plumbing ---------------------------------------------------------------
    def _log(self, msg: str) -> None:
        if self.logger:
            self.logger(msg)

    def describe(self) -> str:
        if self.current is not None:
            return self.current.describe()
        names = ", ".join(c.name for c in self.candidates) or "none"
        return f"unresolved [{self.policy}] among: {names}"

    @property
    def model(self) -> str | None:
        return self.current.model if self.current else None

    def _select(self) -> OllamaClient | None:
        for client in self.candidates:
            if client.available():
                return client
        return None

    # -- the client surface ------------------------------------------------------
    def available(self) -> bool:
        """True if some endpoint is usable. Re-probes when the current pick has
        gone away, so a host that comes back mid-session is picked up again."""
        if self.current is not None and self.current.available():
            return True

        previous = self.current
        self.current = self._select()
        if self.current is None:
            if previous is not None:
                self._log(f"llm: {previous.describe()} went away; no endpoint available")
            return False
        if previous is None:
            self._log(f"llm: using {self.current.describe()} [{self.policy}]")
        elif previous.host != self.current.host:
            self._log(f"llm: {previous.name} unreachable — failed over to "
                      f"{self.current.describe()}")
        return True

    def complete(self, system: str, user: str, max_tokens: int = 2048) -> str:
        if self.current is None and not self.available():
            raise RuntimeError("no Ollama endpoint available")
        try:
            return self.current.complete(system, user, max_tokens=max_tokens)
        except (urllib.error.URLError, OSError) as e:
            # Connection-level failure: the box may have gone away mid-generation.
            # Re-select once and retry. A model error (bad JSON, HTTP 4xx) is not
            # retried here — that is the caller's fix-retry path, not ours.
            self._log(f"llm: {self.current.name} failed ({type(e).__name__}: {e}); re-selecting")
            self.current = None
            if not self.available():
                raise
            return self.current.complete(system, user, max_tokens=max_tokens)


class AnthropicClient:
    def __init__(self, model: str = "claude-sonnet-5"):
        self.model = model
        self.name = "anthropic"
        self._client = None

    def describe(self) -> str:
        return f"anthropic ({self.model})"

    def available(self) -> bool:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            return False
        try:
            import anthropic  # noqa: F401
        except ImportError:
            return False
        return True

    def _ensure(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic()
        return self._client

    def complete(self, system: str, user: str, max_tokens: int = 2048) -> str:
        client = self._ensure()
        resp = client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(block.text for block in resp.content if block.type == "text")


def make_client(cfg: dict, host: str | None = None,
                logger: Callable[[str], None] | None = None):
    """Factory used by the agent loop. Unknown provider -> None (offline mode).

    `host` overrides `agent.host` for this run (the --ollama-host flag).
    """
    provider = (get(cfg, "agent.provider") or "").lower()
    if provider == "ollama":
        endpoints = parse_endpoints(cfg)
        candidates, policy = select_candidates(cfg, endpoints, override=host)
        timeout = int(get(cfg, "agent.request_timeout_s", DEFAULT_REQUEST_TIMEOUT))
        probe_timeout = int(get(cfg, "agent.probe_timeout_s", DEFAULT_PROBE_TIMEOUT))
        clients = [
            OllamaClient(e.model, host=e.url, timeout=timeout,
                         probe_timeout=probe_timeout, name=e.name)
            for e in candidates
        ]
        return OllamaPool(clients, policy=policy, logger=logger)
    if provider == "anthropic":
        return AnthropicClient(get(cfg, "agent.model"))
    return None
