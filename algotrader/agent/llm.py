"""Thin, provider-agnostic LLM client.

Supported providers (chosen via config `agent.provider`):
  * "ollama"    -> local models (Qwen, Llama, etc.) via the Ollama HTTP API. FREE, private.
  * "anthropic" -> Claude via the anthropic SDK.

Every client exposes the same tiny surface: `.available()` and
`.complete(system, user)`. The agent loop doesn't care which one it's talking to.
If none is available it returns False from available() and the loop falls back to
its offline baseline-mutation mode.

For strategy CODE GENERATION, prefer a coder-tuned model, e.g.:
    ollama pull qwen2.5-coder:7b      # good default
    ollama pull qwen2.5-coder:14b     # better, if you have the VRAM/RAM
Small general models (<=3B) tend to produce broken code — expect lots of retries.
"""

from __future__ import annotations

import json
import os
import urllib.request


class OllamaClient:
    """Talks to a local Ollama server (default http://localhost:11434)."""

    def __init__(self, model: str = "qwen2.5-coder:7b", host: str | None = None,
                 timeout: int = 300, think: bool = False):
        self.model = model
        self.host = host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self.timeout = timeout
        # "thinking" models (e.g. qwen3.x) burn the token budget reasoning before
        # emitting the answer. We disable it for code-gen so `response` is non-empty.
        self.think = think

    def available(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.host}/api/tags", timeout=5) as r:
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


class AnthropicClient:
    def __init__(self, model: str = "claude-sonnet-4-5-20250929"):
        self.model = model
        self._client = None

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


def make_client(provider: str, model: str):
    """Factory used by the agent loop. Unknown provider -> None (offline mode)."""
    provider = (provider or "").lower()
    if provider == "ollama":
        return OllamaClient(model)
    if provider == "anthropic":
        return AnthropicClient(model)
    return None
