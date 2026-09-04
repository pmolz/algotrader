"""Thin LLM client wrapper. Anthropic by default; swap freely.

Kept deliberately minimal so the agent loop doesn't care which provider you use.
If no API key / SDK is available, `AnthropicClient.available()` returns False and
the loop falls back to mutating baseline strategies (so you can test the plumbing
offline).
"""

from __future__ import annotations

import os


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

    def complete(self, system: str, user: str, max_tokens: int = 2000) -> str:
        client = self._ensure()
        resp = client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(block.text for block in resp.content if block.type == "text")
