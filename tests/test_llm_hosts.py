"""Endpoint selection: which machine serves the model, and what happens when
it goes away mid-session.

The rules being pinned down here:
  * "auto" takes the first *reachable* endpoint in preference order, and a
    server that answers but lacks the model does not count as reachable.
  * Naming a host pins it. A pinned host that is down means offline fallback,
    never a silent move to a different machine — the run row claims which box
    generated the strategies, so that claim has to be true.
  * A host that dies mid-session is failed over on the next iteration, because
    losing one candidate is much cheaper than losing eight hours.
"""

from __future__ import annotations

import pytest

from algotrader.agent.llm import (
    OllamaPool,
    make_client,
    normalize_url,
    parse_endpoints,
    select_candidates,
)


def cfg(**agent) -> dict:
    base = {
        "provider": "ollama",
        "model": "qwen2.5-coder:7b",
        "host": "auto",
        "host_preference": ["laptop", "local"],
        "hosts": {
            "local": "http://127.0.0.1:11434",
            "laptop": {"url": "http://dell.lan:11434", "model": "qwen2.5-coder:3b"},
        },
    }
    base.update(agent)
    return {"agent": base}


class FakeClient:
    """Stands in for OllamaClient. `up` is flipped by tests to simulate a box
    going away between iterations."""

    def __init__(self, name, up=True, model="m"):
        self.name = name
        self.host = f"http://{name}:11434"
        self.model = model
        self.up = up
        self.calls = 0

    def describe(self):
        return f"{self.name} ({self.host}, {self.model})"

    def available(self):
        return self.up

    def complete(self, system, user, max_tokens=2048):
        if not self.up:
            raise OSError("connection refused")
        self.calls += 1
        return f"from {self.name}"


# -- parsing ---------------------------------------------------------------
def test_url_normalization_accepts_bare_host_port():
    assert normalize_url("192.168.1.42:11434") == "http://192.168.1.42:11434"
    assert normalize_url("http://x:11434/") == "http://x:11434"


def test_endpoints_follow_preference_order_and_inherit_model():
    eps = parse_endpoints(cfg())
    assert [e.name for e in eps] == ["laptop", "local"]
    # per-endpoint override wins; the other inherits agent.model
    assert eps[0].model == "qwen2.5-coder:3b"
    assert eps[1].model == "qwen2.5-coder:7b"


def test_preference_may_name_a_host_that_does_not_exist_yet():
    """The shipped default mentions `laptop` before it is configured."""
    c = cfg(hosts={"local": "http://127.0.0.1:11434"})
    assert [e.name for e in parse_endpoints(c)] == ["local"]


def test_hosts_omitted_from_preference_are_still_reachable_by_name():
    c = cfg(host_preference=["local"])
    eps = parse_endpoints(c)
    assert [e.name for e in eps] == ["local", "laptop"]
    # ...but auto only considers the preference list first
    cands, _ = select_candidates(c, eps)
    assert cands[0].name == "local"


# -- selection policy -------------------------------------------------------
def test_auto_considers_every_endpoint():
    c = cfg()
    cands, policy = select_candidates(c, parse_endpoints(c))
    assert policy == "auto"
    assert [e.name for e in cands] == ["laptop", "local"]


def test_naming_a_host_pins_it_to_exactly_one_candidate():
    c = cfg(host="laptop")
    cands, policy = select_candidates(c, parse_endpoints(c))
    assert policy == "pinned:laptop"
    assert [e.name for e in cands] == ["laptop"]


def test_bare_url_is_accepted_and_matched_back_to_a_known_host():
    c = cfg(host="dell.lan:11434")
    cands, policy = select_candidates(c, parse_endpoints(c))
    assert policy == "pinned:laptop"
    assert cands[0].model == "qwen2.5-coder:3b"


def test_unknown_bare_name_is_rejected_rather_than_invented():
    """`--ollama-host laptp` must fail loudly. Turning a typo into
    http://laptp:11434 buys an endpoint that can never answer, and under a
    pinned policy that silently costs the whole night."""
    c = cfg()
    with pytest.raises(ValueError, match="unknown Ollama host"):
        select_candidates(c, parse_endpoints(c), override="laptp")


def test_unknown_url_becomes_an_adhoc_endpoint():
    c = cfg(host="http://192.168.1.99:11434")
    cands, policy = select_candidates(c, parse_endpoints(c))
    assert policy == "pinned:adhoc"
    assert cands[0].url == "http://192.168.1.99:11434"
    assert cands[0].model == "qwen2.5-coder:7b"


def test_cli_override_beats_env_which_beats_config(monkeypatch):
    c = cfg(host="local")
    eps = parse_endpoints(c)

    monkeypatch.setenv("OLLAMA_HOST", "laptop")
    assert select_candidates(c, eps)[1] == "pinned:laptop"
    assert select_candidates(c, eps, override="local")[1] == "pinned:local"

    monkeypatch.delenv("OLLAMA_HOST")
    assert select_candidates(c, eps)[1] == "pinned:local"


# -- pool behaviour ---------------------------------------------------------
def test_auto_skips_a_down_host_and_takes_the_next():
    laptop, local = FakeClient("laptop", up=False), FakeClient("local")
    pool = OllamaPool([laptop, local], policy="auto")
    assert pool.available()
    assert pool.current is local


def test_pinned_host_that_is_down_does_not_fall_back():
    """The whole point of pinning. Offline mutation is the correct outcome."""
    pool = OllamaPool([FakeClient("laptop", up=False)], policy="pinned:laptop")
    assert pool.available() is False
    assert pool.current is None


def test_failover_when_the_chosen_host_dies_mid_session():
    laptop, local = FakeClient("laptop"), FakeClient("local")
    logged: list[str] = []
    pool = OllamaPool([laptop, local], policy="auto", logger=logged.append)

    assert pool.available() and pool.current is laptop
    laptop.up = False                       # the laptop reboots overnight
    assert pool.available() and pool.current is local
    assert any("failed over" in m for m in logged)


def test_complete_retries_once_on_a_connection_error():
    laptop, local = FakeClient("laptop"), FakeClient("local")
    pool = OllamaPool([laptop, local], policy="auto")
    assert pool.available()

    laptop.up = False                       # dies between probe and generate
    assert pool.complete("sys", "user") == "from local"
    assert local.calls == 1


def test_complete_raises_when_nothing_is_left():
    laptop = FakeClient("laptop")
    pool = OllamaPool([laptop], policy="auto")
    assert pool.available()
    laptop.up = False
    with pytest.raises(OSError):
        pool.complete("sys", "user")


def test_a_host_that_comes_back_is_picked_up_again():
    laptop, local = FakeClient("laptop", up=False), FakeClient("local")
    pool = OllamaPool([laptop, local], policy="auto")
    assert pool.available() and pool.current is local

    local.up = False
    laptop.up = True
    assert pool.available() and pool.current is laptop


# -- factory ----------------------------------------------------------------
def test_make_client_builds_a_pool_with_the_configured_timeouts():
    pool = make_client(cfg(request_timeout_s=123, probe_timeout_s=4))
    assert isinstance(pool, OllamaPool)
    assert [c.name for c in pool.candidates] == ["laptop", "local"]
    assert pool.candidates[0].timeout == 123
    assert pool.candidates[0].probe_timeout == 4


def test_unknown_provider_means_offline_mode():
    assert make_client({"agent": {"provider": "nope"}}) is None
