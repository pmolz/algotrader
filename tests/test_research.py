"""Research briefs: the external idea supply, and its provenance trail.

Two things are easy to get wrong here and both are silent. A brief directory
that has stopped parsing looks exactly like one that is working — the loop just
quietly goes back to inventing ideas — so malformed files must be reported, not
swallowed. And a brief that is handed out forever spends a whole night re-coding
one hypothesis, which is the opposite of the diversity briefs exist to buy.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from algotrader.agent.memory import ExperimentDB
from algotrader.agent.research import Brief, BriefError, BriefLibrary, parse_brief

GOOD = """---
id: some-idea
title: Some Idea
family: microstructure
sources: [a paper]
---

MECHANISM
Something forced happens and price overshoots.
"""


def write(directory, name, text):
    p = Path(directory) / name
    p.write_text(text)
    return p


# -- parsing ---------------------------------------------------------------------

def test_parses_front_matter_and_body():
    b = parse_brief(GOOD)
    assert b.id == "some-idea"
    assert b.family == "microstructure"
    assert b.sources == ("a paper",)
    assert b.body.startswith("MECHANISM")
    assert "---" not in b.body


@pytest.mark.parametrize("text, expected", [
    ("no front matter at all\n", "front matter"),
    ("---\nid: x\ntitle: T\nfamily: f\n---\n\nbody\n", "must be lowercase"),
    ("---\nid: ok-id\ntitle: T\n---\n\nbody\n", "missing required field"),
    ("---\nid: ok-id\ntitle: T\nfamily: f\n---\n", "empty body"),
    ("---\nid: Bad_ID\ntitle: T\nfamily: f\n---\n\nbody\n", "kebab-case"),
    ("---\n- not\n- a mapping\n---\n\nbody\n", "must be a mapping"),
])
def test_rejects_malformed(text, expected):
    with pytest.raises(BriefError, match=expected):
        parse_brief(text)


def test_render_clips_long_bodies():
    b = Brief(id="x-id", title="T", family="f", body="word " * 2000)
    out = b.render(max_chars=100)
    assert len(out) == 100
    assert out.endswith("…")


# -- the library -----------------------------------------------------------------

def test_missing_directory_is_not_an_error(tmp_path):
    lib = BriefLibrary(tmp_path / "nope")
    assert lib.briefs == [] and lib.problems == []
    assert lib.select() is None


def test_malformed_file_is_reported_not_raised(tmp_path):
    write(tmp_path, "ok.md", GOOD)
    write(tmp_path, "broken.md", "this is not a brief")
    lib = BriefLibrary(tmp_path)
    assert [b.id for b in lib.briefs] == ["some-idea"]
    assert len(lib.problems) == 1 and "broken.md" in lib.problems[0]


def test_duplicate_ids_are_rejected(tmp_path):
    write(tmp_path, "a.md", GOOD)
    write(tmp_path, "b.md", GOOD)
    lib = BriefLibrary(tmp_path)
    assert len(lib.briefs) == 1
    assert "duplicate id" in lib.problems[0]


def test_retired_briefs_are_not_served(tmp_path):
    write(tmp_path, "a.md", GOOD.replace("---\n\nMECH", "retired: true\n---\n\nMECH"))
    lib = BriefLibrary(tmp_path)
    assert lib.briefs == [] and lib.select() is None


def _library(tmp_path, ids, **kw):
    for i in ids:
        write(tmp_path, f"{i}.md", GOOD.replace("id: some-idea", f"id: {i}"))
    return BriefLibrary(tmp_path, **kw)


def test_selection_rotates_least_attempted_first(tmp_path):
    lib = _library(tmp_path, ["aaa-idea", "bbb-idea", "ccc-idea"])
    assert lib.select({}).id == "aaa-idea"
    assert lib.select({"aaa-idea": 1}).id == "bbb-idea"
    assert lib.select({"aaa-idea": 1, "bbb-idea": 1}).id == "ccc-idea"
    # a full round trip comes back to the front rather than stalling on the last
    assert lib.select({"aaa-idea": 1, "bbb-idea": 1, "ccc-idea": 1}).id == "aaa-idea"


def test_brief_retires_after_max_attempts(tmp_path):
    lib = _library(tmp_path, ["aaa-idea", "bbb-idea"], max_attempts=2)
    assert lib.select({"aaa-idea": 2}).id == "bbb-idea"
    # exhausting the library is a normal outcome: the loop invents its own idea
    assert lib.select({"aaa-idea": 2, "bbb-idea": 2}) is None


# -- provenance ------------------------------------------------------------------

def test_brief_tally_counts_per_symbol(tmp_path):
    db = ExperimentDB(tmp_path / "e.db")

    def add(symbol, brief):
        db.record(symbol=symbol, source="ccxt", timeframe="15m", strategy_name="s",
                  hypothesis="h", params={}, code="x", promoted=False, metrics={},
                  gauntlet_checks={}, reasons=[], researched_from=brief)

    add("BTC/USD", "aaa-idea")
    add("BTC/USD", "aaa-idea")
    add("BTC/USD", "bbb-idea")
    add("ETH/USD", "aaa-idea")
    add("BTC/USD", None)          # self-generated ideas do not count against a brief

    assert db.brief_tally("BTC/USD") == {"aaa-idea": 2, "bbb-idea": 1}
    assert db.brief_tally("ETH/USD") == {"aaa-idea": 1}
    assert db.brief_tally() == {"aaa-idea": 3, "bbb-idea": 1}


def test_migration_adds_provenance_to_an_existing_db(tmp_path):
    """A log written before briefs existed must still open and stay countable."""
    path = tmp_path / "old.db"
    db = ExperimentDB(path)
    db.conn.execute("ALTER TABLE experiments DROP COLUMN researched_from")
    db.conn.commit()
    db.close()

    db = ExperimentDB(path)
    have = {r["name"] for r in db.conn.execute("PRAGMA table_info(experiments)")}
    assert "researched_from" in have
    assert db.brief_tally() == {}


# -- the seed library that ships with the repo -----------------------------------

def test_shipped_briefs_are_valid():
    """The repo's own briefs are the worked examples for the format; a broken
    one teaches the scheduled researcher to write broken ones."""
    lib = BriefLibrary("research/briefs")
    assert lib.problems == []
    assert len(lib.briefs) >= 3
    for b in lib.briefs:
        # Uncapped rendering means the prompt silently loses the pitfalls
        # section at the bottom, which is the part a 7B most needs.
        assert not b.render().endswith("…"), f"{b.id} is too long for the prompt"
        assert "entry_q" in b.body, f"{b.id} does not pin a starting trade rate"
        assert "rolling" in b.body, f"{b.id} does not specify a rolling threshold"


# -- the loop actually uses them -------------------------------------------------

WORKS = (
    "HYPOTHESIS: implemented the brief\n```python\n"
    'class FromBrief(Strategy):\n'
    '    name = "from_brief"\n'
    '    def generate_signals(self, df):\n'
    '        c = df["close"]\n'
    '        pos = (c > c.rolling(20).mean()).astype(float)\n'
    '        return StrategyResult(positions=pos.fillna(0.0))\n'
    "```"
)


class FakeLLM:
    def __init__(self):
        self.prompts = []

    def describe(self):
        return "fake"

    def available(self):
        return True

    def complete(self, system, user, **kw):
        self.prompts.append(user)
        return WORKS


@pytest.fixture
def loop_cfg(tmp_path):
    from algotrader.config import load_config

    cfg = load_config()
    cfg["agent"]["use_sandbox"] = False
    cfg["experiments"]["db_path"] = str(tmp_path / "exp.db")
    cfg["experiments"]["generated_code_dir"] = str(tmp_path / "gen")
    cfg["research"]["briefs_dir"] = str(tmp_path / "briefs")
    (tmp_path / "briefs").mkdir()
    return cfg


def _loop(cfg, df, monkeypatch, llm):
    from algotrader.agent.loop import AgentLoop

    monkeypatch.setattr("algotrader.agent.loop.make_client", lambda *a, **k: llm)
    return AgentLoop(df, cfg, symbol="T", source="test", timeframe="1d")


def test_a_brief_reaches_the_prompt_and_the_log(loop_cfg, trending_ohlcv, monkeypatch):
    write(loop_cfg["research"]["briefs_dir"], "a.md", GOOD)
    llm = FakeLLM()
    loop = _loop(loop_cfg, trending_ohlcv, monkeypatch, llm)
    r = loop.step()

    prompt = llm.prompts[0]
    assert "RESEARCH BRIEF: Some Idea" in prompt
    assert "Something forced happens" in prompt
    assert "do NOT substitute" in prompt
    # the lessons block and worked example are still there — a brief adds to the
    # context, it does not replace the memory the loop already had
    assert "class VolatilityBreakoutPullback" in prompt
    assert r["brief"] == "some-idea"
    assert loop.db.brief_tally("T") == {"some-idea": 1}


def test_no_briefs_falls_back_to_inventing(loop_cfg, trending_ohlcv, monkeypatch):
    llm = FakeLLM()
    loop = _loop(loop_cfg, trending_ohlcv, monkeypatch, llm)
    r = loop.step()

    assert "RESEARCH BRIEF" not in llm.prompts[0]
    assert "Now propose ONE NEW strategy" in llm.prompts[0]
    assert r["brief"] is None


def test_research_can_be_switched_off(loop_cfg, trending_ohlcv, monkeypatch):
    write(loop_cfg["research"]["briefs_dir"], "a.md", GOOD)
    loop_cfg["research"]["enabled"] = False
    llm = FakeLLM()
    loop = _loop(loop_cfg, trending_ohlcv, monkeypatch, llm)
    assert loop.briefs is None
    loop.step()
    assert "RESEARCH BRIEF" not in llm.prompts[0]


def test_successive_iterations_rotate_through_the_library(
    loop_cfg, trending_ohlcv, monkeypatch
):
    """The tally comes back out of the DB, so the rotation has to survive the
    round trip through `record` — an iteration that forgot to write provenance
    would hand out the same brief forever."""
    d = loop_cfg["research"]["briefs_dir"]
    for i in ("aaa-idea", "bbb-idea"):
        write(d, f"{i}.md", GOOD.replace("id: some-idea", f"id: {i}"))
    loop = _loop(loop_cfg, trending_ohlcv, monkeypatch, FakeLLM())

    assert [loop.step()["brief"] for _ in range(4)] == [
        "aaa-idea", "bbb-idea", "aaa-idea", "bbb-idea",
    ]


def test_exhausted_library_returns_to_inventing(loop_cfg, trending_ohlcv, monkeypatch):
    write(loop_cfg["research"]["briefs_dir"], "a.md", GOOD)
    loop_cfg["research"]["max_attempts_per_brief"] = 1
    llm = FakeLLM()
    loop = _loop(loop_cfg, trending_ohlcv, monkeypatch, llm)

    assert loop.step()["brief"] == "some-idea"
    assert loop.step()["brief"] is None
    assert "Now propose ONE NEW strategy" in llm.prompts[1]
