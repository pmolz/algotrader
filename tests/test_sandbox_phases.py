"""The sandbox runner must attribute failures to a phase, not lump them into one
bucket. Two things depend on it: the agent loop only spends an LLM fix-retry on
`FAIL codegen`, and the morning report distinguishes "the model can't write code"
from "the ideas don't survive validation".
"""

import pytest

from algotrader.agent.sandbox import (
    SandboxLimits,
    docker_available,
    image_exists,
    run_gauntlet_sandboxed,
)
from algotrader.config import load_config

IMAGE = "algotrader-sandbox:latest"
requires_docker = pytest.mark.skipif(
    not (docker_available() and image_exists(IMAGE)),
    reason="Docker or sandbox image not available",
)

DOES_NOT_COMPILE = "class Broken(Strategy:\n    name = 'x'\n"

NO_STRATEGY_CLASS = "x = 1 + 1\n"

CRASHES_AT_RUNTIME = '''
class Exploder(Strategy):
    name = "exploder"
    def generate_signals(self, df):
        df["close"][20]          # KeyError on a timestamp-indexed frame
        return StrategyResult(positions=df["close"] * 0)
'''


@requires_docker
@pytest.mark.parametrize("code", [DOES_NOT_COMPILE, NO_STRATEGY_CLASS])
def test_uncompilable_code_is_labelled_codegen(trending_ohlcv, code):
    rep = run_gauntlet_sandboxed(
        trending_ohlcv, code, load_config(), n_trials=1,
        limits=SandboxLimits(image=IMAGE),
    )
    assert rep["ok"] is False
    assert rep["promoted"] is False
    assert rep["reasons"] == ["FAIL codegen"]
    assert rep["error"]


@requires_docker
def test_runtime_explosion_is_labelled_strategy_runtime(trending_ohlcv):
    rep = run_gauntlet_sandboxed(
        trending_ohlcv, CRASHES_AT_RUNTIME, load_config(), n_trials=1,
        limits=SandboxLimits(image=IMAGE),
    )
    assert rep["ok"] is False
    assert rep["reasons"] == ["FAIL strategy runtime"]
    # it compiled, so we still know what it called itself
    assert rep.get("strategy_name") == "exploder"


@requires_docker
def test_codegen_failure_earns_a_fix_retry(trending_ohlcv, tmp_path, monkeypatch):
    """The loop should re-prompt once on FAIL codegen — the label is what routes it."""
    from algotrader.agent.loop import AgentLoop

    cfg = load_config()
    cfg["experiments"]["db_path"] = str(tmp_path / "exp.db")
    cfg["experiments"]["generated_code_dir"] = str(tmp_path / "gen")
    cfg["agent"]["use_sandbox"] = True
    cfg["sandbox"]["image"] = IMAGE

    calls = []

    class FakeLLM:
        def available(self):
            return True

        def complete(self, system, user, **kw):
            calls.append(user)
            if len(calls) == 1:
                return f"HYPOTHESIS: broken first try\n```python\n{DOES_NOT_COMPILE}```"
            return (
                "HYPOTHESIS: fixed\n```python\n"
                'class Fixed(Strategy):\n'
                '    name = "fixed"\n'
                '    def generate_signals(self, df):\n'
                '        c = df["close"]\n'
                '        pos = (c > c.rolling(20).mean()).astype(float)\n'
                '        return StrategyResult(positions=pos.fillna(0.0))\n'
                "```"
            )

    monkeypatch.setattr("algotrader.agent.loop.make_client", lambda *a: FakeLLM())
    loop = AgentLoop(trending_ohlcv, cfg, symbol="T", source="test", timeframe="1d")
    r = loop.step()

    assert len(calls) == 2, "a codegen failure should trigger exactly one fix-retry"
    assert "failed to run" in calls[1]
    assert r["strategy"] == "fixed"      # the retry's strategy is what got judged
