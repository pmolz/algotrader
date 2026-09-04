"""Sandbox tests. The Docker-dependent test is skipped automatically when Docker
or the image is unavailable, so the suite stays green in any environment.
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
_docker_ready = docker_available() and image_exists(IMAGE)
requires_docker = pytest.mark.skipif(
    not _docker_ready, reason="Docker or sandbox image not available"
)

GOOD_STRATEGY = '''
class GenStrat(Strategy):
    name = "gen_sandbox_test"
    def generate_signals(self, df):
        c = df["close"]
        pos = (c > c.rolling(20).mean()).astype(float)
        return StrategyResult(positions=pos.fillna(0.0))
'''

MALICIOUS_STRATEGY = '''
class Evil(Strategy):
    name = "evil"
    def generate_signals(self, df):
        import socket  # should be blocked by the import whitelist
        return StrategyResult(positions=df["close"] * 0)
'''


@requires_docker
def test_sandbox_runs_good_strategy(trending_ohlcv):
    cfg = load_config()
    rep = run_gauntlet_sandboxed(
        trending_ohlcv, GOOD_STRATEGY, cfg, n_trials=1, limits=SandboxLimits(image=IMAGE)
    )
    assert rep["ok"] is True
    assert "promoted" in rep
    assert isinstance(rep["reasons"], list)


@requires_docker
def test_sandbox_blocks_malicious_import(trending_ohlcv):
    cfg = load_config()
    rep = run_gauntlet_sandboxed(
        trending_ohlcv, MALICIOUS_STRATEGY, cfg, n_trials=1,
        limits=SandboxLimits(image=IMAGE),
    )
    # code screen / import whitelist should stop it; never promoted
    assert rep["promoted"] is False


def test_sandbox_limits_defaults():
    lim = SandboxLimits()
    assert lim.timeout_s > 0 and lim.pids > 0
