# algotrader

A self-improving algorithmic trading research agent.

An LLM proposes trading strategies, codes them, and runs them through a rigorous
validation gauntlet. Only strategies that survive honest out-of-sample testing get
promoted to paper trading. The system records every experiment and *learns from
its own failures* over time.

> **The validation harness is the point of this project.** Markets are adversarial
> and mostly efficient. Any system that "learns what works" will mostly learn
> overfitted noise unless the evaluation underneath is brutally honest. Build and
> trust the gauntlet before you trust anything it produces.

## Architecture

```
Orchestrator (agent loop)
  1. Propose hypothesis        (LLM)
  2. Generate strategy code    (LLM -> sandbox)
  3. Backtest                  (vectorized engine)
  4. Validation gauntlet       (walk-forward, OOS, costs, multiple-testing)
  5. Record to experiment DB
  6. Reflect on past results   (LLM reads memory) -> back to 1

Promoted survivors -> paper trading -> live-vs-backtest monitoring -> (human gate) real capital
```

Package layout:

| Module | Responsibility |
|--------|----------------|
| `algotrader/data`       | Fetch + cache market data (yfinance, ccxt, Alpaca) |
| `algotrader/backtest`   | Backtest engine + performance metrics |
| `algotrader/strategies` | Strategy interface + a couple of hand-written baselines |
| `algotrader/validation` | Walk-forward, out-of-sample, cost sensitivity, deflated Sharpe |
| `algotrader/sentiment`  | Free sentiment sources (Reddit, GDELT, Fear & Greed) |
| `algotrader/agent`      | LLM loop: propose -> code -> evaluate -> reflect |
| `algotrader/live`       | Paper/live trading adapters (Alpaca paper first) |
| `experiments/`          | SQLite experiment log + generated strategy code |

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# 1. pull and cache some data
python scripts/fetch_data.py --symbol BTC/USDT --source ccxt --timeframe 1d

# 2. run the built-in baseline strategies through the gauntlet
python scripts/run_backtest.py --strategy sma_crossover --symbol BTC/USDT

# 3. (later) run the agent loop
python scripts/run_agent.py --iterations 5
```

## Development phases (see ROADMAP.md)

- **Phase 0** — trustworthy data + backtest + validation gauntlet  ← *start here*
- **Phase 1** — manual agent loop (LLM proposes, you review)
- **Phase 2** — automated overnight loop in a sandbox
- **Phase 3** — auto-promote survivors to paper trading
- **Phase 4** — small real capital behind a human approval gate

## Safety

Nothing in this repo should touch real money without a human approval gate,
hard-coded position/loss limits, and a kill switch. See `algotrader/live/safety.py`.

This is research software. Not financial advice. Trade at your own risk.
