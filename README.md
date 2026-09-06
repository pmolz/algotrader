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
Research briefs (Claude, weekly)  ->  research/briefs/*.md
                                            |
Orchestrator (agent loop)                   v
  1. Propose hypothesis        (LLM, implementing a brief if one is available)
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
| `algotrader/agent/research.py` | External research briefs: the idea supply the local model codes from |
| `algotrader/live`       | Paper/live trading adapters (Alpaca paper first) |
| `algotrader/agent/nightly.py` | Unattended overnight session (budgets, locking, reflection) |
| `algotrader/agent/report.py`  | Morning report generator |
| `algotrader/dashboard`  | Local read-only web UI over the experiment log |
| `experiments/`          | SQLite experiment log, generated code, logs, reports |

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# 1. pull and cache some data (a year of 15m bars, paged from the exchange)
python scripts/fetch_data.py --symbol BTC/USD --source ccxt --timeframe 15m

# 2. run the built-in baseline strategies through the gauntlet
python scripts/run_backtest.py --strategy sma_crossover --symbol BTC/USD

# 3. run the agent loop
python scripts/run_agent.py --symbol BTC/USD --timeframe 15m --iterations 5 --sandbox

# 4. hand it the night, read the report over coffee
bash sandbox/build.sh
python scripts/run_nightly.py --hours 0.1 --max-iterations 2   # supervised trial
bash scripts/install_cron.sh                                   # dry run first
```

> `data.ccxt_exchange` defaults to `bitstamp`, not Binance — Binance refuses
> public OHLCV requests from some regions. Any ccxt venue works.

## Intraday crypto, and why costs are the whole problem

The default target is **BTC/USD and ETH/USD on 15-minute bars**, one year of
history (35,040 bars, paged from the exchange 1000 at a time).

At that timeframe transaction costs dominate everything else. A round trip pays
fees plus slippage twice — about **0.30%** at the configured rates — so:

| trades/day | round trips/yr | cost drag |
|-----------:|---------------:|----------:|
| 1 | 365 | ~110% of capital |
| 3 | 1,095 | ~330% |
| 10 | 3,650 | ~1,100% |

A strategy must average more than 0.30% *per trade* before anything is left
over. This is not a tuning detail, it is the constraint the whole search runs
inside, so the gauntlet checks it **before** the expensive validation and
rejects anything above `validation.max_trades_per_day`. The failure messages
distinguish "traded too often to survive costs", "never traded at all", and
"traded sensibly and lost money" — three different problems that need three
different fixes, and all three end up in the agent's memory.

### Stop loss and take profit

Strategies may declare a risk overlay:

```python
return StrategyResult(
    positions=entries.astype(float),
    stop_loss_pct=atr_frac * 1.5,     # float, or a Series for volatility sizing
    take_profit_pct=atr_frac * 3.0,
)
```

When either is set the engine stops doing close-to-close arithmetic and walks
each bar's path. The fill rules are chosen so a backtest cannot flatter itself:

- **If a bar could have hit both your stop and your target, the stop is
  assumed.** OHLC cannot say which came first, and guessing the target is how a
  backtest invents an edge. `ambiguous_exit_frac` counts how often that
  assumption was load-bearing, and a candidate leaning on it too heavily is
  rejected as unmeasurable rather than accepted as profitable.
- **A bar that gaps through a level fills at the open**, not at the level.
- **No immediate re-entry** after a stop: the signal must go flat and fire
  again, or the stop is just a fee generator.

Shorting is off. Perp funding and borrow costs aren't modelled, and leaving
them out would flatter every short strategy.

## Overnight loop + morning report

```bash
python scripts/run_nightly.py       # one unattended session (cron calls this)
python scripts/morning_report.py    # experiments/reports/YYYY-MM-DD.md
bash scripts/install_cron.sh        # show the crontab entries; --install applies
```

A session round-robins your configured symbols, spends its wall-clock budget
proposing and judging candidates, survives crashes and dead data sources, and
reflects on its own results so the next session starts smarter. It **refuses to
run with the sandbox off**. The report is written to be un-flattering: "nothing
promoted" is the healthy headline, a survivor is framed as suspicious until
reviewed, and a night where most candidates failed to compile is called a wasted
night rather than a clean result.

Full details, systemd-timer alternative, and troubleshooting: **[docs/NIGHTLY.md](docs/NIGHTLY.md)**.

## Dashboard

```bash
pip install -e ".[dashboard]"
python scripts/dashboard.py     # http://127.0.0.1:8765
```

Browse the whole experiment log: overview stats, where candidates die, per-day
activity, the full history, and each candidate's evidence — hypothesis, every
gauntlet check, walk-forward fold Sharpes, cost stress, the generated code, and
an on-demand equity curve.

The Overview page also **starts and stops sessions**: pick a duration and an
optional candidate limit, watch live progress, and stop gracefully (the agent
finishes its current candidate, reflects, and closes the run cleanly). Sessions
run as a transient systemd unit, so they survive the dashboard restarting.

It opens the DB **read-only** and cannot write to the log the agent is appending
to. Plotting an equity curve necessarily *runs* strategy code, so that happens
inside the same Docker jail the nightly loop uses, and the result is cached.
Loopback-only with CSRF and DNS-rebinding guards on the mutating endpoints; set
`dashboard.allow_control: false` to make it strictly read-only. See
**[docs/DASHBOARD.md](docs/DASHBOARD.md)**.

## Using a local model (Ollama / Qwen) — free & private

The agent talks to whatever `agent.provider` in `config/default.yaml` points at.
Default is a **local Ollama** model, so no API key or cloud spend is needed.

```bash
# one-time: pull a coder-tuned model (better code = fewer wasted iterations)
ollama pull qwen2.5-coder:7b      # solid default
# qwen3.x "thinking" models also work — the client disables think mode for code-gen
```

Then set the model in `config/default.yaml`:

```yaml
agent:
  provider: ollama
  model: qwen2.5-coder:7b   # or qwen3.5:9b, etc.
```

### Choosing which machine serves the model

The model can live on this box, on another machine on the LAN, or both. Name the
endpoints under `agent.hosts` and pick one with `agent.host`:

```yaml
agent:
  host: auto                        # "auto" | a name from hosts | a bare URL
  host_preference: [laptop, local]  # order "auto" tries them in
  hosts:
    local: http://127.0.0.1:11434
    laptop:                         # mapping form: per-endpoint model override
      url: http://192.168.1.42:11434
      model: qwen2.5-coder:3b       # a 4GB laptop GPU may only fit a smaller one
```

Put machine-specific addresses in `config/local.yaml` (gitignored), not in
`default.yaml`.

```bash
python scripts/check_llm.py                    # who's up, and who'd get the work
python scripts/run_agent.py   --ollama-host laptop --symbol BTC/USD --iterations 3
python scripts/run_nightly.py --ollama-host local
```

Precedence is `--ollama-host` > `$OLLAMA_HOST` > `agent.host`. The rules:

- **`auto`** walks `host_preference` and takes the first endpoint that answers
  *and* has the model pulled — a reachable server missing the model is no more
  use than one that's off.
- **Naming a host pins it.** If a pinned host is down the session drops to the
  offline fallback rather than quietly running somewhere else; the run row
  records which machine generated the strategies, so that claim has to be true.
- **A host that dies mid-session is failed over on the next iteration** (under
  `auto`), because losing one candidate is much cheaper than losing eight hours.
  One that comes back gets picked up again.

A session resolves its endpoint once at startup, logs it, and stores it in
`runs.host`. If nothing is available it says so loudly — an unnoticed night of
offline baseline mutation looks like research and isn't.

Notes:
- Smaller general models (<=3B) tend to emit broken code — expect many retries.
  Use a 7B+ coder model for real work.
- To use Claude instead: set `provider: anthropic`, `pip install -e ".[agent]"`,
  and put `ANTHROPIC_API_KEY` in `.env`. Note that a Claude *subscription* seat
  is not API access — the API bills separately through console.anthropic.com.
- With no working provider, the loop falls back to mutating baselines so you can
  still exercise the whole pipeline offline.

## Research briefs — Claude thinks, the local model codes

The local 7B is good at turning a specified idea into contract-correct Python
and bad at inventing ideas: left alone it converges on the same few textbook
mechanisms. So the idea supply is split off from the coding.

```
Claude (weekly, scheduled)          local 7B (nightly)
  search -> feasibility gate  --->    brief -> contract-correct Python
  -> research/briefs/*.md             -> gauntlet -> experiment log
```

A **brief** is a short, sectioned assignment: the mechanism, the ranked score as
a pandas expression, a starting `entry_q`, ATR multiples, parameter defaults,
and the pitfalls. The 7B supplies the class name, the contract boilerplate, the
hold pattern and the risk wiring — the parts it is actually good at.

```bash
ls research/briefs/                 # three hand-written seeds ship with the repo
cat research/PROMPT.md              # the standing assignment for the scheduler
python scripts/run_agent.py --symbol BTC/USD --iterations 3   # picks briefs up automatically
```

The loop rotates through the library least-attempted-first, retires a brief
after `research.max_attempts_per_brief` tries, and goes back to inventing its
own ideas when the library is exhausted or absent. At the measured 0.58 min per
candidate an 8-hour session runs ~800 of them, so that cap is what decides how
much of a night is researched ideas rather than invented ones. Every candidate records
`experiments.researched_from`, so you can later check whether researched ideas
actually cleared more gauntlet stages than self-generated ones.

The gauntlet does not know where a candidate came from. A brief buys an idea a
place in the queue and nothing else — and since every candidate raises `n_trials`
and the deflated-Sharpe bar for everything else, the supply is deliberately two
or three briefs a week rather than a firehose.

Full details, the format, and the feasibility gate: **[docs/RESEARCH.md](docs/RESEARCH.md)**.

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
