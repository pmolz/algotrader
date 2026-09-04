# Roadmap

Build trust from the bottom up. Do not skip Phase 0.

## Phase 0 — Foundation (trust a backtest number)  ← START HERE
- [x] Repo scaffold
- [x] Data fetch + local cache (Parquet)
- [x] Strategy interface
- [x] Vectorized backtest engine with fees + slippage
- [x] Performance metrics (Sharpe, Sortino, max drawdown, CAGR, etc.)
- [x] Validation gauntlet: walk-forward, OOS holdout, cost sensitivity, deflated Sharpe
- [x] 2 hand-written baseline strategies
- [ ] Automated lookahead-bias check on every strategy
- [ ] Unit tests for the backtest engine on known inputs

## Phase 1 — Manual agent
- [x] Experiment DB (SQLite) recording every run
- [x] LLM strategy proposer (structured hypothesis)
- [x] LLM strategy code generator against the strategy interface
- [ ] Human reviews each generated strategy before it runs
- [ ] "Lessons learned" memory the LLM retrieves before proposing

## Phase 2 — Automated loop
- [ ] Docker sandbox per generated strategy (untrusted code!)
- [ ] Overnight cron loop: propose -> code -> validate -> log -> reflect
- [ ] Morning report generation
- [ ] Multiple-testing bookkeeping across all experiments

## Phase 3 — Forward testing
- [ ] Auto-promote gauntlet survivors to Alpaca paper trading
- [ ] Live-vs-backtest divergence monitoring (overfitting detector)
- [ ] Alerting on drift

## Phase 4 — Real capital (careful)
- [ ] Hard position + loss limits, kill switch
- [ ] Human approval gate for any real-money action
- [ ] Start tiny, scale only on proven live edge

## Guardrails that never come off
- Strict train/OOS separation — the agent never sees holdout data during development.
- Realistic costs modelled in every backtest.
- Deflated Sharpe / multiple-testing correction on all reported edges.
- No real money without a human in the loop.
