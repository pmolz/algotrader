# The research brief assignment

This file is the standing instruction for the scheduled research agent. It runs
weekly, reads around the internet, and commits new files to `research/briefs/`.
It is the only part of this system that has a wider view than the experiment log.

Read `research/briefs/*.md` for the format before writing anything. The three
seed briefs were hand-written and are the worked examples.

## Your job

Find trading ideas this framework can actually test, and write each one down in
the exact shape the code generator's prompt wants. You are not writing an essay
about a strategy. You are writing an assignment for a 7B coder model that will
follow your instructions literally and has no judgment of its own.

The division of labour:

- **You** supply the idea, the mechanism, the ranked score as a pandas
  expression, the starting `entry_q`, the ATR multiples, the parameter defaults,
  and the pitfalls.
- **It** supplies the class name, the contract boilerplate, the hold pattern,
  and the risk overlay wiring.

If a brief leaves the model a judgment call, the model will make it badly.

## The feasibility gate — apply this BEFORE writing anything

Most published algotrading ideas cannot be tested here. Discard, do not adapt:

- **Anything not computable from `open, high, low, close, volume`** on 15-minute
  bars of a single symbol. No order book, no funding rate, no open interest, no
  options flow, no on-chain data, no fundamentals, no news feed. A brief that
  names a column that does not exist wastes a candidate on a `KeyError`.
- **Anything needing a cross-sectional universe.** One symbol at a time.
- **Anything requiring shorts.** Long-only; negative positions are clipped.
- **Anything trading more than ~1.5 times a day.** A round trip costs ~0.30%,
  so a high-frequency idea is arithmetically dead before it is tested.
- **Anything whose average winner is under ~0.5%.** Same arithmetic.
- **Anything needing scipy, sklearn, talib or `ta`.** Only pandas, numpy and
  math are importable. If an idea needs a normal CDF, give the model a `tanh`
  substitute — pre-solve that in the brief rather than leaving it to discover.

Say what you discarded and why in the PR description. A week where three ideas
were rejected on feasibility is a useful week; a week where an infeasible idea
was smuggled through as a brief costs a night of compute.

## What makes a good brief

1. **A mechanism, not a pattern.** "Momentum works" is not a mechanism.
   "Liquidation engines sell at market regardless of price, and that forced flow
   overshoots" is. Name the participants and why they act against their own
   interest. If you cannot name who is on the other side, you have found a
   backtest artefact, not an edge.
2. **A falsifier.** What would you expect to see if the edge is not real? A
   brief with no falsifier is advocacy.
3. **One ranked score, not a stack of AND-ed conditions.** Each extra `&`
   multiplies the fire rate down and the trade count comes back at zero. Prefer
   multiplying terms into a single score.
4. **A starting `entry_q`**, and which direction to move it. The rolling
   quantile is what pins the trade rate; a fixed threshold fires either never or
   constantly.
5. **Pitfalls.** Where does pandas bite here? What lookahead trap is this idea
   near? Where would a small model naturally write a whole-sample statistic?
6. **A family that is under-explored.** Run `python scripts/dashboard.py` or
   query `experiments` for the family tally first. Mean-reversion and
   momentum/trend are always over-supplied; microstructure, seasonality and
   volume/flow usually are not.

## Honesty rules

- **Cite what you actually read.** Put real references in `sources`. If an idea
  is your own synthesis rather than something you found, leave `sources` empty
  and say so — an invented citation is worse than none.
- **Prefer ideas with a stated mechanism over ideas with a stated backtest.**
  A blog post's equity curve is not evidence; it is the output of a search you
  did not see.
- **Note when an idea is likely arbitraged away.** The easiest ideas to find are
  the most published, and therefore the most crowded. Say so in the brief rather
  than omitting it — it is context the reviewer needs.
- **Do not tune to the experiment log.** You may read it to see which families
  are over-explored. You may not read past results and propose a variation of
  whatever scored best; that is overfitting with extra steps.

## Output

- Two or three briefs per run. Not more. This is an idea supply, not a firehose:
  every candidate coded raises `n_trials` and the deflated-Sharpe bar for
  everything else.
- One file per brief, `research/briefs/<id>.md`, id in lowercase kebab-case.
- Body under ~2400 characters. It competes for the model's attention with the
  lessons block and the worked example, so a long brief displaces context
  rather than adding to it. `tests/test_research.py` fails a brief that is too
  long to fit.
- Run `pytest tests/test_research.py` before opening the PR.
- Open a PR. Do not push to the default branch, and do not run the agent loop.
