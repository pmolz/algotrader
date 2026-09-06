# Research briefs

The nightly loop's proposer is a 7B coder model running on a local GPU. It is
good at turning a specified idea into contract-correct Python and bad at
inventing ideas — left to itself it converges on the same few textbook
mechanisms, which is what the family tally in the lessons block exists to push
back against.

A **brief** is the other half of that fix. Something with a wider view of the
literature reads around, discards what this framework cannot test, and writes
the survivors down in the shape the propose prompt already wants. The local
model then codes them.

```
Claude (weekly, scheduled)          local 7B (nightly)
  search -> feasibility gate  --->    brief -> contract-correct Python
  -> research/briefs/*.md             -> gauntlet -> experiment log
```

The gauntlet is unchanged and does not know where a candidate came from. A brief
buys an idea a place in the queue and nothing else.

## The split, and why it is drawn there

| | supplies |
|---|---|
| brief | the mechanism, the ranked score as a pandas expression, the starting `entry_q`, the ATR multiples, the parameter defaults, the pitfalls |
| 7B | the class name, the contract boilerplate, the hold pattern, the risk overlay wiring |

A brief written as an essay does not survive contact with a 7B — the same
reason `prompts.WORKED_EXAMPLE` exists. Anything left to the model's judgment
gets a bad judgment. So briefs are short, sectioned, and capped at
`research.max_chars`; past that they crowd out the lessons block rather than
adding to it.

## Format

YAML front matter, then a body that is pasted into the prompt verbatim.

```markdown
---
id: liquidation-cascade-overshoot   # lowercase kebab-case; this is the provenance key
title: Liquidation cascade overshoot
family: mean-reversion              # matches memory.py's family tally
sources: []                         # real references, or empty — never invented
retired: false                      # optional; true takes it out of rotation
---

MECHANISM / WHY IT CAN CLEAR COSTS / FALSIFIER / SIGNAL / ENTRY / RISK /
PARAMS / PITFALLS
```

Three seed briefs ship with the repo. They were hand-written, not searched, and
exist as worked examples of the format.

## How one gets used

`AgentLoop._propose` asks `BriefLibrary.select` for the next brief. Selection is
least-attempted-first and deterministic, so a session rotates through the
library instead of re-rolling one idea. Attempt counts come back out of
`experiments.researched_from`, per symbol — a brief that died on BTC has not yet
been tried on ETH.

A brief that has been coded `research.max_attempts_per_brief` times is retired
from the rotation. Re-coding an idea that keeps dying spends a night on one
hypothesis; the codegen repair loop already gets `agent.max_fix_attempts` within
each attempt.

When the library is exhausted — or empty, or missing, or `research.enabled` is
false — the loop asks the model to invent an idea, exactly as it did before
briefs existed. That is a normal outcome, not an error.

Malformed brief files are reported and skipped, never raised: one bad file
should not end an unattended session. Watch for `[research] IGNORED malformed
brief` in the session log, because a pipeline that has silently stopped
supplying ideas looks exactly like one that is working.

## Config

```yaml
research:
  enabled: true
  briefs_dir: research/briefs
  max_attempts_per_brief: 3    # per symbol
  max_chars: 2500              # prompt budget for the brief body
```

## The scheduled researcher

`research/PROMPT.md` is the standing assignment. It runs weekly from cron on the
machine that holds the repo:

```bash
bash scripts/research_briefs.sh --dry-run   # show what would run
bash scripts/research_briefs.sh             # run a pass now
bash scripts/install_cron.sh                # includes the Sunday 06:00 entry
```

The script runs `claude -p` headless against `research/PROMPT.md` with WebSearch,
using the CLI login already on the box. `--allowedTools` scopes it so `Write`
cannot leave `research/briefs/` and `Bash` can only run the format checks: with
no TTY a permission prompt would be a hang rather than a question, and the worst
outcome of a bad run should be a bad brief, not a bad commit.

**It writes files and stops.** No commit, no push, no PR. Sunday's briefs sit in
the working tree until you have read them:

```bash
git diff -- research/briefs
```

A research agent that could merge its own ideas into the search would be a
research agent with no review step, and the review is the part that has to stay
human. The script also refuses to start when `research/briefs` already has
uncommitted changes — otherwise a second week's output lands beside an
unreviewed first week's and `git diff` stops telling you which is which.

Cron does not fire while the machine is suspended; the same caveat and the
systemd-timer alternative in [docs/NIGHTLY.md](NIGHTLY.md) apply here.

### Running it in the cloud instead

`research/PROMPT.md` is deliberately the only copy of the assignment, so a
scheduled cloud agent (`/schedule` in Claude Code) can run the same job from the
same file and open a PR instead. That path needs your claude.ai account linked to
GitHub, which on a managed enterprise seat is usually an admin setting rather
than something you can enable yourself.

Two or three briefs a week, deliberately. Every candidate coded raises
`n_trials` and therefore the deflated-Sharpe bar for everything else in the log.
That cost is real but modest — on a 28k-bar dev set the bar for DSR > 0.5 moves
from an annualised Sharpe of 3.55 at 747 trials to 4.16 at 5,600 — so it is a
reason to prefer considered ideas over volume, not a reason to starve the loop.

The binding constraint is throughput, not supply. Measured at 0.58 min per
candidate, an 8-hour session gets through roughly 800 of them, so a library of
three briefs at `max_attempts_per_brief: 25` covers about a fifth of one night
and the rest goes back to invented ideas. Raise the cap to shift that mix;
because sampling is temperature 0.7 with no fixed seed, extra attempts are
different implementations of the same idea rather than repeats of one.

## Reading the results

`experiments.researched_from` is the provenance key. It exists so that after a
few hundred candidates you can ask the question that decides whether any of this
was worth it:

```sql
SELECT researched_from IS NOT NULL AS researched,
       COUNT(*) AS n,
       SUM(promoted) AS promoted
FROM experiments GROUP BY researched;
```

Expect the honest answer to be "no significant difference". Published edges are
the ones most likely to be crowded already, and a brief cannot make a dead idea
work — it can only stop the local model from proposing its ninth mean-reversion
variant. If researched candidates do not clear more gauntlet stages than
self-generated ones, the scheduled agent is costing you a PR review a week for
nothing, and turning it off is the correct response.
