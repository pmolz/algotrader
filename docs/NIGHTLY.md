# The overnight loop

One command, run by cron, that spends the night proposing strategies and judging
them — then leaves you a one-page report to read with coffee.

```
22:00  cron ──> scripts/nightly_cron.sh
                  ├─ run_nightly.py      propose → code → validate → log → reflect
                  └─ morning_report.py   experiments/reports/YYYY-MM-DD.md
07:30  cron ──> morning_report.py        safety net if the session ran long
```

## Setup

```bash
cd ~/Projects/algotrader && source .venv/bin/activate

# 1. the jail (mandatory — see "Why the sandbox is not optional")
bash sandbox/build.sh

# 2. a coder-tuned model
ollama pull qwen2.5-coder:7b        # already the config default

# 3. prove it works while you're watching
python scripts/run_nightly.py --hours 0.1 --max-iterations 2
python scripts/morning_report.py

# 4. see what would go in your crontab, then install it
bash scripts/install_cron.sh
bash scripts/install_cron.sh --install
```

Times are configurable: `NIGHTLY_CRON="0 23 * * *" bash scripts/install_cron.sh`.

## What one session does

Per iteration, against one symbol:

1. **Propose** — the LLM gets the lessons-learned block from the experiment DB
   (best results, recent failures, and its own past reflections) and returns a
   hypothesis plus a `Strategy` subclass.
2. **Code** — the response is parsed; a compile/instantiate failure earns exactly
   one fix-retry with the traceback fed back.
3. **Validate** — the full gauntlet runs *inside the container*. `n_trials` is
   the lifetime experiment count, so the deflated-Sharpe bar keeps rising.
4. **Log** — hypothesis, code, metrics, every check, and the verdict go to
   SQLite, tagged with the run id.
5. **Reflect** — every `reflect_every` iterations and once at the end, the agent
   summarizes its own session. That text is stored and fed back into step 1 of
   future sessions. This is the part that makes the loop cumulative rather than
   just repetitive.

Symbols are round-robined, so an eight-hour night is spread across your markets
instead of being spent entirely on whichever one is listed first.

## Config

```yaml
nightly:
  symbols:                  # round-robined, one iteration each in turn
    - {symbol: BTC/USD, source: ccxt, timeframe: 1d}
    - {symbol: SPY, source: yfinance, timeframe: 1d}
  max_hours: 8.0            # wall-clock budget
  max_iterations: null      # optional hard cap
  reflect_every: 10         # 0 = reflect only at the end
  log_dir: experiments/logs
  report_dir: experiments/reports
  lock_file: experiments/nightly.lock
  report_window_start_hour: 18   # morning report looks back to 18:00 yesterday
```

## Why the sandbox is not optional

`NightlySession` refuses to start with `agent.use_sandbox: false`. Unattended
execution of code a language model wrote, on your machine, with your files and
network, while you sleep, is not a risk worth taking for a backtest. `docker` is
checked and the image verified before the first iteration.

`--allow-unsandboxed` exists for debugging the harness. It is recorded in the
run row and printed in bold in the morning report, because a night that ran that
way is a night you should know about.

## What keeps a night from being wasted

| Failure | What happens |
|---|---|
| A candidate crashes | One iteration lost, labelled `strategy runtime`, session continues |
| Model writes garbage | One fix-retry, then labelled `codegen`, session continues |
| A symbol won't download | Falls back to cached history; skipped only if there is none |
| The session overruns into the next cron slot | Second launch sees the PID lock and exits (code 2) |
| A previous run died and left a lock | PID is dead, lock is taken over |
| Machine sleeps / process killed | `finished_at` stays NULL; the report flags it |
| Wall clock nearly up | A running mean of iteration cost stops it from starting one it can't finish |
| SIGTERM / SIGINT | Current iteration finishes, then it reflects and closes the run cleanly |

## Reading the report

`experiments/reports/YYYY-MM-DD.md`. It is written to be un-flattering:

- **"Nothing promoted"** is the healthy headline. Most nights end here.
- **"Wasted night"** means most candidates never reached a verdict — that says
  something about your model, nothing about the markets. Pull a bigger coder
  model before reading anything into the results.
- **"N candidates survived"** is framed as *suspicious until reviewed*, not as
  good news. After hundreds of trials, a survivor is most often a
  multiple-testing artifact.
- **Where candidates died** is the histogram to watch over time. Ideas dying at
  walk-forward is the normal shape.
- **Closest rejections** exists so you can see the near-misses — explicitly not
  as a shortlist. Re-testing them is how overfitting happens.
- **Multiple-testing bar** only ever goes up. That is intentional.

Regenerate any past window at any time; the report reads only the DB:

```bash
python scripts/morning_report.py --days 7      # the past week
python scripts/morning_report.py --run 12      # one specific session
python scripts/morning_report.py --stdout      # print, write no file
```

## If the machine sleeps at night

Cron does not fire while suspended, and a session interrupted by suspend shows
up as an unfinished run. Either keep the box awake:

```bash
sudo systemctl mask sleep.target suspend.target hibernate.target
```

...or use a systemd timer with `Persistent=true`, which runs a missed job on the
next wake. `~/.config/systemd/user/algotrader-nightly.service`:

```ini
[Unit]
Description=algotrader overnight session

[Service]
Type=oneshot
WorkingDirectory=%h/Projects/algotrader
ExecStart=/bin/bash %h/Projects/algotrader/scripts/nightly_cron.sh
```

`~/.config/systemd/user/algotrader-nightly.timer`:

```ini
[Unit]
Description=Run the algotrader overnight session

[Timer]
OnCalendar=*-*-* 22:00:00
Persistent=true

[Install]
WantedBy=timers.target
```

```bash
systemctl --user daemon-reload
systemctl --user enable --now algotrader-nightly.timer
systemctl --user list-timers algotrader-nightly.timer
loginctl enable-linger "$USER"     # so it runs when you're not logged in
```

## Troubleshooting

**Nothing ran.** The report says so loudly. Check `experiments/logs/cron-*.log`
first — if it is missing entirely, cron never fired (`systemctl status cronie`).

**"another session holds the lock".** Expected if last night overran. Confirm
with `cat experiments/nightly.lock` and `ps -p <pid>`.

**"Docker unavailable" in the cron log but `docker info` works in your shell.**
Group membership from a fresh login isn't in cron's environment the same way —
verify with `id -nG` and re-login after `usermod -aG docker $USER`.

**Every iteration is `codegen`.** The model is too small or not coder-tuned.
`ollama pull qwen2.5-coder:14b` and set `agent.model`.

**Sessions are slower than expected.** Each iteration is one LLM call plus one
container run; ~20–40s per iteration with a 7B model on CPU is normal. The
wall-clock estimator accounts for this automatically.
