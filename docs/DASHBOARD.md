# The dashboard

A local web UI over the experiment log: **http://127.0.0.1:8765**

```bash
python scripts/dashboard.py           # on demand
systemctl --user status algotrader-dashboard   # or as an always-on service
```

## What's on it

| Page | What it answers |
|---|---|
| **Overview** | Start/stop a session; did anything get promoted? Where are candidates dying? Is the model even producing valid code? |
| **Experiments** | The full searchable history, filtered by range / symbol / promoted-only |
| **Experiment detail** | One candidate's whole story: hypothesis, every gauntlet check, fold Sharpes, cost stress, the generated code, and an on-demand equity curve |
| **Sessions** | Every unattended run, plus the agent's own reflections |
| **Reports** | The morning reports, rendered in the browser |

The filter row at the top scopes everything below it — stats, charts and tables
all re-render against the same slice, so the numbers always agree.

## Running sessions from the browser

The **Agent session** card on the Overview page starts and stops the agent.

- **Run for** — a wall-clock budget (15 minutes to 12 hours).
- **Stop after** — an optional candidate limit. Whichever limit is hit first
  ends the session.
- **Stop after this candidate** — a graceful stop. It sends SIGTERM, which
  `NightlySession` handles by finishing the current candidate, writing its
  reflection, and closing the run row. It is not a kill, so it is not instant.
- **Also run automatically every night at 22:00** — toggles the systemd timer.

While a session runs the card shows live progress: elapsed against budget,
candidates so far, and how many were promoted. It polls every 5s, so it is
correct even when the session was started from a terminal or by the timer —
this page is a *view* of the truth, not the owner of it.

Sessions run as a transient systemd unit, not as a child of the web app, so
restarting or crashing the dashboard never kills a four-hour research run.

## What it can and cannot do

The experiment DB is opened with `mode=ro`: the dashboard physically cannot
write the log the agent is appending to. A test asserts the connection rejects a
`DELETE`. Nothing in the app can promote a candidate, edit an experiment, or
place an order.

The endpoints that *do* have side effects are the whole blast radius:

| Endpoint | Effect |
|---|---|
| `POST /api/equity/<id>` | Re-runs one strategy in the Docker jail |
| `POST /api/session/start` | Launches a session as a transient systemd unit |
| `POST /api/session/stop` | SIGTERMs it (graceful) |
| `POST /api/session/schedule` | Enables/disables the nightly timer |

Equity curves execute strategy code — unavoidably, that is what a curve is —
inside the **same jail the nightly loop uses**: no network, read-only root,
dropped capabilities, non-root, memory/CPU/PID caps, hard timeout. The
dashboard process never execs generated code itself, and results are cached by
a hash of (code, data, costs) so a refresh cannot spam container launches.

Session control never passes anything through a shell. Arguments are a
validated list: the duration and iteration count are range-checked and
*rejected* rather than clamped, so a mistyped value is visible instead of
silently becoming something that runs for a fortnight.

Set `dashboard.allow_control: false` to remove the control endpoints and the UI
entirely, making the app strictly read-only.

## No authentication — keep it on loopback

It binds `127.0.0.1` and has no login. That is fine for a local tool and *not*
fine on a network. `scripts/dashboard.py` warns if you pass a non-loopback
`--host`; if you genuinely need remote access, put a reverse proxy with auth in
front rather than exposing it directly — and turn `allow_control` off.

"It's only localhost" is **not** by itself a defence for the mutating
endpoints, so they carry two guards:

- **CSRF.** Any page you visit in the same browser can POST to
  `127.0.0.1:8765` with a plain HTML form — no JavaScript required. A form
  cannot set a custom header, and a custom header on a cross-origin `fetch`
  triggers a CORS preflight this app never approves. So every mutating request
  must carry `X-Algotrader: 1`, and a foreign `Origin` is rejected outright.
- **DNS rebinding.** An attacker's domain can resolve to `127.0.0.1`, making
  their page same-origin with this app. The `Host` header still carries *their*
  hostname, so it is pinned to loopback names.

Both are tested. Neither applies to the read-only pages.

Model-written text (strategy names, hypotheses) is treated as untrusted
throughout: Jinja autoescapes it server-side, and the chart code inserts every
label with `textContent`, never `innerHTML`. Morning reports render as
preformatted text rather than through a Markdown-to-HTML step, which would be an
injection surface for no real gain.

## The charts

Deliberately plain, and chosen by what the data has to do:

- **Where candidates died** — horizontal bars, one hue. The categories (stages)
  have no natural order, so coloring them by size would double-encode the bar
  length and burn the only free channel.
- **Activity by day** — stacked columns, three categorical slots, legend always
  present.
- **Walk-forward fold Sharpes** — diverging columns around zero: blue above, red
  below, with the promotion threshold drawn as a reference line.
- **Cost stress** — a line, single series.
- **Equity curve** — two lines (strategy vs buy & hold) with a marker where the
  out-of-sample holdout begins, so you can see which part of the curve the
  strategy was developed against.

Every chart has a hover layer *and* a "Show as table" twin, so no value is
reachable only by hovering. Light and dark are both first-class: the dark
palette is its own set of steps validated against the dark surface, not an
automatic inversion. The colors pass the CVD, contrast, lightness-band and
chroma checks in both modes.

## Troubleshooting

**Blank charts.** The chart module is an ES module; open the browser console.
If you see a CORS or module error, you are probably opening the HTML from disk
rather than through the server.

**"Docker isn't available" when plotting a curve.** Expected and deliberate —
strategy code is never run outside the sandbox. Start Docker and retry.

**Numbers disagree with the morning report.** They shouldn't; both derive a
candidate's cause of death from `agent.report.fail_stage`, and a test asserts
they agree. If you see a divergence, the dashboard is the one that is wrong.

**Service won't start.** `journalctl --user -u algotrader-dashboard -n 50`.
