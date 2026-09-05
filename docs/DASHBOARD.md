# The dashboard

A local web UI over the experiment log: **http://127.0.0.1:8765**

```bash
python scripts/dashboard.py           # on demand
systemctl --user status algotrader-dashboard   # or as an always-on service
```

## What's on it

| Page | What it answers |
|---|---|
| **Overview** | Did anything get promoted? Where are candidates dying? Is the model even producing valid code? |
| **Experiments** | The full searchable history, filtered by range / symbol / promoted-only |
| **Experiment detail** | One candidate's whole story: hypothesis, every gauntlet check, fold Sharpes, cost stress, the generated code, and an on-demand equity curve |
| **Sessions** | Every unattended run, plus the agent's own reflections |
| **Reports** | The morning reports, rendered in the browser |

The filter row at the top scopes everything below it — stats, charts and tables
all re-render against the same slice, so the numbers always agree.

## It is read-only

The dashboard opens the experiment DB with `mode=ro`, so it physically cannot
write to the log the agent is appending to while you browse. There is a test
asserting the connection rejects a `DELETE`.

The one exception is the equity curve, which has to *execute* strategy code to
produce a curve at all. That runs in the **same Docker jail the nightly loop
uses** — no network, read-only root, dropped capabilities, non-root, memory/CPU
/PID caps, hard timeout. The dashboard process never execs generated code
itself. Results are cached on disk keyed by a hash of (code, data, costs), so a
browser refresh cannot spam container launches.

## No authentication — keep it on loopback

It binds `127.0.0.1` and has no login. That is fine for a local tool and *not*
fine on a network. `scripts/dashboard.py` warns if you pass a non-loopback
`--host`; if you genuinely need remote access, put a reverse proxy with auth in
front rather than exposing it directly.

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
