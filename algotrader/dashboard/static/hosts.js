/* Model host status.
 *
 * Polls /api/hosts and renders one row per configured Ollama endpoint: a state
 * dot, which one a run would currently pick, and — where the host reports a GPU
 * — utilisation, VRAM, temperature and power.
 *
 * The dot is the point. Under `host: auto` a machine that drops off the network
 * fails nothing: the session moves to the fallback and the log looks like any
 * other night. This is the only place that shows it happened.
 */

const POLL_MS = 10000;

const el = (tag, cls, text) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text != null) n.textContent = text;
  return n;
};

// Laptop GPUs throttle before they fail, so a slow night is usually a hot one.
// Thresholds are deliberately conservative for a mobile part.
function tempClass(c) {
  if (c == null) return 'neutral';
  if (c >= 87) return 'bad';
  if (c >= 78) return 'warn';
  return 'good';
}

const STATE = {
  up: ['good', 'up'],
  'no-model': ['warn', 'model missing'],
  down: ['bad', 'unreachable'],
};

function metric(label, value, cls) {
  const m = el('div', 'host-metric');
  m.appendChild(el('span', 'host-metric-label', label));
  m.appendChild(el('span', `host-metric-value ${cls || ''}`.trim(), value));
  return m;
}

function row(host, selected) {
  const wrap = el('div', 'host-row');

  const [dotCls, stateLabel] = STATE[host.state] || ['neutral', host.state];
  const badge = el('span', `badge ${dotCls}`);
  badge.appendChild(el('span', 'dot'));
  badge.appendChild(el('span', null, host.name));
  wrap.appendChild(badge);

  const meta = el('div', 'host-meta');
  let detail = stateLabel;
  if (host.state === 'up' && host.latency_ms != null) detail += ` · ${host.latency_ms}ms`;
  if (host.state === 'no-model') detail += ` · wants ${host.model}`;
  if (host.state === 'down' && host.error) detail += ` · ${host.error.split(':')[0]}`;
  meta.appendChild(el('span', 'host-url', host.url));
  meta.appendChild(el('span', 'host-state', detail));
  if (host.name === selected) meta.appendChild(el('span', 'host-tag', 'in use'));
  // Resident means the weights are in VRAM. Empty on an otherwise-healthy host
  // means OLLAMA_KEEP_ALIVE expired and every iteration re-pays the load.
  else if (host.state === 'up' && !host.resident) {
    meta.appendChild(el('span', 'host-tag idle', 'not loaded'));
  }
  wrap.appendChild(meta);

  const g = host.gpu;
  const metrics = el('div', 'host-metrics');
  if (g && !g.error) {
    metrics.appendChild(metric('GPU', g.util_pct == null ? '—' : `${Math.round(g.util_pct)}%`));
    if (g.mem_total_mb) {
      metrics.appendChild(metric(
        'VRAM',
        `${(g.mem_used_mb / 1024).toFixed(1)}/${(g.mem_total_mb / 1024).toFixed(1)}G`,
      ));
    }
    metrics.appendChild(metric(
      'Temp', g.temp_c == null ? '—' : `${Math.round(g.temp_c)}°C`, tempClass(g.temp_c),
    ));
    if (g.power_w != null) metrics.appendChild(metric('Power', `${Math.round(g.power_w)}W`));
  } else if (g && g.error) {
    metrics.appendChild(el('span', 'host-note', `GPU: ${g.error}`));
  }
  wrap.appendChild(metrics);
  return wrap;
}

async function tick() {
  const card = document.getElementById('hosts-card');
  const list = document.getElementById('hosts-list');
  const sub = document.getElementById('hosts-sub');
  if (!card) return;

  let data;
  try {
    const res = await fetch('/api/hosts');
    data = await res.json();
  } catch {
    sub.textContent = 'Status unavailable — the dashboard could not reach its own API.';
    card.hidden = false;
    return;
  }

  if (data.enabled === false) { card.hidden = true; return; }
  card.hidden = false;

  if (data.error) {
    sub.textContent = `Host config problem: ${data.error}`;
    list.replaceChildren();
    return;
  }
  if (!data.hosts.length) {
    sub.textContent = 'No hosts configured under agent.hosts.';
    list.replaceChildren();
    return;
  }

  const down = data.hosts.filter((h) => h.state !== 'up').map((h) => h.name);
  if (!data.selected) {
    sub.textContent = 'No usable host — a session would fall back to offline '
      + 'baseline mutation, which is a plumbing test rather than research.';
  } else if (down.length) {
    sub.textContent = `Using ${data.selected}. Unavailable: ${down.join(', ')}.`;
  } else {
    sub.textContent = `Using ${data.selected}. All hosts healthy.`;
  }

  list.replaceChildren(...data.hosts.map((h) => row(h, data.selected)));
}

tick();
setInterval(tick, POLL_MS);
