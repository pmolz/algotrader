/* Session control card.
 *
 * Polls /api/session/status and renders one of two states: idle (pick a budget,
 * start) or running (live progress, stop). Polling means the card is correct
 * even when a session was started from a terminal or by the nightly timer —
 * this page is a view of the truth, not the owner of it.
 */

const HEADER = { 'Content-Type': 'application/json', 'X-Algotrader': '1' };

const $ = (id) => document.getElementById(id);

function fmtDur(s) {
  if (s == null) return '—';
  s = Math.floor(s);
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60);
  if (h) return `${h}h ${m}m`;
  if (m) return `${m}m ${s % 60}s`;
  return `${s}s`;
}

async function post(url, body) {
  const res = await fetch(url, {
    method: 'POST', headers: HEADER, body: JSON.stringify(body || {}),
  });
  let data = {};
  try { data = await res.json(); } catch { /* empty body */ }
  if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
  return data;
}

function tile(label, value) {
  const t = document.createElement('div');
  t.className = 'tile';
  const l = document.createElement('div'); l.className = 'label'; l.textContent = label;
  const v = document.createElement('div'); v.className = 'value'; v.textContent = value;
  t.appendChild(l); t.appendChild(v);
  return t;
}

let pollTimer = null;
let lastActive = null;

function render(s) {
  const card = $('control-card');
  card.hidden = false;

  const idle = $('control-idle');
  const running = $('control-running');

  if (!s.available || !s.enabled) {
    $('control-sub').textContent = s.enabled
      ? 'systemd --user is unavailable here — start sessions with '
        + 'python scripts/run_nightly.py instead.'
      : 'Session control is disabled in config (dashboard.allow_control).';
    idle.hidden = true;
    running.hidden = true;
    $('control-schedule-row').hidden = true;
    return;
  }

  // schedule switch reflects the unit, not what you last clicked
  $('ctl-schedule').checked = !!s.scheduled;
  $('ctl-next').textContent = s.next_run ? `next: ${s.next_run}` : '';

  if (s.active && s.run) {
    idle.hidden = true;
    running.hidden = false;
    const r = s.run;
    const budget = r.budget || {};
    $('control-sub').textContent =
      `Session #${r.id} · ${(r.symbols || []).join(', ')} · ${r.model || ''}`;
    $('ctl-state').textContent = 'running';
    $('ctl-progress-text').textContent =
      `${fmtDur(r.elapsed_s)} elapsed` +
      (budget.max_hours ? ` of ${budget.max_hours}h` : '');

    const meter = $('ctl-meter');
    if (r.progress != null) {
      meter.hidden = false;
      $('ctl-fill').style.width = `${Math.round(r.progress * 100)}%`;
    } else {
      meter.hidden = true;
    }

    const live = $('ctl-live');
    live.textContent = '';
    live.appendChild(tile('Candidates so far', String(r.iterations)));
    live.appendChild(tile('Promoted', String(r.promoted)));
    live.appendChild(tile('Elapsed', fmtDur(r.elapsed_s)));
    if (budget.max_iterations) {
      live.appendChild(tile('Iteration limit', String(budget.max_iterations)));
    }
  } else {
    idle.hidden = false;
    running.hidden = true;
    if (s.run && s.run.orphaned) {
      // a run row with no finish time and no live process
      $('control-sub').textContent =
        `No session running. Run #${s.run.id} never finished — it was killed or `
        + 'the machine slept.';
    } else {
      $('control-sub').textContent = 'No session running.';
    }
  }

  // a session that just ended changed the data underneath the page
  if (lastActive === true && s.active === false) {
    $('ctl-msg').textContent = 'Session finished — reload for the new results.';
  }
  lastActive = s.active;
}

async function poll() {
  try {
    const res = await fetch('/api/session/status');
    render(await res.json());
  } catch { /* dashboard restarting; try again on the next tick */ }
}

export function initControl() {
  const card = $('control-card');
  if (!card) return;

  $('ctl-start').addEventListener('click', async () => {
    const btn = $('ctl-start');
    btn.disabled = true;
    $('ctl-msg').textContent = 'Starting…';
    try {
      await post('/api/session/start', {
        hours: $('ctl-hours').value,
        iterations: $('ctl-iters').value || null,
      });
      $('ctl-msg').textContent = '';
      await poll();
    } catch (e) {
      $('ctl-msg').textContent = e.message;
    } finally {
      btn.disabled = false;
    }
  });

  $('ctl-stop').addEventListener('click', async () => {
    const btn = $('ctl-stop');
    btn.disabled = true;
    // the session finishes its current candidate first, so this is not instant
    $('ctl-msg2').textContent = 'Stopping after the current candidate…';
    try {
      await post('/api/session/stop');
    } catch (e) {
      $('ctl-msg2').textContent = e.message;
    } finally {
      btn.disabled = false;
    }
  });

  $('ctl-schedule').addEventListener('change', async (ev) => {
    const on = ev.target.checked;
    try {
      await post('/api/session/schedule', { enabled: on });
    } catch (e) {
      ev.target.checked = !on;            // put the switch back if it failed
      $('ctl-next').textContent = e.message;
    }
    await poll();
  });

  poll();
  // 5s while running is responsive without being chatty; the endpoint only
  // shells out to `systemctl is-active` and runs two indexed queries.
  pollTimer = setInterval(poll, 5000);
  window.addEventListener('beforeunload', () => clearInterval(pollTimer));
}
