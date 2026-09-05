/* Minimal SVG chart library — no dependencies, no CDN, works offline.
 *
 * Four forms, each picked for its data's job:
 *   barsH    magnitude across nominal categories  -> one hue, sequential-safe
 *   stacked  part-to-whole over time              -> categorical, legend + labels
 *   columns  values above/below a baseline        -> diverging blue/red
 *   lines    trend over time                      -> 1-2 series, crosshair
 *
 * Shared rules, applied by construction rather than by eye:
 *   - marks <= 24px thick, 4px rounded data-end, 2px surface gap between fills
 *   - lines 2px, markers >= 8px with a 2px surface ring
 *   - gridlines hairline and solid, never dashed
 *   - hit targets >= 24px, always larger than the mark
 *   - labels use text tokens, never the series color
 *   - every chart gets a table twin, so no value is hover-gated
 *   - all label text goes in via textContent: strategy names and hypotheses are
 *     model-generated, i.e. untrusted
 */

const SVG = 'http://www.w3.org/2000/svg';
const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();

function el(tag, attrs = {}, parent = null) {
  const n = document.createElementNS(SVG, tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  if (parent) parent.appendChild(n);
  return n;
}

/* Render at the container's real pixel width. A fixed viewBox width would be
 * letterboxed by preserveAspectRatio — the chart centres itself and leaves the
 * rest of the card empty. Pages re-draw on resize. */
function width(host, min = 320) {
  return Math.max(min, Math.floor(host.clientWidth || min));
}

function fmt(v, nd = 2) {
  if (v === null || v === undefined || Number.isNaN(v)) return '—';
  const a = Math.abs(v);
  if (a >= 1e9) return (v / 1e9).toFixed(1) + 'B';
  if (a >= 1e6) return (v / 1e6).toFixed(1) + 'M';
  if (a >= 1e4) return (v / 1e3).toFixed(1) + 'K';
  return v.toFixed(nd);
}

function niceTicks(min, max, count = 4) {
  if (min === max) { min -= 1; max += 1; }
  const span = max - min;
  const raw = span / count;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const norm = raw / mag;
  const step = (norm >= 5 ? 10 : norm >= 2 ? 5 : norm >= 1 ? 2 : 1) * mag;
  const out = [];
  for (let t = Math.ceil(min / step) * step; t <= max + 1e-9; t += step) out.push(t);
  return out;
}

/* -- shared chrome ---------------------------------------------------------- */
function tooltipFor(host) {
  let tt = host.querySelector('.tooltip');
  if (!tt) {
    tt = document.createElement('div');
    tt.className = 'tooltip';
    host.appendChild(tt);
  }
  return tt;
}

function showTip(tt, host, x, y, title, rows) {
  tt.textContent = '';
  const t = document.createElement('div');
  t.className = 'tt-title';
  t.textContent = title;                 // untrusted: textContent, never innerHTML
  tt.appendChild(t);
  for (const r of rows) {
    const row = document.createElement('div');
    row.className = 'tt-row';
    if (r.color) {
      const k = document.createElement('span');
      k.className = 'tt-key';
      k.style.background = r.color;
      row.appendChild(k);
    }
    const v = document.createElement('span');
    v.className = 'tt-val';               // value leads
    v.textContent = r.value;
    row.appendChild(v);
    if (r.name) {
      const n = document.createElement('span');
      n.className = 'tt-name';            // label follows
      n.textContent = r.name;
      row.appendChild(n);
    }
    tt.appendChild(row);
  }
  tt.classList.add('show');
  const hw = host.clientWidth;
  const tw = tt.offsetWidth || 150;
  tt.style.left = Math.max(0, Math.min(x + 14, hw - tw - 4)) + 'px';
  tt.style.top = Math.max(0, y - 10) + 'px';
}

const hideTip = (tt) => tt.classList.remove('show');

/* Table twin. Every chart calls this, so every value is reachable without hover. */
function tableTwin(host, columns, rows) {
  const wrap = document.createElement('div');
  wrap.className = 'table-view';
  const table = document.createElement('table');
  const thead = document.createElement('thead');
  const htr = document.createElement('tr');
  columns.forEach((c, i) => {
    const th = document.createElement('th');
    th.textContent = c;
    if (i > 0) th.className = 'num';
    htr.appendChild(th);
  });
  thead.appendChild(htr);
  table.appendChild(thead);
  const tb = document.createElement('tbody');
  for (const r of rows) {
    const tr = document.createElement('tr');
    r.forEach((cell, i) => {
      const td = document.createElement('td');
      td.textContent = cell;
      if (i > 0) td.className = 'num';
      tr.appendChild(td);
    });
    tb.appendChild(tr);
  }
  table.appendChild(tb);
  wrap.appendChild(table);

  const btn = document.createElement('button');
  btn.className = 'table-toggle';
  btn.type = 'button';
  btn.textContent = 'Show as table';
  btn.addEventListener('click', () => {
    const on = wrap.classList.toggle('show');
    btn.textContent = on ? 'Hide table' : 'Show as table';
  });
  host.appendChild(btn);
  host.appendChild(wrap);
}

function legend(host, items, kind = 'swatch') {
  const box = document.createElement('div');
  box.className = 'legend';
  for (const it of items) {
    const k = document.createElement('span');
    k.className = 'key';
    const sw = document.createElement('span');
    sw.className = kind === 'stroke' ? 'stroke' : 'swatch';
    sw.style.background = it.color;
    k.appendChild(sw);
    const lbl = document.createElement('span');
    lbl.textContent = it.label;          // text token color, not the series hue
    k.appendChild(lbl);
    box.appendChild(k);
  }
  host.insertBefore(box, host.firstChild);
}

/* -- 1. horizontal bars: magnitude across nominal categories ---------------- */
export function barsH(host, data, opts = {}) {
  host.textContent = '';
  const { valueKey = 'count', labelKey = 'label', unit = '' } = opts;
  if (!data.length) { host.innerHTML = '<div class="empty">No data in this range.</div>'; return; }

  const rowH = 30, barH = Math.min(24, 18), padL = 132, padR = 56, padT = 4;
  const h = data.length * rowH + padT;
  const W = width(host);
  const svg = el('svg', { viewBox: `0 0 ${W} ${h}`, height: h, role: 'img' }, host);
  const max = Math.max(...data.map(d => d[valueKey])) || 1;
  const w = W - padL - padR;
  const tt = tooltipFor(host);

  data.forEach((d, i) => {
    const y = padT + i * rowH;
    const bw = Math.max(2, (d[valueKey] / max) * w);
    const color = d.color || css('--series-1');

    // category label — text token, never the series color
    const lab = el('text', {
      x: padL - 10, y: y + barH / 2 + 4, 'text-anchor': 'end', class: 'label-text',
    }, svg);
    lab.textContent = d[labelKey];

    // 4px rounded data-end, square at the baseline
    const path = el('path', {
      d: roundedRightBar(padL, y, bw, barH, 4),
      fill: color, class: 'mark',
    }, svg);

    // value at the tip — selective direct labels, not one per data point
    const val = el('text', {
      x: padL + bw + 8, y: y + barH / 2 + 4, class: 'axis-text',
    }, svg);
    val.textContent = fmt(d[valueKey], 0) + unit + (d.share !== undefined ? `  (${Math.round(d.share * 100)}%)` : '');

    // hit target spans the whole row and clears 24px
    const hit = el('rect', { x: 0, y, width: W, height: rowH, class: 'hit' }, svg);
    const enter = (ev) => {
      path.classList.add('hot');
      const r = host.getBoundingClientRect();
      showTip(tt, host, ev.clientX - r.left, y + 4, d[labelKey], [
        { value: fmt(d[valueKey], 0) + unit, name: opts.valueName || 'candidates', color },
      ]);
    };
    hit.addEventListener('pointermove', enter);
    hit.addEventListener('pointerleave', () => { path.classList.remove('hot'); hideTip(tt); });
  });

  tableTwin(host, [opts.categoryName || 'Category', opts.valueName || 'Count'],
    data.map(d => [d[labelKey], String(d[valueKey])]));
}

function roundedRightBar(x, y, w, h, r) {
  r = Math.min(r, w);
  return `M${x},${y} H${x + w - r} Q${x + w},${y} ${x + w},${y + r} V${y + h - r} Q${x + w},${y + h} ${x + w - r},${y + h} H${x} Z`;
}

/* -- 2. stacked columns over time ------------------------------------------- */
export function stacked(host, data, series, opts = {}) {
  host.textContent = '';
  if (!data.length) { host.innerHTML = '<div class="empty">No activity in this range.</div>'; return; }
  legend(host, series.map(s => ({ label: s.label, color: css(s.color) })));

  const W = width(host), H = 220, padL = 34, padR = 8, padT = 10, padB = 30;
  const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, height: H, role: 'img' }, host);
  const plotW = W - padL - padR, plotH = H - padT - padB;
  const totals = data.map(d => series.reduce((a, s) => a + (d[s.key] || 0), 0));
  const max = Math.max(...totals, 1);
  const ticks = niceTicks(0, max, 3);
  // Scale to whichever is larger, the data max or the top gridline. Dividing by
  // the top tick alone lets a taller-than-the-last-tick bar overflow the plot
  // and collide with the legend above it.
  const top = Math.max(max, ...ticks);
  const yOf = (v) => padT + plotH - (v / top) * plotH;

  for (const t of ticks) {
    el('line', { x1: padL, x2: W - padR, y1: yOf(t), y2: yOf(t), class: 'gridline' }, svg);
    const tx = el('text', { x: padL - 7, y: yOf(t) + 4, 'text-anchor': 'end', class: 'axis-text' }, svg);
    tx.textContent = String(Math.round(t));
  }

  const band = plotW / data.length;
  const bw = Math.min(24, band * 0.62);
  const tt = tooltipFor(host);

  data.forEach((d, i) => {
    const cx = padL + band * (i + 0.5);
    let acc = 0;
    series.forEach((s) => {
      const v = d[s.key] || 0;
      if (!v) return;
      const y0 = yOf(acc + v), y1 = yOf(acc);
      // 2px surface gap does the separating — no stroke around the mark
      const hgt = Math.max(1, y1 - y0 - 2);
      el('rect', {
        x: cx - bw / 2, y: y0, width: bw, height: hgt,
        fill: css(s.color), rx: 2, class: 'mark',
      }, svg);
      acc += v;
    });

    const lx = el('text', { x: cx, y: H - 10, 'text-anchor': 'middle', class: 'axis-text' }, svg);
    lx.textContent = (d.day || '').slice(5);

    const hit = el('rect', {
      x: cx - Math.max(12, band / 2), y: padT,
      width: Math.max(24, band), height: plotH, class: 'hit',
    }, svg);
    hit.addEventListener('pointermove', (ev) => {
      const r = host.getBoundingClientRect();
      showTip(tt, host, ev.clientX - r.left, padT, d.day,
        series.map(s => ({ value: String(d[s.key] || 0), name: s.label, color: css(s.color) })));
    });
    hit.addEventListener('pointerleave', () => hideTip(tt));
  });

  tableTwin(host, ['Day', ...series.map(s => s.label)],
    data.map(d => [d.day, ...series.map(s => String(d[s.key] || 0))]));
}

/* -- 3. diverging columns around a baseline --------------------------------- */
export function columns(host, data, opts = {}) {
  host.textContent = '';
  const { labelKey = 'label', valueKey = 'value', threshold = null } = opts;
  if (!data.length) { host.innerHTML = '<div class="empty">Not recorded — this candidate died earlier.</div>'; return; }

  const vals = data.map(d => d[valueKey]);
  // A bar running to the bottom of the plot puts its value label where the
  // category labels live. Reserve a band for it instead of letting them collide.
  const hasNeg = vals.some(v => v < 0);
  const W = width(host), padL = 42, padR = 12, padT = 16;
  const padB = hasNeg ? 44 : 28;
  const H = hasNeg ? 216 : 200;
  const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, height: H, role: 'img' }, host);
  const plotW = W - padL - padR, plotH = H - padT - padB;
  let lo = Math.min(0, ...vals), hi = Math.max(0, ...vals);
  if (threshold !== null) hi = Math.max(hi, threshold);
  const ticks = niceTicks(lo, hi, 4);
  const tLo = Math.min(...ticks, lo), tHi = Math.max(...ticks, hi);
  const yOf = (v) => padT + plotH - ((v - tLo) / (tHi - tLo || 1)) * plotH;

  for (const t of ticks) {
    el('line', { x1: padL, x2: W - padR, y1: yOf(t), y2: yOf(t), class: 'gridline' }, svg);
    const tx = el('text', { x: padL - 7, y: yOf(t) + 4, 'text-anchor': 'end', class: 'axis-text' }, svg);
    tx.textContent = fmt(t, Math.abs(t) < 10 ? 1 : 0);
  }
  el('line', { x1: padL, x2: W - padR, y1: yOf(0), y2: yOf(0), class: 'baseline' }, svg);

  if (threshold !== null) {
    el('line', {
      x1: padL, x2: W - padR, y1: yOf(threshold), y2: yOf(threshold),
      stroke: css('--text-muted'), 'stroke-width': 1, opacity: 0.6,
    }, svg);
    const th = el('text', { x: W - padR, y: yOf(threshold) - 5, 'text-anchor': 'end', class: 'axis-text' }, svg);
    th.textContent = opts.thresholdLabel || `threshold ${fmt(threshold, 1)}`;
  }

  const band = plotW / data.length;
  const bw = Math.min(24, band * 0.5);
  const tt = tooltipFor(host);

  data.forEach((d, i) => {
    const v = d[valueKey];
    const cx = padL + band * (i + 0.5);
    const y0 = yOf(Math.max(0, v)), y1 = yOf(Math.min(0, v));
    const color = v >= 0 ? css('--pos') : css('--neg');
    const hgt = Math.max(2, y1 - y0);
    // rounded end away from the baseline; square where it meets zero
    const path = el('path', {
      d: v >= 0 ? roundedTopBar(cx - bw / 2, y0, bw, hgt, 4)
                : roundedBottomBar(cx - bw / 2, y0, bw, hgt, 4),
      fill: color, class: 'mark',
    }, svg);

    const vl = el('text', {
      x: cx, y: v >= 0 ? y0 - 6 : y1 + 13, 'text-anchor': 'middle', class: 'axis-text',
    }, svg);
    vl.textContent = fmt(v, 2);

    const lx = el('text', { x: cx, y: H - 6, 'text-anchor': 'middle', class: 'axis-text' }, svg);
    lx.textContent = String(d[labelKey]);

    const hit = el('rect', {
      x: cx - Math.max(12, band / 2), y: padT, width: Math.max(24, band), height: plotH, class: 'hit',
    }, svg);
    hit.addEventListener('pointermove', (ev) => {
      path.classList.add('hot');
      const r = host.getBoundingClientRect();
      showTip(tt, host, ev.clientX - r.left, padT, `${opts.categoryName || ''} ${d[labelKey]}`.trim(),
        [{ value: fmt(v, 2), name: opts.valueName || 'Sharpe', color }]);
    });
    hit.addEventListener('pointerleave', () => { path.classList.remove('hot'); hideTip(tt); });
  });

  tableTwin(host, [opts.categoryName || 'Item', opts.valueName || 'Value'],
    data.map(d => [String(d[labelKey]), fmt(d[valueKey], 2)]));
}

function roundedTopBar(x, y, w, h, r) {
  r = Math.min(r, h, w / 2);
  return `M${x},${y + h} V${y + r} Q${x},${y} ${x + r},${y} H${x + w - r} Q${x + w},${y} ${x + w},${y + r} V${y + h} Z`;
}
function roundedBottomBar(x, y, w, h, r) {
  r = Math.min(r, h, w / 2);
  return `M${x},${y} V${y + h - r} Q${x},${y + h} ${x + r},${y + h} H${x + w - r} Q${x + w},${y + h} ${x + w},${y + h - r} V${y} Z`;
}

/* -- 4. line chart with crosshair ------------------------------------------- */
export function lines(host, labels, seriesList, opts = {}) {
  host.textContent = '';
  if (!labels.length) { host.innerHTML = '<div class="empty">No series to plot.</div>'; return; }
  // legend for >= 2 series; a single series is named by the card title instead
  if (seriesList.length >= 2) {
    legend(host, seriesList.map(s => ({ label: s.label, color: css(s.color) })), 'stroke');
  }

  const W = width(host), H = opts.height || 240, padL = 52, padR = 62, padT = 12, padB = 28;
  const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, height: H, role: 'img' }, host);
  const plotW = W - padL - padR, plotH = H - padT - padB;
  const all = seriesList.flatMap(s => s.values)
    .filter(v => v !== null && v !== undefined && Number.isFinite(v));
  if (!all.length) { host.innerHTML = '<div class="empty">No finite values to plot.</div>'; return; }
  let lo = Math.min(...all), hi = Math.max(...all);
  if (opts.includeZero) { lo = Math.min(lo, 0); hi = Math.max(hi, 0); }
  const ticks = niceTicks(lo, hi, 4);
  const tLo = Math.min(...ticks, lo), tHi = Math.max(...ticks, hi);
  const xOf = (i) => padL + (labels.length === 1 ? plotW / 2 : (i / (labels.length - 1)) * plotW);
  const yOf = (v) => padT + plotH - ((v - tLo) / (tHi - tLo || 1)) * plotH;

  for (const t of ticks) {
    el('line', { x1: padL, x2: W - padR, y1: yOf(t), y2: yOf(t), class: 'gridline' }, svg);
    const tx = el('text', { x: padL - 8, y: yOf(t) + 4, 'text-anchor': 'end', class: 'axis-text' }, svg);
    tx.textContent = fmt(t, opts.yDecimals ?? (Math.abs(t) < 10 ? 1 : 0));
  }
  if (tLo < 0 && tHi > 0) {
    el('line', { x1: padL, x2: W - padR, y1: yOf(0), y2: yOf(0), class: 'baseline' }, svg);
  }

  // optional vertical marker (e.g. where the out-of-sample holdout begins)
  if (opts.markerIndex != null && opts.markerIndex >= 0) {
    const mx = xOf(opts.markerIndex);
    el('line', {
      x1: mx, x2: mx, y1: padT, y2: padT + plotH,
      stroke: css('--text-muted'), 'stroke-width': 1, opacity: 0.55,
    }, svg);
    const ml = el('text', { x: mx + 5, y: padT + 11, class: 'axis-text' }, svg);
    ml.textContent = opts.markerLabel || '';
  }

  for (const s of seriesList) {
    const color = css(s.color);
    // Non-finite values arrive as null (the API refuses to emit NaN/Infinity,
    // which are not valid JSON). Break the line at a gap rather than drawing a
    // segment through a value that does not exist.
    let run = [];
    const flush = () => {
      if (run.length > 1) {
        el('polyline', {
          points: run.join(' '), fill: 'none', stroke: color,
          'stroke-width': 2, 'stroke-linejoin': 'round', 'stroke-linecap': 'round',
        }, svg);
      }
      run = [];
    };
    s.values.forEach((v, i) => {
      if (v === null || v === undefined || Number.isNaN(v)) flush();
      else run.push(`${xOf(i)},${yOf(v)}`);
    });
    flush();

    // end marker on the last real value: >= 8px with a 2px surface ring
    let li = s.values.length - 1;
    while (li >= 0 && (s.values[li] === null || Number.isNaN(s.values[li]))) li--;
    if (li < 0) continue;
    el('circle', {
      cx: xOf(li), cy: yOf(s.values[li]), r: 4,
      fill: color, stroke: css('--surface-1'), 'stroke-width': 2,
    }, svg);
    const lbl = el('text', {
      x: xOf(li) + 9, y: yOf(s.values[li]) + 4, class: 'axis-text',
    }, svg);
    lbl.textContent = fmt(s.values[li], opts.yDecimals ?? 0);   // endpoint only
  }

  // x labels: first, middle, last — never one per point
  [0, Math.floor(labels.length / 2), labels.length - 1]
    .filter((v, i, a) => a.indexOf(v) === i && v >= 0 && v < labels.length)
    .forEach((i) => {
      const t = el('text', {
        x: xOf(i), y: H - 8, class: 'axis-text',
        'text-anchor': i === 0 ? 'start' : i === labels.length - 1 ? 'end' : 'middle',
      }, svg);
      t.textContent = labels[i];
    });

  // crosshair: the reader aims at an x position, never at a 2px line
  const cross = el('line', {
    x1: 0, x2: 0, y1: padT, y2: padT + plotH,
    stroke: css('--axis'), 'stroke-width': 1, opacity: 0,
  }, svg);
  const dots = seriesList.map(s => el('circle', {
    r: 4, fill: css(s.color), stroke: css('--surface-1'), 'stroke-width': 2, opacity: 0,
  }, svg));
  const tt = tooltipFor(host);
  const hit = el('rect', { x: padL, y: padT, width: plotW, height: plotH, class: 'hit' }, svg);

  hit.addEventListener('pointermove', (ev) => {
    const r = svg.getBoundingClientRect();
    const px = ((ev.clientX - r.left) / r.width) * W;
    const i = Math.max(0, Math.min(labels.length - 1,
      Math.round(((px - padL) / plotW) * (labels.length - 1))));
    cross.setAttribute('x1', xOf(i)); cross.setAttribute('x2', xOf(i));
    cross.setAttribute('opacity', 1);
    seriesList.forEach((s, k) => {
      dots[k].setAttribute('cx', xOf(i));
      dots[k].setAttribute('cy', yOf(s.values[i]));
      dots[k].setAttribute('opacity', 1);
    });
    const hr = host.getBoundingClientRect();
    // one tooltip lists every series at that x — no need to land on a line
    showTip(tt, host, ev.clientX - hr.left, padT, labels[i],
      seriesList.map(s => ({
        value: fmt(s.values[i], opts.yDecimals ?? 0), name: s.label, color: css(s.color),
      })));
  });
  hit.addEventListener('pointerleave', () => {
    cross.setAttribute('opacity', 0);
    dots.forEach(d => d.setAttribute('opacity', 0));
    hideTip(tt);
  });

  const step = Math.max(1, Math.ceil(labels.length / 60));   // keep the twin readable
  tableTwin(host, [opts.xName || 'x', ...seriesList.map(s => s.label)],
    labels.filter((_, i) => i % step === 0)
      .map((lab, k) => {
        const i = k * step;
        return [lab, ...seriesList.map(s => fmt(s.values[i], opts.yDecimals ?? 2))];
      }));
}

export { fmt };
