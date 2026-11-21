import {DEFAULTS, ageLaw, returnCdf} from './engine.mjs';
import {presets, applyPreset} from './models.mjs';
import {contextSeries, survivalView, pingProfile} from './charts.mjs';
import {EXPORT_SCHEMA, parseLink, parseFile, resolveScenario, shareHash} from './scenario.mjs';

const $ = id => document.getElementById(id), form = $('scenario'), tip = $('tip');
const KINDS = [['input', 'Questions'], ['retrieval', 'Retrieved text'], ['output', 'Answers']];
const PRICES_CHECKED = '2026-11-20', BASELINE_COMMIT = 'c49b553b85b82bc700465d4607949574f892366c';
// Why the planner sends no ping after a turn, in the page's words.
const WHY_NO_PING = {'disabled': 'keep-warm is off', 'no pings allowed': 'pings per wait is 0', 'ping interval is not positive': 'the ping lead is longer than the cache lifetime', 'prefix is below cache minimum': 'the request is too short to cache',
  'prefix exceeds keep-warm ceiling': 'the next turn rewrites the request anyway', 'no modeled future return': 'no further turn is expected', 'no positive expected value': 'no ping pays for itself'};

// Numbers as text.
const fixed = digits => new Intl.NumberFormat('en-US', {maximumFractionDigits: digits});
const num = n => Number.isFinite(n) ? fixed(1).format(n) : '—';
const pct = n => n === null || !Number.isFinite(n) ? '—' : `${num(n * 100)}%`;
const compact = n => n >= 1e6 ? `${fixed(2).format(n / 1e6)}M` : n >= 1000 ? `${num(n / 1000)}k` : num(n);
const usd = (n, digits) => new Intl.NumberFormat('en-US', {style: 'currency', currency: 'USD', minimumFractionDigits: digits, maximumFractionDigits: digits}).format(n);
// Whole dollars from $10 and cents below. An amount that cents would round to nothing says so.
const dollars = n => { if (!Number.isFinite(n)) return '—'; const a = Math.abs(n); return a > 0 && a < 0.005 ? `${n < 0 ? '−' : ''}<$0.01` : usd(n, a === 0 || a >= 10 ? 0 : 2); };
const esc = text => String(text).replace(/[&<>"]/g, ch => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'})[ch]);
const plural = (n, word) => `${num(n)} ${word}${n === 1 ? '' : 's'}`;

let current = null, baseConfig = DEFAULTS, latest = 0, timer, worker = null;
let priorTokens = DEFAULTS.tokenMedian, priorLength = DEFAULTS.plannedTurns, retrievalSpread = DEFAULTS.tokenP95 / DEFAULTS.tokenMedian;
// Where the scenario came from, whether it was changed since, and a load failure that stays on screen until the next edit.
let source = '', edited = false, loadError = '';

// A field shown in percent stores a share. Twelve significant digits survive the trip both ways.
const tidy = n => Number(n.toPrecision(12));
const shown = (el, value) => el.dataset.scale ? tidy(value * Number(el.dataset.scale)) : value;
const stored = el => el.dataset.scale ? tidy(Number(el.value) / Number(el.dataset.scale)) : Number(el.value);
const sliders = [...form.querySelectorAll('[data-slider]')];
const syncSliders = () => { for (const slider of sliders) slider.value = form.elements[slider.dataset.slider].value; };

// A setting's name as the page labels it, for error messages.
function labelOf(key) {
  const el = key.startsWith('model.') ? form.querySelector(`[data-model="${key.slice(6)}"]`) : form.elements[key];
  const text = el?.getAttribute?.('aria-label') || el?.closest?.('label')?.firstChild?.textContent || '';
  return text.trim().replace(/\s+/g, ' ') || key;
}
const readable = message => message.replace(/\b(?:model\.)?[A-Za-z][A-Za-z0-9]*\b/g, word => (word !== 'model' && Object.hasOwn(DEFAULTS, word)) || (word.startsWith('model.') && Object.hasOwn(DEFAULTS.model, word.slice(6))) ? `“${labelOf(word)}”` : word);

function setForm(c) {
  baseConfig = {...DEFAULTS, ...c, model: {...DEFAULTS.model, ...c.model}};
  for (const el of form.elements) {
    if (el.name && baseConfig[el.name] !== undefined) { if (el.type === 'checkbox') el.checked = baseConfig[el.name]; else el.value = shown(el, baseConfig[el.name]); }
    if (el.dataset.model) el.value = baseConfig.model[el.dataset.model];
  }
  priorTokens = Number(form.elements.tokenMedian.value); priorLength = Number(form.elements.plannedTurns.value);
  if (priorTokens > 0) retrievalSpread = Number(form.elements.tokenP95.value) / priorTokens;
  labels();
}
function readForm() {
  const c = {...DEFAULTS, ...baseConfig, model: {...DEFAULTS.model, ...baseConfig.model}};
  for (const el of form.elements) {
    if (el.name) c[el.name] = el.type === 'checkbox' ? el.checked : el.tagName === 'SELECT' ? el.value : stored(el);
    if (el.dataset.model) c.model[el.dataset.model] = Number(el.value);
  }
  return applyPreset(c, $('modelPreset').value);
}
function labels() {
  const fields = form.elements, kind = fields.lengthDistribution.value, noRetrieval = Number(fields.tokenMedian.value) === 0;
  fields.plannedTurnsP95.disabled = kind !== 'lognormal';
  $('lengthLabel').firstChild.textContent = kind === 'geometric' ? 'Mean turns' : kind === 'fixed' ? 'Turns' : 'Median turns';
  for (const name of ['tokenP95', 'retrievalDependencyProbability', 'retrievalDependencyTurnsBack', 'retrievalDependencyTurnsBackP95']) fields[name].disabled = noRetrieval;
  $('retrievalCaption').hidden = !noRetrieval; $('retrievalCaption').textContent = noRetrieval ? 'No retrieved text, so no later turn can need any.' : '';
  syncSliders();
}
const busy = value => document.querySelectorAll('.results').forEach(el => el.setAttribute('aria-busy', value));
function invalidate(message) {
  current = null; busy('false');
  $('selectedPolicy').textContent = 'No policy fitted. Check the inputs.';
  $('errors').textContent = message; $('errors').hidden = false; $('status').textContent = 'Check inputs';
  for (const id of ['compacheCost', 'openclawCost', 'savingPercent', 'compacheKept', 'openclawKept', 'compacheHit', 'openclawHit']) $(id).textContent = '—';
  $('savingAmount').textContent = ''; $('export').disabled = true; $('share').disabled = true;
}
function update() {
  clearTimeout(timer); worker?.terminate(); latest++; labels();
  if (!form.checkValidity()) {
    const bad = Array.from(form.elements).find(e => e.validity && !e.validity.valid);
    invalidate(`${labelOf(bad?.dataset.model ? `model.${bad.dataset.model}` : bad?.name || '') || 'Input'}: ${bad?.validationMessage || 'check this value'}`);
    return;
  }
  current = null; $('errors').textContent = loadError; $('errors').hidden = !loadError;
  $('status').textContent = 'Fitting…'; $('selectedPolicy').textContent = 'Fitting the cheapest policy…';
  busy('true'); $('export').disabled = true; $('share').disabled = true;
  const c = readForm();
  worker = new Worker(new URL('./worker.mjs', import.meta.url), {type: 'module'});
  worker.onmessage = ({data}) => { if (data.id !== latest) return; if (data.error) { invalidate(readable(data.error)); return; } render(data.result); };
  worker.onerror = () => invalidate('Could not load the calculator. Serve this folder over HTTP and reload.');
  worker.postMessage({id: latest, config: c});
}

// Chart pieces. Every colour is a class that resolves to a design token in style.css.
// Clean axis values from lo to at least hi, about `count` steps apart.
function ticks(lo, hi, count = 4) {
  const raw = (hi - lo || 1) / count, mag = 10 ** Math.floor(Math.log10(raw)), norm = raw / mag;
  const step = (norm <= 1 ? 1 : norm <= 2 ? 2 : norm <= 2.5 ? 2.5 : norm <= 5 ? 5 : 10) * mag, first = Math.floor(lo / step + 1e-9), values = [];
  for (let i = first; ; i++) { const v = tidy(i * step); values.push(v); if (v >= hi - step * 1e-9) break; }
  let digits = 0; while (digits < 6 && Math.abs(step * 10 ** digits - Math.round(step * 10 ** digits)) > 1e-9) digits++;
  return {step, values, digits};
}
// A bar from the baseline y0 to the value y1, rounded at the data end only.
function bar(x, w, y0, y1) {
  const r = Math.min(4, w / 2, Math.abs(y1 - y0)), toward = y1 < y0 ? -1 : 1;
  return `M${x},${y0}V${y1 - toward * r}Q${x},${y1} ${x + r},${y1}H${x + w - r}Q${x + w},${y1} ${x + w},${y1 - toward * r}V${y0}Z`;
}
function showTip(anchor, title, rows) {
  tip.replaceChildren();
  const head = document.createElement('h4'); head.textContent = title; tip.append(head);
  for (const [key, value, label] of rows) {
    const row = document.createElement('div'), mark = document.createElement('i'), strong = document.createElement('b'), name = document.createElement('span');
    if (key) mark.className = key; strong.textContent = value; name.textContent = label; row.append(mark, strong, name); tip.append(row);
  }
  tip.hidden = false;
  // At a pointer, or beside a focused mark.
  const box = anchor instanceof Event ? null : anchor.getBoundingClientRect(), px = box ? box.right : anchor.clientX, py = box ? box.top : anchor.clientY, gap = 14;
  let left = px + gap, top = py + gap;
  if (left + tip.offsetWidth > innerWidth - 8) left = (box ? box.left : px) - tip.offsetWidth - gap;
  if (top + tip.offsetHeight > innerHeight - 8) top = py - tip.offsetHeight - gap;
  tip.style.left = `${Math.max(8, left)}px`; tip.style.top = `${Math.max(8, top)}px`;
}
const hideTip = () => { tip.hidden = true; };

// The small curves under the inputs: neutral ink, since they describe the workload and not either side.
function miniLine(id, points, {label = '', xLabel = '', reference = null, referenceLabel = '', referenceLow = false} = {}) {
  const W = 260, H = 70, L = 4, R = 4, T = 6, B = 16, xmax = Math.max(1e-9, ...points.map(p => p.x));
  const x = v => L + v / xmax * (W - L - R), y = v => H - B - v * (H - T - B);
  let svg = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(label)}"><line class="g-axis" x1="${L}" y1="${H - B}" x2="${W - R}" y2="${H - B}"/>`;
  if (reference !== null && reference <= xmax) { const far = x(reference) > W * .6; svg += `<line class="g-axis" x1="${x(reference)}" y1="${T}" x2="${x(reference)}" y2="${H - B}"/><text x="${x(reference) + (far ? -4 : 4)}" y="${referenceLow ? H - B - 5 : T + 8}" text-anchor="${far ? 'end' : 'start'}">${esc(referenceLabel)}</text>`; }
  svg += `<polyline class="line s-ink" points="${points.map(p => `${x(p.x).toFixed(1)},${y(p.y).toFixed(1)}`).join(' ')}"/><text x="${L}" y="${H - 3}">0</text><text x="${W - R}" y="${H - 3}" text-anchor="end">${esc(xLabel)}</text></svg>`;
  $(id).innerHTML = svg;
}
function renderMini(c) {
  // Request size over the first six turns if nothing were removed.
  const W = 260, H = 70, step = c.tokenMedian + c.inputTokens + c.outputTokens, last = c.systemTokens + step * 6;
  let svg = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Request size over six turns with nothing removed, reaching ${compact(last)} tokens"><line class="g-axis" x1="4" y1="54" x2="256" y2="54"/>`;
  for (let i = 0; i < 6; i++) svg += `<path class="f-neutral" d="${bar(8 + i * 42, 24, 54, 54 - Math.max(2, (c.systemTokens + step * (i + 1)) / last * 46))}"/>`;
  $('growthMini').innerHTML = `${svg}<text x="4" y="67">turn 1</text><text x="256" y="67" text-anchor="end">turn 6 · ${compact(last)} tokens</text></svg>`;

  // Of everything later turns need, the share that lies within so many turns back.
  const needs = KINDS.map(([kind]) => ({chance: kind === 'retrieval' && c.tokenMedian === 0 ? 0 : c[`${kind}DependencyProbability`], law: ageLaw(c[`${kind}DependencyTurnsBack`], c[`${kind}DependencyTurnsBackP95`], 1000), far: c[`${kind}DependencyTurnsBackP95`]})).filter(n => n.chance > 0);
  const weight = needs.reduce((sum, n) => sum + n.chance, 0), within = age => needs.reduce((sum, n) => sum + n.chance * n.law.cdf(age), 0) / weight;
  if (weight > 0) {
    const reach = Math.max(2, Math.ceil(Math.max(...needs.map(n => n.far)))); let half = 1; while (half < 1000 && within(half) < .5) half++;
    miniLine('needsMini', Array.from({length: reach + 1}, (_, age) => ({x: age, y: within(age)})), {label: 'Share of needed earlier text within so many turns back', xLabel: `${reach} turns back`});
    $('needsCaption').textContent = `${pct(1 - needs.reduce((none, n) => none * (1 - n.chance), 1))} of turns need something earlier · half of it within ${plural(half, 'turn')}`;
  } else { $('needsMini').innerHTML = ''; $('needsCaption').textContent = 'No turn needs anything earlier.'; }

  const v = survivalView(c);
  miniLine('survivalMini', v.points.map(p => ({x: p.turn, y: p.realized})), {label: `Share of threads that reach each turn, through turn ${v.max}`, xLabel: `${v.max} turns`, reference: v.reference, referenceLabel: `${v.referenceLabel} ${num(v.reference)}`});
  $('survivalCaption').textContent = `${pct(v.tail)} of threads reach turn ${v.max}`;

  const reach = Math.max(30, c.ttlMinutes, c.gapP95Minutes);
  miniLine('returnMini', Array.from({length: 101}, (_, i) => ({x: i * reach / 100, y: returnCdf(i * reach / 100, c)})), {label: 'Share of returning users back within so many minutes', xLabel: `${num(reach)} min`, reference: c.ttlMinutes, referenceLabel: 'cache expires', referenceLow: true});
  $('returnCaption').textContent = `${pct(returnCdf(c.ttlMinutes, c))} back within the cache lifetime of ${num(c.ttlMinutes)} min`;
}

function contextChart() {
  if (!current) return;
  const id = $('threadPicker').value, average = id === 'average', selected = Number($('runPicker').value), c = current.config, total = current.workload.threads.length;
  const [ours, theirs] = contextSeries(current, id), all = [...ours, ...theirs];
  // A wide chart names the thresholds in a gutter on the right, clear of the data.
  const box = $('contextChart'), W = Math.max(320, box.clientWidth), gutter = W >= 560, H = box.clientHeight || 280, L = 52, R = gutter ? 132 : 14, T = 14, B = 26;
  const maxX = Math.max(2, ...all.map(p => p.turn + 1)), scale = ticks(0, Math.max(1, ...all.map(p => p.inputTokens)), 5), top = scale.values.at(-1);
  const x = n => L + (n - 1) / (maxX - 1) * (W - L - R), y = n => H - B - n / top * (H - T - B);
  let svg = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${average ? 'Mean input tokens per turn across threads' : `Input tokens per turn for ${esc(id)}`}, Compache and OpenClaw">`;
  for (const v of scale.values) svg += `<line class="${v ? 'g-grid' : 'g-axis'}" x1="${L}" x2="${W - R}" y1="${y(v)}" y2="${y(v)}"/><text x="${L - 8}" y="${y(v) + 4}" text-anchor="end">${compact(v)}</text>`;
  // Where each side starts rewriting the request, when that lies inside the plotted range.
  const openclawAt = c.toolClearing ? Math.max(c.clearTriggerFloor, Math.floor(c.model.context * c.clearTriggerShare)) : Math.min(c.summaryTriggerShare * c.model.context, c.model.context - Math.min(c.baselineReserve, .25 * c.model.context));
  const marks = [['compache', c.warmTrigger * c.model.context, 'compacts warm'], ['compache', (c.target + .02) * c.model.context, 'compacts cold'], ['baseline', openclawAt, c.toolClearing ? 'clears results' : 'summarizes']].filter(m => m[1] <= top).sort((m, n) => n[1] - m[1]);
  let marksSvg = '', taken = -Infinity;
  marks.forEach(([cls, value, text], i) => {
    svg += `<line class="g-axis" x1="${L}" x2="${W - R}" y1="${y(value)}" y2="${y(value)}"/>`;
    if (gutter) {
      // Beside its line, pushed down just enough to clear the label above.
      const base = Math.max(y(value) + 4, taken + 14); taken = base;
      marksSvg += `<rect class="f-${cls}" x="${W - R + 8}" y="${base - 7}" width="7" height="7" rx="1.5"/><text class="t-2" x="${W - R + 19}" y="${base}">${text} · ${compact(value)}</text>`;
    } else {
      // Inside the plot: above its line, or below when the line above is too close for that.
      const below = i > 0 && y(value) - y(marks[i - 1][1]) < 15, base = y(value) + (below ? 12 : -5);
      marksSvg += `<rect class="f-${cls} halo" x="${L + 6}" y="${base - 7}" width="7" height="7" rx="1.5"/><text class="t-2 halo" x="${L + 17}" y="${base}">${text} · ${compact(value)}</text>`;
    }
  });
  const xTicks = [...new Set([1, Math.round(maxX / 4), Math.round(maxX / 2), Math.round(3 * maxX / 4), maxX].filter(n => n > 0))];
  for (const n of xTicks) svg += `<text x="${x(n)}" y="${H - B + 17}" text-anchor="middle">${n}</text>`;
  if (ours.some(p => p.turn === selected)) svg += `<line class="g-axis" x1="${x(selected + 1)}" x2="${x(selected + 1)}" y1="${T}" y2="${H - B}"/>`;
  const rewrites = {compache: new Set(current.compache.decisions.filter(d => d.thread === id && d.removedTokens > 0).map(d => d.turn)), baseline: new Set(current.openclaw.ledger.filter(p => p.thread === id && p.role === 'summary' && p.admitted).map(p => p.turn))};
  for (const [cls, points] of [['baseline', theirs], ['compache', ours]]) {
    if (points.length > 1) svg += `<polyline class="line s-${cls}" points="${points.map(p => `${x(p.turn + 1).toFixed(1)},${y(p.inputTokens).toFixed(1)}`).join(' ')}"/>`;
    // Dots at the end of the line, at the inspected turn and, for one thread, wherever the request was rewritten.
    for (const [i, p] of points.entries()) if (i === points.length - 1 || p.turn === selected || (!average && rewrites[cls].has(p.turn))) svg += `<circle class="dot f-${cls}" cx="${x(p.turn + 1)}" cy="${y(p.inputTokens)}" r="4"/>`;
  }
  svg += `${marksSvg}<line class="g-cross" id="contextCross" y1="${T}" y2="${H - B}" visibility="hidden"/><rect class="hit" x="${L}" y="${T}" width="${W - L - R}" height="${H - T - B}"/></svg>`;
  box.innerHTML = svg;
  const drawn = box.firstElementChild, cross = drawn.querySelector('#contextCross'), area = drawn.querySelector('.hit');
  const turnAt = e => { const r = drawn.getBoundingClientRect(); return Math.max(1, Math.min(maxX, Math.round(1 + ((e.clientX - r.left) * W / r.width - L) / (W - L - R) * (maxX - 1)))); };
  area.addEventListener('pointermove', e => {
    const turn = turnAt(e), a = ours.find(p => p.turn + 1 === turn), b = theirs.find(p => p.turn + 1 === turn);
    cross.setAttribute('x1', x(turn)); cross.setAttribute('x2', x(turn)); cross.setAttribute('visibility', 'visible');
    showTip(e, `Turn ${turn} · input tokens${average && a ? ` · ${a.count} of ${total} threads` : ''}`, [['compache', a ? num(Math.round(a.inputTokens)) : '—', 'Compache'], ['openclaw', b ? num(Math.round(b.inputTokens)) : '—', 'OpenClaw']]);
  });
  area.addEventListener('pointerleave', () => { cross.setAttribute('visibility', 'hidden'); hideTip(); });
  area.addEventListener('click', e => { const turn = turnAt(e) - 1; if ([...$('runPicker').options].some(o => Number(o.value) === turn)) { $('runPicker').value = turn; hideTip(); renderPing(); } });
  $('contextCaption').textContent = average ? 'Mean request size among the threads that reach each turn. Click a turn to inspect its ping plan.' : 'Request size at each turn. Dots mark a rewritten request. Click a turn to inspect its ping plan.';

  $('threadCountPanel').hidden = !average;
  if (!average) return;
  const CH = $('threadCountChart').clientHeight || 150, CT = 8, CB = 30, counts = ticks(0, Math.max(1, total), 4), cy = n => CH - CB - n / counts.values.at(-1) * (CH - CT - CB);
  const slot = (W - L - R) / (maxX - 1), bw = Math.max(1, Math.min(20, slot * .65));
  let bars = `<svg viewBox="0 0 ${W} ${CH}" role="img" aria-label="Number of threads that reach each turn, out of ${total}">`;
  for (const n of counts.values) bars += `<line class="${n ? 'g-grid' : 'g-axis'}" x1="${L}" x2="${W - R}" y1="${cy(n)}" y2="${cy(n)}"/><text x="${L - 8}" y="${cy(n) + 4}" text-anchor="end">${num(n)}</text>`;
  for (const [i, p] of ours.entries()) bars += `<rect class="hit" data-i="${i}" x="${x(p.turn + 1) - slot / 2}" y="${CT}" width="${slot}" height="${CH - CT - CB}"/><path class="mark ${p.turn === selected ? 'f-neutral-strong' : 'f-neutral'}" d="${bar(x(p.turn + 1) - bw / 2, bw, cy(0), cy(p.count))}"/>`;
  for (const n of xTicks) bars += `<text x="${x(n)}" y="${CH - 13}" text-anchor="middle">${n}</text>`;
  $('threadCountChart').innerHTML = `${bars}<text x="${W - R}" y="${CH - 1}" text-anchor="end">turn</text></svg>`;
  const countSvg = $('threadCountChart').firstElementChild;
  countSvg.addEventListener('pointermove', e => { const p = ours[e.target.dataset?.i]; if (p) showTip(e, `Turn ${p.turn + 1}`, [['', `${num(p.count)} of ${num(total)}`, `threads · ${pct(p.count / total)}`]]); else hideTip(); });
  countSvg.addEventListener('pointerleave', hideTip);
  const picked = ours.find(p => p.turn === selected);
  $('threadCountNote').textContent = picked ? `Turn ${selected + 1}: ${picked.count} of ${total} threads · ${pct(picked.count / total)}` : 'No thread reaches this turn';
}
function chooseThread() {
  if (!current) return;
  const id = $('threadPicker').value, previous = Number($('runPicker').value);
  const turns = [...new Set(current.compache.decisions.filter(d => id === 'average' || d.thread === id).map(d => d.turn))].sort((a, b) => a - b);
  $('runPicker').replaceChildren(...turns.map(turn => { const o = document.createElement('option'); o.value = turn; o.textContent = `Turn ${turn + 1}`; return o; }));
  $('runPicker').value = turns.includes(previous) ? previous : turns[0] ?? '';
  renderPing();
}
function renderPing() {
  if (!current) return;
  contextChart();
  const id = $('threadPicker').value, average = id === 'average', turn = Number($('runPicker').value), p = pingProfile(current, id, turn);
  // Amounts are per 1,000 waits after this turn, the page's unit for cost.
  const rows = p.rows.map(r => ({pings: r.pings, gain: 1000 * r.expectedGain, cost: 1000 * r.expectedCost, net: 1000 * r.net}));
  $('pingScope').textContent = average ? `Average of ${p.eligible} of ${p.decisions} threads at this turn` : id;
  $('pingChoiceLabel').textContent = average ? 'Best average plan' : 'Plan';
  $('pingChoice').textContent = p.eligible ? plural(p.best.pings, 'ping') : 'No ping possible';
  $('pingValue').textContent = p.eligible ? dollars(1000 * p.best.net) : '—';
  $('pingTiming').textContent = p.eligible ? (p.best.pings ? `every ${fixed(2).format(p.interval)} min` : 'no ping pays for itself') : '';
  $('pingReturn').textContent = `${pct(p.returnProbability)} chance of another turn`;
  $('pingActual').textContent = p.decisions ? `${num(p.plannedPings)} → ${num(p.actualPings)}` : '—';
  const reasons = Object.entries(p.reasons).map(([reason, n]) => `${plural(n, 'thread')}: ${WHY_NO_PING[reason] || reason}`).join(' · ');
  $('pingExplanation').hidden = !reasons && p.decisions > 0; $('pingExplanation').textContent = p.decisions ? (reasons ? `No ping for ${reasons}.` : '') : 'No completed turn to inspect.';

  const box = $('pingChart'), W = Math.max(320, box.clientWidth), H = box.clientHeight || 260, L = 56, R = 14, T = 14, B = 40;
  if (rows.length < 2) { box.innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="No ping plan to chart"><text class="t-2" x="${W / 2}" y="${H / 2}" text-anchor="middle">No ping plan to chart at this turn</text></svg>`; return; }
  const scale = ticks(Math.min(0, ...rows.map(r => r.net)), Math.max(1e-9, ...rows.flatMap(r => [r.gain, r.cost, r.net])), 4), low = scale.values[0], high = scale.values.at(-1);
  const y = v => T + (high - v) / (high - low) * (H - T - B), slot = (W - L - R) / rows.length, x = i => L + (i + .5) * slot, bw = Math.min(24, slot * .55);
  let svg = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Net saving, saving from a warm return and cost of pings for zero to ${rows.at(-1).pings} pings, per 1,000 waits">`;
  for (const v of scale.values) svg += `<line class="${v ? 'g-grid' : 'g-axis'}" x1="${L}" x2="${W - R}" y1="${y(v)}" y2="${y(v)}"/><text x="${L - 8}" y="${y(v) + 4}" text-anchor="end">${usd(v, scale.digits)}</text>`;
  rows.forEach((r, i) => {
    svg += `<rect class="hit" tabindex="0" data-i="${i}" x="${L + i * slot}" y="${T}" width="${slot}" height="${H - T - B}" aria-label="${plural(r.pings, 'ping')}: net ${dollars(r.net)}, saving ${dollars(r.gain)}, cost ${dollars(r.cost)}"/>`;
    if (Math.abs(y(r.net) - y(0)) >= .5) svg += `<path class="mark ${r.pings === p.best.pings ? 'f-compache' : 'f-compache-soft'}" d="${bar(x(i) - bw / 2, bw, y(0), y(r.net))}"/>`;
    svg += `<text x="${x(i)}" y="${H - B + 17}" text-anchor="middle">${r.pings}</text>`;
  });
  for (const [key, cls] of [['gain', '2'], ['cost', '3']]) svg += `<polyline class="line s-${cls}" points="${rows.map((r, i) => `${x(i).toFixed(1)},${y(r[key]).toFixed(1)}`).join(' ')}"/><circle class="dot f-${cls}" cx="${x(rows.length - 1)}" cy="${y(rows.at(-1)[key])}" r="4"/>`;
  box.innerHTML = `${svg}<text x="${(L + W - R) / 2}" y="${H - 4}" text-anchor="middle">pings while waiting</text></svg>`;
  const drawn = box.firstElementChild, describe = (anchor, target) => { const r = rows[target.dataset?.i]; if (!r) return hideTip(); showTip(anchor, plural(r.pings, 'ping'), [['compache', dollars(r.net), 'Net saving'], ['s2', dollars(r.gain), 'Saving from a warm return'], ['s3', dollars(r.cost), 'Cost of pings']]); };
  drawn.addEventListener('pointermove', e => describe(e, e.target)); drawn.addEventListener('pointerleave', hideTip);
  drawn.addEventListener('focusin', e => describe(e.target, e.target)); drawn.addEventListener('focusout', hideTip);
}
function renderCache() {
  // Why answer input was, or was not, read from the cache. Each column sums to 100%.
  const rows = [['cached', 'Read from cache', 'Input that matched a live cache entry.'], ['newContext', 'New this turn', 'Text no request of the thread had carried before: the new question and what was retrieved for it.'], ['firstRequest', 'First turn of a thread', 'Nothing of the thread is cached yet.'], ['expiredPrefix', 'Cache expired', 'It would have matched, but the entry had expired when the user returned.'], ['changedPrefix', 'Request changed', 'Sent before, but no longer at the same place in the request: a rewrite moved or removed what came before it.']];
  const share = (arm, key) => { const mix = arm.cacheBreakdown || {}, total = Object.values(mix).reduce((n, v) => n + v, 0); return total ? (mix[key] || 0) / total : null; };
  const cell = (cls, value) => `<td>${pct(value)}<i class="${cls}" style="--share:${(100 * (value || 0)).toFixed(2)}%"></i></td>`;
  $('cacheRows').innerHTML = rows.map(([key, label, title]) => `<tr><th title="${esc(title)}">${label}</th>${cell('compache', share(current.compache, key))}${cell('openclaw', share(current.openclaw, key))}</tr>`).join('');
}
const sourceLabel = () => source ? `${source}${edited && !source.endsWith('(edited)') ? ' (edited)' : ''}` : '';
function render(r) {
  current = r; busy('false'); $('export').disabled = false; $('share').disabled = false;
  const a = r.compache, b = r.openclaw, c = r.config, opt = r.optimization, den = r.workload.totalRuns, full = a.complete && b.complete && opt.feasible, preset = presets[$('modelPreset').value];
  const per1000 = x => 1000 * x / den, role = (arm, key, field = 'cost') => arm.byRole[key]?.[field] || 0;
  $('status').textContent = `Seed ${c.seed} · updated`;
  $('modelWindow').textContent = `${compact(c.model.context)} context · ${compact(c.model.maxInput)} max input`;
  $('cacheDefault').textContent = `cache lifetime ${num(c.ttlMinutes)} min`;
  $('sourceNote').hidden = !source; $('sourceNote').textContent = source ? `From your traces: ${sourceLabel()}` : '';
  $('workloadNote').textContent = `${preset?.name || 'Custom prices'} · ${num(c.threads)} threads · ${num(den)} turns · policy fitted on ${num(opt.fittedThreads)} other threads`;
  $('baselineNote').textContent = c.toolClearing ? `OpenClaw: tool results cleared above ${pct(c.clearTriggerShare)} of the window, newest ${c.clearKeepResults} kept · no keep-warm` : `OpenClaw: summary at ${pct(c.summaryTriggerShare)} of the window, ${compact(c.tailTokens)} tail · no keep-warm`;

  $('compacheCost').textContent = full ? dollars(per1000(a.totals.cost)) : '—'; $('openclawCost').textContent = full ? dollars(per1000(b.totals.cost)) : '—';
  $('compacheKept').textContent = full ? pct(a.retention.observed) : '—'; $('openclawKept').textContent = full ? pct(b.retention.observed) : '—';
  const hit = arm => arm.byRole.answer?.inputTokens ? arm.byRole.answer.cachedTokens / arm.byRole.answer.inputTokens : null;
  $('compacheHit').textContent = full ? pct(hit(a)) : '—'; $('openclawHit').textContent = full ? pct(hit(b)) : '—';
  const dearer = r.savings?.absolute < 0, known = r.savings && r.savings.percent !== null;
  $('savingPercent').textContent = known ? `${dearer ? '−' : ''}${num(Math.abs(r.savings.percent))}%` : '—';
  $('savingAmount').className = `delta ${r.savings ? (dearer ? 'bad' : 'good') : ''}`;
  $('savingAmount').textContent = r.savings ? `${dearer ? '↑' : '↓'} ${dollars(per1000(Math.abs(r.savings.absolute)))} ${dearer ? 'more' : 'less'} per 1,000 turns` : 'No comparison for these inputs';
  $('warnings').hidden = full;
  $('warnings').textContent = full ? '' : !opt.feasible ? 'No candidate policy keeps that much in view within the input limit. Lower the minimum, the turns back or the context per turn.' : `Some requests did not fit the model: Compache finished ${a.completedRuns} of ${den} turns and OpenClaw ${b.completedRuns}. There is no comparable cost.`;

  const cells = (format, x, y) => `<td>${full ? format(x) : '—'}</td><td>${full ? format(y) : '—'}</td>`, group = title => `<tr class="group"><th colspan="3">${title}</th></tr>`;
  const line = (label, format, x, y, cls = '') => `<tr class="${cls}"><th>${label}</th>${cells(format, x, y)}</tr>`, millions = x => num(x / 1e6);
  $('ledgerRows').innerHTML = [group('Cost'),
    line('Answers', dollars, per1000(role(a, 'answer')), per1000(role(b, 'answer'))), line('Keep-warm pings', dollars, per1000(role(a, 'ping')), per1000(role(b, 'ping'))),
    line('Fetching sources back', dollars, per1000(role(a, 'recovery')), per1000(role(b, 'recovery'))), line('Summaries and flushes', dollars, per1000(role(a, 'summary') + role(a, 'flush')), per1000(role(b, 'summary') + role(b, 'flush'))),
    line('Total', dollars, per1000(a.totals.cost), per1000(b.totals.cost), 'total'),
    group('Calls'),
    line('Answers', num, per1000(role(a, 'answer', 'calls')), per1000(role(b, 'answer', 'calls'))), line('Keep-warm pings', num, per1000(a.pings), per1000(b.pings)),
    line('Fetch-backs', num, per1000(role(a, 'recovery', 'calls')), per1000(role(b, 'recovery', 'calls'))), line('Context rewrites', num, per1000(a.compactions), per1000(b.compactions + b.clearings)),
    group('Tokens'),
    line('Input, millions', millions, per1000(a.totals.inputTokens), per1000(b.totals.inputTokens)), line('read from cache, millions', millions, per1000(a.totals.cachedTokens), per1000(b.totals.cachedTokens)),
    line('written at full price, millions', millions, per1000(a.totals.freshTokens), per1000(b.totals.freshTokens)), line('Output, millions', millions, per1000(a.totals.outputTokens), per1000(b.totals.outputTokens)),
    line('Mean answer request', compact, role(a, 'answer', 'inputTokens') / Math.max(1, role(a, 'answer', 'calls')), role(b, 'answer', 'inputTokens') / Math.max(1, role(b, 'answer', 'calls')))].join('');

  $('selectedPolicy').classList.toggle('infeasible', !opt.feasible);
  $('selectedPolicy').textContent = opt.feasible ? `Fitted policy · compact to ${pct(c.target)} when cold · at ${pct(c.warmTrigger)} when warm · always keep ${plural(c.retrievalRetentionRuns, 'turn')}${opt.retention === null ? '' : ` · expected in view ${pct(opt.fittedRetention)} fitted, ${pct(opt.retention)} here`}` : 'No candidate policy meets the minimum.';
  $('searchNote').textContent = `Cheapest of ${opt.candidateCount} on the ${num(opt.fittedThreads)} fitted threads · cost per 1,000 turns`;
  const candidates = opt.candidates.filter(t => t.feasible).sort((x, y) => x.cost - y.cost).slice(0, 8), isChosen = t => t.target === c.target && t.warmTrigger === c.warmTrigger && t.retrievalRetentionRuns === c.retrievalRetentionRuns;
  $('policyRows').innerHTML = candidates.length ? candidates.map(t => `<tr class="${isChosen(t) ? 'selected' : ''}"><td>${pct(t.target)}</td><td>${pct(t.warmTrigger)}</td><td>${t.retrievalRetentionRuns}</td><td>${pct(t.retention)}</td><td>${dollars(1000 * t.cost / opt.fittedRuns)}</td></tr>`).join('') : '<tr><td colspan="5">No candidate meets the minimum</td></tr>';
  const fetched = arm => role(arm, 'recovery', 'calls') / den, gone = arm => arm.rawSourceAvailability === null ? null : 1 - arm.rawSourceAvailability;
  $('retentionRows').innerHTML = [group('Before fetching back'), ...KINDS.map(([kind, label]) => line(label, pct, a.retention.byKind[kind].observed, b.retention.byKind[kind].observed)), line('All', pct, a.retention.observed, b.retention.observed, 'total'),
    group('Fetching back'), line('Turns with a fetch', pct, fetched(a), fetched(b)), line('Still not in view', pct, gone(a), gone(b))].join('');

  const picker = $('threadPicker'), previous = picker.value, everything = document.createElement('option'); everything.value = 'average'; everything.textContent = 'Average across threads';
  picker.replaceChildren(everything, ...r.workload.threads.map(t => { const o = document.createElement('option'); o.value = t.id; o.textContent = `${t.id} · ${plural(t.turns.length, 'turn')}`; return o; }));
  picker.value = previous === 'average' || r.workload.threads.some(t => t.id === previous) ? previous : 'average';
  renderMini(c); chooseThread(); renderCache();
  $('priceNote').textContent = `${preset?.name || 'Custom'}: $${c.model.inputRate} input / $${c.model.cachedRate} cache read / $${c.model.outputRate} output per 1M tokens. A cache write costs ${c.model.writeMultiplier}× input.${c.model.longThreshold > 0 ? ` Above ${compact(c.model.longThreshold)} input: ${c.model.longInputMultiplier}× input and cache, ${c.model.longOutputMultiplier}× output.` : ''}`;
  $('modelSource').href = preset?.url || 'https://platform.claude.com/docs/en/about-claude/pricing';
}

// Editing.
form.addEventListener('submit', e => { e.preventDefault(); update(); });
form.addEventListener('input', e => {
  const el = e.target, fields = form.elements, name = el.dataset.slider || el.name;
  if (el.dataset.slider) fields[name].value = el.value;
  // A price or cache rule typed by hand is no longer the preset's.
  if (el.hasAttribute('data-preset')) $('modelPreset').value = 'custom';
  if (name === 'tokenMedian' && fields.tokenMedian.value !== '') {
    const next = Number(fields.tokenMedian.value);
    if (priorTokens > 0) retrievalSpread = Math.max(1, Number(fields.tokenP95.value) / priorTokens);
    fields.tokenP95.value = next === 0 ? 0 : Math.min(1000000, Math.max(next, Number((next * retrievalSpread).toPrecision(3))));
    priorTokens = next;
  }
  if (name === 'plannedTurns' && fields.lengthDistribution.value === 'lognormal' && Number(fields.plannedTurns.value) > 0) {
    const next = Number(fields.plannedTurns.value), ratio = Math.max(1, Number(fields.plannedTurnsP95.value) / Math.max(1, priorLength));
    fields.plannedTurnsP95.value = Math.min(320, Math.max(next, Math.round(next * ratio))); priorLength = next;
  }
  if (source) edited = true;
  loadError = ''; labels(); worker?.terminate(); latest++; current = null; $('export').disabled = true; $('share').disabled = true; busy('true');
  try { renderMini(readForm()); } catch { /* an unfinished entry: the check in update() reports it */ }
  clearTimeout(timer); timer = setTimeout(update, 250);
});
$('modelPreset').addEventListener('change', () => { if (presets[$('modelPreset').value]) { setForm(applyPreset(readForm(), $('modelPreset').value)); update(); } });
$('threadPicker').addEventListener('change', chooseThread);
$('runPicker').addEventListener('change', renderPing);

// Loading a scenario: defaults, a link, or a file.
function note(text) { $('actionStatus').textContent = text; }
function load(carried, origin) {
  const {preset, config, source: label, legacy} = resolveScenario(carried);
  $('modelPreset').value = preset; source = label; edited = false; loadError = '';
  setForm(config);
  note(legacy ? 'Made with an earlier version of the calculator. Its numbers were computed differently.' : origin === 'file' ? 'Imported.' : '');
  update();
}
function defaults() { $('modelPreset').value = 'sonnet'; source = ''; edited = false; setForm(applyPreset(DEFAULTS, 'sonnet')); }
function start() {
  try { const carried = parseLink(location.hash); if (carried) { load(carried, 'link'); return; } }
  catch (error) { loadError = `This link was not loaded: ${readable(error.message)}. The defaults are shown instead.`; }
  defaults(); update();
}
async function importFile(file) {
  try {
    let data; try { data = JSON.parse(await file.text()); } catch { throw new Error('it is not valid JSON'); }
    load(parseFile(data), 'file'); history.replaceState(null, '', location.pathname);
  } catch (error) { loadError = `${file.name} was not imported: ${readable(error.message)}.`; $('errors').textContent = loadError; $('errors').hidden = false; note(''); }
}
$('reset').addEventListener('click', () => { $('threadPicker').value = 'average'; $('runPicker').value = '0'; loadError = ''; defaults(); history.replaceState(null, '', location.pathname); note(''); update(); });
$('import').addEventListener('click', () => $('importFile').click());
$('importFile').addEventListener('change', e => { const file = e.target.files[0]; e.target.value = ''; if (file) importFile(file); });
document.addEventListener('dragover', e => { if (e.dataTransfer?.types.includes('Files')) e.preventDefault(); });
document.addEventListener('drop', e => { const file = e.dataTransfer?.files[0]; if (file) { e.preventDefault(); importFile(file); } });
$('share').addEventListener('click', async () => {
  if (!current) return;
  const url = new URL(location.href); url.hash = shareHash(current.config, $('modelPreset').value, sourceLabel()); history.replaceState(null, '', url);
  try { await navigator.clipboard.writeText(url.href); note('Scenario link copied.'); } catch { note('Scenario saved in the address bar. Copy the URL to share.'); }
});
$('export').addEventListener('click', () => {
  if (!current) return;
  const blob = new Blob([JSON.stringify({schemaVersion: EXPORT_SCHEMA, sourceDate: PRICES_CHECKED, baselineCommit: BASELINE_COMMIT, modelPreset: $('modelPreset').value, source: sourceLabel(), ...current}, null, 2)], {type: 'application/json'});
  const url = URL.createObjectURL(blob), a = document.createElement('a'); a.href = url; a.download = `compache-scenario-${current.config.seed}.json`; a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  note('Exported settings, workload and cost ledger.');
});
window.addEventListener('hashchange', start);
let resizeTimer; window.addEventListener('resize', () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(() => { if (current) renderPing(); }, 100); });
start();
