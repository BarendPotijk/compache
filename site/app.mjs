import {BINS, DEFAULTS, DRAWN, WINDOW, configure, describePolicy, drawnStats, longAvailable, lawWeights, needLaw, warmReturnShare} from './engine.mjs';
import {RULES, comparisons, configureRules} from './baselines.mjs';
import {presets, applyPreset} from './models.mjs';
import {contextSeries, coverage, ticks} from './charts.mjs';
import {EXPORT_SCHEMA, LINK_VERSION, parseLink, parseFile, resolveScenario, shareHash} from './scenario.mjs';
import {STARTS} from './starts.mjs';
import {isLabel, isThread, measure, parseRows} from './traces.mjs';
import {isPolicy, readPolicy} from './policy.mjs';

const $ = id => document.getElementById(id), form = $('scenario'), tip = $('tip');
const KINDS = [['input', 'Questions'], ['retrieval', 'Retrieved text'], ['output', 'Answers']];
const PRICES_CHECKED = '2026-11-20';
// Why the planner sends no ping after a turn, in the page's words.
const WHY_NO_PING = {'disabled': 'Keep-warm is off.', 'no pings allowed': 'This policy sends none.', 'ping interval is not positive': 'The ping lead is longer than the cache lifetime.', 'answer outlasts the ping interval': 'The answer is still running when the first ping is due.', 'prefix is below cache minimum': 'The request is too short to cache.',
  'the request leaves no room for a ping': 'The request leaves no room for one.', 'no modeled future return': 'No further turn is expected.', 'no positive expected value': 'No ping pays for itself.'};

// Numbers as text.
const fixed = digits => new Intl.NumberFormat('en-US', {maximumFractionDigits: digits});
const num = n => Number.isFinite(n) ? fixed(1).format(n) : '—';
const whole = n => Number.isFinite(n) ? fixed(0).format(n) : '—';
const pct = (n, digits = 1) => n === null || !Number.isFinite(n) ? '—' : `${fixed(digits).format(n * 100)}%`;
const compact = n => n >= 1e6 ? `${fixed(2).format(n / 1e6)}M` : n >= 1000 ? `${num(n / 1000)}k` : num(n);
const usd = (n, digits) => new Intl.NumberFormat('en-US', {style: 'currency', currency: 'USD', minimumFractionDigits: digits, maximumFractionDigits: digits}).format(n);
// Whole dollars from $100, cents below. An amount that cents would round to nothing says so.
const dollars = (n, cents = false) => { if (!Number.isFinite(n)) return '—'; const a = Math.abs(n); return a > 0 && a < 0.005 ? `${n < 0 ? '−' : ''}<$0.01` : usd(n, !cents && (a === 0 || a >= 100) ? 0 : 2); };
const esc = text => String(text).replace(/[&<>"]/g, ch => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'})[ch]);
const plural = (n, word) => `${num(n)} ${word}${n === 1 ? '' : 's'}`;
const minutes = m => m < 1 ? `${num(m * 60)} s` : m < 90 ? `${num(m)} min` : m < 2880 ? `${num(m / 60)} h` : `${num(m / 1440)} d`;

let current = null, baseConfig = DEFAULTS, latest = 0, timer, worker = null, running = false;
let priorTokens = 0, priorLength = 0, retrievalSpread = 1.8;
// Where the scenario came from, whether it was changed since, and a load failure that stays on screen until the next edit.
let source = '', edited = false, loadError = '', startId = '';
// What was loaded from files, the policy being tried, and which of the three is priced: fit, file or own.
let rows = [], labelRows = [], filePolicy = null, ownPolicy = null, mode = 'fit', turnIndex = 0;

// A field shown in percent stores a share. Twelve significant digits survive the trip both ways.
const tidy = n => Number(n.toPrecision(12));
const shown = (el, value) => el.dataset.scale ? tidy(value * Number(el.dataset.scale)) : value;
const stored = el => el.dataset.scale ? tidy(Number(el.value) / Number(el.dataset.scale)) : Number(el.value);
const ruleFields = [...form.querySelectorAll('[data-rule]')];
const readField = el => el.type === 'checkbox' ? el.checked : el.tagName === 'SELECT' ? el.value : stored(el);
const writeField = (el, value) => { if (el.type === 'checkbox') el.checked = value; else el.value = shown(el, value); };

// A setting's name as the page labels it, for error messages.
function labelOf(key) {
  const el = key.startsWith('model.') ? form.querySelector(`[data-model="${key.slice(6)}"]`) : key.startsWith('app.') ? form.querySelector(`[data-rule="${key.slice(4)}"]`) : form.elements[key];
  const text = el?.getAttribute?.('aria-label') || el?.closest?.('label')?.firstChild?.textContent || '';
  return text.trim().replace(/\s+/g, ' ') || key;
}
const readable = message => message.replace(/\b(?:(?:model|app)\.)?[A-Za-z][A-Za-z0-9]*\b/g, word => (!['model', 'app', 'policy', 'compare'].includes(word) && Object.hasOwn(DEFAULTS, word)) || (word.startsWith('model.') && Object.hasOwn(DEFAULTS.model, word.slice(6))) || (word.startsWith('app.') && Object.hasOwn(RULES, word.slice(4))) ? `“${labelOf(word)}”` : word);

// The numbers beside a drawn chart are its own.
const STATS = {tokenDrawn: ['tokenMedian', 'tokenP95'], gapDrawn: ['gapMedianMinutes', 'gapP95Minutes'], lengthDrawn: ['plannedTurns', 'plannedTurnsP95'], inputNeedDrawn: ['inputDependencyTurnsBack', 'inputDependencyTurnsBackP95'], retrievalNeedDrawn: ['retrievalDependencyTurnsBack', 'retrievalDependencyTurnsBackP95'], outputNeedDrawn: ['outputDependencyTurnsBack', 'outputDependencyTurnsBackP95']};
// The numbers of the curve a drawing replaced, for when the curve is taken again.
const curveOf = {};
function remember(name) { if (!baseConfig[name]) curveOf[name] = STATS[name].map(field => form.elements[field].value); }
function showStats(name) {
  const weights = baseConfig[name]; if (!weights) return;
  const {median, p95} = drawnStats(name, weights), round = v => DRAWN[name] === 'token' ? Math.round(v) : DRAWN[name] === 'gap' ? Number(v.toPrecision(3)) : Math.round(v);
  const [low, high] = STATS[name]; form.elements[low].value = round(median); form.elements[high].value = Math.max(round(median), round(p95));
}
function setForm(c) {
  baseConfig = {...DEFAULTS, ...c, model: {...DEFAULTS.model, ...c.model}};
  for (const el of form.elements) {
    if (el.name && baseConfig[el.name] !== undefined) writeField(el, baseConfig[el.name]);
    if (el.dataset.model) el.value = baseConfig.model[el.dataset.model];
  }
  const rules = configureRules(baseConfig.app); for (const el of ruleFields) writeField(el, rules[el.dataset.rule]);
  priorTokens = Number(form.elements.tokenMedian.value); priorLength = Number(form.elements.plannedTurns.value);
  if (priorTokens > 0) retrievalSpread = Number(form.elements.tokenP95.value) / priorTokens;
  labels();
}
function readForm() {
  const c = {...DEFAULTS, ...baseConfig, model: {...DEFAULTS.model, ...baseConfig.model}};
  for (const el of form.elements) {
    if (el.name && !el.dataset.rule) c[el.name] = readField(el);
    if (el.dataset.model) c.model[el.dataset.model] = Number(el.value);
  }
  c.app = Object.fromEntries(ruleFields.map(el => [el.dataset.rule, readField(el)]).filter(([key, value]) => value !== RULES[key]));
  c.policy = mode === 'file' && filePolicy ? filePolicy.policy : mode === 'own' ? ownPolicy : null;
  return applyPreset(c, $('modelPreset').value);
}
function labels() {
  const f = form.elements, drawn = !!baseConfig.lengthDrawn, kind = f.lengthDistribution.value, byChance = kind === 'continuation' && !drawn;
  const noRetrieval = !baseConfig.tokenDrawn && Number(f.tokenMedian.value) === 0 || (f.retrievalShare.value !== '' && Number(f.retrievalShare.value) === 0);
  $('lengthByLaw').hidden = byChance; $('lengthByChance').hidden = !byChance;
  for (const name of ['plannedTurns']) f[name].disabled = byChance;
  f.continuationProbability.disabled = byChance || drawn; f.lengthDistribution.disabled = drawn;
  for (const name of ['continueFirst', 'continueLong', 'continueHalfTurn']) f[name].disabled = !byChance;
  f.plannedTurnsP95.disabled = !drawn && kind !== 'lognormal';
  $('lengthLabel').firstChild.textContent = drawn ? 'Median turns' : kind === 'geometric' ? 'Mean turns' : kind === 'fixed' ? 'Turns' : 'Median turns';
  for (const name of ['retrievalDependencyProbability', 'retrievalDependencyTurnsBack', 'retrievalDependencyTurnsBackP95']) f[name].disabled = noRetrieval;
  f.gapHalfTurn.disabled = Number(f.gapFirstFactor.value) === 1;
  f.breakMedianMinutes.disabled = f.breakShare.value !== '' && Number(f.breakShare.value) === 0;
  $('appSummary').textContent = summary();
  for (const b of $('starts').children) b.setAttribute('aria-pressed', b.dataset.id === startId && !edited);
}
// Your app in a few words.
function summary() {
  const r = Object.fromEntries(ruleFields.map(el => [el.dataset.rule, readField(el)])), parts = [];
  parts.push(r.keepTurns ? `last ${r.keepTurns} turns` : 'whole history');
  if (r.dropRetrievedAfter >= 0) parts.push(`retrieved text dropped after ${plural(r.dropRetrievedAfter, 'turn')}`);
  if (r.compress) parts.push(`retrieved text −${pct(r.compress, 0)}`);
  if (r.clear) parts.push('clears tool results'); if (r.summarize) parts.push('summarizes');
  parts.push(r.cache === 'off' ? 'no caching' : r.cache === 'long' ? 'long cache' : 'cached'); if (r.pings) parts.push(plural(r.pings, 'ping'));
  return parts.join(' · ');
}
const busy = value => document.querySelectorAll('.results').forEach(el => el.setAttribute('aria-busy', value));
function invalidate(message) {
  current = null; busy('false');
  $('errors').textContent = message; $('errors').hidden = false; $('status').textContent = 'Check inputs';
  $('savingPercent').textContent = '—'; $('savingAmount').textContent = ''; $('savingRange').textContent = ''; $('savingFacts').innerHTML = ''; $('sides').innerHTML = '';
  $('baselineNote').textContent = '';
  $('export').disabled = true; $('share').disabled = true;
}
// An idle worker holds the fit for the current workload. Interrupt only an unfinished calculation.
function cancelCalculation() {
  latest++;
  if (running) { worker.terminate(); worker = null; running = false; }
}
function update() {
  clearTimeout(timer); cancelCalculation(); labels();
  if (!form.checkValidity()) {
    const bad = Array.from(form.elements).find(e => e.validity && !e.validity.valid); bad?.closest('details')?.setAttribute('open', '');
    invalidate(`${labelOf(bad?.dataset.model ? `model.${bad.dataset.model}` : bad?.dataset.rule ? `app.${bad.dataset.rule}` : bad?.name || '') || 'Input'}: ${bad?.validationMessage || 'check this value'}`);
    return;
  }
  current = null; $('errors').textContent = loadError; $('errors').hidden = !loadError;
  $('status').textContent = 'Calculating…'; busy('true'); $('export').disabled = true; $('share').disabled = true;
  const id = latest;
  function workerFailed() {
    if (id !== latest) return;
    worker?.terminate(); worker = null; running = false;
    invalidate('The calculator could not start its background worker. Reload the page to try again.');
    $('status').textContent = 'Calculation unavailable';
  }
  try {
    worker ??= new Worker(new URL('./worker.mjs', import.meta.url), {type: 'module'});
    worker.onmessage = ({data}) => {
      if (data.id !== latest) return;
      if (data.candidate) { $('status').textContent = `Fitting policy ${data.candidate} of ${data.total}…`; return; }
      running = false;
      if (data.error) { invalidate(readable(data.error)); return; }
      render(data.result);
    };
    worker.onerror = workerFailed;
    worker.onmessageerror = workerFailed;
    running = true;
    worker.postMessage({id, config: readForm(), reportProgress: true});
  } catch { workerFailed(); }
}
const changed = () => { if (source) edited = true; startId = ''; };
function soon() {
  loadError = ''; labels(); cancelCalculation(); current = null; $('export').disabled = true; $('share').disabled = true; busy('true');
  try { renderInputs(configure(readForm())); } catch { /* an unfinished entry: the check in update() reports it */ }
  clearTimeout(timer); timer = setTimeout(update, 250);
}

// Chart pieces. Every colour is a class that resolves to a design token in style.css.
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


// -- distributions you can draw ---------------------------------------------

const DRAW_TITLE = {tokenDrawn: 'Retrieved tokens per turn', gapDrawn: 'Time until the next message', lengthDrawn: 'Turns in a thread', inputNeedDrawn: 'Turns back to a needed question', retrievalNeedDrawn: 'Turns back to needed retrieved text', outputNeedDrawn: 'Turns back to a needed answer'};
const binLabel = {
  token: k => `${compact(BINS.token.edges[k])} to ${compact(BINS.token.edges[k + 1])} tokens`, gap: k => `${minutes(BINS.gap.edges[k])} to ${minutes(BINS.gap.edges[k + 1])}`,
  length: k => { const a = BINS.length.edges[k], b = BINS.length.edges[k + 1] - 1; return a === b ? plural(a, 'turn') : `${a} to ${b} turns`; }, need: k => { const a = BINS.need.edges[k], b = BINS.need.edges[k + 1] - 1; return a === b ? `${plural(a, 'turn')} back` : `${a} to ${b} turns back`; }
};
// Which bin edges get a label under the chart.
const AXIS = {token: [[1, '125'], [4, '1k'], [7, '8k'], [10, '64k']], gap: [[2, '30 s'], [5, '5 min'], [8, '30 min'], [10, '2 h'], [13, '1 d']], length: [[0, '1'], [4, '5'], [7, '11'], [9, '20'], [12, '45'], [15, '110']], need: [[0, '1'], [4, '5'], [7, '11'], [9, '21'], [11, '51']]};
// Where a value falls along the bins, in bins.
function binAt(kind, value) {
  const {edges} = BINS[kind]; if (value <= edges[0]) return 0; if (value >= edges.at(-1)) return edges.length - 1;
  let k = 0; while (value >= edges[k + 1]) k++;
  return k + Math.log(value / edges[k]) / Math.log(edges[k + 1] / edges[k]);
}
let painting = null;
function drawChart(box, c) {
  const name = box.dataset.draw, kind = DRAWN[name], weights = lawWeights(name, c), total = weights.reduce((a, b) => a + b, 0) || 1, share = weights.map(w => w / total), off = name === 'tokenDrawn' && !c.tokenDrawn && c.tokenMedian === 0 || name.startsWith('retrieval') && !c.tokenDrawn && c.tokenMedian === 0;
  const W = 260, H = 104, T = 8, B = 18, n = share.length, slot = W / n, top = painting?.name === name ? painting.top : Math.max(.08, ...share) * 1.18, y = v => H - B - Math.min(1, v / top) * (H - T - B);
  let svg = `<svg viewBox="0 0 ${W} ${H}" role="group" aria-label="${DRAW_TITLE[name]}. Drag to draw, or use the arrow keys on a bar."><line class="g-axis" x1="0" y1="${H - B}" x2="${W}" y2="${H - B}"/>`;
  share.forEach((s, k) => {
    svg += `<rect class="hit" data-k="${k}" x="${k * slot}" y="0" width="${slot}" height="${H - B}" ${off ? '' : `tabindex="0" role="slider" aria-label="${binLabel[kind](k)}" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${Math.round(s * 100)}" aria-valuetext="${pct(s)}"`}/>`;
    if (!off && s > 0) svg += `<path class="mark ${c[name] ? 'f-drawn' : 'f-neutral'}" d="${bar(k * slot + 1.5, slot - 3, H - B, Math.min(H - B - 1, y(s)))}"/>`;
  });
  for (const [k, text] of AXIS[kind]) svg += `<text x="${BINS[kind].whole ? (k + .5) * slot : k * slot}" y="${H - 4}" text-anchor="${k === 0 && !BINS[kind].whole ? 'start' : 'middle'}">${text}</text>`;
  // The cache lifetime of the policy in force among the delays, less the answer time it has already spent.
  const lives = current?.policy.long && longAvailable(c) ? c.longTtlMinutes : c.ttlMinutes;
  if (kind === 'gap') { const at = binAt('gap', Math.max(.1, lives - c.answerSeconds / 60)) * slot; svg += `<line class="g-mark" x1="${at}" x2="${at}" y1="${T - 4}" y2="${H - B}"/><text class="halo t-2" x="${at + 4}" y="${T + 4}">cache expires</text>`; }
  const caption = off ? 'No retrieved text.' : kind === 'gap' ? `${pct(warmReturnShare({...c, ttlMinutes: lives}), 0)} back before the cache expires` : c[name] ? 'Drawn' : 'Drag the bars to draw your own';
  box.innerHTML = `${svg}</svg><p class="caption"><span>${caption}</span>${c[name] ? `<button type="button" class="link" data-reset="${name}">use the curve</button>` : ''}</p>`;
  if (off) return;
  const el = box.firstElementChild;
  const paint = (k, value) => { const next = [...(baseConfig[name] ?? share)]; next[k] = Math.max(0, value); if (!next.some(w => w > 0)) next[k] = 1e-6; baseConfig = {...baseConfig, [name]: next}; showStats(name); };
  el.addEventListener('pointerdown', e => {
    const k = e.target.dataset?.k; if (k === undefined) return;
    e.preventDefault(); changed(); remember(name);
    baseConfig = {...baseConfig, [name]: [...share]}; painting = {name, top, scale: 1};
    const move = ev => { const r = box.firstElementChild.getBoundingClientRect(), x = (ev.clientX - r.left) / r.width * W, v = Math.max(0, (H - B - (ev.clientY - r.top) / r.height * H) / (H - T - B)) * top, at = Math.max(0, Math.min(n - 1, Math.floor(x / slot))); paint(at, v < top * .02 ? 0 : v); labels(); drawChartRaw(box); };
    // While drawing, heights stay as drawn: the shares are worked out when the pointer is released.
    const drawChartRaw = host => { const raw = baseConfig[name]; host.querySelectorAll('path.mark').forEach(p => p.remove()); const svgEl = host.firstElementChild; raw.forEach((v, i) => { if (v <= 0) return; const p = document.createElementNS('http://www.w3.org/2000/svg', 'path'); p.setAttribute('class', 'mark f-drawn'); p.setAttribute('d', bar(i * slot + 1.5, slot - 3, H - B, Math.min(H - B - 1, y(v)))); svgEl.insertBefore(p, svgEl.querySelector('text')); }); };
    const up = () => { removeEventListener('pointermove', move); removeEventListener('pointerup', up); removeEventListener('pointercancel', up); painting = null; const sum = baseConfig[name].reduce((a, b) => a + b, 0); baseConfig = {...baseConfig, [name]: baseConfig[name].map(w => tidy(Number((w / sum).toPrecision(4))))}; showStats(name); soon(); };
    addEventListener('pointermove', move); addEventListener('pointerup', up); addEventListener('pointercancel', up); hideTip(); move(e);
  });
  el.addEventListener('keydown', e => {
    const k = Number(e.target.dataset?.k), step = {ArrowUp: .02, ArrowDown: -.02, PageUp: .1, PageDown: -.1}[e.key];
    if (step !== undefined) { e.preventDefault(); changed(); remember(name); paint(k, (baseConfig[name] ?? share)[k] + step); soon(); box.querySelector(`[data-k="${k}"]`)?.focus(); }
    else if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') { e.preventDefault(); box.querySelector(`[data-k="${k + (e.key === 'ArrowRight' ? 1 : -1)}"]`)?.focus(); }
  });
  const describe = (anchor, target) => { const k = target.dataset?.k; if (k === undefined || painting) return hideTip(); showTip(anchor, binLabel[kind](Number(k)), [['', pct(share[k]), kind === 'length' ? 'of threads' : kind === 'need' ? 'of these needs' : 'of turns']]); };
  el.addEventListener('pointermove', e => describe(e, e.target)); el.addEventListener('pointerleave', hideTip); el.addEventListener('focusin', e => describe(e.target, e.target)); el.addEventListener('focusout', hideTip);
}
function renderInputs(c) {
  for (const box of form.querySelectorAll('[data-draw]')) drawChart(box, c);
  const none = KINDS.reduce((p, [kind]) => p * (1 - (kind === 'retrieval' && !c.tokenDrawn && c.tokenMedian === 0 ? 0 : c[`${kind}DependencyProbability`])), 1);
  $('needsCaption').textContent = `${pct(1 - none, 0)} of turns need something earlier`;
}

// -- results ----------------------------------------------------------------

const role = (arm, key, field = 'cost') => arm.byRole[key]?.[field] || 0;
function renderLead(r) {
  const ours = r.compache.complete ? 1000 * r.compache.totals.cost / r.workload.totalRuns : null, s = r.savings, known = s && s.percent !== null, dearer = s?.absolute < 0;
  $('savingPercent').textContent = known ? `${dearer ? '−' : ''}${num(Math.abs(s.percent))}%` : '—';
  $('savingAmount').className = `delta ${s ? (dearer ? 'bad' : 'good') : ''}`;
  if (s) {
    const difference = s.absolute === 0 ? 'Same cost' : `${dollars(Math.abs(s.absolute), true)} ${dearer ? 'more' : 'less'}`;
    $('savingAmount').textContent = `${difference} per 1,000 turns`;
  } else {
    $('savingAmount').textContent = r.compache.complete && r.selected.complete ? 'No eligible fitted policy' : 'Requests exceed the context window';
  }
  $('savingRange').textContent = s?.range ? `${pct(s.range.low, 0)} to ${pct(s.range.high, 0)} when the sampled threads are drawn again` : '';
  const facts = [['Fetches trimmed text back', `${pct(role(r.compache, 'recovery', 'calls') / r.workload.totalRuns)} of turns`], ['Needed text in view before that', pct(r.compache.retention.observed, 0)]];
  if (r.selected.missing !== null) facts.push(['Needed text missing with the baseline', pct(r.selected.missing, 0)]);
  if (r.fit.pinned && r.fit.fittedHere !== null) facts.push(['The policy fitted here would cost', dollars(r.fit.fittedHere)]);
  $('savingFacts').innerHTML = facts.map(([k, v]) => `<div><dt>${esc(k)}</dt><dd>${v}</dd></div>`).join('');
  document.querySelectorAll('[data-against]').forEach(el => { el.textContent = r.baseline.name; });
  const picker = $('baselinePicker');
  picker.replaceChildren(...r.sides.map(side => new Option(side.name, side.id)));
  picker.value = r.selected.id;
  picker.disabled = false;
  const costs = [['Baseline', r.selected.per1000], ['Compache', ours]];
  const scale = Math.max(r.selected.per1000 ?? 0, ours ?? 0) || 1;
  $('sides').innerHTML = costs.map(([name, cost]) => `
    <div><dt>${name}</dt><dd>
      <span>${cost === null ? 'Does not fit' : dollars(cost, true)}</span>
      <span class="cost-track" aria-hidden="true"><i style="width:${100 * (cost ?? 0) / scale}%"></i></span>
    </dd></div>`).join('');
}
function renderPolicy(r) {
  const p = r.policy, c = r.config, file = mode === 'file' && filePolicy;
  for (const b of $('policyTabs').children) { b.setAttribute('aria-selected', b.dataset.mode === mode); if (b.dataset.mode === 'file') b.hidden = !filePolicy; }
  $('policyText').textContent = describePolicy(p, c.longTtlMinutes).join(' ');
  $('policyNote').textContent = file ? `${filePolicy.model}${filePolicy.threads ? ` · fitted on ${plural(filePolicy.threads, 'thread')}, ${plural(filePolicy.turns, 'turn')}` : ''}${filePolicy.cost !== null ? ` · there ${dollars(filePolicy.cost)} against ${dollars(filePolicy.everything)} keeping everything` : ''}${filePolicy.coverage !== null ? ` · ${pct(filePolicy.coverage, 0)} of labelled needs in view` : ''}. ${filePolicy.notes.join(' ')}${filePolicy.workload ? '' : ' The file does not say what its workload was: it is priced on the one set here.'}` : mode === 'own' ? 'Drag a window, or set the pings. It is priced like any other policy.' : `The cheapest of ${r.fit.candidates.length} on ${num(r.fit.threads)} threads, priced on ${num(r.workload.threads.length)} others.`;
  if (document.activeElement !== $('ownPings')) $('ownPings').value = p.maxPings;
  $('ownLongRow').hidden = !longAvailable(c); $('ownLong').checked = p.long;
  // One lane per kind: where what later turns need lies, and how much of it the window holds.
  const {edges} = BINS.need, n = edges.length - 1, W = Math.max(320, Math.min(620, $('windowsChart').clientWidth || 520)), narrow = W < 460, L = narrow ? 84 : 98, R = narrow ? 92 : 128, lane = 74, H = lane * 3 + 22, slot = (W - L - R) / n, longest = Math.max(...r.workload.threads.map(t => t.turns.length));
  let svg = `<svg viewBox="0 0 ${W} ${H}" role="group" aria-label="What the policy keeps of each kind, over what later turns need">`;
  KINDS.forEach(([kind, label], row) => {
    const window = p[WINDOW[kind]], none = kind === 'retrieval' && !c.tokenDrawn && c.tokenMedian === 0, law = needLaw(c, kind, 160), share = Array.from({length: n}, (_, k) => law.cdf(Math.min(160, edges[k + 1] - 1)) - (k ? law.cdf(edges[k] - 1) : 0)), top = Math.max(...share, .05), base = lane * row + lane - 14, y = v => base - v / top * (lane - 30);
    svg += `<text class="t-ink" x="0" y="${base - 20}">${label}</text><text x="0" y="${base - 5}">${none ? 'none here' : `${pct(c[`${kind}DependencyProbability`], 0)} of turns`}</text><line class="g-axis" x1="${L}" x2="${W - R}" y1="${base}" y2="${base}"/>`;
    const reach = window === null ? n : (() => { let k = 0; while (k < n && edges[k + 1] - 1 <= window) k++; return k + (k < n && window >= edges[k] ? (window - edges[k] + 1) / (edges[k + 1] - edges[k]) : 0); })(), edge = L + reach * slot;
    share.forEach((s, k) => {
      if (none || s <= 0) return;
      const x = L + k * slot + 1.5, w = slot - 3, inside = Math.max(0, Math.min(w, edge - x));
      if (inside < w) svg += `<path class="f-neutral" d="${bar(x, w, base, Math.min(base - 1, y(s)))}"/>`;
      if (inside > 0) svg += `<clipPath id="w${kind}${k}"><rect x="${x}" y="0" width="${inside}" height="${H}"/></clipPath><path class="f-compache" clip-path="url(#w${kind}${k})" d="${bar(x, w, base, Math.min(base - 1, y(s)))}"/>`;
    });
    if (!none) svg += `<rect class="hit wide" data-kind="${kind}" x="${L - slot}" y="${lane * row + 4}" width="${W - L - R + 2 * slot}" height="${lane - 12}"/><line class="g-edge" x1="${edge}" x2="${edge}" y1="${lane * row + 12}" y2="${base + 4}"/><circle class="knob" data-kind="${kind}" tabindex="0" role="slider" aria-label="${label} kept, turns" aria-valuemin="0" aria-valuemax="161" aria-valuenow="${window ?? 161}" aria-valuetext="${window === null ? 'all' : plural(window, 'turn')}" cx="${edge}" cy="${lane * row + 12}" r="5.5"/>`;
    svg += `<text class="t-ink" x="${W - R + 12}" y="${base - 20}">${none ? '' : window === null ? 'all kept' : window === 0 ? 'none kept' : `last ${plural(window, 'turn')}`}</text><text x="${W - R + 12}" y="${base - 5}">${none ? '' : `${pct(coverage(r, kind), 0)} ${narrow ? 'in view' : 'of needs in view'}`}</text>`;
    if (row === 2) for (const [k, text] of AXIS.need) svg += `<text x="${L + (k + .5) * slot}" y="${H - 4}" text-anchor="middle">${text}</text>`;
  });
  $('windowsChart').innerHTML = `${svg}<text x="${W - R + 12}" y="${H - 4}">turns back</text></svg>`;
  const el = $('windowsChart').firstElementChild, snap = k => k <= 0 ? 0 : k >= n ? null : edges[k] - 1 || 0;
  const set = (kind, window) => { ownPolicy = {...r.policy, ...(mode === 'own' ? ownPolicy : {}), [WINDOW[kind]]: window}; mode = 'own'; r.policy = {...ownPolicy}; renderPolicy(r); };
  el.addEventListener('pointerdown', e => {
    const kind = e.target.dataset?.kind; if (!kind) return; e.preventDefault();
    // The window follows the pointer bin by bin: after bin k it holds everything up to that bin's last turn.
    const at = ev => { const box = $('windowsChart').firstElementChild.getBoundingClientRect(), k = Math.round(((ev.clientX - box.left) / box.width * W - L) / slot); set(kind, snap(k)); };
    const up = () => { removeEventListener('pointermove', at); removeEventListener('pointerup', up); removeEventListener('pointercancel', up); update(); };
    addEventListener('pointermove', at); addEventListener('pointerup', up); addEventListener('pointercancel', up); at(e);
  });
  el.addEventListener('keydown', e => {
    const kind = e.target.dataset?.kind, step = {ArrowRight: 1, ArrowLeft: -1}[e.key]; if (!kind || !step) return; e.preventDefault();
    const now = r.policy[WINDOW[kind]] ?? longest, next = now + step; set(kind, next >= Math.max(longest, 1) ? null : Math.max(0, next)); $('windowsChart').querySelector(`.knob[data-kind="${kind}"]`)?.focus(); clearTimeout(timer); timer = setTimeout(update, 400);
  });
  renderMap(r, file);
}
function renderMap(r, file) {
  const same = (a, b) => ['questions', 'answers', 'retrieved', 'maxPings', 'keepUnder', 'long'].every(k => (a[k] ?? null) === (b[k] ?? null));
  const points = file && filePolicy.candidates.length && filePolicy.cost !== null ? filePolicy.candidates.map(x => ({...x, kept: x.coverage})) : r.fit.candidates.filter(x => x.complete).map(x => ({policy: x.policy, everything: x.everything, coverage: x.coverage, cost: 1000 * x.cost / r.fit.turns, chosen: same(x.policy, r.policy)}));
  const usable = points.filter(x => Number.isFinite(x.cost)).map(x => ({...x, x: x.coverage ?? 1}));
  const label = n => n === null ? 'all' : n === 0 ? 'none' : `${n}`, cache = long => minutes((file ? filePolicy.lives[long ? 1 : 0] : null) ?? (long ? r.config.longTtlMinutes : r.config.ttlMinutes)), lasting = usable.some(p => p.policy.long);
  $('searchNote').textContent = usable.length ? `Cost per 1,000 turns against the needed text that stays in view${file && filePolicy.cost !== null ? ', as the package fitted it' : ''}.${lasting ? ` Dashed: the ${cache(true)} cache.` : ''} ${usable.some(x => x.everything) ? '' : 'Keeping everything does not fit the model here.'}` : 'No candidate fits the model.';
  $('policyRows').innerHTML = [...usable].sort((a, b) => a.cost - b.cost).map(x => `<tr class="${x.chosen ? 'selected' : ''}"><td>${label(x.policy.questions)}</td><td>${label(x.policy.answers)}</td><td>${label(x.policy.retrieved)}</td><td>${x.policy.maxPings}</td><td>${cache(x.policy.long)}</td><td>${pct(x.coverage, 0)}</td><td>${dollars(x.cost)}</td></tr>`).join('');
  const box = $('mapChart'), W = Math.max(300, box.clientWidth), H = box.clientHeight || 280, L = 52, R = 16, T = 24, B = 40; if (!usable.length) { box.innerHTML = ''; return; }
  const low = Math.min(.95, Math.floor(Math.min(...usable.map(p => p.x)) * 20) / 20), ys = ticks(0, Math.max(...usable.map(p => p.cost)), 4), top = ys.values.at(-1), x = v => L + (v - low) / (1 - low || 1) * (W - L - R), y = v => H - B - v / top * (H - T - B), counts = [...new Set(usable.map(p => p.policy.maxPings))].sort((a, b) => a - b), tone = n => n === 0 ? 'baseline' : n === counts.at(-1) ? 'compache' : 'compache-soft';
  let svg = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Cost of every candidate policy against the share of needed text it keeps in view">`;
  for (const v of ys.values) svg += `<line class="${v ? 'g-grid' : 'g-axis'}" x1="${L}" x2="${W - R}" y1="${y(v)}" y2="${y(v)}"/><text x="${L - 8}" y="${y(v) + 4}" text-anchor="end">${usd(v, ys.digits)}</text>`;
  for (let v = low; v <= 1.0001; v += (1 - low) / 4) svg += `<text x="${x(v)}" y="${H - B + 17}" text-anchor="${v > .999 ? 'end' : 'middle'}">${pct(v, 0)}</text>`;
  for (const long of lasting ? [false, true] : [false]) for (const n of counts) { const line = usable.filter(p => p.policy.maxPings === n && p.policy.long === long).sort((a, b) => a.x - b.x || a.cost - b.cost); svg += `<polyline class="line thin s-${tone(n)}"${long ? ' stroke-dasharray="5 4"' : ''} points="${line.map(p => `${x(p.x).toFixed(1)},${y(p.cost).toFixed(1)}`).join(' ')}"/>`; }
  usable.forEach((p, i) => { svg += `<circle class="hit" data-i="${i}" cx="${x(p.x)}" cy="${y(p.cost)}" r="11"/><circle class="mark dot f-${p.chosen ? 'compache' : tone(p.policy.maxPings)}" cx="${x(p.x)}" cy="${y(p.cost)}" r="${p.chosen ? 6.5 : 4}"/>`; });
  const chosen = usable.find(p => p.chosen); if (chosen) svg += `<text class="t-ink halo" x="${Math.max(L + 24, x(chosen.x))}" y="${y(chosen.cost) + (y(chosen.cost) > H - B - 30 ? -13 : 22)}" text-anchor="middle">${dollars(chosen.cost)}</text>`;
  let at = L; for (const n of counts) { svg += `<circle class="f-${tone(n)}" cx="${at + 4}" cy="10" r="4"/><text class="t-2" x="${at + 13}" y="14">${n ? `up to ${plural(n, 'ping')}` : 'no pings'}</text>`; at += 92; }
  box.innerHTML = `${svg}<text x="${(L + W - R) / 2}" y="${H - 4}" text-anchor="middle">needed text in view</text></svg>`;
  const el = box.firstElementChild; el.addEventListener('pointermove', e => { const p = usable[e.target.dataset?.i]; if (!p) return hideTip(); showTip(e, p.everything ? 'Keep everything' : `Questions ${label(p.policy.questions)} · answers ${label(p.policy.answers)} · retrieved ${label(p.policy.retrieved)}`, [['', dollars(p.cost), 'per 1,000 turns'], ['', pct(p.coverage, 0), 'in view'], ['', String(p.policy.maxPings), 'pings at most'], ...(lasting ? [['', cache(p.policy.long), 'cache lifetime']] : [])]); }); el.addEventListener('pointerleave', hideTip);
}

// -- one conversation --------------------------------------------------------

const decisionsOf = id => current.compache.decisions.filter(d => d.thread === id && d.admitted);
// The first turn that trims says most.
const firstCut = () => { turnIndex = Math.max(0, decisionsOf($('threadPicker').value).findIndex(d => d.action === 'cut')); };
function renderThread() {
  if (!current) return;
  const id = $('threadPicker').value, list = decisionsOf(id), box = $('threadChart'), W = Math.max(320, box.clientWidth), H = box.clientHeight || 300, L = 52, R = 16, T = 12, B = 66; if (!list.length) { box.innerHTML = ''; return; }
  turnIndex = Math.min(turnIndex, list.length - 1);
  const ys = ticks(0, Math.max(...list.map(d => Math.max(d.fullTokens, d.sentTokens))), 4), top = ys.values.at(-1), slot = (W - L - R) / list.length, bw = Math.max(1.5, Math.min(22, slot * .72)), x = i => L + (i + .5) * slot, y = v => H - B - v / top * (H - T - B);
  const sent = list.reduce((n, d) => n + d.sentTokens, 0), full = list.reduce((n, d) => n + d.fullTokens, 0), cuts = list.filter(d => d.action === 'cut').length, pinged = list.filter((d, i) => i && d.cache === 'warm' && list[i - 1].actualPings > 0).length;
  $('threadCaption').textContent = `${plural(list.length, 'turn')} · ${pct(1 - sent / full, 0)} fewer tokens sent than the whole history · trimmed ${plural(cuts, 'time')}, pings kept the cache warm ${plural(pinged, 'time')} · click a turn`;
  let svg = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Tokens sent on each turn of one conversation, over its whole history">`;
  for (const v of ys.values) svg += `<line class="${v ? 'g-grid' : 'g-axis'}" x1="${L}" x2="${W - R}" y1="${y(v)}" y2="${y(v)}"/><text x="${L - 8}" y="${y(v) + 4}" text-anchor="end">${compact(v)}</text>`;
  svg += `<text x="${L - 8}" y="${H - B + 18}" text-anchor="end">cache</text><text x="${L - 8}" y="${H - B + 34}" text-anchor="end">text</text>`;
  list.forEach((d, i) => {
    const left = x(i) - bw / 2, recall = d.byKind.recall || 0, own = d.sentTokens - recall;
    svg += `<rect class="hit" tabindex="0" data-i="${i}" x="${L + i * slot}" y="${T}" width="${slot}" height="${H - T - B + 40}" aria-label="Turn ${i + 1}: ${whole(d.sentTokens)} of ${whole(d.fullTokens)} tokens sent"/>${i === turnIndex ? `<rect class="pick" x="${L + i * slot}" y="${T}" width="${slot}" height="${H - T - B + 40}"/>` : ''}`;
    if (d.fullTokens > d.sentTokens) svg += `<path class="mark f-ghost" d="${bar(left, bw, y(0), y(d.fullTokens))}"/>`;
    if (recall) svg += `<path class="mark f-3" d="${bar(left, bw, y(0), y(d.sentTokens))}"/>`;
    svg += `<path class="mark f-compache" d="${bar(left, bw, y(0), Math.min(y(0) - 1, y(own)))}"/>`;
    const cy = H - B + 14, ty = H - B + 30, small = Math.min(4, slot * .3);
    if (d.cache === 'cold') svg += `<circle class="mark f-2" cx="${x(i)}" cy="${cy}" r="${small}"/>`;
    else if (i && list[i - 1].actualPings > 0) svg += `<path class="mark line s-compache" d="M${x(i) - small},${cy} l${small * .8},${small * .8} l${small * 1.3},${-small * 1.6}"/>`;
    if (d.action === 'cut') svg += `<path class="mark f-compache" d="M${x(i) - small},${ty - small} h${2 * small} l${-small},${2 * small} z"/>`;
    if (d.fetched.length) svg += `<circle class="mark f-3" cx="${x(i) + (d.action === 'cut' ? small * 2.4 : 0)}" cy="${ty}" r="${small}"/>`;
    const every = Math.max(1, Math.ceil(list.length / 12)); if (i % every === 0) svg += `<text x="${x(i)}" y="${H - 6}" text-anchor="middle">${i + 1}</text>`;
  });
  box.innerHTML = `${svg}</svg>`;
  const el = box.firstElementChild, pick = target => { const i = target.dataset?.i; if (i !== undefined) { turnIndex = Number(i); hideTip(); renderThread(); box.querySelector(`[data-i="${i}"]`)?.focus({preventScroll: true}); } };
  el.addEventListener('click', e => pick(e.target)); el.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); pick(e.target); } const s = {ArrowRight: 1, ArrowLeft: -1}[e.key]; if (s) { e.preventDefault(); box.querySelector(`[data-i="${turnIndex + s}"]`) && pick(box.querySelector(`[data-i="${turnIndex + s}"]`)); } });
  el.addEventListener('pointermove', e => { const d = list[e.target.dataset?.i]; if (!d) return hideTip(); showTip(e, `Turn ${d.turn + 1}`, [['compache', whole(d.sentTokens), 'tokens sent'], ['', whole(d.fullTokens), 'whole history'], ...(d.removedTokens ? [['', whole(d.removedTokens), 'trimmed now']] : [])]); }); el.addEventListener('pointerleave', hideTip);
  renderTurn(list[turnIndex], list);
}
function renderTurn(d, list) {
  const parts = Object.values(d.parts).join(''), archived = [...parts].filter(ch => ch === 'a' || ch === 'f').length;
  $('turnTitle').textContent = `Turn ${d.turn + 1}`;
  const gap = d.turn ? d.time - list[d.turn - 1].time : null;
  $('turnWhy').textContent = d.turn === 0 ? 'The first turn: nothing to trim yet.' : `${minutes(gap)} after the turn before. The cache was ${d.cache === 'cold' ? 'gone' : list[d.turn - 1].actualPings ? `warm, after ${plural(list[d.turn - 1].actualPings, 'ping')}` : 'warm'}. ${d.action === 'cut' ? `Trimmed ${plural(d.removed, 'part')}, ${compact(d.removedTokens)} tokens.` : d.cutTokens === null ? 'Nothing was outside its window.' : 'Kept as it was: trimming now would rewrite what the cache holds.'}${d.fetched.length ? ` Fetched back ${plural(d.fetched.length, 'part')}.` : ''}`;
  // Every earlier part, by kind and turn.
  const n = d.turn, W = 300, L = 66, cell = Math.min(14, (W - L) / Math.max(1, n + 1)), colour = {k: 'f-compache', a: 'f-ghost', f: 'f-3'};
  let svg = `<svg viewBox="0 0 ${W} 74" role="img" aria-label="Which earlier parts are in the request: ${parts.length - archived} kept, ${archived} archived">`;
  KINDS.forEach(([kind, label], row) => { svg += `<text class="t-2" x="0" y="${20 + row * 20}">${label === 'Retrieved text' ? 'Retrieved' : label}</text>`; [...d.parts[kind]].forEach((ch, t) => { if (ch !== '-') svg += `<rect class="${colour[ch]}" x="${L + t * cell}" y="${10 + row * 20}" width="${Math.max(.8, cell - 1.5)}" height="12" rx="${Math.min(2, cell / 4)}"/>`; }); if (kind !== 'output') svg += `<rect class="f-neutral-strong" x="${L + n * cell}" y="${10 + row * 20}" width="${Math.max(.8, cell - 1.5)}" height="12" rx="${Math.min(2, cell / 4)}"/>`; });
  $('turnParts').innerHTML = `${svg}</svg><p class="legend"><span class="key compache">kept</span><span class="key ghost">archived</span><span class="key s3">fetched back</span><span class="key ink">this turn</span></p>`;
  const per = v => dollars(1000 * v), line = (k, v, cls = '') => `<tr class="${cls}"><th>${k}</th><td>${v}</td></tr>`;
  $('turnDecisionTitle').title = 'Expected cost of this turn and the next two, per 1,000 such turns.';
  $('turnRows').innerHTML = [line('Send as it is, three turns ahead', per(d.keepCost), d.action === 'keep' ? 'total' : ''), line('Trim now, three turns ahead', d.cutCost === null ? '—' : per(d.cutCost), d.action === 'cut' ? 'total' : ''), line('Sent', `${whole(d.sentTokens)} tokens`), line('read from cache', pct(d.cachedTokens / d.sentTokens, 0)), line('Cost of the answer call', per(d.cost)), line('Chance of another turn', pct(d.planner.returnProbability, 0))].join('');
  // The pings of this wait.
  const p = d.planner, box = $('pingChart'), W2 = Math.max(220, box.clientWidth), H = box.clientHeight || 160, L2 = 50, R = 8, T = 10, B = 34, data = p.rows.map(r => ({...r, net: 1000 * r.net, cost: 1000 * r.expectedCost, gain: 1000 * r.expectedGain}));
  $('pingExplanation').textContent = p.chosen ? `Up to ${plural(p.chosen, 'ping')}, every ${num(p.interval)} min. ${d.actualPings < p.chosen ? `${d.actualPings || 'None'} sent before the user was back` : 'All sent'}.` : WHY_NO_PING[p.reason] ?? p.reason;
  if (data.length < 2) { box.innerHTML = ''; return; }
  if (data.some(row => !Number.isFinite(row.net))) {
    $('pingExplanation').textContent = 'The projected requests exceed the context window. A keep-warm saving is unavailable for this turn.';
    box.replaceChildren();
    return;
  }
  const ys = ticks(Math.min(0, ...data.map(r => r.net)), Math.max(1e-9, ...data.map(r => r.net)), 3), lo = ys.values[0], hi = ys.values.at(-1), y = v => T + (hi - v) / (hi - lo || 1) * (H - T - B), slot = (W2 - L2 - R) / data.length;
  svg = `<svg viewBox="0 0 ${W2} ${H}" role="img" aria-label="Expected net saving of 0 to ${data.length - 1} pings, per 1,000 such waits">`;
  for (const v of ys.values) svg += `<line class="${v ? 'g-grid' : 'g-axis'}" x1="${L2}" x2="${W2 - R}" y1="${y(v)}" y2="${y(v)}"/><text x="${L2 - 6}" y="${y(v) + 4}" text-anchor="end">${usd(v, ys.digits)}</text>`;
  data.forEach((r, i) => { const cx = L2 + (i + .5) * slot, w = Math.min(24, slot * .55); svg += `<rect class="hit" data-i="${i}" x="${L2 + i * slot}" y="${T}" width="${slot}" height="${H - T - B}"/>`; if (Math.abs(y(r.net) - y(0)) > .5) svg += `<path class="mark ${r.pings === p.chosen ? 'f-compache' : 'f-compache-soft'}" d="${bar(cx - w / 2, w, y(0), y(r.net))}"/>`; svg += `<text x="${cx}" y="${H - B + 15}" text-anchor="middle">${r.pings}</text>`; });
  box.innerHTML = `${svg}<text x="${(L2 + W2 - R) / 2}" y="${H - 4}" text-anchor="middle">pings, expected net saving</text></svg>`;
  box.firstElementChild.addEventListener('pointermove', e => { const r = data[e.target.dataset?.i]; if (!r) return hideTip(); showTip(e, plural(r.pings, 'ping'), [['compache', dollars(r.net), 'net per 1,000 waits'], ['', dollars(r.gain), 'saved on the next turns'], ['', dollars(r.cost), 'cost of pings']]); }); box.firstElementChild.addEventListener('pointerleave', hideTip);
}

// -- the breakdown ------------------------------------------------------------

function contextChart() {
  const [ours, theirs] = contextSeries(current), all = [...ours, ...theirs], box = $('contextChart'), W = Math.max(320, box.clientWidth), H = box.clientHeight || 280, L = 52, R = 16, T = 14, B = 26; if (!all.length) { box.innerHTML = ''; return; }
  const maxX = Math.max(2, ...all.map(p => p.turn + 1)), scale = ticks(0, Math.max(1, ...all.map(p => p.inputTokens)), 4), top = scale.values.at(-1), x = n => L + (n - 1) / (maxX - 1) * (W - L - R), y = n => H - B - n / top * (H - T - B), total = current.workload.threads.length;
  let svg = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Mean input tokens per turn, Compache and ${esc(current.baseline.name)}">`;
  for (const v of scale.values) svg += `<line class="${v ? 'g-grid' : 'g-axis'}" x1="${L}" x2="${W - R}" y1="${y(v)}" y2="${y(v)}"/><text x="${L - 8}" y="${y(v) + 4}" text-anchor="end">${compact(v)}</text>`;
  for (const n of [...new Set([1, Math.round(maxX / 4), Math.round(maxX / 2), Math.round(3 * maxX / 4), maxX].filter(n => n > 0))]) svg += `<text x="${x(n)}" y="${H - B + 17}" text-anchor="middle">${n}</text>`;
  for (const [cls, points] of [['baseline', theirs], ['compache', ours]]) { if (points.length > 1) svg += `<polyline class="line s-${cls}" points="${points.map(p => `${x(p.turn + 1).toFixed(1)},${y(p.inputTokens).toFixed(1)}`).join(' ')}"/>`; const p = points.at(-1); if (p) svg += `<circle class="dot f-${cls}" cx="${x(p.turn + 1)}" cy="${y(p.inputTokens)}" r="4"/>`; }
  box.innerHTML = `${svg}<line class="g-cross" y1="${T}" y2="${H - B}" visibility="hidden"/><rect class="hit" x="${L}" y="${T}" width="${W - L - R}" height="${H - T - B}"/></svg>`;
  const el = box.firstElementChild, cross = el.querySelector('.g-cross'), area = el.querySelector('.hit');
  area.addEventListener('pointermove', e => { const r = el.getBoundingClientRect(), turn = Math.max(1, Math.min(maxX, Math.round(1 + ((e.clientX - r.left) * W / r.width - L) / (W - L - R) * (maxX - 1)))), a = ours.find(p => p.turn + 1 === turn), b = theirs.find(p => p.turn + 1 === turn); for (const k of ['x1', 'x2']) cross.setAttribute(k, x(turn)); cross.setAttribute('visibility', 'visible'); showTip(e, `Turn ${turn}${a ? ` · ${a.count} of ${total} threads` : ''}`, [['compache', a ? whole(a.inputTokens) : '—', 'Compache'], ['openclaw', b ? whole(b.inputTokens) : '—', current.baseline.name]]); });
  area.addEventListener('pointerleave', () => { cross.setAttribute('visibility', 'hidden'); hideTip(); });
}
function renderBreakdown(r) {
  const a = r.compache, b = r.baseline.arm, c = r.config, den = r.workload.totalRuns, per = x => 1000 * x / den, full = a.complete && b.complete;
  const cells = (format, x, y) => `<td>${a.complete ? format(x) : '—'}</td><td>${b.complete ? format(y) : '—'}</td>`, group = title => `<tr class="group"><th colspan="3">${title}</th></tr>`, line = (label, format, x, y, cls = '') => `<tr class="${cls}"><th>${label}</th>${cells(format, x, y)}</tr>`, millions = x => num(x / 1e6);
  const further = (label, format, field) => c.callsPerTurn > 1 ? [line(label, format, per(role(a, 'further', field)), per(role(b, 'further', field)))] : [];
  $('ledgerRows').innerHTML = [group('Cost'),
    line('Answers', dollars, per(role(a, 'answer')), per(role(b, 'answer'))), ...further('Further calls of a turn', dollars, 'cost'), line('Keep-warm pings', dollars, per(role(a, 'ping')), per(role(b, 'ping'))),
    line('Fetching text back', dollars, per(role(a, 'recovery')), per(role(b, 'recovery'))), line('Summaries and flushes', dollars, 0, per(role(b, 'summary') + role(b, 'flush'))), line('Total', dollars, per(a.totals.cost), per(b.totals.cost), 'total'),
    group('Calls'), line('Answers', num, per(role(a, 'answer', 'calls')), per(role(b, 'answer', 'calls'))), ...further('Further calls of a turn', num, 'calls'), line('Keep-warm pings', num, per(a.pings), per(b.pings)), line('Fetch-backs', num, per(role(a, 'recovery', 'calls')), per(role(b, 'recovery', 'calls'))), line('Context rewrites', num, per(a.compactions), per(b.compactions + b.clearings)),
    group('Tokens'), line('Input, millions', millions, per(a.totals.inputTokens), per(b.totals.inputTokens)), line('read from cache, millions', millions, per(a.totals.cachedTokens), per(b.totals.cachedTokens)), line('Output, millions', millions, per(a.totals.outputTokens), per(b.totals.outputTokens)), line('Mean answer request', compact, role(a, 'answer', 'inputTokens') / Math.max(1, role(a, 'answer', 'calls')), role(b, 'answer', 'inputTokens') / Math.max(1, role(b, 'answer', 'calls'))),
    group('Needed text in view, before fetching back'), ...KINDS.map(([kind, label]) => line(label, x => pct(x, 0), a.retention.byKind[kind].observed, b.retention.byKind[kind].observed)), line('Missing in the end', x => pct(x, 0), a.dependencyCoverage === null ? null : 1 - a.dependencyCoverage, b.dependencyCoverage === null ? null : 1 - b.dependencyCoverage)].join('');
  // Why answer input was, or was not, read from the cache. Each column sums to 100%.
  const why = [['cached', 'Read from cache', 'Input that matched a live cache entry.'], ['newContext', 'New this turn', 'Text no request of the thread had carried before.'], ['firstRequest', 'First turn of a thread', 'Nothing of the thread is cached yet.'], ['expiredPrefix', 'Cache expired', 'It would have matched, but the entry had expired when the user returned.'], ['changedPrefix', 'Request changed', 'Sent before, but a rewrite moved or removed what came before it.']];
  if (c.cacheHitShare < 1) why.push(['missed', 'Live entry not read', 'A live entry matched, but the request did not read it.']);
  const share = (arm, key) => { const mix = arm.cacheBreakdown || {}, sum = Object.values(mix).reduce((n, v) => n + v, 0); return sum ? (mix[key] || 0) / sum : null; }, cell = (cls, v) => `<td>${pct(v, 0)}<i class="${cls}" style="--share:${(100 * (v || 0)).toFixed(2)}%"></i></td>`;
  $('cacheRows').innerHTML = why.map(([key, label, title]) => `<tr><th title="${esc(title)}">${label}</th>${cell('compache', share(a, key))}${cell('openclaw', share(b, key))}</tr>`).join('');
  $('warnings').hidden = full;
  $('warnings').textContent = full ? '' : `Some requests do not fit the model's context window: ${[!a.complete && `Compache finished ${a.completedRuns} of ${den} turns`, !b.complete && `${r.baseline.name} ${b.completedRuns}`].filter(Boolean).join(', ')}.`;
}
// The address bar always holds the scenario on screen, so copying it is sharing it. What the page opens with needs no fragment.
const OPENS_WITH = shareHash(configure(applyPreset({...STARTS[0].scenario}, 'sonnet')), 'sonnet', '', STARTS[0].id);
function address() {
  // Named from whichever is nearest: the defaults, or one of the starting points.
  const url = new URL(location.href), config = configure(readForm()), hash = ['', ...STARTS.map(s => s.id)].map(id => shareHash(config, $('modelPreset').value, sourceLabel(), id)).reduce((a, b) => b.length < a.length ? b : a);
  url.hash = hash === OPENS_WITH ? '' : hash; return url;
}
// A link that could not be read stays in the bar, beside the message about it, until something is changed or shared.
function showAddress() { if (!loadError) history.replaceState(null, '', address()); }
// The result in one line of text, to send along with the link.
function resultLine(r) {
  const c = r.config, ours = dollars(1000 * r.compache.totals.cost / r.workload.totalRuns), retrieves = c.tokenDrawn || c.tokenMedian > 0;
  const workload = [presets[$('modelPreset').value]?.name || 'Custom prices', retrieves ? `${c.tokenDrawn ? 'drawn' : compact(c.tokenMedian)} retrieved tokens per turn` : 'no retrieval', `users back after ${c.gapDrawn ? 'drawn times' : minutes(c.gapMedianMinutes)}`].join(', ');
  return r.savings ? `Modeled chat cost per 1,000 turns: Compache ${ours}, ${r.selected.name} ${dollars(r.selected.per1000)}. ${num(Math.abs(r.savings.percent))}% ${r.savings.absolute < 0 ? 'more' : 'less'}. (${workload})` : `Chat cost calculator (${workload})`;
}
const sourceLabel = () => source ? `${source}${edited && !source.endsWith('(edited)') ? ' (edited)' : ''}` : '';
function render(r) {
  baseConfig = {...baseConfig, compare: r.selected.id};
  current = r; busy('false'); $('export').disabled = false; $('share').disabled = false; showAddress();
  const c = r.config, preset = presets[$('modelPreset').value];
  $('status').textContent = `Seed ${c.seed} · updated`;
  $('modelWindow').textContent = `${compact(c.model.context)} context · ${compact(c.model.maxInput)} max input`; $('cacheDefault').textContent = `cache lifetime ${num(c.ttlMinutes)} min`;
  $('sourceNote').textContent = source ? `From your data: ${sourceLabel()}` : STARTS.find(s => s.id === startId)?.from ?? '';
  $('workloadNote').textContent = `${preset?.name || 'Custom prices'} · ${num(c.threads)} threads · ${num(r.workload.totalRuns)} turns · modeled, nothing is run`;
  $('baselineNote').textContent = r.selected.note;
  renderLead(r); renderInputs(c); renderPolicy(r); renderBreakdown(r);
  const picker = $('threadPicker'), previous = picker.value;
  picker.replaceChildren(...r.workload.threads.map(t => { const o = document.createElement('option'); o.value = t.id; o.textContent = `${t.id} · ${plural(t.turns.length, 'turn')}`; return o; }));
  // A thread long enough to show something and short enough to read.
  const readable = [...r.workload.threads].filter(t => t.turns.length <= 30).sort((x, y) => y.turns.length - x.turns.length)[0] ?? r.workload.threads[0];
  picker.value = r.workload.threads.some(t => t.id === previous) ? previous : readable.id; if (picker.value !== previous) firstCut();
  renderThread(); contextChart();
  $('priceNote').textContent = `${preset?.name || 'Custom'}: $${c.model.inputRate} input / $${c.model.cachedRate} cache read / $${c.model.outputRate} output per 1M tokens. A cache write costs ${c.model.writeMultiplier}× input.${c.model.longThreshold > 0 ? ` Above ${compact(c.model.longThreshold)} input: ${c.model.longInputMultiplier}× input and cache, ${c.model.longOutputMultiplier}× output.` : ''}`;
  $('modelSource').href = preset?.url || 'https://platform.claude.com/docs/en/about-claude/pricing';
}

// -- editing --------------------------------------------------------------------

form.addEventListener('submit', e => { e.preventDefault(); update(); });
form.addEventListener('input', e => {
  const el = e.target, f = form.elements, name = el.name;
  if (el.id === 'modelPreset' || el.id === 'appFrom') return;
  // A price or cache rule typed by hand is no longer the preset's.
  if (el.hasAttribute('data-preset')) $('modelPreset').value = 'custom';
  // A number typed over a drawn chart takes the curve again.
  for (const [drawn, fields] of Object.entries(STATS)) if (fields.includes(name) && baseConfig[drawn]) baseConfig = {...baseConfig, [drawn]: null};
  if (name === 'tokenMedian' && f.tokenMedian.value !== '') {
    const next = Number(f.tokenMedian.value); if (priorTokens > 0) retrievalSpread = Math.max(1, Number(f.tokenP95.value) / priorTokens);
    f.tokenP95.value = next === 0 ? 0 : Math.min(1000000, Math.max(next, Number((next * retrievalSpread).toPrecision(3)))); priorTokens = next;
  }
  if (name === 'plannedTurns' && f.lengthDistribution.value === 'lognormal' && Number(f.plannedTurns.value) > 0) {
    const next = Number(f.plannedTurns.value), ratio = Math.max(1, Number(f.plannedTurnsP95.value) / Math.max(1, priorLength)); f.plannedTurnsP95.value = Math.min(320, Math.max(next, Math.round(next * ratio))); priorLength = next;
  }
  // The rules of your app, or what you pay, are what to compare with once they are touched.
  if (el.dataset.rule) { baseConfig = {...baseConfig, compare: 'app'}; $('appFrom').value = ''; }
  if (name === 'measuredCost') baseConfig = {...baseConfig, compare: Number(el.value) > 0 ? 'measured' : baseConfig.compare === 'measured' ? 'app' : baseConfig.compare};
  changed(); soon();
});
form.addEventListener('click', e => { const name = e.target.dataset?.reset; if (name) { baseConfig = {...baseConfig, [name]: null}; STATS[name].forEach((field, i) => { if (curveOf[name]) form.elements[field].value = curveOf[name][i]; }); delete curveOf[name]; changed(); soon(); } });
$('modelPreset').addEventListener('change', () => { if (presets[$('modelPreset').value]) { setForm(applyPreset(readForm(), $('modelPreset').value)); fillFrom(); update(); } });
$('needTabs').addEventListener('click', e => { const kind = e.target.dataset?.kind; if (!kind) return; for (const b of $('needTabs').children) b.setAttribute('aria-selected', b.dataset.kind === kind); for (const box of form.querySelectorAll('.need')) box.hidden = box.dataset.kind !== kind; });
$('baselinePicker').addEventListener('change', e => {
  baseConfig = {...baseConfig, compare: e.target.value};
  if (e.target.value === 'app') $('appSettings').open = true;
  update();
});
$('policyTabs').addEventListener('click', e => { const next = e.target.dataset?.mode; if (!next || next === mode) return; if (next === 'own') ownPolicy = {...(current?.policy ?? {questions: null, answers: null, retrieved: null, maxPings: 0, keepUnder: 0})}; mode = next; update(); });
$('ownPings').addEventListener('change', e => { const n = Math.max(0, Math.min(60, Math.round(Number(e.target.value) || 0))); ownPolicy = {...(current?.policy ?? {}), maxPings: n}; mode = 'own'; update(); });
$('ownLong').addEventListener('change', e => { ownPolicy = {...(current?.policy ?? {}), long: e.target.checked}; mode = 'own'; update(); });
$('threadPicker').addEventListener('change', () => { firstCut(); renderThread(); });
function fillFrom() { const list = comparisons(configure(applyPreset({}, $('modelPreset').value))); $('appFrom').replaceChildren(new Option('Rules below', ''), ...list.filter(b => b.id !== 'app').map(b => new Option(b.name, b.id))); }
$('appFrom').addEventListener('change', e => { const b = comparisons(configure(readForm())).find(x => x.id === e.target.value); if (!b) return; for (const el of ruleFields) writeField(el, b.rules[el.dataset.rule]); baseConfig = {...baseConfig, compare: 'app'}; changed(); update(); });

// -- loading a scenario: a start, a link, or files ---------------------------------

function note(text) { $('actionStatus').textContent = text; }
// What the workload is compared with stays when the workload changes.
function compared() { try { const {app, compare, measuredCost} = readForm(); return {app, compare, measuredCost}; } catch { return {}; } }
function adopt(config) { mode = 'fit'; ownPolicy = null; if (config.policy) { mode = 'own'; ownPolicy = {...config.policy}; } }
function load(carried, origin) {
  const {preset, config, source: label, legacy} = resolveScenario(carried);
  $('modelPreset').value = preset; source = label; edited = false; loadError = ''; startId = ''; adopt(config); setForm(config); fillFrom();
  note(legacy ? 'Loaded settings from an earlier calculator version.' : origin === 'file' ? 'Imported.' : ''); update();
}
function begin(id) {
  const start = STARTS.find(s => s.id === id) ?? STARTS[0], preset = $('modelPreset').value;
  startId = start.id; source = ''; edited = false; loadError = ''; filePolicy = null; rows = []; labelRows = []; mode = 'fit'; ownPolicy = null; turnIndex = 0; $('threadPicker').replaceChildren();
  setForm(applyPreset({...DEFAULTS, ...start.scenario, ...compared()}, presets[preset] ? preset : 'sonnet')); if (!presets[preset]) $('modelPreset').value = 'sonnet'; fillFrom(); update();
}
function start() {
  try { const carried = parseLink(location.hash); if (carried) { load(carried, 'link'); return; } }
  catch (error) { loadError = `This link was not loaded: ${readable(error.message)}. A starting point is shown instead.`; }
  const kept = loadError; setForm(applyPreset(DEFAULTS, 'sonnet')); begin('documents'); loadError = kept; if (kept) { $('errors').textContent = kept; $('errors').hidden = false; }
}
async function textOf(file) {
  if (!/\.gz$/i.test(file.name)) return file.text();
  return new Response(file.stream().pipeThrough(new DecompressionStream('gzip'))).text();
}
async function importFiles(files) {
  try {
    const threads = [], labelsIn = []; let policy = null, scenario = null;
    for (const file of files) {
      let data; try { data = parseRows(await textOf(file)); } catch (e) { throw new Error(`${file.name}: ${e.message}`); }
      for (const row of data) {
        if (isPolicy(row)) policy = readPolicy(row); else if (isLabel(row)) labelsIn.push(row); else if (isThread(row)) threads.push(row);
        else if (row?.schema_version === 'compache-profile-v1' || row?.schemaVersion) scenario = parseFile(row); else throw new Error(`${file.name}: not a chat log, labels, a policy, a profile or an export`);
      }
    }
    if (threads.length) { rows = threads; labelRows = labelsIn; } else if (labelsIn.length) { if (!rows.length) throw new Error('labels need the chat log they belong to: open both'); labelRows = labelsIn; }
    const told = [];
    if (scenario) load(scenario, 'file');
    if (threads.length || labelsIn.length) {
      const m = measure(rows, labelRows), preset = policy?.preset ?? (m.model && Object.keys(presets).find(k => presets[k].id === m.model)) ?? $('modelPreset').value;
      const config = configure(applyPreset({...DEFAULTS, ...m.scenario, ...compared(), ...(presets[preset] ? {} : {model: baseConfig.model, ttlMinutes: baseConfig.ttlMinutes, cacheMinTokens: baseConfig.cacheMinTokens, cacheBlockTokens: baseConfig.cacheBlockTokens})}, preset));
      $('modelPreset').value = preset; source = m.source; edited = false; loadError = ''; startId = ''; mode = 'fit'; filePolicy = null; setForm(config); fillFrom();
      told.push(`Measured ${m.source}.`, m.assessed ? `${m.assessed} labelled turns, ${m.needing} need something earlier.` : 'No labels: what later turns need keeps its defaults.');
    }
    if (policy) {
      if (!threads.length && policy.workload) { const w = resolveScenario({version: LINK_VERSION, model: policy.workload.model ?? policy.preset, scenario: policy.workload.scenario, source: policy.workload.source ?? ''}); $('modelPreset').value = w.preset; source = w.source; edited = false; startId = ''; setForm({...w.config, ...compared()}); fillFrom(); }
      else if (policy.preset) { $('modelPreset').value = policy.preset; setForm(applyPreset(readForm(), policy.preset)); }
      filePolicy = policy; mode = 'file'; told.push('Showing the policy from the file.');
    }
    history.replaceState(null, '', location.pathname); turnIndex = 0; $('threadPicker').replaceChildren(); note(told.join(' ')); if (!scenario || told.length) update();
  } catch (error) { loadError = `Not loaded. ${readable(error.message)}.`; $('errors').textContent = loadError; $('errors').hidden = false; note(''); }
}
$('starts').replaceChildren(...STARTS.map(s => { const b = document.createElement('button'); b.type = 'button'; b.dataset.id = s.id; b.textContent = s.name; b.title = s.from; b.addEventListener('click', () => { history.replaceState(null, '', location.pathname); note(''); begin(s.id); }); return b; }));
$('reset').addEventListener('click', () => { current = null; setForm(applyPreset(DEFAULTS, 'sonnet')); $('modelPreset').value = 'sonnet'; history.replaceState(null, '', location.pathname); note(''); begin('documents'); });
$('import').addEventListener('click', () => $('importFile').click());
$('importFile').addEventListener('change', e => { const files = [...e.target.files]; e.target.value = ''; if (files.length) importFiles(files); });
document.addEventListener('dragover', e => { if (e.dataTransfer?.types.includes('Files')) e.preventDefault(); });
document.addEventListener('drop', e => { const files = [...e.dataTransfer?.files ?? []]; if (files.length) { e.preventDefault(); importFiles(files); } });
$('share').addEventListener('click', async () => {
  if (!current) return;
  const url = address().href, text = resultLine(current); history.replaceState(null, '', url);
  // A phone or tablet has a share sheet. Elsewhere the result and its link go to the clipboard.
  if (navigator.share && matchMedia('(pointer: coarse)').matches) { try { await navigator.share({title: document.title, text, url}); } catch { /* closed without sharing */ } return; }
  try { await navigator.clipboard.writeText(`${text}\n${url}`); note('Result and link copied.'); } catch { note('Copy the address bar to share this scenario.'); }
});
$('export').addEventListener('click', () => {
  if (!current) return;
  const blob = new Blob([JSON.stringify({schemaVersion: EXPORT_SCHEMA, sourceDate: PRICES_CHECKED, modelPreset: $('modelPreset').value, source: sourceLabel(), ...current}, null, 2)], {type: 'application/json'});
  const url = URL.createObjectURL(blob), a = document.createElement('a'); a.href = url; a.download = `compache-scenario-${current.config.seed}.json`; a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000); note('Exported settings, workload and cost ledger.');
});
addEventListener('hashchange', start);
let resizeTimer; addEventListener('resize', () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(() => { if (current) { renderPolicy(current); renderMap(current, mode === 'file' && filePolicy); renderThread(); contextChart(); } }, 100); });
start();
