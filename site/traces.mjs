// Measure a chat log in the browser, the way the package reads it. Nothing leaves the page: only counts, sizes and times are kept.
import {weightsFrom} from './drawn.mjs';

const RETRIEVED = new Set(['retrieval', 'tool', 'context', 'document', 'function']);
const TIMES = ['at', 'created_at', 'timestamp', 'time'];
const encoder = new TextEncoder();
// Four bytes a token, plus a part's overhead, as the package estimates.
const textTokens = text => Math.ceil(encoder.encode(text).length / 4), partTokens = text => textTokens(text) + 6;
const isObject = v => v !== null && typeof v === 'object' && !Array.isArray(v);
const rank = (values, q) => { const a = [...values].sort((x, y) => x - y); return a[Math.min(a.length - 1, Math.max(0, Math.ceil(q * a.length) - 1))]; };
function seconds(value) {
  if (value === undefined || value === null) return null;
  if (typeof value === 'number') return value > 1e11 ? value / 1000 : value;
  const text = String(value), at = Date.parse(/[zZ]|[+-]\d\d:?\d\d$/.test(text) ? text : `${text}Z`);
  if (Number.isNaN(at)) throw new Error(`“${text.slice(0, 40)}” is not a time`);
  return at / 1000;
}
function content(value) {
  if (value === null || value === undefined) return ['', []];
  if (typeof value === 'string') return [value, []];
  if (!Array.isArray(value)) throw new Error('message content must be text or a list of blocks');
  const texts = [], results = [];
  for (const b of value) {
    if (typeof b === 'string') texts.push(b);
    else if (isObject(b) && b.type === 'tool_result') results.push(content(b.content)[0]);
    else if (isObject(b) && typeof b.text === 'string') texts.push(b.text);
  }
  return [texts.filter(Boolean).join('\n\n'), results.filter(Boolean)];
}
function turnsOf(messages) {
  const runs = []; let current = null, waiting = [];
  const close = () => { if (current?.answers.length) runs.push({start: current.start, end: current.end ?? current.start, input: current.questions.join('\n\n'), retrieved: current.retrieved, output: current.answers.join('\n\n'), calls: current.calls}); current = null; };
  messages.forEach((m, i) => {
    if (!isObject(m)) throw new Error(`message ${i + 1} is not an object`);
    if (m.role === 'system') return;
    const [text, found] = content(m.content), at = TIMES.map(k => m[k]).find(v => v !== null && v !== undefined);
    if (RETRIEVED.has(m.role)) { if (text) found.push(text); }
    else if (m.role === 'user') {
      if (text) {
        if (current?.answers.length) close();
        if (!current) {
          if (at === undefined) throw new Error(`message ${i + 1}: a user message needs a time (“at”)`);
          current = {start: seconds(at), end: null, questions: [], retrieved: [], answers: [], calls: 0}; found.unshift(...waiting); waiting = [];
        }
        current.questions.push(text);
      }
    } else if (m.role === 'assistant') {
      if (current) { if (text) current.answers.push(text); if (at !== undefined) current.end = seconds(at); current.calls++; }
    } else throw new Error(`message ${i + 1}: unknown role “${m.role}”`);
    (current ? current.retrieved : waiting).push(...found);
  });
  close();
  return runs;
}
// One part of a turn that is written out: text, an object with text, or its size alone.
const partText = p => typeof p === 'string' ? p : isObject(p) ? (typeof p.text === 'string' ? p.text : Number.isInteger(p.bytes) ? '.'.repeat(p.bytes) : '') : '';
export function readThread(row) {
  if (!isObject(row)) throw new Error('a thread must be an object');
  const runs = Array.isArray(row.messages) ? turnsOf(row.messages) : Array.isArray(row.runs) ? row.runs.map(r => ({start: seconds(r.started_at), end: seconds(r.ended_at) ?? seconds(r.started_at), input: partText(r.input), retrieved: (r.retrievals ?? []).filter(p => p?.exposed_to_model !== false).map(partText), output: partText(r.output), calls: r.calls ?? null})) : null;
  if (!runs) throw new Error('a thread needs “messages” or “runs”');
  const system = Array.isArray(row.messages) ? row.messages.find(m => m?.role === 'system') : null, models = Array.isArray(row.messages) ? row.messages.map(m => m?.model).filter(Boolean) : [];
  return {id: row.thread_id ?? row.id ?? null, runs, model: row.model ?? models.at(-1) ?? null, system: typeof row.system_prompt === 'string' ? row.system_prompt : system ? content(system.content)[0] : '', tools: Array.isArray(row.tools) ? row.tools : []};
}
// JSON Lines, or one JSON value.
export function parseRows(text) {
  const body = text.trim();
  try { const value = JSON.parse(body); return Array.isArray(value) ? value : [value]; } catch { /* a line each */ }
  return body.split(/\r?\n/).filter(line => line.trim()).map((line, i) => { try { return JSON.parse(line); } catch { throw new Error(`line ${i + 1} is not JSON`); } });
}
export const isThread = row => isObject(row) && (Array.isArray(row.messages) || Array.isArray(row.runs));
export const isLabel = row => isObject(row) && 'needs' in row && typeof row.thread_id === 'string';
// How far back each kind lies in the alternative whose evidence is nearest.
function nearest(alternatives, threadId, target) {
  let best = null;
  for (const alternative of alternatives) {
    const far = {};
    for (const part of alternative) {
      const id = typeof part === 'string' ? part : part?.part_id ?? '', m = id.startsWith(`${threadId}:`) ? /^(\d+):(input|output|retrieval)/.exec(id.slice(threadId.length + 1)) : null;
      if (m && Number(m[1]) < target) far[m[2]] = Math.max(far[m[2]] ?? 0, target - Number(m[1]));
    }
    const lags = Object.values(far); if (!lags.length) continue;
    const key = [Math.max(...lags), alternative.length];
    if (!best || key[0] < best.key[0] || (key[0] === best.key[0] && key[1] < best.key[1])) best = {key, far};
  }
  return best?.far ?? {};
}
// The settings of the calculator that a chat log, with labels or without, fills in.
export function measure(rows, labels = []) {
  const threads = rows.map((row, i) => { try { return readThread(row); } catch (e) { throw new Error(`thread ${i + 1}: ${e.message}`); } }).filter(t => t.runs.length);
  if (!threads.length) throw new Error('no thread holds an answered user message');
  const sizes = {input: [], retrieval: [], output: []}, lengths = [], gaps = [], durations = [], calls = [], pieces = []; let turns = 0, retrieving = 0;
  for (const t of threads) {
    lengths.push(t.runs.length); let previous = null;
    for (const r of t.runs) {
      turns++; sizes.input.push(partTokens(r.input)); sizes.output.push(partTokens(r.output));
      if (r.retrieved.length) { retrieving++; pieces.push(r.retrieved.length); sizes.retrieval.push(r.retrieved.reduce((n, p) => n + partTokens(p), 0)); }
      if (r.calls) calls.push(r.calls);
      durations.push(Math.max(0, r.end - r.start)); if (previous !== null) gaps.push(Math.max(0, r.start - previous) / 60); previous = r.end;
    }
  }
  const mean = a => a.reduce((x, y) => x + y, 0) / a.length, scenario = {inputTokens: Math.max(1, rank(sizes.input, .5)), outputTokens: rank(sizes.output, .5), continuationProbability: 1, lengthDrawn: weightsFrom('length', lengths)};
  scenario.plannedTurns = Math.min(160, rank(lengths, .5)); scenario.plannedTurnsP95 = Math.min(320, Math.max(scenario.plannedTurns, rank(lengths, .95)));
  if (sizes.retrieval.length) Object.assign(scenario, {tokenMedian: Math.min(500000, rank(sizes.retrieval, .5)), tokenP95: Math.min(1000000, rank(sizes.retrieval, .95)), tokenDrawn: weightsFrom('token', sizes.retrieval), retrievalShare: Number((retrieving / turns).toFixed(3)), archivedTokens: Math.min(10000, Math.round(28 * mean(pieces)))});
  else Object.assign(scenario, {tokenMedian: 0, tokenP95: 0});
  if (gaps.length) { scenario.gapMedianMinutes = Number(Math.max(.01, rank(gaps, .5)).toFixed(2)); scenario.gapP95Minutes = Number(Math.max(scenario.gapMedianMinutes, rank(gaps, .95)).toFixed(2)); scenario.gapDrawn = weightsFrom('gap', gaps); }
  const answer = rank(durations, .5); if (answer > 0) scenario.answerSeconds = Number(Math.min(86400, answer).toFixed(1));
  const perTurn = calls.length ? mean(calls) : 1; if (perTurn > 1) scenario.callsPerTurn = Number(Math.min(100, perTurn).toFixed(2));
  const system = threads[0].system, tools = threads[0].tools; scenario.systemTokens = textTokens(system) + (tools.length ? textTokens(JSON.stringify(tools)) : 0);
  // Labels: per kind, the share of assessed turns that need an earlier part of it and how far back it lies.
  const known = new Map(threads.map(t => [t.id, t])), lags = {input: [], retrieval: [], output: []}; let assessed = 0, needing = 0;
  for (const row of labels) {
    const t = known.get(row.thread_id); if (!t || row.needs === null || !(row.run > 1)) continue;
    assessed++; const far = nearest(row.needs, t.id, row.run); if (Object.keys(far).length) needing++;
    for (const [kind, lag] of Object.entries(far)) lags[kind].push(lag);
  }
  if (assessed) for (const [kind, values] of Object.entries(lags)) {
    scenario[`${kind}DependencyProbability`] = Number((values.length / assessed).toFixed(3));
    if (values.length) { const median = Math.min(160, rank(values, .5)); scenario[`${kind}DependencyTurnsBack`] = median; scenario[`${kind}DependencyTurnsBackP95`] = Math.max(median, Math.min(1000, rank(values, .95))); scenario[`${kind}NeedDrawn`] = weightsFrom('need', values); }
  }
  const tally = {}; for (const t of threads) if (t.model) tally[t.model] = (tally[t.model] ?? 0) + 1;
  return {scenario, model: Object.entries(tally).sort((a, b) => b[1] - a[1])[0]?.[0] ?? null, threads: threads.length, turns, assessed, needing, source: `${threads.length.toLocaleString('en-US')} threads, ${turns.toLocaleString('en-US')} turns`};
}
