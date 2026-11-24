import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {gunzipSync} from 'node:zlib';
import {BINS, histogram, weightsFrom, weightsOf} from './drawn.mjs';
import {configure, DEFAULTS, generateWorkload, lawWeights, returnCdf, simulate, theoreticalSurvival, drawnStats} from './engine.mjs';
import {RULES, comparisons, configureRules} from './baselines.mjs';
import {measure, parseRows, readThread} from './traces.mjs';
import {readPolicy} from './policy.mjs';
import {applyPreset, presets} from './models.mjs';
import {STARTS} from './starts.mjs';

const file = path => readFileSync(new URL(`../examples/quickstart/${path}`, import.meta.url));
const near = (a, b, tolerance, note) => assert.ok(Math.abs(a - b) <= tolerance, `${note ?? ''} ${a} is not within ${tolerance} of ${b}`);

test('a histogram is a distribution: its chances rise from 0 to 1 and its quantiles invert them', () => {
  for (const [kind, {edges, whole}] of Object.entries(BINS)) {
    const weights = edges.slice(1).map((_, i) => 1 + i % 3), h = histogram(kind, weights);
    near(h.share.reduce((a, b) => a + b, 0), 1, 1e-12); assert.equal(h.cdf(edges.at(-1)), 1); assert.equal(h.cdf(edges[0] / 2), 0);
    let before = 0; for (const e of edges) { assert.ok(h.cdf(e) >= before); before = h.cdf(e); }
    for (const q of [.03, .2, .5, .77, .99]) whole ? assert.ok(h.cdf(h.quantile(q)) >= q - 1e-9 && h.cdf(h.quantile(q) - 1) <= q + 1e-9) : near(h.cdf(h.quantile(q)), q, 1e-9);
  }
  // All the weight in one bin, the last one included.
  assert.equal(histogram('length', [0, 0, 0, 1, ...new Array(10).fill(0)]).quantile(.999), 4);
  const last = histogram('need', [...new Array(11).fill(0), 1]); assert.ok(last.quantile(0) >= 51 && last.quantile(.9999) <= 160);
});
test('drawing starts from the law in force, and drawing exactly the law changes little', () => {
  const c = applyPreset({threads: 200, plannedTurns: 8, plannedTurnsP95: 20, tokenMedian: 3700, tokenP95: 9000, gapMedianMinutes: 4, gapP95Minutes: 60, continuationProbability: 1}, 'sonnet');
  const drawn = Object.fromEntries(['tokenDrawn', 'gapDrawn', 'lengthDrawn', 'inputNeedDrawn', 'retrievalNeedDrawn', 'outputNeedDrawn'].map(name => [name, lawWeights(name, c)]));
  for (const x of [1, 5, 30, 120]) near(returnCdf(x, {...c, gapDrawn: drawn.gapDrawn}), returnCdf(x, c), .02, 'return');
  for (const turn of [2, 5, 11, 21]) near(theoreticalSurvival({...c, lengthDrawn: drawn.lengthDrawn})[turn - 1].realized, theoreticalSurvival(c)[turn - 1].realized, .02, 'survival');
  near(drawnStats('tokenDrawn', drawn.tokenDrawn).median, 3700, 250);
  // Other numbers are drawn, so single runs differ by what any two seeds differ: the mean of a few does not.
  const mean = o => [100, 110, 120, 130].reduce((n, seed) => n + simulate({...c, ...o, seed}).savings.percent / 4, 0); near(mean(drawn), mean({}), 5, 'saving');
});
test('a drawn distribution is what is sampled', () => {
  const only = (kind, k) => BINS[kind].edges.slice(1).map((_, i) => i === k ? 1 : 0);
  const w = generateWorkload({...DEFAULTS, threads: 40, lengthDrawn: only('length', 3), gapDrawn: only('gap', 4), tokenDrawn: only('token', 5), retrievalNeedDrawn: only('need', 1), retrievalDependencyProbability: 1});
  for (const t of w.threads) {
    assert.equal(t.turns.length, 4);
    t.turns.forEach((turn, i) => { assert.ok(turn.retrievalTokens >= 2000 && turn.retrievalTokens <= 4000); if (i) { const gap = turn.time - t.turns[i - 1].time; assert.ok(gap >= 2 && gap <= 5); } if (i >= 2) assert.equal(i - turn.dependencies.find(d => d.kind === 'retrieval').turn, 2); });
  }
  assert.throws(() => configure({gapDrawn: [1, 2]}), /gapDrawn must be 14 weights/); assert.throws(() => configure({gapDrawn: new Array(14).fill(0)}), /some weight/); assert.throws(() => configure({gapDrawn: new Array(14).fill(-1)}), /at least 0/);
});
test('binning keeps every value, and weights of a law sum to one', () => {
  assert.equal(weightsFrom('gap', [0, .05, 3, 1e9]).reduce((a, b) => a + b, 0), 4); assert.equal(weightsFrom('gap', [1e9]).at(-1), 1);
  near(weightsOf('gap', x => returnCdf(x, DEFAULTS)).reduce((a, b) => a + b, 0), 1, 1e-9);
});
test('a chat log is measured as the package measures it', () => {
  const rows = parseRows(gunzipSync(file('threads.jsonl.gz')).toString()), labels = parseRows(file('labels.jsonl').toString()), m = measure(rows, labels);
  // The numbers `compache profile` prints for the same files.
  const s = m.scenario; assert.deepEqual([m.threads, m.turns], [4, 71]); assert.equal(m.source, '4 threads, 71 turns');
  assert.deepEqual([s.inputTokens, s.outputTokens, s.tokenMedian, s.tokenP95], [36, 117, 3209, 3850]);
  assert.deepEqual([s.systemTokens, s.archivedTokens, s.answerSeconds], [14, 168, 3]);
  assert.deepEqual([s.inputDependencyProbability, s.retrievalDependencyProbability, s.outputDependencyProbability], [0.21, 0.129, 0.21]);
  assert.deepEqual([s.gapMedianMinutes, s.gapP95Minutes, s.plannedTurns, s.plannedTurnsP95, s.inputDependencyTurnsBackP95, s.retrievalDependencyTurnsBackP95, s.outputDependencyTurnsBackP95], [3.92, 34.95, 10, 26, 7, 15, 9]);
  assert.equal(s.lengthDrawn.reduce((a, b) => a + b, 0), 4); assert.equal(s.gapDrawn.reduce((a, b) => a + b, 0), 67);
  configure(applyPreset(s, 'sonnet'));
  // Without labels the needs keep their defaults.
  assert.equal(measure(rows).scenario.inputDependencyProbability, undefined);
});
test('message shapes: blocks, tool results, merged questions, unanswered ends and what cannot be read', () => {
  const t = readThread({thread_id: 't', model: 'm', messages: [{role: 'system', content: 'S'}, {role: 'retrieval', content: 'early'}, {role: 'user', content: [{type: 'text', text: 'a'}], at: 1000}, {role: 'user', content: 'b', at: 1001}, {role: 'assistant', content: 'x', at: 1005}, {role: 'user', content: [{type: 'tool_result', content: 'found'}], at: 1006}, {role: 'assistant', content: 'y', at: 1010}, {role: 'user', content: 'never answered', at: 2000}]});
  assert.equal(t.runs.length, 1); assert.deepEqual(t.runs[0], {start: 1000, end: 1010, input: 'a\n\nb', retrieved: ['early', 'found'], output: 'x\n\ny', calls: 2}); assert.equal(t.system, 'S');
  assert.equal(readThread({runs: [{started_at: '2026-01-01T00:00:00', ended_at: '2026-01-01T00:00:05Z', input: 'q', retrievals: [{text: 'r'}, {text: 'hidden', exposed_to_model: false}], output: 'a'}]}).runs[0].retrieved.length, 1);
  assert.throws(() => measure([{messages: [{role: 'user', content: 'no time'}, {role: 'assistant', content: 'x'}]}]), /needs a time/);
  assert.throws(() => measure([{messages: [{role: 'wizard', content: 'x'}]}]), /unknown role/); assert.throws(() => measure([{thread_id: 'x'}]), /messages/); assert.throws(() => parseRows('{"a":1}\nnope'), /line 2/);
});
test('a fitted policy file gives its decision, its candidates and its costs', () => {
  const recipe = {input_runs: 4, output_runs: null, retrieval_runs: 1, protect_query_matches: true, keep_under_tokens: 0};
  const data = {schema_version: 'compache-workload-planner-v1', model_spec: {model_id: 'claude-sonnet-5-5'}, recipes: [{eligible: false, recipe: {}}, {eligible: true, recipe}], planner_options: {max_pings: 3}, training_threads: ['a', 'b'], pricing: {preset: 'sonnet', input_price_per_million: 2},
    solve: {turns: 10, keep_everything_cost: 100000, saving_fraction: .5, chosen: {coverage: .9, expected_cost: 50000}, candidates: [{recipe: null, max_pings: 0, coverage: 1, expected_cost: 100000, complete: true}, {recipe, max_pings: 3, coverage: .9, expected_cost: 50000, complete: true}, {recipe, max_pings: 0, expected_cost: 1, complete: false}]}};
  const p = readPolicy(data);
  assert.deepEqual({...p.policy}, {questions: 4, answers: null, retrieved: 1, maxPings: 3, keepUnder: 0, long: false}); assert.equal(p.preset, 'sonnet'); near(p.cost, 10, 1e-9); near(p.everything, 20, 1e-9);
  assert.deepEqual(p.candidates.map(c => c.chosen), [false, true]); assert.equal(p.notes.length, 1);
  // A policy that asks for a longer cache carries the provider's default beside it, and each candidate its own.
  const lasting = readPolicy({...data, model_spec: {model_id: 'claude-sonnet-5-5', cache_ttl_seconds: 3600}, standard_spec: {model_id: 'claude-sonnet-5-5', cache_ttl_seconds: 300},
    solve: {...data.solve, candidates: [300, 3600].map(seconds => ({recipe, max_pings: 3, cache_ttl_seconds: seconds, coverage: .9, expected_cost: 50000, complete: true}))}});
  assert.equal(lasting.policy.long, true); assert.deepEqual(lasting.lives, [5, 60]); assert.deepEqual(lasting.candidates.map(c => [c.policy.long, c.chosen]), [[false, false], [true, true]]);
  assert.throws(() => readPolicy({}), /not a policy/);
});
test('every comparison is a valid set of rules for every model, and your app starts as sending everything', () => {
  for (const id of Object.keys(presets)) {
    const list = comparisons(configure(applyPreset({}, id))); assert.equal(list[0].id, 'app'); assert.deepEqual(list[0].rules, configureRules({})); assert.deepEqual(list.find(b => b.id === 'everything').rules, configureRules({}));
    assert.equal(list.some(b => b.id === 'compaction'), presets[id].anthropic); assert.equal(list.find(b => b.id === 'openclaw').rules.clear, presets[id].anthropic);
    for (const b of list) assert.deepEqual(Object.keys(b.rules).sort(), Object.keys(RULES).sort());
  }
  assert.throws(() => configureRules({pings: -1}), /app.pings/); assert.throws(() => configureRules({nope: 1}), /not a rule/); assert.throws(() => configureRules({cache: 'forever'}), /app.cache/);
});
test('every starting point is a valid scenario that fits', () => {
  for (const s of STARTS) for (const id of ['sonnet', 'luna']) { const r = simulate(applyPreset({...s.scenario, threads: 12}, id)); assert.ok(r.compache.complete && r.savings, `${s.id} ${id}`); }
});
