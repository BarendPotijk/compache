import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {gunzipSync} from 'node:zlib';
import {DEFAULTS, candidateCuts, configure, createFitCache, describePolicy, fitPolicy, fitScenario, generateWorkload, pingCounts, runArm, simulate, theoreticalSurvival, continueChance, warmReturnShare, longAvailable} from './engine.mjs';
import {comparisons} from './baselines.mjs';
import {applyPreset, presets} from './models.mjs';
import {measure, parseRows, readThread} from './traces.mjs';
import {STARTS} from './starts.mjs';

// Mechanics are tested without the notice, so that token counts are the parts' own.
const config = o => configure({threads: 1, maxTurns: 8, plannedTurns: 8, plannedTurnsP95: 8, lengthDistribution: 'fixed', continuationProbability: 1, keepWarm: false, archivedTokens: 0, safetyReserve: 0, cacheMinTokens: 1, systemTokens: 100, inputTokens: 10, outputTokens: 20, tokenMedian: 300, tokenP95: 300, ttlMinutes: 5, gapMedianMinutes: 4, gapP95Minutes: 4, ...o});
const load = turns => ({threads: [{id: 'thread-1', turns}], totalRuns: turns.length});
const turn = (index, time, retrievalTokens = 300, dependencies = []) => ({index, time, inputTokens: 10, retrievalTokens, outputTokens: 20, dependencies});
const ours = (w, c, policy) => runArm(w, c, {kind: 'compache', policy});
const rules = (w, c, r) => runArm(w, c, {kind: 'rules', rules: r});
const answers = r => r.ledger.filter(x => x.role === 'answer');
const near = (a, b, tolerance) => assert.ok(Math.abs(a - b) <= tolerance, `${a} is not within ${tolerance} of ${b}`);

test('a saved fit prices new baselines and pinned policies exactly like a fresh fit', () => {
  const base = config({threads: 3}), published = JSON.parse(JSON.stringify(fitScenario(base)));
  const cached = createFitCache(published);
  for (const edits of [{}, {compare: 'uncached'}, {compare: 'app', app: {keepTurns: 2}},
    {compare: 'measured', measuredCost: 12}, {policy: {retrieved: 1}}]) {
    const c = configure({...base, ...edits});
    assert.strictEqual(cached(c, () => assert.fail('unexpected refit')), published);
    assert.deepEqual(simulate(c, cached), simulate(c));
  }
});

test('workload, model and planner changes invalidate the fit; repeated uses do no fitting work', () => {
  const base = config({threads: 2}), starter = fitScenario(base), cached = createFitCache(starter);
  for (const edits of [{seed: 25}, {inputTokens: 44}, {gapMedianMinutes: 1},
    {inputDependencyProbability: .1}, {retentionRequirement: .95}, {model: {...base.model, cachedRate: 2}}]) {
    const c = configure({...base, ...edits}), progress = [];
    const fitted = cached(c, (index, total) => progress.push([index, total]));
    assert.notStrictEqual(fitted, starter);
    assert.deepEqual(progress, fitted.candidates.map((_, i) => [i + 1, fitted.candidates.length]));
    assert.deepEqual(fitted, fitScenario(c));
    assert.strictEqual(cached(configure({...c, compare: 'uncached'})), fitted);
  }
  assert.strictEqual(cached(base), starter);
});

test('a cut keeps each kind for its own number of turns and waits for a cold cache', () => {
  const c = config({}), w = load([turn(0, 0), turn(1, 2), turn(2, 4), turn(3, 60), turn(4, 62)]), r = ours(w, c, {questions: 2, answers: 1, retrieved: 1});
  assert.deepEqual(r.decisions.map(d => [d.cache, d.action]), [['first', 'keep'], ['warm', 'keep'], ['warm', 'keep'], ['cold', 'cut'], ['warm', 'keep']]);
  // At turn 4: the question of the first turn, the answers and the retrieved text of the first two are archived.
  assert.deepEqual(r.decisions[3].parts, {input: 'akk', retrieval: 'aak', output: 'aak'});
  // Sent: the system prompt, two earlier questions, one answer, one retrieved text and this turn's own question and text.
  assert.equal(r.decisions[3].sentTokens, 100 + 20 + 20 + 300 + 310); assert.equal(r.decisions[3].fullTokens, 100 + 3 * 330 + 310); assert.equal(r.compactions, 1);
  // What was cut stays cut, and what came after is sent.
  assert.deepEqual(r.decisions[4].parts, {input: 'akkk', retrieval: 'aakk', output: 'aakk'});
  assert.equal(ours(w, c, {}).compactions, 0); assert.ok(r.totals.cost < ours(w, c, {}).totals.cost);
});
test('retrieved text is counted in turns that retrieved, and small parts stay', () => {
  const w = load([turn(0, 0), turn(1, 60, 0), turn(2, 120, 0), turn(3, 180)]), c = config({retrievalShare: .5});
  assert.equal(ours(w, c, {retrieved: 1}).decisions[3].parts.retrieval, 'k--');
  assert.equal(ours(w, c, {retrieved: 0}).decisions[3].parts.retrieval, 'a--');
  assert.equal(ours(w, c, {retrieved: 0, keepUnder: 300}).compactions, 0);
  // A part no larger than its notice is never archived: one part leaves 28, a turn's retrieved text what is set.
  assert.deepEqual(ours(w, config({archivedTokens: 28}), {questions: 0, answers: 0, retrieved: 0}).decisions[3].parts, {input: 'kkk', retrieval: 'a--', output: 'kkk'});
  assert.equal(ours(w, config({archivedTokens: 300}), {retrieved: 0}).compactions, 0);
});
test('a needed part that is archived costs one more call, for this turn only', () => {
  const c = config({recoveryOutputTokens: 32}), w = load([turn(0, 0), turn(1, 60), turn(2, 120, 300, [{kind: 'retrieval', turn: 0}, {kind: 'input', turn: 0}]), turn(3, 122)]), r = ours(w, c, {questions: 0, retrieved: 0});
  assert.equal(r.recalls, 2); assert.equal(r.byRole.recovery.calls, 1); assert.deepEqual(r.decisions[2].fetched, ['r0', 'i0']); assert.deepEqual(r.decisions[2].parts, {input: 'fa', retrieval: 'fa', output: 'kk'});
  const ask = r.ledger.find(x => x.role === 'recovery'), given = answers(r)[2];
  assert.equal(ask.outputTokens, 32); assert.equal(given.inputTokens, ask.inputTokens + 32 + 310); assert.equal(given.cachedTokens, ask.inputTokens);
  // Not kept: the next request holds the system prompt, the three answers and the new turn. Nothing but a one-turn text was in the way, so it cuts warm.
  assert.equal(answers(r)[3].inputTokens, 100 + 60 + 310); assert.equal(r.decisions[3].action, 'cut'); assert.equal(r.decisions[3].cache, 'warm'); assert.equal(r.retention.observed, 0); assert.equal(r.dependencyCoverage, 1);
});
test('the same chat and policy: the calculator cuts where the package does, and sends what it sends', () => {
  const read = path => readFileSync(new URL(`../${path}`, import.meta.url)), rows = parseRows(gunzipSync(read('examples/quickstart/threads.jsonl.gz')).toString()), recorded = JSON.parse(read('docs/assets/data/requests.json')).turns, t = readThread(rows[1]), tokens = s => Math.ceil(Buffer.byteLength(s) / 4) + 6;
  const turns = t.runs.map((r, i) => ({index: i, time: (r.start - t.runs[0].start) / 60, inputTokens: tokens(r.input), retrievalTokens: r.retrieved.reduce((n, p) => n + tokens(p), 0), outputTokens: tokens(r.output), dependencies: []}));
  const r = ours({threads: [{id: 'x', turns}], totalRuns: 26}, configure(applyPreset({...measure(rows).scenario, safetyReserve: 0}, 'sonnet')), {questions: 4, answers: 9, retrieved: 1, maxPings: 1});
  let alike = 0;
  r.decisions.forEach((d, i) => {
    const p = recorded[i], cut = p.archived > (i ? recorded[i - 1].archived : 0);
    near(d.fullTokens, p.full + 14, 1);
    if (i) { assert.equal(d.action === 'cut', d.cache === 'cold', `calculator, turn ${i + 1}`); assert.equal(cut, p.cache === 'cold', `package, turn ${i + 1}`); }
    if (d.cache === p.cache) alike++;
    // From turn 12 on both have cut on the same turns. The sizes differ by notices and reopened handles.
    if (i >= 11) { assert.equal(d.cache, p.cache); near(d.sentTokens / p.sent, 1, .04); }
  });
  assert.ok(alike >= 23, String(alike));
});
test('pings are planned in expectation, charged when nobody returns and cancelled by a return', () => {
  const c = config({keepWarm: true, maxPings: 3, systemTokens: 2048, tokenMedian: 100000, tokenP95: 100000, gapMedianMinutes: 10, gapP95Minutes: 10, pingLeadMinutes: .75, plannedTurns: 8, plannedTurnsP95: 8}), big = (i, at) => turn(i, at, 100000);
  const left = ours(load([big(0, 0)]), c, {maxPings: 3}); assert.deepEqual(left.ledger.filter(x => x.role === 'ping').map(x => x.time), [4.25, 8.5]);
  const back = ours(load([big(0, 0), big(1, 6)]), c, {maxPings: 3}); assert.equal(back.ledger.filter(x => x.role === 'ping' && x.turn === 0).length, 1); assert.ok(answers(back)[1].cachedTokens > 100000);
  const plan = left.decisions[0].planner; assert.equal(plan.chosen, 2); assert.equal(plan.rows.length, 4); assert.deepEqual(plan.rows[0], {pings: 0, expectedCost: 0, expectedGain: 0, net: 0}); near(plan.rows[2].net, plan.rows[2].expectedGain - plan.rows[2].expectedCost, 1e-12);
  // None without a return to wait for, when off, when the answer outlasts the interval or when the policy has none.
  for (const [o, p, why] of [[{plannedTurns: 1, plannedTurnsP95: 1}, 3, 'no modeled future return'], [{keepWarm: false}, 3, 'disabled'], [{answerSeconds: 300}, 3, 'answer outlasts the ping interval'], [{}, 0, 'no pings allowed']]) { const r = ours(load([big(0, 0)]), configure({...c, ...o}), {maxPings: p}); assert.equal(r.pings, 0); assert.equal(r.decisions[0].planner.reason, why); }
});
test('a decision knows nothing of the needs to come', () => {
  const c = config({}), make = deps => load([turn(0, 0), turn(1, 60), turn(2, 120), turn(3, 180, 300, deps)]), strip = r => r.decisions.slice(0, 3).map(d => [d.action, d.sentTokens, d.plannedPings]);
  assert.deepEqual(strip(ours(make([]), c, {retrieved: 1})), strip(ours(make([{kind: 'retrieval', turn: 0}]), c, {retrieved: 1})));
});
test('the rules of a comparison: caching off, a long cache, a window, dropped, shrunk and cleared text, pings', () => {
  const c = config({longTtlMinutes: 60, longWriteMultiplier: 2}), w = load([turn(0, 0), turn(1, 2), turn(2, 30), turn(3, 32, 300, [{kind: 'retrieval', turn: 0}])]), full = rules(w, c, {});
  assert.deepEqual(answers(full).map(x => x.inputTokens), [410, 740, 1070, 1400]); assert.deepEqual(answers(full).map(x => x.cachedTokens), [0, 410, 0, 1070]);
  const off = rules(w, c, {cache: 'off'}); assert.equal(off.totals.cachedTokens, 0); near(answers(off)[0].inputCost, 410 * c.model.inputRate / 1e6, 1e-15);
  const long = rules(w, c, {cache: 'long'}); assert.deepEqual(answers(long).map(x => x.cachedTokens), [0, 410, 740, 1070]); near(answers(long)[0].inputCost, 410 * c.model.inputRate * 2 / 1e6, 1e-15);
  const last = rules(w, c, {keepTurns: 1}); assert.deepEqual(answers(last).map(x => x.inputTokens), [410, 740, 740, 740]); assert.equal(last.dependencyCoverage, 0); assert.equal(last.byRole.recovery, undefined);
  const rag = rules(w, c, {dropRetrievedAfter: 0}); assert.deepEqual(answers(rag).map(x => x.inputTokens), [410, 440, 470, 500]); assert.equal(rag.dependencyCoverage, 0);
  const again = rules(w, c, {dropRetrievedAfter: 0, refetch: true}); assert.equal(again.byRole.recovery.calls, 1); assert.equal(again.dependencyCoverage, 1);
  assert.deepEqual(answers(rules(w, c, {compress: .5})).map(x => x.inputTokens), [260, 440, 620, 800]);
  const clear = rules(w, c, {clear: true, clearTriggerFloor: 700, clearKeepResults: 2, clearedTokens: 5, refetch: true}); assert.equal(clear.clearings, 2); assert.equal(clear.dependencyCoverage, 1); assert.equal(clear.byRole.recovery.calls, 1);
  const pinged = rules(load([turn(0, 0), turn(1, 8)]), config({pingLeadMinutes: 1}), {pings: 2}); assert.deepEqual(pinged.ledger.filter(x => x.role === 'ping').map(x => [x.turn, x.time]), [[0, 4], [1, 12], [1, 16]]); assert.equal(answers(pinged)[1].cachedTokens, 430);
});
test('a summary replaces what is older than the tail, from the request itself or from a digest, with its own calls', () => {
  const c = config({model: {context: 1000, maxInput: 1000}}), w = load([turn(0, 0, 200), turn(1, 1, 200), turn(2, 2, 200), turn(3, 3, 200, [{kind: 'retrieval', turn: 0}])]);
  const self = rules(w, c, {summarize: true, summaryTriggerShare: .75, tailTokens: 100, summaryTokens: 80}); const ask = self.ledger.find(x => x.role === 'summary'); assert.equal(ask.outputTokens, 80); assert.ok(ask.cachedTokens > 0); assert.equal(self.byRole.flush, undefined); assert.ok(self.compactions >= 1); assert.equal(self.dependencyCoverage, 0);
  const digest = rules(w, c, {summarize: true, tailTokens: 100, summaryTokens: 80, summaryOfRequest: false, summarySystemTokens: 100, summaryInputCap: 60, toolSummaryTokens: 50, flushOutputTokens: 5}); assert.equal(digest.ledger.find(x => x.role === 'summary').inputTokens, 160); assert.equal(digest.ledger.find(x => x.role === 'summary').cachedTokens, 0); assert.equal(digest.byRole.flush.calls, digest.byRole.summary.calls);
  const none = rules(w, c, {summarize: true, summaryTriggerTokens: 500, summaryTriggerShare: 1, tailTokens: 0, summaryTokens: 80}); assert.equal(answers(none)[2].inputTokens, 100 + 80 + 210);
});
test('the fit is deterministic, keeps everything as a candidate and takes the cheapest', () => {
  const c = configure(applyPreset({...STARTS[0].scenario, threads: 15}, 'sonnet')), w = generateWorkload(c), fit = fitPolicy(w, c);
  assert.deepEqual(fit, fitPolicy(w, c)); assert.ok(fit.candidates.filter(x => x.everything).length === 2 * pingCounts(c).length);
  for (const x of fit.candidates) { assert.ok(fit.chosen.cost <= x.cost + 1e-12); if (!x.everything) assert.ok(x.coverage >= .8 && x.kept < 1); }
  // Each cut on the frontier keeps more than every smaller one.
  const cuts = candidateCuts(w, c); assert.ok(cuts.length > 1 && cuts.length <= 6); cuts.slice(1).forEach((x, i) => { assert.ok(x.coverage >= cuts[i].coverage + .01 - 1e-9 && x.kept >= cuts[i].kept); });
  // A higher floor leaves fewer cuts and cannot be cheaper. No pings when they are off.
  assert.ok(fitPolicy(w, {...c, retentionRequirement: .99}).chosen.cost >= fit.chosen.cost); assert.deepEqual(pingCounts({...c, keepWarm: false}), [0]); assert.deepEqual(pingCounts({...c, maxPings: 5}), [0, 1, 3, 5]);
  // Nothing is needed and nothing can be archived.
  const tiny = {...c, archivedTokens: 1000000 / 100, inputTokens: 5, outputTokens: 5}; assert.deepEqual(candidateCuts(generateWorkload(tiny), tiny), []);
});
test('the policy is fitted on one set of threads and every side is costed on another', () => {
  const c = configure(applyPreset({...STARTS[0].scenario, threads: 10}, 'sonnet')), r = simulate(c), held = generateWorkload({...c, seed: c.seed + 1});
  assert.deepEqual(r.workload.threads, held.threads); assert.deepEqual(r.fit.chosen, fitPolicy(generateWorkload(c), c).chosen); near(r.compache.totals.cost, ours(held, c, r.policy).totals.cost, 1e-12);
  for (const s of r.sides) near(s.per1000, 1000 * rules(held, c, s.rules).totals.cost / held.totalRuns, 1e-9);
});
test('the package and the calculator agree on DocChat: the saving against sending everything is within four points', () => {
  // compache replay on DocChat's threads, users back after 4 minutes: 83%.
  const r = simulate(applyPreset({...STARTS[0].scenario, compare: 'everything'}, 'sonnet')); near(r.savings.percent, 83, 4); assert.equal(r.policy.retrieved, 1); assert.equal(r.policy.maxPings, 3);
  assert.ok(r.savings.range.low < r.savings.percent / 100 && r.savings.range.high > r.savings.percent / 100);
});
test('a policy of your own is priced instead of the fitted one, which is still reported', () => {
  const c = applyPreset({...STARTS[0].scenario, threads: 10, policy: {questions: 2, retrieved: 1}}, 'sonnet'), r = simulate(c);
  assert.deepEqual({...r.policy}, {questions: 2, answers: null, retrieved: 1, maxPings: 0, keepUnder: 0, long: false}); assert.equal(r.fit.pinned, true); assert.ok(r.fit.fittedHere > 0);
  assert.deepEqual(describePolicy(r.policy), ['Keep questions from the last 2 turns, all answers and retrieved text from the last turn.', 'Send no keep-warm pings.']);
  assert.deepEqual(describePolicy({questions: null, answers: null, retrieved: null, maxPings: 1, keepUnder: 0}), ['Keep the whole history in view.', 'Send up to 1 keep-warm ping while a user is away.']);
  for (const bad of [{questions: -1}, {answers: 1.5}, {other: 1}, {maxPings: null}, {long: 1}]) assert.throws(() => configure({policy: bad}), /policy/);
});
test('what is compared: a named comparison, your app, or a cost you measured', () => {
  const base = {...STARTS[0].scenario, threads: 8}, run = o => simulate(applyPreset({...base, ...o}, 'sonnet'));
  const app = run({compare: 'app', app: {keepTurns: 3}}); assert.equal(app.selected.id, 'app'); assert.equal(app.baseline.rules.keepTurns, 3); assert.deepEqual(app.config.app, {keepTurns: 3});
  const paid = run({compare: 'measured', measuredCost: 100}); assert.equal(paid.selected.id, 'measured'); assert.equal(paid.savings.range, null); near(paid.savings.percent, 100 - 1000 * paid.compache.totals.cost / paid.workload.totalRuns, 1e-9); assert.equal(paid.baseline.id, 'app');
  // The headline, ledger and shared config must agree even after a baseline becomes unavailable.
  for (const result of [run({}), run({compare: 'measured', measuredCost: 0}), simulate(applyPreset({...base, compare: 'compaction'}, 'luna'))]) {
    assert.equal(result.selected.id, 'openclaw');
    assert.equal(result.baseline.id, result.selected.id);
    assert.equal(result.config.compare, result.selected.id);
  }
  assert.throws(() => configure({compare: 'nope'}), /not a comparison/);
});
test('a side that does not fit the model has no cost and no saving', () => {
  const r = simulate({threads: 2, maxTurns: 3, plannedTurns: 3, plannedTurnsP95: 3, lengthDistribution: 'fixed', tokenMedian: 400, tokenP95: 400, systemTokens: 100, safetyReserve: 0, keepWarm: false, cacheMinTokens: 1, model: {context: 1200, maxInput: 1200}, outputTokens: 20, inputTokens: 10, compare: 'everything'});
  assert.equal(r.selected.complete, false); assert.equal(r.selected.per1000, null); assert.equal(r.savings, null);
  const none = simulate({threads: 1, maxTurns: 1, plannedTurns: 1, plannedTurnsP95: 1, lengthDistribution: 'fixed', tokenMedian: 2000, tokenP95: 2000, systemTokens: 100, safetyReserve: 0, keepWarm: false, model: {context: 500, maxInput: 500}}); assert.equal(none.fit.chosen, null); assert.equal(none.savings, null); assert.equal(none.compache.ledger[0].admitted, false);
});
test('answer time, further calls, reasoning, a shared prompt and a partly read entry are priced on both sides', () => {
  const w = load([turn(0, 0), turn(1, 2)]), c = config({});
  for (const arm of [{kind: 'compache', policy: {}}, {kind: 'rules', rules: {}}]) {
    const run = o => runArm(w, config(o), arm), plain = run({});
    const more = run({callsPerTurn: 3}); assert.equal(more.byRole.further.calls, 4); near(more.byRole.further.cachedTokens, 2 * (410 + 740), 1e-9);
    const thought = run({reasoningTokens: 50}); near(thought.totals.outputTokens - plain.totals.outputTokens, 100, 1e-9); assert.equal(answers(thought)[1].inputTokens, answers(plain)[1].inputTokens);
    assert.equal(answers(run({sharedPrefixWarm: .5}))[0].cachedTokens, 50); assert.equal(answers(run({cacheHitShare: .25}))[1].cachedTokens, 410 * .25);
    const row = answers(run({cacheHitShare: .25, sharedPrefixWarm: .5}))[1]; near(Object.values(row.cacheBreakdown).reduce((a, b) => a + b, 0), row.inputTokens, 1e-9);
  }
  assert.ok(warmReturnShare({...c, gapP95Minutes: 40, answerSeconds: 120}) < warmReturnShare({...c, gapP95Minutes: 40}));
});
test('thread lengths: the laws, position effects and the chance of another turn', () => {
  const s = theoreticalSurvival({plannedTurns: 2, plannedTurnsP95: 2, maxTurns: 4, lengthDistribution: 'fixed', continuationProbability: .8}); assert.deepEqual(s.map(x => x.planned), [1, 1, 0, 0]); near(s[1].realized, .8, 1e-12);
  const c = configure({lengthDistribution: 'continuation', continueFirst: .5, continueLong: .95, continueHalfTurn: 2}); near(continueChance(1, c), .5, 1e-12); near(continueChance(3, c), .725, 1e-12);
  const lengths = generateWorkload({...c, threads: 200}).threads.map(t => t.turns.length); near(lengths.filter(n => n === 1).length / 200, .5, .1);
  assert.ok(warmReturnShare({gapFirstFactor: .2, ttlMinutes: 5}, 1) > warmReturnShare({gapFirstFactor: .2, ttlMinutes: 5}, 50)); assert.ok(warmReturnShare({breakShare: .5}) < warmReturnShare({}));
});
test('the published result is pinned', () => {
  const r = simulate(applyPreset({compare: 'openclaw'}, 'sonnet')); assert.equal(r.selected.id, 'openclaw'); near(r.selected.per1000, 207.77, .01);
  assert.equal(presets.sonnet.anthropic, true); assert.ok(comparisons(r.config).length >= 10); assert.equal(Object.keys(DEFAULTS).includes('target'), false);
});
test('the long cache is fitted where a model has one to ask for', () => {
  const fitted = (preset, changes) => { const c = configure(applyPreset({...STARTS[0].scenario, threads: 25, ...changes}, preset)); return {c, fit: fitPolicy(generateWorkload(c), c)}; };
  // Users back in seconds find the default cache warm. After half an hour only the long one is.
  const quick = fitted('sonnet', {gapMedianMinutes: .5, gapP95Minutes: 3}), slow = fitted('sonnet', {gapMedianMinutes: 25, gapP95Minutes: 55});
  assert.equal(quick.fit.chosen.policy.long, false); assert.equal(slow.fit.chosen.policy.long, true);
  assert.ok(describePolicy(slow.fit.chosen.policy).includes('Ask for one-hour cache entries.'));
  // The package cannot ask OpenAI for its longer one on every API, so no policy here does either.
  const luna = fitted('luna', {gapMedianMinutes: 60, gapP95Minutes: 300});
  assert.ok(luna.fit.candidates.every(x => !x.policy.long)); assert.equal(longAvailable(luna.c), false);
  const pinned = long => simulate(applyPreset({...STARTS[0].scenario, threads: 10, policy: {questions: 2, answers: 2, retrieved: 1, maxPings: 1, long}}, 'luna'));
  assert.equal(pinned(true).policy.long, false); assert.equal(pinned(true).compache.totals.cost, pinned(false).compache.totals.cost);
  assert.equal(describePolicy({questions: null, answers: null, retrieved: null, maxPings: 0, keepUnder: 0, long: true}, 120)[1], 'Ask for 2-hour cache entries.');
  // Keeping everything under it is the comparison with the long cache.
  const everything = slow.fit.candidates.find(x => x.everything && x.policy.long && !x.policy.maxPings);
  const rules = runArm(generateWorkload(slow.c), slow.c, {kind: 'rules', rules: {cache: 'long'}}, {collectLedger: false});
  assert.ok(Math.abs(everything.cost - rules.totals.cost) < 1e-9 * rules.totals.cost);
});
