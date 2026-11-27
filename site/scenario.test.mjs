import test from 'node:test';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {DEFAULTS, configure, simulate} from './engine.mjs';
import {presets, applyPreset} from './models.mjs';
import {STARTS} from './starts.mjs';
import {LINK_VERSION, EXPORT_SCHEMA, migrateScenario, parseLink, parseFile, resolveScenario, shareHash} from './scenario.mjs';

// The full settings a version 5 link carried: the calculator's defaults of that version.
const V5 = {seed: 42, companies: 1, threads: 100, maxTurns: 160, lengthDistribution: 'lognormal', plannedTurns: 8, plannedTurnsP95: 20,
  continuationProbability: 0.99, tokenMedian: 25000, tokenP95: 45000, companyVariation: 0, gapMedianMinutes: 4, gapP95Minutes: 60,
  dependencyProbability: 0.25, dependencyLookback: 24, inputDependencyProbability: 0.25, retrievalDependencyProbability: 0.25, outputDependencyProbability: 0.25,
  inputDependencyLookback: 24, retrievalDependencyLookback: 24, outputDependencyLookback: 24, retentionRequirement: 0.8, retrievalRetentionRuns: 0,
  inputTokens: 200, outputTokens: 500, ttlMinutes: 5, target: 0.2, warmTrigger: 0.5, maxPings: 12, keepWarm: true, pingLeadMinutes: 0.75,
  pingOutputTokens: 16, recoveryOutputTokens: 32, toolClearing: true, clearedTokens: 12, safetyReserve: 32000, systemTokens: 2048,
  cacheMinTokens: 512, cacheBlockTokens: 1, tailTokens: 20000, summaryTokens: 4000, summaryInputCap: 40000, toolSummaryTokens: 500,
  summarySystemTokens: 2048, recordTokens: 1500, flushOutputTokens: 256,
  model: {context: 1000000, maxInput: 1000000, inputRate: 2, cachedRate: 0.1, outputRate: 10, writeMultiplier: 1.25, longThreshold: 0, longInputMultiplier: 1, longOutputMultiplier: 1}};
const link = (version, model, scenario, source) => `#${new URLSearchParams({v: String(version), model, scenario: JSON.stringify(scenario), ...(source ? {source} : {})})}`;

test('a version 5 link loads: each look-back becomes a median of half and a 95th percentile of the whole', () => {
  const old = {...V5, retrievalDependencyLookback: 10, outputDependencyLookback: 5, inputDependencyProbability: 0.4};
  const {preset, config, legacy} = resolveScenario(parseLink(link(5, 'sonnet', old)));
  assert.equal(preset, 'sonnet'); assert.equal(legacy, true);
  assert.deepEqual([config.inputDependencyTurnsBack, config.inputDependencyTurnsBackP95], [12, 24]);
  assert.deepEqual([config.retrievalDependencyTurnsBack, config.retrievalDependencyTurnsBackP95], [5, 10]);
  assert.deepEqual([config.outputDependencyTurnsBack, config.outputDependencyTurnsBackP95], [3, 5]);
  assert.deepEqual([config.inputDependencyProbability, config.retrievalDependencyProbability, config.outputDependencyProbability], [0.4, 0.25, 0.25]);
  // Everything else is carried over, and removed settings are gone.
  for (const key of ['tokenMedian', 'gapP95Minutes', 'retentionRequirement', 'seed', 'ttlMinutes']) assert.equal(config[key], old[key], key);
  assert.equal(config.anthropic, true); assert.equal(config.compare, 'openclaw'); assert.deepEqual({...config.app}, {});
  assert.deepEqual(Object.keys(config).sort(), Object.keys(DEFAULTS).sort());
  // Read as a current link, the same settings are refused instead of silently dropped.
  assert.throws(() => resolveScenario(parseLink(link(LINK_VERSION, 'sonnet', old))), /is not a setting/);
});
test('versions 1 and 2 had one chance and one look-back for all kinds and no floor', () => {
  const {config} = resolveScenario({version: 2, model: 'luna', scenario: {dependencyProbability: 0.4, dependencyLookback: 9, plannedTurns: 12, plannedTurnsP95: 30}});
  for (const kind of ['input', 'retrieval', 'output']) assert.deepEqual([config[`${kind}DependencyProbability`], config[`${kind}DependencyTurnsBack`], config[`${kind}DependencyTurnsBackP95`]], [0.4, 5, 9]);
  assert.equal(config.retentionRequirement, 0.95); assert.equal(config.plannedTurns, 12);
  const bare = migrateScenario(1, {});
  assert.deepEqual([bare.inputDependencyProbability, bare.inputDependencyTurnsBack, bare.inputDependencyTurnsBackP95], [0.25, 12, 24]);
  assert.deepEqual(migrateScenario(4, {retrievalDependencyLookback: 1, unknownOldKey: 3, model: {context: 5, gone: 1}}).model, {context: 5});
  assert.equal(migrateScenario(4, {retrievalDependencyLookback: 1}).retrievalDependencyTurnsBack, 1);
});
test('a version 6 link with a changed OpenClaw constant carries it over as the rules of your app', () => {
  const six = resolveScenario(parseLink(link(6, 'sonnet', {tokenMedian: 4000, tokenP95: 9000, target: .2, warmTrigger: .5, maxPings: 12, toolClearing: true, clearKeepResults: 5, baselineReserve: 20000, tailTokens: 20000})));
  assert.equal(six.legacy, true); assert.equal(six.config.compare, 'app'); assert.equal(six.config.app.clearKeepResults, 5); assert.equal(six.config.app.clear, true); assert.equal(six.config.app.summarize, true); assert.equal(six.config.maxPings, DEFAULTS.maxPings);
  const usual = resolveScenario(parseLink(link(6, 'luna', {toolClearing: false, clearKeepResults: 3, tailTokens: 20000, inputDependencyTurnsBack: 2, inputDependencyTurnsBackP95: 9}))); assert.equal(usual.config.compare, 'openclaw'); assert.equal(usual.config.inputDependencyTurnsBack, 2);
});
test('drawn distributions, a policy and your app travel in a link', () => {
  const gapDrawn = new Array(14).fill(0); gapDrawn[3] = 2; gapDrawn[8] = 1;
  const config = configure(applyPreset({gapDrawn, policy: {questions: 4, retrieved: 1, maxPings: 1}, app: {keepTurns: 6, cache: 'long'}, compare: 'app', measuredCost: 12.5}, 'sonnet'));
  const hash = shareHash(config, 'sonnet'); assert.ok(hash.length < 8000, String(hash.length)); assert.deepEqual(resolveScenario(parseLink(`#${hash}`)).config, config);
});
test('a share link carries every setting and the source label back unchanged', () => {
  const config = configure(applyPreset({tokenMedian: 3615, tokenP95: 4207, gapMedianMinutes: 4.18, gapP95Minutes: 31.3, plannedTurns: 27, plannedTurnsP95: 87, retrievalDependencyTurnsBack: 3, retrievalDependencyTurnsBackP95: 29}, 'sonnet'));
  const back = resolveScenario(parseLink(`#${shareHash(config, 'sonnet', '30 threads, 1,227 turns')}`));
  assert.deepEqual(back, {preset: 'sonnet', config, source: '30 threads, 1,227 turns', legacy: false});
  assert.equal(parseLink(''), null); assert.equal(parseLink('#section'), null);
  assert.equal(resolveScenario(parseLink(`#${shareHash(config, 'sonnet')}`)).source, '');
});
// The defaults a version 7 link counts on for every setting it leaves out. When one changes, raise LINK_VERSION and
// let migrateScenario fill in the old one, or the links people have shared will say something else.
const V7 = {
  seed: 42, threads: 100, maxTurns: 160, lengthDistribution: 'lognormal', plannedTurns: 8, plannedTurnsP95: 20, continuationProbability: 0.99,
  continueFirst: 0.6, continueLong: 0.95, continueHalfTurn: 3, tokenMedian: 25000, tokenP95: 45000, retrievalShare: 1, inputTokens: 200, outputTokens: 500,
  reasoningTokens: 0, systemTokens: 2048, gapMedianMinutes: 4, gapP95Minutes: 60, gapFirstFactor: 1, gapHalfTurn: 3, breakShare: 0, breakMedianMinutes: 1440,
  answerSeconds: 0, callsPerTurn: 1, sharedPrefixWarm: 0, inputDependencyProbability: 0.15, inputDependencyTurnsBack: 1, inputDependencyTurnsBackP95: 20,
  retrievalDependencyProbability: 0.15, retrievalDependencyTurnsBack: 3, retrievalDependencyTurnsBackP95: 30, outputDependencyProbability: 0.2,
  outputDependencyTurnsBack: 1, outputDependencyTurnsBackP95: 35, tokenDrawn: null, gapDrawn: null, lengthDrawn: null, inputNeedDrawn: null,
  retrievalNeedDrawn: null, outputNeedDrawn: null, ttlMinutes: 30, cacheMinTokens: 1024, cacheBlockTokens: 1, cacheHitShare: 1, longTtlMinutes: 720,
  longWriteMultiplier: 2.5, anthropic: false, retentionRequirement: 0.8, keepWarm: true, maxPings: 3, pingLeadMinutes: 0.75, pingOutputTokens: 16,
  recoveryOutputTokens: 32, safetyReserve: 32000, archivedTokens: 28, policy: null, compare: 'everything', app: {}, measuredCost: 0,
  model: {context: 1050000, maxInput: 922000, inputRate: 0.1, cachedRate: 0.01, outputRate: 0.5, writeMultiplier: 1.25, longThreshold: 272000, longInputMultiplier: 2, longOutputMultiplier: 1.5}};
const V7_STARTS = '80729d006b86cf64';
test('version 7 links and exports keep the original baseline and settings', () => {
  const {config} = resolveScenario(parseLink('#v=7&model=luna'));
  assert.deepEqual(config, configure(V7));
  const old = resolveScenario(parseLink('#v=7&model=sonnet&compare=window&maxPings=1'));
  assert.equal(old.config.compare, 'window');
  assert.equal(old.config.maxPings, 1);
  const file = parseFile({schemaVersion: 'compache-cost-explorer-v7', modelPreset: 'sonnet', config: {compare: 'app', app: {keepTurns: 4}}});
  assert.equal(resolveScenario(file).config.app.keepTurns, 4);
});

test('a share link names only what it changes', () => {
  assert.equal(LINK_VERSION, 8); assert.deepEqual(JSON.parse(JSON.stringify(DEFAULTS)), {...V7, compare: 'openclaw'});
  assert.equal(shareHash(configure(applyPreset({}, 'sonnet')), 'sonnet'), 'v=8&model=sonnet');
  const config = configure(applyPreset({gapMedianMinutes: 30, gapP95Minutes: 240, keepWarm: false}, 'haiku'));
  assert.equal(shareHash(config, 'haiku'), 'v=8&model=haiku&gapMedianMinutes=30&gapP95Minutes=240&keepWarm=false');
  assert.deepEqual(resolveScenario(parseLink('#v=8&model=claude-haiku-5-5&gapMedianMinutes=30&gapP95Minutes=240&keepWarm=false')), {preset: 'haiku', config, source: '', legacy: false});
  // From a starting point, only what differs from it. Its numbers are part of what a link counts on, as the defaults are.
  assert.equal(createHash('sha256').update(JSON.stringify(STARTS.map(s => [s.id, s.scenario]))).digest('hex').slice(0, 16), V7_STARTS);
  const chat = configure(applyPreset({...STARTS[0].scenario, gapMedianMinutes: 30, gapP95Minutes: 240}, 'sonnet'));
  assert.equal(shareHash(chat, 'sonnet', '', 'documents'), 'v=8&model=sonnet&start=documents&gapMedianMinutes=30&gapP95Minutes=240');
  assert.deepEqual(resolveScenario(parseLink('#v=8&model=sonnet&start=documents&gapMedianMinutes=30&gapP95Minutes=240')).config, chat);
  assert.throws(() => parseLink('#v=8&model=sonnet&start=somewhere'), /somewhere is not a starting point/);
  // Your own rules, a policy and a drawn distribution, each by name.
  const own = configure(applyPreset({app: {keepTurns: 6, cache: 'long', clear: true}, compare: 'app', policy: {questions: 4, retrieved: 1}, gapDrawn: [0, 0, 2, 1, ...new Array(10).fill(0)]}, 'sonnet'));
  assert.equal(shareHash(own, 'sonnet'), 'v=8&model=sonnet&gapDrawn=0,0,2,1,0,0,0,0,0,0,0,0,0,0&compare=app&app.cache=long&app.keepTurns=6&app.clear=true&policy.questions=4&policy.retrieved=1&policy.maxPings=0');
  assert.deepEqual(resolveScenario(parseLink(`#${shareHash(own, 'sonnet')}`)).config, own);
  // A policy that keeps everything is still a policy, and a window without an end can be written out.
  const all = configure(applyPreset({policy: {}}, 'sonnet'));
  assert.equal(shareHash(all, 'sonnet'), 'v=8&model=sonnet&policy.maxPings=0');
  assert.deepEqual(resolveScenario(parseLink('#v=8&model=sonnet&policy.maxPings=0&policy.answers=all')).config, all);
  // Custom prices travel whole.
  const prices = configure({ttlMinutes: 60, model: {context: 400000, maxInput: 380000, inputRate: 3, cachedRate: 0.3, outputRate: 15, writeMultiplier: 2, longThreshold: 0, longInputMultiplier: 1, longOutputMultiplier: 1}});
  assert.deepEqual(resolveScenario(parseLink(`#${shareHash(prices, 'custom')}`)).config, prices);
  // What cannot be read is named.
  assert.throws(() => parseLink('#v=8&model=sonnet&tokenMedian=lots'), /tokenMedian must be a number/);
  assert.throws(() => parseLink('#v=8&model=sonnet&tokenMedian='), /tokenMedian must be a number/);
  assert.throws(() => parseLink('#v=8&model=sonnet&gapDrawn=1,,2'), /gapDrawn must be a number/);
  assert.throws(() => parseLink('#v=8&model=sonnet&keepWarm=maybe'), /keepWarm must be true or false/);
  assert.throws(() => parseLink('#v=8&model=custom&model.inputRate=cheap'), /model.inputRate must be a number/);
  assert.throws(() => resolveScenario(parseLink('#v=8&model=sonnet&tokenMedain=1')), /tokenMedain is not a setting/);
  assert.throws(() => resolveScenario(parseLink('#v=8&model=sonnet&app.keepTurn=1')), /app.keepTurn is not a rule/);
  assert.throws(() => resolveScenario(parseLink('#v=8&model=sonnet&policy.question=1')), /policy.question is not part of a policy/);
  assert.throws(() => resolveScenario(parseLink('#model=sonnet&tokenMedian=4000&tokenP95=9000')), /version is missing/);
  assert.throws(() => resolveScenario(parseLink('#v=9&model=sonnet&somethingNew=1')), /newer version/);
});
test('a link written by hand may name a median alone, and the spread goes with it', () => {
  const at = hash => resolveScenario(parseLink(hash)).config, pair = c => [c.gapMedianMinutes, c.gapP95Minutes];
  assert.deepEqual(pair(STARTS[0].scenario), [4, 22]);
  assert.deepEqual(pair(at('#v=7&model=sonnet&start=documents&gapMedianMinutes=30')), [30, 165]);
  assert.deepEqual(pair(at('#v=7&model=claude-haiku-5-5&gapMedianMinutes=120')), [120, 1800]);
  // One that fits the 95th percentile it was given with leaves it, so a shared link comes back as it was. A named one stays.
  assert.deepEqual(pair(at('#v=7&model=sonnet&start=documents&gapMedianMinutes=20')), [20, 22]);
  assert.throws(() => at('#v=7&model=sonnet&gapMedianMinutes=30&gapP95Minutes=20'), /gapP95Minutes must be at least gapMedianMinutes/);
});
test('every calculator link in the docs opens', () => {
  const links = ['README.md', 'docs/calculator.md', 'docs/guide.md', 'docs/comparisons.md'].flatMap(file => readFileSync(new URL(`../${file}`, import.meta.url), 'utf8').match(/barendpotijk\.github\.io\/compache\/#[^\s)`"]+/g) ?? []);
  assert.ok(links.length >= 1);
  for (const link of links) assert.ok(simulate(resolveScenario(parseLink(link.slice(link.indexOf('#')))).config).savings, link);
});
test('a profile made from traces fills in the workload and keeps the preset prices', () => {
  const scenario = {inputTokens: 30, outputTokens: 132, tokenMedian: 3615, tokenP95: 4207, systemTokens: 14, lengthDistribution: 'lognormal', plannedTurns: 27, plannedTurnsP95: 87,
    continuationProbability: 1, gapMedianMinutes: 4.18, gapP95Minutes: 31.3, inputDependencyProbability: 0.166, inputDependencyTurnsBack: 1, inputDependencyTurnsBackP95: 22,
    retrievalDependencyProbability: 0.132, retrievalDependencyTurnsBack: 3, retrievalDependencyTurnsBackP95: 29, outputDependencyProbability: 0.195, outputDependencyTurnsBack: 1,
    outputDependencyTurnsBackP95: 37, retentionRequirement: 0.8, pingOutputTokens: 16, recoveryOutputTokens: 32};
  const profile = {schema_version: 'compache-profile-v1', source: {threads: 30, turns: 1227}, calculator: {model: 'sonnet', scenario, source: '30 threads, 1,227 turns'}};
  const {preset, config, source, legacy} = resolveScenario(parseFile(profile));
  assert.equal(preset, 'sonnet'); assert.equal(source, '30 threads, 1,227 turns'); assert.equal(legacy, false);
  for (const [key, value] of Object.entries(scenario)) assert.equal(config[key], value, key);
  assert.deepEqual(config.model, presets.sonnet.model); assert.equal(config.ttlMinutes, 5); assert.equal(config.anthropic, true);
  // The package's model ID names the same preset as the calculator's key.
  assert.deepEqual(resolveScenario(parseFile({...profile, calculator: {...profile.calculator, model: 'claude-sonnet-5-5'}})), {preset: 'sonnet', config, source, legacy});
  // The same profile through a link, as `compache profile` prints it, and as it printed it before.
  const named = new URLSearchParams({v: String(LINK_VERSION), model: 'sonnet', ...scenario, source: '30 threads, 1,227 turns'});
  for (const made of [`#${named}`, link(LINK_VERSION, 'sonnet', scenario, '30 threads, 1,227 turns')]) assert.deepEqual(resolveScenario(parseLink(made)).config, config);
  // It runs: the workload and both arms complete.
  const r = simulate({...config, threads: 6});
  assert.ok(r.compache.complete && r.baseline.arm.complete && r.savings);
  // A model the calculator has no preset for brings its own prices and cache rules.
  const own = {...scenario, ttlMinutes: 60, cacheMinTokens: 2048, cacheBlockTokens: 64, model: {context: 400000, maxInput: 380000, inputRate: 3, cachedRate: 0.3, outputRate: 15, writeMultiplier: 2, longThreshold: 0, longInputMultiplier: 1, longOutputMultiplier: 1}};
  const custom = resolveScenario(parseFile({...profile, calculator: {model: 'custom', scenario: own}}));
  assert.equal(custom.preset, 'custom'); assert.deepEqual(custom.config.model, own.model); assert.equal(custom.config.ttlMinutes, 60); assert.equal(custom.config.cacheBlockTokens, 64);
});
test('a profile carries what depends on the position in a thread, and a link without it behaves as before', () => {
  // What `compache profile` writes when the threads show these effects.
  const measured = {lengthDistribution: 'continuation', continueFirst: 0.42, continueLong: 0.96, continueHalfTurn: 5, plannedTurns: 2, plannedTurnsP95: 31, continuationProbability: 1,
    retrievalShare: 0.83, reasoningTokens: 310, callsPerTurn: 2.4, answerSeconds: 11.5, gapMedianMinutes: 3.2, gapP95Minutes: 41, gapFirstFactor: 2.6, gapHalfTurn: 3,
    breakShare: 0.07, breakMedianMinutes: 960.5, sharedPrefixWarm: 0.25};
  const {config, legacy} = resolveScenario(parseLink(link(LINK_VERSION, 'claude-sonnet-5-5', measured, '212 threads, 1,904 turns')));
  assert.equal(legacy, false);
  for (const [key, value] of Object.entries(measured)) assert.equal(config[key], value, key);
  // It runs, and the share link gives the same scenario back.
  const r = simulate({...config, threads: 8});
  assert.ok(r.compache.complete && r.baseline.arm.complete && r.savings);
  assert.ok(r.compache.byRole.further.calls > 0 && r.baseline.arm.byRole.further.calls > 0);
  assert.deepEqual(resolveScenario(parseLink(`#${shareHash(config, 'sonnet')}`)).config, config);
  // The same through a profile file.
  assert.deepEqual(resolveScenario(parseFile({schema_version: 'compache-profile-v1', calculator: {v: LINK_VERSION, model: 'sonnet', scenario: measured}})).config, config);
  // A link made before these settings existed takes their defaults, which change nothing.
  const before = resolveScenario(parseLink(link(6, 'sonnet', {tokenMedian: 3615, tokenP95: 4207}))).config;
  for (const key of ['continueFirst', 'continueLong', 'continueHalfTurn', 'gapFirstFactor', 'gapHalfTurn', 'breakShare', 'breakMedianMinutes', 'answerSeconds', 'callsPerTurn', 'sharedPrefixWarm', 'retrievalShare', 'reasoningTokens', 'cacheHitShare', 'archivedTokens'])
    assert.equal(before[key], DEFAULTS[key], key);
  assert.equal(before.lengthDistribution, 'lognormal');
  // A value out of range is refused by name.
  assert.throws(() => resolveScenario({version: LINK_VERSION, model: 'sonnet', scenario: {callsPerTurn: 0}}), /callsPerTurn must be between 1 and 100/);
  assert.throws(() => resolveScenario({version: LINK_VERSION, model: 'sonnet', scenario: {lengthDistribution: 'chance'}}), /lengthDistribution must be/);
});
test('an export of this page loads again, and an older export is migrated', () => {
  const config = configure(applyPreset({seed: 9}, 'opus'));
  const now = resolveScenario(parseFile({schemaVersion: EXPORT_SCHEMA, modelPreset: 'opus', source: 'my traces', config, workload: {}, compache: {}}));
  assert.deepEqual(now.config, config); assert.equal(now.source, 'my traces');
  const before = resolveScenario(parseFile({schemaVersion: 'compache-cost-explorer-v5', modelPreset: 'sonnet', config: V5}));
  assert.equal(before.legacy, true); assert.equal(before.config.inputDependencyTurnsBack, 12);
});
test('whatever cannot be loaded is an error that names the cause, never a default', () => {
  const ok = {tokenMedian: 3615};
  assert.throws(() => resolveScenario({version: LINK_VERSION, model: 'sonnet', scenario: {...ok, tokenMedain: 1}}), /tokenMedain is not a setting/);
  assert.throws(() => resolveScenario({version: LINK_VERSION, model: 'sonnet', scenario: {...ok, gapMedianMinutes: 0}}), /gapMedianMinutes must be between/);
  assert.throws(() => resolveScenario({version: LINK_VERSION, model: 'sonnet', scenario: {...ok, inputTokens: 29.5}}), /inputTokens must be a whole number/);
  assert.throws(() => resolveScenario({version: LINK_VERSION, model: 'sonet', scenario: ok}), /sonet is not a model preset/);
  assert.throws(() => resolveScenario({version: LINK_VERSION, model: 'custom', scenario: ok}), /custom prices need a model/);
  assert.throws(() => resolveScenario({version: LINK_VERSION, model: 'sonnet', scenario: [1]}), /must be an object/);
  assert.throws(() => resolveScenario({version: LINK_VERSION + 1, model: 'sonnet', scenario: ok}), /newer version/);
  assert.throws(() => resolveScenario({version: null, model: 'sonnet', scenario: ok}), /version is missing/);
  assert.throws(() => parseLink('#v=6&model=sonnet&scenario={oops'), /not valid JSON/);
  assert.throws(() => parseFile({hello: 1}), /not a Compache profile/); assert.throws(() => parseFile(null), /not a Compache profile/);
  assert.throws(() => parseFile({schema_version: 'compache-profile-v1'}), /no calculator section/);
  // A long source label is cut, not refused.
  assert.equal(resolveScenario({version: LINK_VERSION, model: 'sonnet', scenario: ok, source: 'x'.repeat(500)}).source.length, 120);
});
