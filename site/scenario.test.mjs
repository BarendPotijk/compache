import test from 'node:test';
import assert from 'node:assert/strict';
import {DEFAULTS, configure, simulate} from './engine.mjs';
import {presets, applyPreset} from './models.mjs';
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
  // Everything else is carried over, removed settings are gone, and the new OpenClaw constants take their defaults.
  for (const key of ['tokenMedian', 'gapP95Minutes', 'retentionRequirement', 'seed', 'toolClearing', 'ttlMinutes']) assert.equal(config[key], old[key], key);
  assert.deepEqual(Object.keys(config).sort(), Object.keys(DEFAULTS).sort());
  assert.equal(config.clearKeepResults, 3); assert.equal(config.summaryTriggerShare, 0.75);
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
test('a share link carries every setting and the source label back unchanged', () => {
  const config = configure(applyPreset({tokenMedian: 3615, tokenP95: 4207, gapMedianMinutes: 4.18, gapP95Minutes: 31.3, plannedTurns: 27, plannedTurnsP95: 87, retrievalDependencyTurnsBack: 3, retrievalDependencyTurnsBackP95: 29}, 'sonnet'));
  const back = resolveScenario(parseLink(`#${shareHash(config, 'sonnet', '30 threads, 1,227 turns')}`));
  assert.deepEqual(back, {preset: 'sonnet', config, source: '30 threads, 1,227 turns', legacy: false});
  assert.equal(parseLink(''), null); assert.equal(parseLink('#section'), null);
  assert.equal(resolveScenario(parseLink(`#${shareHash(config, 'sonnet')}`)).source, '');
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
  assert.deepEqual(config.model, presets.sonnet.model); assert.equal(config.ttlMinutes, 5); assert.equal(config.toolClearing, true);
  // The package's model ID names the same preset as the calculator's key.
  assert.deepEqual(resolveScenario(parseFile({...profile, calculator: {...profile.calculator, model: 'claude-sonnet-5-5'}})), {preset: 'sonnet', config, source, legacy});
  // The same profile through a link, as `compache profile` prints it.
  assert.deepEqual(resolveScenario(parseLink(link(6, 'sonnet', scenario, '30 threads, 1,227 turns'))).config, config);
  // It runs: the workload and both arms complete.
  const r = simulate({...config, threads: 6});
  assert.ok(r.compache.complete && r.openclaw.complete && r.optimization.feasible);
  // A model the calculator has no preset for brings its own prices and cache rules.
  const own = {...scenario, ttlMinutes: 60, cacheMinTokens: 2048, cacheBlockTokens: 64, model: {context: 400000, maxInput: 380000, inputRate: 3, cachedRate: 0.3, outputRate: 15, writeMultiplier: 2, longThreshold: 0, longInputMultiplier: 1, longOutputMultiplier: 1}};
  const custom = resolveScenario(parseFile({...profile, calculator: {model: 'custom', scenario: own}}));
  assert.equal(custom.preset, 'custom'); assert.deepEqual(custom.config.model, own.model); assert.equal(custom.config.ttlMinutes, 60); assert.equal(custom.config.cacheBlockTokens, 64);
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
  assert.throws(() => resolveScenario({version: 6, model: 'sonnet', scenario: {...ok, tokenMedain: 1}}), /tokenMedain is not a setting/);
  assert.throws(() => resolveScenario({version: 6, model: 'sonnet', scenario: {...ok, gapMedianMinutes: 0}}), /gapMedianMinutes must be between/);
  assert.throws(() => resolveScenario({version: 6, model: 'sonnet', scenario: {...ok, inputTokens: 29.5}}), /inputTokens must be a whole number/);
  assert.throws(() => resolveScenario({version: 6, model: 'sonet', scenario: ok}), /sonet is not a model preset/);
  assert.throws(() => resolveScenario({version: 6, model: 'custom', scenario: ok}), /custom prices need a model/);
  assert.throws(() => resolveScenario({version: 6, model: 'sonnet', scenario: [1]}), /must be an object/);
  assert.throws(() => resolveScenario({version: 7, model: 'sonnet', scenario: ok}), /newer version/);
  assert.throws(() => resolveScenario({version: null, model: 'sonnet', scenario: ok}), /version is missing/);
  assert.throws(() => parseLink('#v=6&model=sonnet&scenario={oops'), /not valid JSON/);
  assert.throws(() => parseFile({hello: 1}), /not a Compache profile/); assert.throws(() => parseFile(null), /not a Compache profile/);
  assert.throws(() => parseFile({schema_version: 'compache-profile-v1'}), /no calculator section/);
  // A long source label is cut, not refused.
  assert.equal(resolveScenario({version: 6, model: 'sonnet', scenario: ok, source: 'x'.repeat(500)}).source.length, 120);
});
