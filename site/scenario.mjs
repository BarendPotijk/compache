// What carries a scenario into the calculator: a share link, a profile made from traces, or a file this page exported.
import {DEFAULTS, configure} from './engine.mjs';
import {applyPreset, presetKey} from './models.mjs';

export const LINK_VERSION = 6;
export const EXPORT_SCHEMA = `compache-cost-explorer-v${LINK_VERSION}`;
export const PROFILE_SCHEMA = 'compache-profile-v1';
const KINDS = ['input', 'retrieval', 'output'];
const isObject = value => value !== null && typeof value === 'object' && !Array.isArray(value);

// Settings saved by an earlier version of the page, rewritten for this one.
export function migrateScenario(version, scenario) {
  const v = Number(version), s = {...scenario};
  for (const kind of KINDS) {
    const chance = `${kind}DependencyProbability`, lookback = `${kind}DependencyLookback`;
    // Versions 1 and 2 had one chance and one look-back for every kind, and no floor on what stays in view.
    if (v <= 2) { s[chance] = s.dependencyProbability ?? .25; s[lookback] = s.dependencyLookback ?? 24; }
    s[chance] ??= s.dependencyProbability ?? .25;
    // A look-back of L turns meant any of the last L, equally likely. Its middle and its far end carry over.
    const turns = s[lookback] ?? s.dependencyLookback ?? 24;
    s[`${kind}DependencyTurnsBack`] = Math.max(1, Math.round(turns/2));
    s[`${kind}DependencyTurnsBackP95`] = turns;
  }
  if (v <= 2) s.retentionRequirement = .95;
  // Settings that no longer exist are dropped.
  const known = Object.fromEntries(Object.entries(s).filter(([key]) => Object.hasOwn(DEFAULTS, key)));
  if (isObject(known.model)) known.model = Object.fromEntries(Object.entries(known.model).filter(([key]) => Object.hasOwn(DEFAULTS.model, key)));
  return known;
}

// The scenario in a link's fragment, or null when the fragment carries none.
export function parseLink(hash) {
  const text = hash.replace(/^#/, '');
  if (text.length > 20000) throw new Error('the link is too long');
  const query = new URLSearchParams(text);
  if (!query.has('scenario')) return null;
  let scenario;
  try { scenario = JSON.parse(query.get('scenario')); } catch { throw new Error('the scenario in the link is not valid JSON'); }
  return {version: query.get('v'), model: query.get('model') ?? 'custom', scenario, source: query.get('source') ?? ''};
}

// The scenario in a profile made from traces or in a file this page exported.
export function parseFile(data) {
  if (isObject(data) && data.schema_version === PROFILE_SCHEMA) {
    if (!isObject(data.calculator)) throw new Error('the profile has no calculator section');
    const {v = LINK_VERSION, model = 'custom', scenario, source = ''} = data.calculator;
    return {version: v, model, scenario, source};
  }
  const exported = isObject(data) && /^compache-cost-explorer-v(\d+)$/.exec(data.schemaVersion ?? '');
  if (exported) return {version: exported[1], model: data.modelPreset ?? 'custom', scenario: data.config, source: data.source ?? ''};
  throw new Error('this file is not a Compache profile or a calculator export');
}

// Check what was carried in and complete it with defaults. An unknown or out-of-range setting is an error.
export function resolveScenario({version, model, scenario, source = ''}) {
  const v = Number(version);
  if (!Number.isInteger(v) || v < 1) throw new Error('the version is missing');
  if (v > LINK_VERSION) throw new Error('this was made with a newer version of the calculator');
  if (!isObject(scenario)) throw new Error('the scenario must be an object of settings');
  // A preset by its key or by the package's model ID, or custom prices carried in the scenario.
  const preset = model === 'custom' ? 'custom' : presetKey(model);
  if (!preset) throw new Error(`${model} is not a model preset`);
  const legacy = v < LINK_VERSION, settings = legacy ? migrateScenario(v, scenario) : scenario;
  if (preset === 'custom' && !isObject(settings.model)) throw new Error('custom prices need a model object with the prices');
  return {preset, config: configure(applyPreset(settings, preset)), source: String(source).slice(0, 120), legacy};
}

export const shareHash = (config, preset, source = '') =>
  new URLSearchParams({v: String(LINK_VERSION), model: preset, scenario: JSON.stringify(config), ...(source ? {source} : {})}).toString();
