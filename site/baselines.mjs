// What Compache is compared with. Every comparison is a set of the same rules, so any of them can be edited into your own application.
// Defaults were read from the linked pages on the date given. Nothing here runs the products.
export const RULES = Object.freeze({
  // The provider's prompt cache: its standard lifetime, its long one at a higher write price, or none.
  cache: 'standard',
  // Keep-warm pings sent after every answer, whatever the expected return.
  pings: 0,
  // Only so many earlier turns are sent. 0 sends all of them.
  keepTurns: 0,
  // Retrieved text is dropped from the request this many turns after its own. -1 never drops it.
  dropRetrievedAfter: -1,
  // The share of retrieved text removed when it arrives.
  compress: 0,
  // Whether a cleared or dropped source that a turn needs is fetched again with one more call.
  refetch: false,
  // Old tool results are replaced by a placeholder once the request reaches a size.
  clear: false, clearTriggerShare: 0, clearTriggerFloor: 100000, clearKeepResults: 3, clearMinFreedShare: 0, clearMinFreedFloor: 0, clearedTokens: 12,
  // Older turns are replaced by a summary once the request reaches a size: a share of the window, a number of tokens, or whichever is less.
  summarize: false, summaryTriggerShare: 0.75, summaryTriggerTokens: 0, reserveTokens: 0, tailTokens: 20000, summaryTokens: 4000,
  // Whether the summary call sends the conversation as it stands, which can be read from the cache, or a digest of its older part.
  summaryOfRequest: true, summaryInputCap: 40000, toolSummaryTokens: 500, summarySystemTokens: 2048, recordTokens: 0, flushOutputTokens: 0
});
const RANGES = Object.freeze({
  pings: [0, 60, true], keepTurns: [0, 160, true], dropRetrievedAfter: [-1, 160, true], compress: [0, 0.99],
  clearTriggerShare: [0, 1], clearTriggerFloor: [0, 10000000, true], clearKeepResults: [1, 160, true], clearMinFreedShare: [0, 1], clearMinFreedFloor: [0, 10000000, true], clearedTokens: [0, 10000, true],
  summaryTriggerShare: [0.01, 1], summaryTriggerTokens: [0, 10000000, true], reserveTokens: [0, 1000000, true], tailTokens: [0, 1000000, true], summaryTokens: [1, 100000, true],
  summaryInputCap: [1, 1000000, true], toolSummaryTokens: [1, 100000, true], summarySystemTokens: [0, 100000, true], recordTokens: [0, 100000, true], flushOutputTokens: [0, 100000, true]
});
export function configureRules(overrides = {}) {
  if (overrides === null || typeof overrides !== 'object' || Array.isArray(overrides)) throw new TypeError('app must be an object of rules');
  for (const key of Object.keys(overrides)) if (!Object.hasOwn(RULES, key)) throw new RangeError(`app.${key} is not a rule`);
  const rules = {...RULES, ...overrides};
  for (const [key, [min, max, whole]] of Object.entries(RANGES)) {
    if (!Number.isFinite(rules[key]) || rules[key] < min || rules[key] > max) throw new RangeError(`app.${key} must be between ${min} and ${max}`);
    if (whole && !Number.isInteger(rules[key])) throw new RangeError(`app.${key} must be a whole number`);
  }
  for (const key of ['refetch', 'clear', 'summarize', 'summaryOfRequest']) if (typeof rules[key] !== 'boolean') throw new TypeError(`app.${key} must be boolean`);
  if (!['standard', 'long', 'off'].includes(rules.cache)) throw new RangeError('app.cache must be standard, long or off');
  return Object.freeze(rules);
}
const CHECKED = '2026-11-24';
const OPENCLAW = {reserveTokens: 20000, summarize: true, summaryTriggerShare: 0.75, tailTokens: 20000, summaryTokens: 4000, summaryOfRequest: false, summaryInputCap: 40000, toolSummaryTokens: 500, summarySystemTokens: 2048, recordTokens: 1500, flushOutputTokens: 256, refetch: true,
  clearTriggerShare: 0.30, clearTriggerFloor: 50000, clearKeepResults: 3, clearMinFreedShare: 0.05, clearMinFreedFloor: 12500};
const minutes = n => n % 60 === 0 ? `${n / 60}-hour` : `${n}-minute`;
// `rules` depends on the model: the provider decides the cache and, for some, what is done by default. `when` leaves out what a model does not have.
const LIST = [
  {id: 'everything', name: 'Send everything', note: 'The whole history, with prompt caching.', rules: () => ({})},
  {id: 'long', name: c => `Send everything, ${minutes(c.longTtlMinutes)} cache`, when: c => c.longTtlMinutes > c.ttlMinutes, note: c => `The whole history. Cache writes cost ${c.longWriteMultiplier}× input.`, rules: () => ({cache: 'long'})},
  {id: 'uncached', name: 'Send everything, no caching', note: 'The whole history at the full input price.', rules: () => ({cache: 'off'})},
  {id: 'window', name: 'Last 10 turns', note: 'Older turns are dropped and not fetched back.', rules: () => ({keepTurns: 10})},
  {id: 'rag', name: 'Retrieved text for one turn', note: 'Questions and answers stay. Retrieved text is dropped after its turn and not fetched back.', rules: () => ({dropRetrievedAfter: 0})},
  {id: 'openclaw', name: 'OpenClaw', url: 'https://github.com/openclaw/openclaw/tree/c49b553b85b82bc700465d4607949574f892366c', checked: '2026-11-20',
    note: c => c.anthropic ? 'Clears tool results above 30% of the window, keeps the newest 3. Summary at 75%.' : 'Summary at 75% of the window, 20k tail.', rules: c => ({...OPENCLAW, clear: c.anthropic})},
  {id: 'opencode', name: 'opencode', url: 'https://opencode.ai/docs/compaction', checked: CHECKED, note: 'Summary at 85% of the window, 15k tail. Warming is off by default.', rules: c => ({summarize: true, summaryTriggerShare: 1 - Math.min(0.99, Math.max(0.15, 16000 / c.model.context)), tailTokens: 15000, refetch: true})},
  {id: 'clearing', name: c => c.anthropic ? 'Anthropic context editing' : 'LangChain context editing', url: 'https://platform.claude.com/docs/en/build-with-claude/context-editing', checked: CHECKED, note: 'Clears tool results above 100k tokens, keeps the newest 3. LangChain\'s middleware has the same defaults.', rules: () => ({clear: true, refetch: true})},
  {id: 'compaction', name: 'Anthropic compaction', when: c => c.anthropic, url: 'https://platform.claude.com/docs/en/build-with-claude/compaction-threshold', checked: CHECKED, note: 'Everything becomes a summary above 150k tokens.', rules: () => ({summarize: true, summaryTriggerShare: 1, summaryTriggerTokens: 150000, tailTokens: 0})},
  {id: 'headroom', name: 'Headroom', url: 'https://github.com/headroomlabs-ai/headroom', checked: CHECKED, note: 'Keeps everything, shrinks retrieved text by 37%: the mean of its four published agent scenarios. Prose shrinks far less.', rules: () => ({compress: 0.37})}
];
const read = (value, c) => typeof value === 'function' ? value(c) : value;
// Every comparison this scenario can make, your own application first.
export function comparisons(c) {
  const app = {id: 'app', name: 'Your app', note: 'The rules you set.', rules: configureRules(c.app)};
  return [app, ...LIST.filter(b => !b.when || b.when(c)).map(b => ({id: b.id, name: read(b.name, c), note: read(b.note, c), url: b.url, checked: b.checked, rules: configureRules(b.rules(c))}))];
}
export const COMPARISON_IDS = Object.freeze(['measured', 'app', ...LIST.map(b => b.id)]);
export const OPENCLAW_RULES = Object.freeze(OPENCLAW);
