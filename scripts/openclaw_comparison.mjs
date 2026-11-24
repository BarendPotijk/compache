#!/usr/bin/env node
// Price compache against OpenClaw's defaults with the calculator's cost model.
//   node scripts/openclaw_comparison.mjs [--requirement 0.8] [--model sonnet] [--json docs/assets/data/openclaw.json]
// Prints cost per 1,000 turns for every model preset at three thread lengths. --json also writes the
// rows of one model, which scripts/figures.py draws as docs/assets/openclaw-cost.svg.
import {writeFileSync} from 'node:fs';
import {parseArgs} from 'node:util';
import {simulate} from '../site/engine.mjs';
import {applyPreset, presets} from '../site/models.mjs';

const {values: args} = parseArgs({options: {requirement: {type: 'string', default: '0.8'}, json: {type: 'string'}, model: {type: 'string', default: 'sonnet'}}});
const requirement = Number(args.requirement);
// Median and 95th-percentile turns per thread. The first pair is the calculator default.
const LENGTHS = [[8, 20], [30, 60], [80, 140]];
const MODELS = ['fable', 'opus', 'sonnet', 'haiku', 'astra', 'terra', 'luna'];

const money = x => `$${x.toLocaleString('en-US', {minimumFractionDigits: x < 10 ? 2 : 0, maximumFractionDigits: x < 10 ? 2 : 0})}`;
const cells = {};
for (const id of MODELS) cells[id] = LENGTHS.map(([turns, p95]) => {
  const r = simulate(applyPreset({plannedTurns: turns, plannedTurnsP95: p95, retentionRequirement: requirement, compare: 'openclaw'}, id)), openclaw = r.baseline.arm;
  if (!r.compache.complete || !openclaw.complete) throw new Error(`${id}, ${turns} turns: a side did not finish`);
  const per1000 = arm => 1000 * arm.totals.cost / arm.completedRuns;
  return {turns, openclaw: per1000(openclaw), compache: per1000(r.compache),
    kept: r.compache.retention.observed, openclawKept: openclaw.retention.observed,
    fetched: (r.compache.byRole.recovery?.calls || 0) / r.workload.totalRuns, openclawFetched: (openclaw.byRole.recovery?.calls || 0) / r.workload.totalRuns,
    openclawGone: 1 - openclaw.dependencyCoverage};
});

console.log(`| Model | ${LENGTHS.map(([t]) => `${t}-turn threads`).join(' | ')} |`);
console.log(`|---|${LENGTHS.map(() => '---').join('|')}|`);
for (const id of MODELS) console.log(`| ${presets[id].name} | ${cells[id].map(c => `${money(c.openclaw)} → ${money(c.compache)}, ${Math.round(100 * (1 - c.compache / c.openclaw))}% less`).join(' | ')} |`);

const share = xs => { const p = xs.map(x => Math.round(100 * x)); return Math.min(...p) === Math.max(...p) ? `${p[0]}%` : `${Math.min(...p)}% to ${Math.max(...p)}%`; };
console.log(`\nCompache keeps at least ${requirement} of what later turns need in view. Shares are by provider, as OpenClaw's defaults differ.`);
for (const [family, ids] of [['Claude', MODELS.filter(id => presets[id].anthropic)], ['OpenAI', MODELS.filter(id => !presets[id].anthropic)]]) LENGTHS.forEach(([turns], i) => {
  const of = key => share(ids.map(id => cells[id][i][key]));
  console.log(`  ${family}, ${turns}-turn threads: in view before fetching back, compache ${of('kept')}, OpenClaw ${of('openclawKept')}; turns with a fetch, compache ${of('fetched')}, OpenClaw ${of('openclawFetched')}; left to an OpenClaw summary ${of('openclawGone')}`);
});

if (args.json) {
  const rows = cells[args.model].map(({turns, openclaw, compache}) => ({turns, openclaw, compache}));
  writeFileSync(args.json, JSON.stringify({model: presets[args.model].name, requirement, rows}, null, 1) + '\n');
  console.log(`\nwrote ${args.json}`);
}
