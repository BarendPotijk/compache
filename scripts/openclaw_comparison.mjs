#!/usr/bin/env node
// Reproduces the README comparison with the calculator's cost model:
//   node scripts/openclaw_comparison.mjs [--requirement 0.8] [--svg docs/assets/openclaw-cost.svg] [--model sonnet]
// Prints a Markdown table of cost per 1,000 turns, OpenClaw against compache, and optionally draws one model's row.
import {writeFileSync} from 'node:fs';
import {parseArgs} from 'node:util';
import {simulate} from '../site/engine.mjs';
import {applyPreset, presets} from '../site/models.mjs';

const {values: args} = parseArgs({options: {requirement: {type: 'string', default: '0.8'}, svg: {type: 'string'}, model: {type: 'string', default: 'sonnet'}}});
const requirement = Number(args.requirement);
// Median and 95th-percentile turns per thread. The first pair is the calculator default.
const LENGTHS = [[8, 20], [30, 60], [80, 140]];
const MODELS = ['fable', 'opus', 'sonnet', 'haiku', 'astra', 'terra', 'luna'];

const money = (x, cents = x < 10) => `$${x.toLocaleString('en-US', {minimumFractionDigits: cents ? 2 : 0, maximumFractionDigits: cents ? 2 : 0})}`;
const pair = (theirs, ours) => { const cents = Math.min(theirs, ours) < 10; return `${money(theirs, cents)} → ${money(ours, cents)}`; };
// Non-breaking spaces and hyphens keep a cell on one line, so a narrow screen scrolls the table instead of wrapping it.
const nb = s => s.replaceAll(' ', '&nbsp;').replaceAll('-', '&#8209;');
const change = (ours, theirs) => { const p = Math.round(100 * (ours / theirs - 1)); return `${p > 0 ? '+' : '−'}${Math.abs(p)}%`; };

const cells = {};
for (const id of MODELS) cells[id] = LENGTHS.map(([turns, p95]) => {
  const r = simulate(applyPreset({plannedTurns: turns, plannedTurnsP95: p95, retentionRequirement: requirement}, id));
  if (!r.compache.complete || !r.openclaw.complete) throw new Error(`${id}, ${turns} turns: an arm did not finish`);
  const per1000 = arm => 1000 * arm.totals.cost / arm.completedRuns;
  return {turns, openclaw: per1000(r.openclaw), compache: per1000(r.compache),
    kept: r.compache.retention.observed, openclawKept: r.openclaw.retention.observed,
    fetched: (r.compache.byRole.recovery?.calls || 0) / r.workload.totalRuns, openclawFetched: (r.openclaw.byRole.recovery?.calls || 0) / r.workload.totalRuns,
    openclawGone: 1 - r.openclaw.rawSourceAvailability};
});

const lines = [`| ${nb('Cost per 1,000 turns')}<br><sub>${nb('OpenClaw → compache')}</sub> | ${LENGTHS.map(([t]) => nb(`${t}-turn threads`)).join(' | ')} |`, `|:--|${LENGTHS.map(() => ':-:').join('|')}|`];
for (const id of MODELS) lines.push(`| **${nb(presets[id].name)}**<br><sub>${nb(`${presets[id].ttlMinutes}-minute cache`)}</sub> | ${cells[id].map(c => `**${change(c.compache, c.openclaw)}**<br><sub>${nb(pair(c.openclaw, c.compache))}</sub>`).join(' | ')} |`);
console.log(lines.join('\n'));

const share = xs => { const p = xs.map(x => Math.round(100 * x)); return Math.min(...p) === Math.max(...p) ? `${p[0]}%` : `${Math.min(...p)}% to ${Math.max(...p)}%`; };
console.log(`\nCompache keeps at least ${requirement} of what later turns need in view. Shares below are by family, as the OpenClaw defaults differ.`);
for (const [family, ids] of [['Claude', MODELS.filter(id => presets[id].toolClearing)], ['OpenAI', MODELS.filter(id => !presets[id].toolClearing)]]) LENGTHS.forEach(([turns], i) => {
  const of = key => share(ids.map(id => cells[id][i][key]));
  console.log(`  ${family}, ${turns}-turn threads: in view before fetching back, compache ${of('kept')}, OpenClaw ${of('openclawKept')}; turns with a fetch, compache ${of('fetched')}, OpenClaw ${of('openclawFetched')}; left to an OpenClaw summary ${of('openclawGone')}`);
});

if (args.svg) { writeFileSync(args.svg, chart(presets[args.model].name, cells[args.model])); console.log(`\nwrote ${args.svg}`); }

// Horizontal bars, two per thread length. OpenClaw is the grey context series and compache the accent.
function chart(name, rows) {
  const x0 = 132, full = 360, bar = 16, gap = 2, top = 86, step = 58, max = Math.max(...rows.map(r => r.openclaw));
  const mark = (cls, y, v) => { const w = Math.max(8, full * v / max); return `<path class="${cls}" d="M${x0} ${y}h${(w - 4).toFixed(1)}a4 4 0 0 1 4 4v${bar - 8}a4 4 0 0 1-4 4H${x0}z"/>`; };
  const groups = rows.map((r, i) => { const y = top + i * step, less = Math.round(100 * (1 - r.compache / r.openclaw));
    return `<text class="t row" x="24" y="${y + bar + gap / 2 + 4.5}">${r.turns}-turn threads</text>${mark('oc', y, r.openclaw)}${mark('cp', y + bar + gap, r.compache)}` +
      `<text class="s val" x="${(x0 + full * r.openclaw / max + 8).toFixed(1)}" y="${y + 12.5}">${money(r.openclaw)}</text>` +
      `<text class="t val" x="${(x0 + full * r.compache / max + 8).toFixed(1)}" y="${y + bar + gap + 12.5}"><tspan font-weight="600">${money(r.compache)}</tspan><tspan class="s" dx="6">${less > 0 ? `${less}% less` : `${-less}% more`}</tspan></text>`; });
  const height = top + (rows.length - 1) * step + 2 * bar + gap + 28;
  const summary = rows.map(r => `${r.turns}-turn threads: OpenClaw ${money(r.openclaw)}, compache ${money(r.compache)}`).join('; ');
  return `<svg xmlns="http://www.w3.org/2000/svg" width="640" height="${height}" viewBox="0 0 640 ${height}" role="img" aria-labelledby="oc-title oc-desc">
<title id="oc-title">Cost per 1,000 turns on ${name}: OpenClaw against compache</title>
<desc id="oc-desc">Modeled cost per 1,000 turns at three thread lengths. ${summary}.</desc>
<style>
  text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans", Helvetica, Arial, sans-serif; }
  .bg { fill: #ffffff; } .t { fill: #1f2328; } .s { fill: #656d76; }
  .oc { fill: #8c959f; } .cp { fill: #2a78d6; } .base { stroke: #d0d7de; }
  .head { font-size: 15px; font-weight: 600; } .sub, .key, .val { font-size: 12.5px; } .row { font-size: 13px; }
  @media (prefers-color-scheme: dark) {
    .bg { fill: #0d1117; } .t { fill: #e6edf3; } .s { fill: #9198a1; }
    .oc { fill: #6e7681; } .cp { fill: #3987e5; } .base { stroke: #30363d; }
  }
</style>
<rect class="bg" width="640" height="${height}" rx="10"/>
<text class="t head" x="24" y="34">Cost per 1,000 turns · ${name}</text>
<text class="s sub" x="24" y="54">Modeled on the calculator's default workload, against OpenClaw's defaults.</text>
<rect class="oc" x="446" y="24" width="10" height="10" rx="2"/><text class="s key" x="462" y="33.5">OpenClaw</text>
<rect class="cp" x="538" y="24" width="10" height="10" rx="2"/><text class="s key" x="554" y="33.5">compache</text>
<line class="base" x1="${x0}" y1="${top - 8}" x2="${x0}" y2="${top + (rows.length - 1) * step + 2 * bar + gap + 8}"/>
${groups.join('\n')}
</svg>
`;
}
