/* Deterministic hypothetical cost model. This does not execute either product. */
const KINDS = ['input', 'retrieval', 'output'];
// The candidate policies, as shares of the context window: what a cold return is compacted down to, and the size
// at which a warm request is compacted. Together with the kept turns they span keeping almost nothing old to
// keeping nearly everything.
export const TARGETS = Object.freeze([.01, .03, .05, .10, .20, .45]);
export const WARM_TRIGGERS = Object.freeze([.05, .10, .25, .50, .60]);
export const DEFAULTS = Object.freeze({
  // Sampling.
  seed: 42, threads: 100, maxTurns: 160,
  // Workload: thread length, what each turn adds, and how long users stay away.
  lengthDistribution: 'lognormal', plannedTurns: 8, plannedTurnsP95: 20, continuationProbability: 0.99,
  // The 'continuation' length law: the chance of another turn after the first turn and in a long thread, and how
  // many turns after the first it is halfway between the two.
  continueFirst: 0.6, continueLong: 0.95, continueHalfTurn: 3,
  // Retrieved tokens describe the turns that retrieve, and retrievalShare is the share of turns that do.
  tokenMedian: 25000, tokenP95: 45000, retrievalShare: 1, inputTokens: 200, outputTokens: 500, reasoningTokens: 0, systemTokens: 2048,
  // Return delay in a long thread. After the first turn it is gapFirstFactor times that, and halfway on a log scale
  // gapHalfTurn turns later. With chance breakShare a return comes after a long break instead, whatever the turn.
  gapMedianMinutes: 4, gapP95Minutes: 60, gapFirstFactor: 1, gapHalfTurn: 3, breakShare: 0, breakMedianMinutes: 1440,
  // One turn: how long its answer takes, how many model calls it makes, and the chance another conversation has
  // already cached the system prompt and tools when this one's own cache is cold.
  answerSeconds: 0, callsPerTurn: 1, sharedPrefixWarm: 0,
  // What later turns need, per kind: the chance, and how many turns back it lies (median and 95th percentile).
  // Measured on the DocChat benchmark.
  inputDependencyProbability: 0.15, inputDependencyTurnsBack: 1, inputDependencyTurnsBackP95: 20,
  retrievalDependencyProbability: 0.15, retrievalDependencyTurnsBack: 3, retrievalDependencyTurnsBackP95: 30,
  outputDependencyProbability: 0.20, outputDependencyTurnsBack: 1, outputDependencyTurnsBackP95: 35,
  // Prompt cache: an entry's lifetime from when its request was sent, the least a request must hold, the size of a
  // cached block, and the share of a live entry that the next request actually reads.
  ttlMinutes: 30, cacheMinTokens: 1024, cacheBlockTokens: 1, cacheHitShare: 1,
  // Compache: the floor, the fitted policy (chosen by simulate) and its fixed costs.
  retentionRequirement: 0.95, target: 0.20, warmTrigger: 0.5, retrievalRetentionRuns: 0,
  keepWarm: true, maxPings: 12, pingLeadMinutes: 0.75, pingOutputTokens: 16, recoveryOutputTokens: 32, safetyReserve: 32000,
  // Tokens of the notice left in place of one turn's archived retrieved text: what the package leaves for one part.
  archivedTokens: 22,
  // OpenClaw's documented defaults. Clearing old tool results is on for its Anthropic provider only.
  toolClearing: false, clearTriggerShare: 0.30, clearTriggerFloor: 50000, clearKeepResults: 3, clearMinFreedShare: 0.05, clearMinFreedFloor: 12500, clearedTokens: 12,
  summaryTriggerShare: 0.75, baselineReserve: 20000, tailTokens: 20000,
  summaryTokens: 4000, summaryInputCap: 40000, toolSummaryTokens: 500, summarySystemTokens: 2048, recordTokens: 1500, flushOutputTokens: 256,
  model: Object.freeze({context: 1050000, maxInput: 922000, inputRate: 0.1,
    cachedRate: 0.01, outputRate: 0.5, writeMultiplier: 1.25,
    longThreshold: 272000, longInputMultiplier: 2, longOutputMultiplier: 1.5})
});
// Each setting's least value, greatest value and whether it must be a whole number.
const RANGES = Object.freeze({
  seed: [0, 4294967295, true], threads: [1, 200, true], maxTurns: [1, 160, true],
  plannedTurns: [1, 160], plannedTurnsP95: [1, 320], continuationProbability: [0, 1],
  continueFirst: [0, 1], continueLong: [0, 1], continueHalfTurn: [0.1, 160],
  tokenMedian: [0, 500000], tokenP95: [0, 1000000], retrievalShare: [0, 1], inputTokens: [1, 100000, true], outputTokens: [0, 200000, true], reasoningTokens: [0, 1000000, true], systemTokens: [0, 1000000, true],
  gapMedianMinutes: [0.001, 100000], gapP95Minutes: [0.001, 1000000], gapFirstFactor: [0.01, 1000], gapHalfTurn: [0.1, 160], breakShare: [0, 1], breakMedianMinutes: [1, 1000000],
  answerSeconds: [0, 86400], callsPerTurn: [1, 100], sharedPrefixWarm: [0, 1],
  ...Object.fromEntries(KINDS.flatMap(kind => [[`${kind}DependencyProbability`, [0, 1]], [`${kind}DependencyTurnsBack`, [1, 160]], [`${kind}DependencyTurnsBackP95`, [1, 1000]]])),
  ttlMinutes: [0.01, 100000], cacheMinTokens: [0, 1000000, true], cacheBlockTokens: [1, 100000, true], cacheHitShare: [0, 1],
  retentionRequirement: [0, 1], target: [0.01, 0.45], warmTrigger: [0.03, 1], retrievalRetentionRuns: [0, 160, true],
  maxPings: [0, 60, true], pingLeadMinutes: [0, 100000], pingOutputTokens: [0, 10000, true], recoveryOutputTokens: [0, 10000, true], safetyReserve: [0, 1000000, true], archivedTokens: [0, 10000, true],
  clearTriggerShare: [0, 1], clearTriggerFloor: [0, 10000000, true], clearKeepResults: [1, 160, true], clearMinFreedShare: [0, 1], clearMinFreedFloor: [0, 10000000, true], clearedTokens: [0, 10000, true],
  summaryTriggerShare: [0.01, 1], baselineReserve: [0, 1000000, true], tailTokens: [0, 1000000, true],
  summaryTokens: [1, 100000, true], summaryInputCap: [1, 1000000, true], toolSummaryTokens: [1, 100000, true], summarySystemTokens: [0, 100000, true], recordTokens: [0, 100000, true], flushOutputTokens: [0, 100000, true]
});
const MODEL_RANGES = Object.freeze({context: [1, 10000000], maxInput: [1, 10000000], inputRate: [0, 100000], cachedRate: [0, 100000], outputRate: [0, 100000], writeMultiplier: [0, 100], longThreshold: [0, 10000000], longInputMultiplier: [0, 100], longOutputMultiplier: [0, 100]});
const z95 = 1.6448536269514722;
const COST_KEYS = ['inputTokens', 'cachedTokens', 'freshTokens', 'outputTokens', 'inputCost', 'cachedCost', 'outputCost', 'cost'];
const sum = blocks => blocks.reduce((n, b) => n + b.tokens, 0);
const byKind = blocks => Object.freeze(blocks.reduce((totals,b)=>{totals[b.kind]=(totals[b.kind]||0)+b.tokens;return totals;},{}));
const ceil = n => Math.ceil(n);
function finite(n, name, min, max) {
  if (!Number.isFinite(n) || n < min || n > max) throw new RangeError(`${name} must be between ${min} and ${max}`);
}
// Every setting, checked. An unknown key is an error: a mistyped or outdated setting must not fall back to a default unnoticed.
export function configure(overrides = {}) {
  for (const key of Object.keys(overrides)) if (!Object.hasOwn(DEFAULTS, key)) throw new RangeError(`${key} is not a setting`);
  if (overrides.model !== undefined && (overrides.model === null || typeof overrides.model !== 'object')) throw new TypeError('model must be an object');
  for (const key of Object.keys(overrides.model || {})) if (!Object.hasOwn(DEFAULTS.model, key)) throw new RangeError(`model.${key} is not a setting`);
  const c = {...DEFAULTS, ...overrides, model: {...DEFAULTS.model, ...(overrides.model || {})}};
  for (const [key, [min, max, whole]] of Object.entries(RANGES)) {
    finite(c[key], key, min, max);
    if (whole && !Number.isInteger(c[key])) throw new RangeError(`${key} must be a whole number`);
  }
  for (const [key, [min, max]] of Object.entries(MODEL_RANGES)) finite(c.model[key], `model.${key}`, min, max);
  if (!['fixed','geometric','lognormal','continuation'].includes(c.lengthDistribution)) throw new RangeError('lengthDistribution must be fixed, geometric, lognormal or continuation');
  if (c.lengthDistribution === 'fixed' && !Number.isInteger(c.plannedTurns)) throw new RangeError('plannedTurns must be a whole number for a fixed length');
  const atLeast = (high, low) => { if (c[high] < c[low]) throw new RangeError(`${high} must be at least ${low}`); };
  if (c.lengthDistribution === 'lognormal') atLeast('plannedTurnsP95', 'plannedTurns');
  atLeast('tokenP95', 'tokenMedian'); atLeast('gapP95Minutes', 'gapMedianMinutes');
  for (const kind of KINDS) atLeast(`${kind}DependencyTurnsBackP95`, `${kind}DependencyTurnsBack`);
  if (c.target + 0.02 >= c.warmTrigger) throw new RangeError('target must be more than two points below warmTrigger');
  if (c.warmTrigger > 1 - c.safetyReserve/c.model.context) throw new RangeError(`safetyReserve must be at most ${Math.floor((1 - c.warmTrigger)*c.model.context)} for this context window`);
  if (typeof c.keepWarm !== 'boolean') throw new TypeError('keepWarm must be boolean');
  if (typeof c.toolClearing !== 'boolean') throw new TypeError('toolClearing must be boolean');
  return c;
}
function rng(seed) {
  let state = seed >>> 0;
  return () => {state = (state + 0x6D2B79F5) >>> 0; let t=state; t=Math.imul(t ^ t>>>15,t|1); t^=t+Math.imul(t ^ t>>>7,t|61); return ((t ^ t>>>14)>>>0)/4294967296;};
}
function normal(random) {return Math.sqrt(-2*Math.log(Math.max(random(),1e-12)))*Math.cos(2*Math.PI*random());}
function lognormal(random, median, p95) {return median*Math.exp(normal(random)*Math.log(p95/median)/z95);}
function quantile(sorted, q) {if (!sorted.length) return null; const x=(sorted.length-1)*q, i=Math.floor(x); return sorted[i]+(sorted[Math.ceil(x)]-sorted[i])*(x-i);}
export function distribution(values) {
  const a=[...values].sort((x,y)=>x-y);
  return {count:a.length, mean:a.length ? a.reduce((x,y)=>x+y,0)/a.length : null, median:quantile(a,.5), p95:quantile(a,.95), min:a[0] ?? null, max:a.at(-1) ?? null};
}
// How many turns back a needed item lies: a lognormal with this median and 95th percentile, rounded to a
// whole turn and at least one. F[a] is the chance that it lies at most `a` turns back.
export function ageLaw(median, p95, maxAge = DEFAULTS.maxTurns) {
  const sigma = Math.log(p95/median)/z95, F = new Float64Array(maxAge + 1);
  for (let a = 1; a <= maxAge; a++) F[a] = Math.max(F[a - 1], sigma > 0 ? normalCdf(Math.log((a + .5)/median)/sigma) : median < a + .5 ? 1 : 0);
  return Object.freeze({
    cdf: age => F[Math.max(0, Math.min(maxAge, Math.floor(age)))],
    // The weight on exactly `age` turns back, before any thread length is taken into account.
    weight: age => age >= 1 && age <= maxAge ? F[age] - F[age - 1] : 0,
    // Only `oldest` earlier turns exist. Among them, the share that lie exactly `age` back.
    // When the law puts no weight on any of them, the oldest one stands in.
    pmf: (age, oldest) => F[oldest] > 0 ? (F[age] - F[age - 1])/F[oldest] : age === oldest ? 1 : 0,
    // The age for a uniform draw `u` in [0, 1), among the `oldest` turns that exist.
    sample: (u, oldest) => { if (!(F[oldest] > 0)) return oldest; const x = u*F[oldest]; let age = 1; while (age < oldest && F[age] <= x) age++; return age; },
    // The fewest turns back that cover a share `q` of the needs, on a thread long enough to hold them all.
    quantile: q => { let age = 1; while (age < maxAge && F[age] < q) age++; return age; }
  });
}
const ageLaws = (c, maxAge) => Object.fromEntries(KINDS.map(kind => [kind, ageLaw(c[`${kind}DependencyTurnsBack`], c[`${kind}DependencyTurnsBackP95`], maxAge)]));
// Whether any turn retrieves anything.
const retrieves = c => c.tokenMedian !== 0 && c.retrievalShare !== 0;
// Under the 'continuation' length law, the chance of another turn after turn `turn`, counted from 1.
export function continueChance(turn, c) { return c.continueLong - (c.continueLong - c.continueFirst)*0.5**((turn - 1)/c.continueHalfTurn); }
// How many times longer than in a long thread the usual return delay is after turn `turn`, counted from 1.
function gapFactor(turn, c) { return c.gapFirstFactor**(0.5**((turn - 1)/c.gapHalfTurn)); }
// Long breaks are lognormal around their median, with a 95th percentile four times that.
const BREAK_SIGMA = Math.log(4)/z95;
// Of the earlier turns that retrieved (`origins`, oldest first), the one a need of turn `i` points at for a uniform draw
// `u`: each weighted by the age law. When the law puts no weight on any of them, the oldest stands in.
function pickOrigin(law, origins, i, u) {
  let total = 0; for (const origin of origins) total += law.weight(i - origin);
  if (!(total > 0)) return origins[0];
  let x = u*total;
  for (let k = origins.length - 1; k > 0; k--) { x -= law.weight(i - origins[k]); if (x < 0) return origins[k]; }
  return origins[0];
}
export function generateWorkload(config) {
  const c=configure(config), random=rng(c.seed), laws=ageLaws(c,c.maxTurns), threads=[], gaps=[], planned=[], realized=[];
  const byChance=c.lengthDistribution==='continuation', answerMinutes=c.answerSeconds/60, retrieving=retrieves(c);
  for(let ti=0;ti<c.threads;ti++) {
    // By chance of another turn, a thread has no length in advance: it runs until a turn is the last, or to the cap.
    let length=byChance ? c.maxTurns : c.lengthDistribution==='fixed' ? c.plannedTurns : c.lengthDistribution==='geometric' ? Math.ceil(Math.log(Math.max(random(),1e-12))/Math.log(1-1/Math.max(1.00000001,c.plannedTurns))) : Math.round(lognormal(random,c.plannedTurns,c.plannedTurnsP95));
    length=Math.max(1,Math.min(c.maxTurns,length));
    // The earlier turns that retrieved, oldest first.
    const turns=[], retrieved=[]; let time=0;
    for(let i=0;i<length;i++) {
      if(i>0) {
        if(random()>=(byChance?continueChance(i,c):c.continuationProbability)) break;
        // A long break with its chance, otherwise the usual delay for this position. The question arrives once the answer is there and the user is back.
        const g=c.breakShare>0 && random()<c.breakShare ? lognormal(random,c.breakMedianMinutes,4*c.breakMedianMinutes) : lognormal(random,c.gapMedianMinutes,c.gapP95Minutes)*gapFactor(i,c);
        gaps.push(g); time+=answerMinutes+g;
      }
      const found=retrieving && (c.retrievalShare>=1 || random()<c.retrievalShare);
      const retrieval=found?Math.max(1,Math.min(1000000,Math.round(lognormal(random,c.tokenMedian,c.tokenP95)))):0, deps=[];
      // Each kind independently: with its chance, this turn needs one earlier item of that kind, as far back as its law says.
      // Retrieved text can only be needed from a turn that retrieved. With no such turn there is no need.
      if(i) for(const kind of KINDS) {
        if(kind==='retrieval') {
          if(retrieved.length && random()<c.retrievalDependencyProbability) deps.push({kind,turn:retrieved.length===i?i-laws.retrieval.sample(random(),i):pickOrigin(laws.retrieval,retrieved,i,random())});
        } else if(random()<c[`${kind}DependencyProbability`]) deps.push({kind,turn:i-laws[kind].sample(random(),i)});
      }
      if(found)retrieved.push(i);
      turns.push({index:i,time,inputTokens:c.inputTokens,retrievalTokens:retrieval,outputTokens:c.outputTokens,dependencies:deps});
    }
    planned.push(byChance?turns.length:length); realized.push(turns.length); threads.push({id:`thread-${ti+1}`,plannedTurns:byChance?turns.length:length,turns});
  }
  const survival=Array.from({length:c.maxTurns},(_,i)=>({turn:i+1,planned:planned.filter(x=>x>i).length/c.threads,realized:realized.filter(x=>x>i).length/c.threads})).filter(x=>x.planned||x.realized);
  return {threads,plannedLengths:distribution(planned),realizedLengths:distribution(realized),gaps:distribution(gaps),survival,totalRuns:realized.reduce((a,b)=>a+b,0)};
}
export function sharedPrefix(a,b) {
  let total=0;
  for(let i=0;i<Math.min(a.length,b.length);i++) {if(a[i].id!==b[i].id || a[i].tokens!==b[i].tokens) break; total+=a[i].tokens;}
  return total;
}
// Whether two requests share at least `min` tokens from the start.
function sharesAtLeast(a,b,min) {
  let total=0;
  for(let i=0;total<min && i<Math.min(a.length,b.length);i++) {if(a[i].id!==b[i].id || a[i].tokens!==b[i].tokens) break; total+=a[i].tokens;}
  return total>=min;
}
export function priceCall(input,cached,output,model,{cacheable=true}={}) {
  if (![input,cached,output].every(Number.isFinite) || input<0 || output<0 || cached<0 || cached>input) throw new RangeError('invalid call token counts');
  const long=model.longThreshold>0 && input>model.longThreshold, im=long?model.longInputMultiplier:1, om=long?model.longOutputMultiplier:1;
  const fresh=input-cached, inputCost=fresh*model.inputRate*(cacheable?model.writeMultiplier:1)*im/1e6, cachedCost=cached*model.cachedRate*im/1e6, outputCost=output*model.outputRate*om/1e6;
  return {inputTokens:input,cachedTokens:cached,freshTokens:fresh,outputTokens:output,inputCost,cachedCost,outputCost,cost:inputCost+cachedCost+outputCost,long};
}
// Normal CDF approximation used only by the explicitly modeled planner.
function normalCdf(x) {const sign=x<0?-1:1, t=1/(1+.3275911*Math.abs(x)/Math.SQRT2); const e=1-(((((1.061405429*t-1.453152027)*t+1.421413741)*t-.284496736)*t+.254829592)*t)*Math.exp(-x*x/2);return .5*(1+sign*e);}
// The chance that a returning user is back within `t` minutes of the answer to turn `turn`, counted from 1: the usual
// delay at that position, mixed with long breaks.
function gapCdf(t,c,turn=Infinity) {
  if(t<=0)return 0;
  const median=c.gapMedianMinutes*gapFactor(turn,c), s=Math.log(c.gapP95Minutes/c.gapMedianMinutes)/z95;
  const usual=s===0 ? (t>=median?1:0) : normalCdf(Math.log(t/median)/s);
  return c.breakShare>0 ? (1-c.breakShare)*usual+c.breakShare*normalCdf(Math.log(t/c.breakMedianMinutes)/BREAK_SIGMA) : usual;
}
// Retrieved text can be archived when that frees anything: a turn that retrieved nothing, or less than the notice
// an archived source leaves behind, keeps its place.
const archivable=(b,c)=>b.kind==='retrieval' && b.tokens>c.archivedTokens;
const notice=(b,c)=>block(`archived-${b.id}`,c.archivedTokens,'archived',b.turn,'archived');
// The request a cold return would carry once old retrieved text is archived down to the target. No future
// question or dependency is known: only earlier retrieved text can go, and questions and answers keep their place.
export function coldPreview(history, c) {
  const system=block('system',c.systemTokens,'system',-1);
  let size=c.systemTokens+sum(history);
  if(size<=(c.target+.02)*c.model.context) return [system,...history];
  const lastTurn=history.reduce((n,b)=>Math.max(n,b.turn),-1), limit=c.target*c.model.context, view=[system];
  for(const b of history) {
    if(size>limit && archivable(b,c) && lastTurn+1-Number(b.source.slice(1))>c.retrievalRetentionRuns) {size-=b.tokens-c.archivedTokens; if(c.archivedTokens)view.push(notice(b,c));}
    else view.push(b);
  }
  return view;
}
// The size of that request, for a view that starts with its system block.
function coldSize(full, c) {
  let size=c.systemTokens, lastTurn=-1;
  for(let j=1;j<full.length;j++) {size+=full[j].tokens; if(full[j].turn>lastTurn)lastTurn=full[j].turn;}
  if(size<=(c.target+.02)*c.model.context) return size;
  const limit=c.target*c.model.context;
  for(let j=1;j<full.length && size>limit;j++) {const b=full[j]; if(archivable(b,c) && lastTurn+1-Number(b.source.slice(1))>c.retrievalRetentionRuns) size-=b.tokens-c.archivedTokens;}
  return size;
}
// Planned length is rounded and capped in generateWorkload; return probability
// is conditional on reaching this turn, without inspecting a realized future.
function plannedSurvival(turn,c) {
  if(turn<=1)return 1;
  if(turn>c.maxTurns)return 0;
  if(c.lengthDistribution==='continuation'){let alive=1;for(let n=1;n<turn;n++)alive*=continueChance(n,c);return alive;}
  if(c.lengthDistribution==='fixed')return turn<=Math.min(c.plannedTurns,c.maxTurns)?1:0;
  if(c.lengthDistribution==='geometric')return (1-1/Math.max(1.00000001,c.plannedTurns))**(turn-1);
  const sigma=Math.log(c.plannedTurnsP95/c.plannedTurns)/z95;
  if(sigma===0)return turn<=Math.round(c.plannedTurns)?1:0;
  return 1-normalCdf(Math.log((turn-.5)/c.plannedTurns)/sigma);
}
// The chance that the user comes back after the turn with this index, given that the thread got that far.
function returnChance(runIndex,c) {
  if(runIndex+2>c.maxTurns)return 0;
  if(c.lengthDistribution==='continuation')return continueChance(runIndex+1,c);
  const currentSurvival=plannedSurvival(runIndex+1,c), nextSurvival=plannedSurvival(runIndex+2,c);
  return currentSurvival>0?c.continuationProbability*nextSurvival/currentSurvival:0;
}
export function theoreticalSurvival(config={}) {
  const c=configure(config);
  // By chance of another turn there is no planned length apart from the realized one, and no flat chance on top of it.
  const flat=c.lengthDistribution==='continuation'?1:c.continuationProbability;
  return Array.from({length:c.maxTurns},(_,i)=>({turn:i+1,planned:plannedSurvival(i+1,c),realized:plannedSurvival(i+1,c)*flat**i}));
}
// CDF of the return delay after turn `turn` (counted from 1, a long thread when left out), conditional on a
// continuation: not an unconditional return probability.
export function returnCdf(minutes,config={},turn=Infinity) {return gapCdf(minutes,configure(config),turn);}
// The share of returning users whose next question finds the cache alive without a ping: the answer time and
// the delay together stay inside the cache lifetime, which runs from when the request was sent.
export function warmReturnShare(config={},turn=Infinity) {const c=configure(config);return gapCdf(c.ttlMinutes-c.answerSeconds/60,c,turn);}
// For n pings after a turn: the chance a returning user is back before the n-th goes out, and the chance the return
// falls in the extra time those pings keep the cache alive. Pings are due every interval from when the request was
// sent, and the delay starts when the answer is there. Both depend on the settings and the turn's position only.
let pingCurveKey='', pingCurves=new Map();
function pingCurve(c,turn) {
  const key=`${c.ttlMinutes}|${c.pingLeadMinutes}|${c.gapMedianMinutes}|${c.gapP95Minutes}|${c.maxPings}|${c.gapFirstFactor}|${c.gapHalfTurn}|${c.breakShare}|${c.breakMedianMinutes}|${c.answerSeconds}`;
  if(key!==pingCurveKey) {pingCurves=new Map();pingCurveKey=key;}
  // Without an effect of position one curve serves every turn.
  const position=c.gapFirstFactor===1?0:turn;
  let rows=pingCurves.get(position);
  if(!rows) {
    const interval=c.ttlMinutes-c.pingLeadMinutes, answer=c.answerSeconds/60, unaided=gapCdf(c.ttlMinutes-answer,c,turn);
    rows=Array.from({length:c.maxPings+1},(_,n)=>({back:gapCdf(n*interval-answer,c,turn),bridged:Math.max(0,gapCdf(n*interval+c.ttlMinutes-answer,c,turn)-unaided)}));
    pingCurves.set(position,rows);
  }
  return rows;
}
// The tokens of the system block that another conversation may have left in the cache: none when no conversation
// shares it or it is too short to cache.
function sharedSystem(c) {return c.sharedPrefixWarm>0 && c.systemTokens>=c.cacheMinTokens ? Math.floor(c.systemTokens/c.cacheBlockTokens)*c.cacheBlockTokens : 0;}
// The tokens a call can expect to read from the cache. `own` is the live prefix of its own thread, read with chance
// cacheHitShare. `shared` is the system block another conversation may hold, there with chance sharedPrefixWarm.
// The call reads the longer of the two that it finds. Priced by expectation, so no outcome is drawn.
function expectedRead(own,shared,c) {
  const hit=c.cacheHitShare, there=shared?c.sharedPrefixWarm:0;
  return hit===1 && shared<=own ? own : hit*there*Math.max(own,shared)+hit*(1-there)*own+(1-hit)*there*shared;
}
// The same for a call that starts with the thread's system block.
const expectedCached=(own,c)=>expectedRead(own,sharedSystem(c),c);
export function evaluatePingPlan(held, full, c, runIndex=0) {
  if(!Number.isInteger(runIndex)||runIndex<0)throw new RangeError('runIndex must be nonnegative');
  const interval=c.ttlMinutes-c.pingLeadMinutes, heldInput=sum(held), k=sum(full);
  const returnProbability=returnChance(runIndex,c);
  const coldInput=coldSize(full,c);
  // Above this size the next warm turn would compact anyway, so the prefix is not worth keeping warm.
  const aboveCeiling=k>(c.warmTrigger-c.safetyReserve/c.model.context)*c.model.context;
  // A ping is due one interval after the request was sent. An answer still running then leaves nothing to keep warm.
  let reason=!c.keepWarm?'disabled':!c.maxPings?'no pings allowed':interval<=0?'ping interval is not positive':c.answerSeconds/60>=interval?'answer outlasts the ping interval':heldInput<c.cacheMinTokens?'prefix is below cache minimum':aboveCeiling?'prefix exceeds keep-warm ceiling':returnProbability<=0?'no modeled future return':null;
  const rows=[{pings:0,expectedGain:0,expectedCost:0,net:0}];
  if(reason)return {chosen:0,reason,coldTokens:coldInput,heldTokens:heldInput,interval,ttl:c.ttlMinutes,returnProbability,aboveCeiling,rows};
  const heldShared=sharedPrefix(held,full);
  const cachedHeld=heldShared>=c.cacheMinTokens ? Math.floor(heldShared/c.cacheBlockTokens)*c.cacheBlockTokens : 0;
  const fullCached=k>=c.cacheMinTokens ? Math.floor(k/c.cacheBlockTokens)*c.cacheBlockTokens : 0;
  // Expired structural prefixes receive no cold cache credit of their own: a cold return reads at most the system
  // block another conversation cached. One later call approximates the cost of carrying a larger warm prefix.
  const coldCost=priceCall(coldInput,expectedCached(0,c),0,c.model,{cacheable:coldInput>=c.cacheMinTokens}).cost;
  const warmCost=priceCall(k,expectedCached(fullCached,c),0,c.model).cost;
  const laterExtra=Math.max(0,warmCost-priceCall(coldInput,expectedCached(coldInput>=c.cacheMinTokens?Math.floor(coldInput/c.cacheBlockTokens)*c.cacheBlockTokens:0,c),0,c.model,{cacheable:coldInput>=c.cacheMinTokens}).cost);
  const saving=coldCost-warmCost-laterExtra;
  let best=0, value=0, cost=0;
  const curve=pingCurve(c,runIndex+1);
  for(let n=1;n<=c.maxPings;n++) {
    const sent=1-returnProbability*curve[n].back;
    cost+=sent*priceCall(k,expectedCached(n===1?cachedHeld:fullCached,c),c.pingOutputTokens,c.model).cost;
    const bridged=returnProbability*curve[n].bridged;
    const expectedGain=bridged*saving, net=expectedGain-cost;
    rows.push({pings:n,expectedGain,expectedCost:cost,net});
    if(net>value){value=net;best=n;}
  }
  return {interval,ttl:c.ttlMinutes,chosen:best,reason:best?'positive expected value':'no positive expected value',coldTokens:coldInput,heldTokens:heldInput,returnProbability,aboveCeiling,rows};
}
export function planPings(held,full,c,runIndex=0) {return evaluatePingPlan(held,full,c,runIndex).chosen;}

const block=(id,tokens,kind,turn,source=id)=>Object.freeze({id,tokens,kind,turn,source});
function admitted(input,output,c,arm,role){const reserve=arm==='compache'?c.safetyReserve:role==='answer'?Math.min(c.baselineReserve,.25*c.model.context):0;return input<=c.model.maxInput && input+Math.max(output,reserve)<=c.model.context;}
export function runArm(workload, config, arm, {collectDecisions=true,collectLedger=true}={}) {
  const c=configure(config); if(!['compache','openclaw'].includes(arm)) throw new RangeError('invalid arm');
  const laws=ageLaws(c,Math.max(c.maxTurns,...workload.threads.map(t=>t.turns.length)));
  const ledger=[], curves=[], decisions=[], roles={}, cacheBreakdown=collectLedger?{cached:0,newContext:0,firstRequest:0,expiredPrefix:0,changedPrefix:0,missed:0}:null, total={inputTokens:0,cachedTokens:0,freshTokens:0,outputTokens:0,inputCost:0,cachedCost:0,outputCost:0,cost:0,calls:0};
  let completedThreads=0,completedRuns=0,attemptedRuns=0,compactions=0,clearings=0,recalls=0,required=0,covered=0,failedCalls=0,pings=0,totalRemovedTokens=0,totalRecallTokens=0;
  // Whether the warm trigger ever decided anything: a warm request above it, or a prefix above the ceiling it puts on pings.
  // A run in which it never did is, call for call, the run of every higher warm trigger.
  let warmTriggerReached=false;
  const pre=Object.fromEntries(KINDS.map(kind=>[kind,{expectedRequired:0,expectedRetained:0,observedRequired:0,observedRetained:0}]));
  for(const thread of workload.threads) {
    let history=[],snapshots=[],expiredSnapshots=[],seenBlocks=new Set(),cleared=new Set(),callIndex=0,threadCost=0,summaryIndex=0,failed=false;
    // Compache side: the turns whose retrieved text is in the request.
    const inView=new Set();
    // The earlier turns that retrieved anything, oldest first.
    const retrievedBefore=[];
    const system=block('system',c.systemTokens,'system',-1);
    function expireSnapshots(time) {
      const stillLive=[]; for(const snapshot of snapshots) {if(time-snapshot.time<c.ttlMinutes)stillLive.push(snapshot);else if(collectLedger)expiredSnapshots.push(snapshot);}
      snapshots=stillLive; if(expiredSnapshots.length>16)expiredSnapshots=expiredSnapshots.slice(-16);
    }
    // Price one call. `blocks` is kept as the cached request, so the caller must not change it afterwards.
    function request(blocks,role,time,output,turn) {
      const input=sum(blocks);
      if(!admitted(input,output,c,arm,role)){failedCalls++;if(collectLedger)ledger.push({arm,thread:thread.id,turn,role,time,inputTokens:input,outputTokens:output,admitted:false});return null;}
      expireSnapshots(time);
      let prefix=0; for(const s of snapshots) prefix=Math.max(prefix,s.blocks===blocks?input:sharedPrefix(s.blocks,blocks));
      const own=prefix>=c.cacheMinTokens ? Math.floor(prefix/c.cacheBlockTokens)*c.cacheBlockTokens : 0;
      // The call reads its own thread's live prefix when it hits it, and otherwise the system block when another
      // conversation has that cached. It is priced at the expected tokens read: the blend of those outcomes.
      const shared=blocks[0]===system?sharedSystem(c):0, cached=expectedRead(own,shared,c);
      let attribution=null;
      if(collectLedger && role==='answer') {
        let best=prefix; if(callIndex)for(const snapshot of expiredSnapshots) best=Math.max(best,sharedPrefix(snapshot.blocks,blocks));
        const bestCached=best>=c.cacheMinTokens?Math.floor(best/c.cacheBlockTokens)*c.cacheBlockTokens:0;
        // Why the tokens beyond the `read` ones were not read from the cache. `live` is how far a live entry reached.
        const explain=(read,live)=>{
          const parts={cached:read,newContext:0,firstRequest:0,expiredPrefix:0,changedPrefix:0,missed:live-read};
          if(callIndex===0){parts.firstRequest=input-live;return parts;}
          const from=Math.max(bestCached,live); parts.expiredPrefix=from-live;
          let offset=0; for(const b of blocks) {
            const tail=Math.max(0,offset+b.tokens-Math.max(offset,from));
            if(tail){if(seenBlocks.has(`${b.id}:${b.tokens}`))parts.changedPrefix+=tail;else parts.newContext+=tail;}
            offset+=b.tokens;
          }
          return parts;
        };
        if(cached===own)attribution=explain(own,own);
        else {
          // Each outcome with its chance: both entries found, only the thread's own, only the shared one, neither.
          const hit=c.cacheHitShare, there=shared?c.sharedPrefixWarm:0, longer=Math.max(own,shared);
          attribution={cached:0,newContext:0,firstRequest:0,expiredPrefix:0,changedPrefix:0,missed:0};
          for(const [chance,read] of [[hit*there,longer],[hit*(1-there),own],[(1-hit)*there,shared],[(1-hit)*(1-there),0]]) {
            if(!chance)continue;
            const parts=explain(read,Math.max(read,own)); for(const key of Object.keys(attribution))attribution[key]+=chance*parts[key];
          }
        }
        for(const key of Object.keys(cacheBreakdown))cacheBreakdown[key]+=attribution[key];
      }
      const costs=priceCall(input,cached,output,c.model,{cacheable:input>=c.cacheMinTokens});
      const row=collectLedger?{arm,thread:thread.id,turn,role,time,admitted:true,...costs,...(attribution?{cacheBreakdown:attribution}:{})}:costs; if(collectLedger)ledger.push(row); callIndex++;
      tally(role,costs);
      if(collectLedger)for(const b of blocks)seenBlocks.add(`${b.id}:${b.tokens}`); if(input>=c.cacheMinTokens) snapshots.push({time,blocks});
      return row;
    }
    function tally(role,costs,calls=1) {
      const roleTotal=roles[role] ||= {calls:0,inputTokens:0,cachedTokens:0,freshTokens:0,outputTokens:0,inputCost:0,cachedCost:0,outputCost:0,cost:0};
      total.calls+=calls; roleTotal.calls+=calls; for(const k of COST_KEYS) {total[k]+=costs[k]; roleTotal[k]+=costs[k];}
      threadCost+=costs.cost;
    }
    // The further model calls of a turn. Each sends the answer's request again with no output: read from the cache
    // where the request is long enough to cache, at the full input price otherwise. Priced together, as callsPerTurn - 1 calls.
    function furtherCalls(blocks,time,turn) {
      const count=c.callsPerTurn-1, input=sum(blocks), cacheable=input>=c.cacheMinTokens;
      const one=priceCall(input,expectedRead(cacheable?Math.floor(input/c.cacheBlockTokens)*c.cacheBlockTokens:0,blocks[0]===system?sharedSystem(c):0,c),0,c.model,{cacheable});
      const costs=Object.fromEntries(COST_KEYS.map(key=>[key,count*one[key]]));
      if(collectLedger)ledger.push({arm,thread:thread.id,turn,role:'further',time,admitted:true,calls:count,...costs,long:one.long});
      tally('further',costs,count);
    }
    for(let i=0;i<thread.turns.length;i++) {
      const turn=thread.turns[i], input=block(`i${i}`,turn.inputTokens,'input',i), retrieval=block(`r${i}`,turn.retrievalTokens,'retrieval',i), output=block(`o${i}`,turn.outputTokens,'output',i);
      attemptedRuns++;
      const protectedSources=new Set(turn.dependencies.filter(d=>d.kind==='retrieval').map(d=>`r${d.turn}`));
      let decision=null, added=[];
      if(arm==='compache') {
        // Evaluate availability before current dependencies are recovered. The
        // expectation integrates over the configured age law, not future labels.
        if(i) {
          for(const kind of KINDS) {
            // Retrieved text can only be needed from an earlier turn that retrieved.
            const p=kind==='retrieval' && (!retrieves(c) || !retrievedBefore.length)?0:c[`${kind}DependencyProbability`];
            // Questions and answers are never archived. Retrieved text counts for the share of its age law that is still in the request.
            let share=1;
            if(kind==='retrieval' && retrievedBefore.length===i){share=0;for(const origin of inView)share+=laws.retrieval.pmf(i-origin,i);}
            else if(kind==='retrieval' && p) {
              // Some earlier turns retrieved nothing: the age law is renormalised over the ones that did.
              let all=0, held=0; for(const origin of retrievedBefore){const w=laws.retrieval.weight(i-origin);all+=w;if(inView.has(origin))held+=w;}
              share=all>0?held/all:inView.has(retrievedBefore[0])?1:0;
            }
            pre[kind].expectedRequired+=p; pre[kind].expectedRetained+=p*share;
          }
        }
        for(const source of protectedSources) if(!inView.has(Number(source.slice(1)))) {
          const sourceTurn=thread.turns[Number(source.slice(1))];
          added.push(block(`recall-${i}-${source}`,sourceTurn.retrievalTokens,'retrieval',i,source)); recalls++;
        }
        let view=[system,...history,input,retrieval,...added];
        const before=collectDecisions?view.slice():null, removed=[];
        const recallTokens=sum(added); totalRecallTokens+=recallTokens;
        expireSnapshots(turn.time);
        const warm=snapshots.some(s=>sharesAtLeast(s.blocks,view,c.cacheMinTokens));
        const threshold=(warm?c.warmTrigger:c.target+.02)*c.model.context;
        let projectedSize=sum(view);
        if(projectedSize>threshold) {
          if(warm)warmTriggerReached=true;
          // Oldest first, archive retrieved text that is past the kept turns and not needed now, until the request is down to the target.
          const limit=c.target*c.model.context, kept=[];
          for(const b of history) {
            if(projectedSize>limit && archivable(b,c) && i-Number(b.source.slice(1))>c.retrievalRetentionRuns && !protectedSources.has(b.source)) {
              removed.push(b);projectedSize-=b.tokens-c.archivedTokens;inView.delete(Number(b.source.slice(1)));
              if(c.archivedTokens)kept.push(notice(b,c));
            } else kept.push(b);
          }
          if(removed.length){history=kept;compactions++;if(collectDecisions)view=[system,...history,input,retrieval,...added];}
        }
        const removedTokens=sum(removed); totalRemovedTokens+=removedTokens;
        if(collectDecisions) {const beforeBlocks=before.slice(-16), afterBlocks=view.slice(-16); decision={thread:thread.id,turn:i,beforeTokens:sum(before),afterTokens:sum(view),cacheState:warm?'warm':'cold',triggerTokens:threshold,targetTokens:c.target*c.model.context,removedTokens,removedSources:Object.freeze(removed.slice()),protectedSources:Object.freeze([...protectedSources,retrieval.source]),recallTokens,plannedPings:0,actualPings:0,beforeBlocks:Object.freeze(beforeBlocks),afterBlocks:Object.freeze(afterBlocks),beforeByKind:byKind(before),afterByKind:byKind(view),beforeOmittedCount:before.length-beforeBlocks.length,afterOmittedCount:view.length-afterBlocks.length,beforeOmittedTokens:sum(before)-sum(beforeBlocks),afterOmittedTokens:sum(view)-sum(afterBlocks),planner:null,admitted:null};}
        history.push(input,retrieval); inView.add(i);
      } else {
        if(c.toolClearing) {
          // Anthropic server-side clearing with OpenClaw's derived parameters. By default: above 30% of the
          // window, clear all but the three newest tool results (this turn's and two before it), if that frees at least 5%.
          // A turn that retrieved nothing has no tool result to keep or to clear.
          const results=history.filter(b=>b.kind==='retrieval' && b.tokens>0), old=results.slice(0,Math.max(0,results.length-(c.clearKeepResults-(retrieval.tokens>0?1:0))));
          if(old.length && c.systemTokens+sum(history)+input.tokens+retrieval.tokens>=Math.max(c.clearTriggerFloor,Math.floor(c.model.context*c.clearTriggerShare)) && sum(old)-old.length*c.clearedTokens>=Math.max(c.clearMinFreedFloor,Math.floor(c.model.context*c.clearMinFreedShare))) {
            const gone=new Set(old); for(const b of old)cleared.add(b.source);
            history=history.map(b=>gone.has(b)?block(`cleared-${b.id}`,c.clearedTokens,'cleared',b.turn,'cleared'):b); clearings++;
          }
        }
        // A cleared result that this run needs is fetched again. A summarized one is not: the summary stands in for it.
        for(const source of protectedSources) if(cleared.has(source)) {added.push(block(`refetch-${i}-${source}`,thread.turns[Number(source.slice(1))].retrievalTokens,'retrieval',i,source)); cleared.delete(source); recalls++; totalRecallTokens+=added.at(-1).tokens;}
        let view=[system,...history,input,retrieval,...added];
        const threshold=Math.min(c.summaryTriggerShare*c.model.context,c.model.context-Math.min(c.baselineReserve,.25*c.model.context));
        if(sum(view)>=threshold && history.length) {
          // Retain complete newest turns until the tail target is met. The
          // summary sees only the older head, with bounded tool projections.
          const kept=[]; let size=0;
          for(let j=i-1;j>=0;j--){const blocks=history.filter(b=>b.turn===j && b.kind!=='summary' && b.kind!=='record'); const n=sum(blocks); if(!blocks.length)continue; kept.unshift(...blocks);size+=n;if(size>=c.tailTokens)break;}
          const keptIds=new Set(kept.map(b=>b.id)), head=history.filter(b=>!keptIds.has(b.id));
          const projected=[]; let budget=c.summaryInputCap;
          for(const b of head) {
            const available=b.kind==='retrieval'?Math.min(b.tokens,c.toolSummaryTokens):b.tokens;
            const n=Math.min(available,budget); if(n>0)projected.push(block(`summarized-${b.id}`,n,b.kind,b.turn,b.source)); budget-=n;
            if(budget<=0)break;
          }
          // Flush and summary have separate system identities; neither is an answer snapshot.
          const flush=[block('openclaw-flush-system',c.systemTokens,'system',-1),...history,input,retrieval,...added];
          const summary=[block('openclaw-summary-system',c.summarySystemTokens,'system',-1),...projected];
          if(!request(flush,'flush',turn.time,c.flushOutputTokens,i) || !request(summary,'summary',turn.time,c.summaryTokens,i)){failed=true;break;}
          const sequence=++summaryIndex;
          history=[block(`summary-${sequence}`,c.summaryTokens,'summary',-1),...(c.recordTokens?[block(`record-${sequence}`,c.recordTokens,'record',-1)]:[]),...kept]; compactions++;
        }
        history.push(input,retrieval);
      }
      const sourceOf=d=>`${d.kind==='input'?'i':d.kind==='retrieval'?'r':'o'}${d.turn}`, needs=turn.dependencies.filter(d=>d.kind!=='retrieval'||thread.turns[d.turn]?.retrievalTokens!==0);
      for(const d of needs) {pre[d.kind].observedRequired++; if(history.some(b=>b.source===sourceOf(d)))pre[d.kind].observedRetained++;}
      if(added.length) {
        // The model asks for the missing sources in one short reply; the answer call then carries them.
        if(!request([system,...history],'recovery',turn.time,c.recoveryOutputTokens,i)){if(decision){decision.admitted=false;decisions.push(Object.freeze(decision));}failed=true;break;}
        history.push(block(`reopen-${i}`,c.recoveryOutputTokens,'recovery',i,'recovery'),...added);
        if(arm==='compache')for(const b of added)inView.add(Number(b.source.slice(1)));
      }
      const view=[system,...history];
      required+=needs.length;
      for(const d of needs) if(view.some(b=>b.source===sourceOf(d))) covered++;
      // Reasoning is billed as output with the answer and is not sent again.
      const row=request(view,'answer',turn.time,turn.outputTokens+c.reasoningTokens,i);
      if(!row){if(decision){decision.admitted=false;if(collectLedger)Object.assign(ledger.at(-1),{beforeTokens:decision.beforeTokens,afterTokens:decision.afterTokens,cacheState:decision.cacheState,triggerTokens:decision.triggerTokens,targetTokens:decision.targetTokens,removedTokens:decision.removedTokens,recallTokens:decision.recallTokens});decisions.push(Object.freeze(decision));}failed=true;break;}
      if(decision)decision.admitted=true;
      if(c.callsPerTurn>1)furtherCalls(view,turn.time,i);
      completedRuns++; history.push(output); if(turn.retrievalTokens>0)retrievedBefore.push(i);
      if(collectLedger && thread===workload.threads[0]) curves.push({turn:i+1,inputTokens:row.inputTokens,cachedTokens:row.cachedTokens,cost:row.cost,cumulativeCost:threadCost});
      if(arm==='compache') {
        const full=[system,...history], plan=evaluatePingPlan(view,full,c,i), count=plan.chosen, next=thread.turns[i+1]?.time ?? Infinity, interval=c.ttlMinutes-c.pingLeadMinutes;
        if(c.keepWarm && plan.aboveCeiling)warmTriggerReached=true;
        if(decision){decision.planner=Object.freeze({...plan,rows:Object.freeze(plan.rows.map(r=>Object.freeze({...r})))});decision.plannedPings=count;}
        for(let n=1;n<=count;n++) {const time=turn.time+n*interval;if(time>=next)break;const ping=request(full,'ping',time,c.pingOutputTokens,i);if(!ping){failed=true;break;}pings++;if(decision)decision.actualPings++;}
        if(decision){Object.assign(row,{beforeTokens:decision.beforeTokens,afterTokens:decision.afterTokens,cacheState:decision.cacheState,triggerTokens:decision.triggerTokens,targetTokens:decision.targetTokens,removedTokens:decision.removedTokens,recallTokens:decision.recallTokens,plannedPings:decision.plannedPings,actualPings:decision.actualPings});decisions.push(Object.freeze(decision));}
        if(failed)break;
      }
    }
    if(!failed)completedThreads++;
  }
  const expectedRequired=Object.values(pre).reduce((n,x)=>n+x.expectedRequired,0), expectedRetained=Object.values(pre).reduce((n,x)=>n+x.expectedRetained,0), observedRequired=Object.values(pre).reduce((n,x)=>n+x.observedRequired,0), observedRetained=Object.values(pre).reduce((n,x)=>n+x.observedRetained,0);
  const byKindRetention=Object.fromEntries(Object.entries(pre).map(([kind,row])=>[kind,{...row,expected:row.expectedRequired?row.expectedRetained/row.expectedRequired:null,observed:row.observedRequired?row.observedRetained/row.observedRequired:null}]));
  const retention={expected:expectedRequired?expectedRetained/expectedRequired:null,observed:observedRequired?observedRetained/observedRequired:null,expectedRequired,expectedRetained,observedRequired,observedRetained,byKind:byKindRetention};
  return {totals:total,byRole:roles,ledger,decisions,retention,cacheBreakdown,removedTokens:totalRemovedTokens,recallTokens:totalRecallTokens,completedThreads,plannedThreads:workload.threads.length,completedRuns,plannedRuns:workload.totalRuns,attemptedRuns,failedCalls,complete:completedThreads===workload.threads.length,compactions,clearings,recalls,pings,warmTriggerReached,rawSourceAvailability:required?covered/required:null,dependencyCoverage:required?covered/required:null,dependencies:{required,covered},curve:curves};
}
export function simulate(overrides={}) {
  // The policy is fitted on one set of threads and both arms are costed on a second set drawn the same way.
  const base=configure(overrides), fitted=generateWorkload(base), workload=generateWorkload({...base,seed:(base.seed+1)>>>0}), candidates=[];
  // Latest turns whose retrieved text is never archived: none, or just enough to cover a given share of the
  // retrieved text later turns need. Without retrieval there is nothing to archive and one horizon suffices.
  const law=ageLaw(base.retrievalDependencyTurnsBack,base.retrievalDependencyTurnsBackP95,base.maxTurns);
  const horizons=!retrieves(base)?[0]:[...new Set([0,...[.25,.5,.75,.9,.95,.99].map(q=>law.quantile(q))])];
  for(const target of TARGETS) for(const retrievalRetentionRuns of horizons) {
    let stands=null;
    for(const warmTrigger of WARM_TRIGGERS) {
      if(target+.02>=warmTrigger) continue;
      const policy={target,warmTrigger,retrievalRetentionRuns};
      if(warmTrigger>1-base.safetyReserve/base.model.context) {candidates.push({...policy,cost:null,retention:null,complete:false,feasible:false,completedRuns:0});continue;}
      // A run in which the warm trigger never decided anything is also the run of every higher one.
      const result=stands ?? runArm(fitted,{...base,...policy},'compache',{collectDecisions:false,collectLedger:false});
      if(!result.warmTriggerReached) stands=result;
      const retention=result.retention.expected;
      // No modeled dependencies means the constraint is vacuous; report null, not 100%.
      const feasible=result.complete && (retention===null || retention+1e-12>=base.retentionRequirement);
      candidates.push({...policy,cost:result.totals.cost,retention,observedRetention:result.retention.observed,complete:result.complete,feasible,completedRuns:result.completedRuns});
    }
  }
  const feasible=candidates.filter(x=>x.feasible);
  // The cheapest. Among equally cheap ones, the one that removes least: more in view, then a higher target, a higher warm trigger, more kept turns.
  feasible.sort((a,b)=>a.cost-b.cost || (b.retention??1)-(a.retention??1) || b.target-a.target || b.warmTrigger-a.warmTrigger || b.retrievalRetentionRuns-a.retrievalRetentionRuns);
  // For diagnostics choose the most-retentive complete candidate, otherwise the
  // candidate with the greatest completed-run count. Never claim it is feasible.
  const diagnostic=[...candidates].filter(x=>x.cost!==null).sort((a,b)=>Number(b.complete)-Number(a.complete) || b.completedRuns-a.completedRuns || (b.retention??-1)-(a.retention??-1) || a.cost-b.cost)[0];
  const pick=feasible[0] || diagnostic || {target:base.target,warmTrigger:base.warmTrigger,retrievalRetentionRuns:base.retrievalRetentionRuns};
  const selected={target:pick.target,warmTrigger:pick.warmTrigger,retrievalRetentionRuns:pick.retrievalRetentionRuns};
  const config=configure({...base,...selected});
  const compache=runArm(workload,config,'compache'), openclaw=runArm(workload,config,'openclaw');
  const optimization={feasible:feasible.length>0,retentionRequirement:base.retentionRequirement,retention:compache.retention.expected,fittedRetention:pick.retention??null,fittedThreads:fitted.threads.length,fittedRuns:fitted.totalRuns,candidateCount:candidates.length,selected:feasible.length?selected:null,diagnosticPolicy:feasible.length?null:selected,candidates};
  const savings=optimization.feasible&&compache.complete&&openclaw.complete ? {absolute:openclaw.totals.cost-compache.totals.cost,percent:openclaw.totals.cost?100*(openclaw.totals.cost-compache.totals.cost)/openclaw.totals.cost:null} : null;
  const warnings=['A model of token counts and list prices: neither Compache, OpenClaw nor a provider is run, and answer quality is not measured.', 'The policy is the cheapest feasible candidate of a finite set on the fitting threads; among equally cheap ones, the one that removes least. Costs are reported on a second set of threads drawn with the next seed.', 'When a run needs a source that was archived or cleared, one extra call asks for it and the answer call carries it. Both arms pay this, and both are assumed to know what the run needs. A source that OpenClaw summarized is not fetched again.', 'The floor is on the expected share of needed sources left in the request before fetching back, averaged over the configured chances and how far back each kind of need lies. A zero denominator is reported as null and imposes no constraint.', 'Prices and token counts are assumptions. A cache entry is modeled as living exactly its configured lifetime from when its request was sent. The share of a live entry that is read, and a system prompt that another conversation has cached, are priced by expectation with no outcome drawn.', 'Each further model call of a turn sends its request again with no output. Reasoning is billed as output with the answer and is not sent again.', 'OpenClaw starts from its documented defaults at the pinned revision, and each constant is a setting. Claude presets: server-side tool-result clearing above 30% of the window, keeping the newest three, with the client summary at 75% behind it. OpenAI presets: the client-side summary path; the default OpenAI compact endpoint, retries and summary quality are not modeled. No keep-warm requests.', 'Keep-warm planning uses the configured chance of another turn and return delay at each turn\'s position, and an expected-value approximation, not the package\'s fitted policy. Pings can be sent after a user has left for good.', 'An incomplete arm or no feasible policy suppresses the savings figure; partial costs are not comparable totals.'];
  return {config,workload,compache,openclaw,optimization,savings,curves:{compache:compache.curve,openclaw:openclaw.curve},warnings};
}
