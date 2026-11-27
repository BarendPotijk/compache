/* Deterministic hypothetical cost model. This does not execute any product. */
import {BINS, DRAWN, checkDrawn, histogram, weightsOf} from './drawn.mjs';
import {RULES, COMPARISON_IDS, comparisons, configureRules} from './baselines.mjs';

export const KINDS = Object.freeze(['input', 'retrieval', 'output']);
// What a policy file calls the window on each kind.
export const WINDOW = Object.freeze({input: 'questions', retrieval: 'retrieved', output: 'answers'});
// How many turns ahead a decision looks, as the package does.
const LOOKAHEAD = 3;
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
  // Distributions drawn bin by bin, see drawn.mjs. Null takes the law above.
  tokenDrawn: null, gapDrawn: null, lengthDrawn: null, inputNeedDrawn: null, retrievalNeedDrawn: null, outputNeedDrawn: null,
  // Prompt cache: an entry's lifetime from when its request was sent, the least a request must hold, the size of a
  // cached block, and the share of a live entry that the next request actually reads. The long lifetime and its write price.
  ttlMinutes: 30, cacheMinTokens: 1024, cacheBlockTokens: 1, cacheHitShare: 1, longTtlMinutes: 720, longWriteMultiplier: 2.5, anthropic: false,
  // Compache: the floor, the most pings a candidate may have, and its fixed costs.
  retentionRequirement: 0.8, keepWarm: true, maxPings: 3, pingLeadMinutes: 0.75, pingOutputTokens: 16, recoveryOutputTokens: 32, safetyReserve: 32000,
  // Tokens of the notice left in place of one turn's archived retrieved text: 28 for each part the package archives.
  archivedTokens: 28,
  // A policy to price instead of the fitted one: {questions, answers, retrieved} in turns, null for all, with {maxPings, keepUnder, long}.
  policy: null,
  // What Compache is compared with, the rules of your own application, and what it really costs per 1,000 turns when you know (0 when not).
  compare: 'openclaw', app: Object.freeze({}), measuredCost: 0,
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
  ttlMinutes: [0.01, 100000], cacheMinTokens: [0, 1000000, true], cacheBlockTokens: [1, 100000, true], cacheHitShare: [0, 1], longTtlMinutes: [0.01, 100000], longWriteMultiplier: [0, 100],
  retentionRequirement: [0, 1], maxPings: [0, 60, true], pingLeadMinutes: [0, 100000], pingOutputTokens: [0, 10000, true], recoveryOutputTokens: [0, 10000, true], safetyReserve: [0, 1000000, true], archivedTokens: [0, 10000, true],
  measuredCost: [0, 1000000000]
});
const MODEL_RANGES = Object.freeze({context: [1, 10000000], maxInput: [1, 10000000], inputRate: [0, 100000], cachedRate: [0, 100000], outputRate: [0, 100000], writeMultiplier: [0, 100], longThreshold: [0, 10000000], longInputMultiplier: [0, 100], longOutputMultiplier: [0, 100]});
const z95 = 1.6448536269514722;
const COST_KEYS = ['inputTokens', 'cachedTokens', 'freshTokens', 'outputTokens', 'inputCost', 'cachedCost', 'outputCost', 'cost'];
const sum = blocks => { let n = 0; for (const b of blocks) n += b.tokens; return n; };
function finite(n, name, min, max) {
  if (!Number.isFinite(n) || n < min || n > max) throw new RangeError(`${name} must be between ${min} and ${max}`);
}
export function configurePolicy(policy) {
  if (policy === null) return null;
  if (typeof policy !== 'object' || Array.isArray(policy)) throw new TypeError('policy must be an object');
  const p = {questions: null, answers: null, retrieved: null, maxPings: 0, keepUnder: 0, long: false, ...policy};
  for (const key of Object.keys(p)) if (!['questions', 'answers', 'retrieved', 'maxPings', 'keepUnder', 'long'].includes(key)) throw new RangeError(`policy.${key} is not part of a policy`);
  if (typeof p.long !== 'boolean') throw new TypeError('policy.long must be boolean');
  for (const key of ['questions', 'answers', 'retrieved']) if (p[key] !== null && (!Number.isInteger(p[key]) || p[key] < 0 || p[key] > 100000)) throw new RangeError(`policy.${key} must be a whole number of turns, or null for all`);
  for (const [key, max] of [['maxPings', 60], ['keepUnder', 1000000]]) if (!Number.isInteger(p[key]) || p[key] < 0 || p[key] > max) throw new RangeError(`policy.${key} must be a whole number between 0 and ${max}`);
  return Object.freeze(p);
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
  for (const key of Object.keys(DRAWN)) checkDrawn(key, c[key]);
  if (!['fixed','geometric','lognormal','continuation'].includes(c.lengthDistribution)) throw new RangeError('lengthDistribution must be fixed, geometric, lognormal or continuation');
  if (c.lengthDistribution === 'fixed' && !Number.isInteger(c.plannedTurns)) throw new RangeError('plannedTurns must be a whole number for a fixed length');
  const atLeast = (high, low) => { if (c[high] < c[low]) throw new RangeError(`${high} must be at least ${low}`); };
  if (c.lengthDistribution === 'lognormal') atLeast('plannedTurnsP95', 'plannedTurns');
  atLeast('tokenP95', 'tokenMedian'); atLeast('gapP95Minutes', 'gapMedianMinutes');
  for (const kind of KINDS) atLeast(`${kind}DependencyTurnsBackP95`, `${kind}DependencyTurnsBack`);
  if (c.safetyReserve >= c.model.context) throw new RangeError('safetyReserve must be less than the context window');
  for (const key of ['keepWarm', 'anthropic']) if (typeof c[key] !== 'boolean') throw new TypeError(`${key} must be boolean`);
  if (!COMPARISON_IDS.includes(c.compare)) throw new RangeError(`${c.compare} is not a comparison`);
  // The rules of your own application are kept as given, checked, with their defaults left out.
  const app = configureRules(c.app); c.app = Object.freeze(Object.fromEntries(Object.entries(app).filter(([key, value]) => value !== RULES[key])));
  c.policy = configurePolicy(c.policy);
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
// Normal CDF approximation.
function normalCdf(x) {const sign=x<0?-1:1, t=1/(1+.3275911*Math.abs(x)/Math.SQRT2); const e=1-(((((1.061405429*t-1.453152027)*t+1.421413741)*t-.284496736)*t+.254829592)*t)*Math.exp(-x*x/2);return .5*(1+sign*e);}
function lognormalCdf(x, median, p95) {const s=Math.log(p95/median)/z95; return x<=0?0:s===0?(x>=median?1:0):normalCdf(Math.log(x/median)/s);}

// How many turns back a needed item lies: a lognormal with this median and 95th percentile rounded to a
// whole turn and at least one, or drawn weights. F[a] is the chance that it lies at most `a` turns back.
export function ageLaw(median, p95, maxAge = DEFAULTS.maxTurns, drawn = null) {
  const sigma = Math.log(p95/median)/z95, F = new Float64Array(maxAge + 1), hist = drawn && histogram('need', drawn);
  for (let a = 1; a <= maxAge; a++) F[a] = hist ? hist.cdf(a) : Math.max(F[a - 1], sigma > 0 ? normalCdf(Math.log((a + .5)/median)/sigma) : median < a + .5 ? 1 : 0);
  return Object.freeze({
    cdf: age => F[Math.max(0, Math.min(maxAge, Math.floor(age)))],
    // The weight on exactly `age` turns back, before any thread length is taken into account.
    weight: age => age >= 1 && age <= maxAge ? F[age] - F[age - 1] : 0,
    // Only `oldest` earlier turns exist. Among them, the share that lie exactly `age` back.
    // When the law puts no weight on any of them, the oldest one stands in.
    pmf: (age, oldest) => F[oldest] > 0 ? (F[age] - F[age - 1])/F[oldest] : age === oldest ? 1 : 0,
    // Among the `oldest` turns that exist, the share that lie at most `age` back.
    within: (age, oldest) => age >= oldest ? 1 : age < 1 ? 0 : F[oldest] > 0 ? F[Math.floor(age)]/F[oldest] : 0,
    // The age for a uniform draw `u` in [0, 1), among the `oldest` turns that exist.
    sample: (u, oldest) => { if (!(F[oldest] > 0)) return oldest; const x = u*F[oldest]; let age = 1; while (age < oldest && F[age] <= x) age++; return age; },
    // The fewest turns back that cover a share `q` of the needs, on a thread long enough to hold them all.
    quantile: q => { let age = 1; while (age < maxAge && F[age] < q) age++; return age; }
  });
}
export const needLaw = (c, kind, maxAge = c.maxTurns) => ageLaw(c[`${kind}DependencyTurnsBack`], c[`${kind}DependencyTurnsBackP95`], maxAge, c[`${kind}NeedDrawn`]);
const needLaws = (c, maxAge) => Object.fromEntries(KINDS.map(kind => [kind, needLaw(c, kind, maxAge)]));
// Whether any turn retrieves anything.
const retrieves = c => (c.tokenDrawn !== null || c.tokenMedian !== 0) && c.retrievalShare !== 0;
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
// Histograms are built once per drawn array.
const histograms = new WeakMap();
const drawnOf = (kind, weights) => { let h = histograms.get(weights); if (!h) { h = histogram(kind, weights); histograms.set(weights, h); } return h; };
export function generateWorkload(config) {
  const c=configure(config), random=rng(c.seed), laws=needLaws(c,c.maxTurns), threads=[], gaps=[], planned=[], realized=[];
  const lengths=c.lengthDrawn && drawnOf('length',c.lengthDrawn), minutes=c.gapDrawn && drawnOf('gap',c.gapDrawn), sizes=c.tokenDrawn && drawnOf('token',c.tokenDrawn);
  const byChance=!lengths && c.lengthDistribution==='continuation', answerMinutes=c.answerSeconds/60, retrieving=retrieves(c);
  for(let ti=0;ti<c.threads;ti++) {
    // By chance of another turn, a thread has no length in advance: it runs until a turn is the last, or to the cap.
    let length=lengths ? lengths.quantile(random()) : byChance ? c.maxTurns : c.lengthDistribution==='fixed' ? c.plannedTurns : c.lengthDistribution==='geometric' ? Math.ceil(Math.log(Math.max(random(),1e-12))/Math.log(1-1/Math.max(1.00000001,c.plannedTurns))) : Math.round(lognormal(random,c.plannedTurns,c.plannedTurnsP95));
    length=Math.max(1,Math.min(c.maxTurns,length));
    // The earlier turns that retrieved, oldest first.
    const turns=[], retrieved=[]; let time=0;
    for(let i=0;i<length;i++) {
      if(i>0) {
        // A drawn length is the length.
        if(!lengths && random()>=(byChance?continueChance(i,c):c.continuationProbability)) break;
        // A long break with its chance, otherwise the usual delay for this position. The question arrives once the answer is there and the user is back.
        const g=c.breakShare>0 && random()<c.breakShare ? lognormal(random,c.breakMedianMinutes,4*c.breakMedianMinutes) : (minutes?minutes.quantile(random()):lognormal(random,c.gapMedianMinutes,c.gapP95Minutes))*gapFactor(i,c);
        gaps.push(g); time+=answerMinutes+g;
      }
      const found=retrieving && (c.retrievalShare>=1 || random()<c.retrievalShare);
      const retrieval=found?Math.max(1,Math.min(1000000,Math.round(sizes?sizes.quantile(random()):lognormal(random,c.tokenMedian,c.tokenP95)))):0, deps=[];
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
  for(let i=0,n=Math.min(a.length,b.length);i<n;i++) {const x=a[i], y=b[i]; if(x!==y && (x.id!==y.id || x.tokens!==y.tokens)) break; total+=x.tokens;}
  return total;
}
export function priceCall(input,cached,output,model,{cacheable=true}={}) {
  if (![input,cached,output].every(Number.isFinite) || input<0 || output<0 || cached<0 || cached>input) throw new RangeError('invalid call token counts');
  const long=model.longThreshold>0 && input>model.longThreshold, im=long?model.longInputMultiplier:1, om=long?model.longOutputMultiplier:1;
  const fresh=input-cached, inputCost=fresh*model.inputRate*(cacheable?model.writeMultiplier:1)*im/1e6, cachedCost=cached*model.cachedRate*im/1e6, outputCost=output*model.outputRate*om/1e6;
  return {inputTokens:input,cachedTokens:cached,freshTokens:fresh,outputTokens:output,inputCost,cachedCost,outputCost,cost:inputCost+cachedCost+outputCost,long};
}
// The same, as one number.
function costOf(input,cached,output,model,cacheable) {
  const long=model.longThreshold>0 && input>model.longThreshold, im=long?model.longInputMultiplier:1;
  return ((input-cached)*model.inputRate*(cacheable?model.writeMultiplier:1)*im+cached*model.cachedRate*im+output*model.outputRate*(long?model.longOutputMultiplier:1))/1e6;
}
// The chance that a returning user is back within `t` minutes of the answer to turn `turn`, counted from 1: the usual
// delay at that position, mixed with long breaks.
function gapCdf(t,c,turn=Infinity) {
  if(t<=0)return 0;
  const factor=gapFactor(turn,c), usual=c.gapDrawn ? drawnOf('gap',c.gapDrawn).cdf(t/factor) : lognormalCdf(t,c.gapMedianMinutes*factor,c.gapP95Minutes*factor);
  return c.breakShare>0 ? (1-c.breakShare)*usual+c.breakShare*normalCdf(Math.log(t/c.breakMedianMinutes)/BREAK_SIGMA) : usual;
}
// Planned length is rounded and capped in generateWorkload; return probability
// is conditional on reaching this turn, without inspecting a realized future.
function plannedSurvival(turn,c) {
  if(turn<=1)return 1;
  if(turn>c.maxTurns)return 0;
  if(c.lengthDrawn)return 1-drawnOf('length',c.lengthDrawn).cdf(turn-1);
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
  if(!c.lengthDrawn && c.lengthDistribution==='continuation')return continueChance(runIndex+1,c);
  const currentSurvival=plannedSurvival(runIndex+1,c), nextSurvival=plannedSurvival(runIndex+2,c);
  return currentSurvival>0?(c.lengthDrawn?1:c.continuationProbability)*nextSurvival/currentSurvival:0;
}
export function theoreticalSurvival(config={}) {
  const c=configure(config);
  // Drawn, or by chance of another turn, there is no planned length apart from the realized one, and no flat chance on top of it.
  const flat=c.lengthDrawn || c.lengthDistribution==='continuation'?1:c.continuationProbability;
  return Array.from({length:c.maxTurns},(_,i)=>({turn:i+1,planned:plannedSurvival(i+1,c),realized:plannedSurvival(i+1,c)*flat**i}));
}
// CDF of the return delay after turn `turn` (counted from 1, a long thread when left out), conditional on a
// continuation: not an unconditional return probability.
export function returnCdf(minutes,config={},turn=Infinity) {return gapCdf(minutes,configure(config),turn);}
// The share of returning users whose next question finds the cache alive without a ping: the answer time and
// the delay together stay inside the cache lifetime, which runs from when the request was sent.
export function warmReturnShare(config={},turn=Infinity) {const c=configure(config);return gapCdf(c.ttlMinutes-c.answerSeconds/60,c,turn);}
// The weights of each bin under the law in force, which is where drawing starts from.
export function lawWeights(name,config={}) {
  const c=configure(config);
  if(c[name])return [...c[name]];
  if(name==='tokenDrawn')return weightsOf('token',x=>c.tokenMedian>0?lognormalCdf(x,c.tokenMedian,c.tokenP95):1);
  if(name==='gapDrawn')return weightsOf('gap',x=>lognormalCdf(x,c.gapMedianMinutes,c.gapP95Minutes));
  if(name==='lengthDrawn'){const s=theoreticalSurvival(c);return weightsOf('length',x=>x>=c.maxTurns?1:1-(s[x]?.realized??0));}
  return weightsOf('need',needLaw(c,name.replace('NeedDrawn','')).cdf);
}
// The median and the 95th percentile of a drawn distribution.
export function drawnStats(name,weights) {const h=histogram(DRAWN[name],weights);return {median:h.quantile(.5),p95:h.quantile(.95),mean:h.mean};}
export {BINS, DRAWN};
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
// The part of a matching prefix that the cache can serve.
const cacheable=(tokens,c)=>tokens>=c.cacheMinTokens?Math.floor(tokens/c.cacheBlockTokens)*c.cacheBlockTokens:0;

// -- the decisions ------------------------------------------------------------

// What a decision looks up: the chance of a return after each turn, and how soon one comes.
function forecast(c) {
  const interval=c.ttlMinutes-c.pingLeadMinutes, answer=c.answerSeconds/60, rows=[], shared=sharedSystem(c);
  const drawnTokens=c.tokenDrawn && drawnOf('token',c.tokenDrawn).mean, sigma=c.tokenMedian>0?Math.log(c.tokenP95/c.tokenMedian)/z95:0;
  // The mean of what a turn adds before its answer.
  const arrival=c.inputTokens+(retrieves(c)?c.retrievalShare*(drawnTokens??c.tokenMedian*Math.exp(sigma*sigma/2)):0);
  return {interval,answer,shared,arrival,
    at(pos,maxPings) {
      const row=rows[pos] ||= {returns:returnChance(pos,c),warm:gapCdf(c.ttlMinutes-answer,c,pos+1),before:[0],until:[]};
      for(let n=row.before.length;n<=maxPings;n++) {row.before[n]=gapCdf(n*interval-answer,c,pos+1);row.until[n]=gapCdf(n*interval+c.ttlMinutes-answer,c,pos+1);}
      return row;
    }};
}
// The package's question before every request and after every answer, in expectation: what does it cost, three turns
// ahead, to send the history as it is or to cut it, and to keep the cache warm with 0, 1 or more pings while the user is away?
// `views` holds the request as it is and, when a cut would remove something, the cut one: its size, and how much of the
// uncut request it still shares. Later turns add their expected size to whichever view was chosen.
function lookahead(c,views,maxPings,future) {
  const {interval,answer,shared,arrival}=future, cap=Math.min(c.model.maxInput,c.model.context-c.safetyReserve), history=c.outputTokens, billed=c.outputTokens+c.reasoningTokens;
  const charge=(size,own,out)=>costOf(size,expectedRead(cacheable(Math.min(own,size),c),shared,c),out,c.model,size>=c.cacheMinTokens);
  function arrive(pos,depth,v,growth,own) {
    const size=views[v].size+growth;
    if(size>cap)return Infinity;
    let total=charge(size,own,billed);
    if(c.callsPerTurn>1)total+=(c.callsPerTurn-1)*charge(size,size,0);
    if(depth>1)total+=Math.min(...after(pos,depth-1,v,growth+history,size).map(row=>row.total));
    return total;
  }
  // One row per number of pings: what the pings cost, and what they and the turns after them cost together.
  function after(pos,depth,v,growth,asked) {
    const held=views[v].size+growth, possible=c.keepWarm && maxPings>0 && interval>0 && answer<interval && asked>=c.cacheMinTokens && held<=cap;
    const last=possible?maxPings:0, {returns,warm,before,until}=future.at(pos,last);
    const next=own=>{let best=Infinity;for(let w=0;w<views.length;w++)if(w>=v || v===0)best=Math.min(best,arrive(pos+1,depth,w,growth+arrival,w===v?own:Math.min(own,views[w].shared)));return best;};
    const any=returns>0, cold=any?next(0):0, plain=any?next(asked):0, kept=any && last?next(held):0, rows=[];
    let pings=0;
    for(let n=0;n<=last;n++) {
      if(n)pings+=(1-returns*before[n])*charge(held,n===1?asked:held,c.pingOutputTokens);
      const early=n?before[1]:warm, late=n?Math.max(0,until[n]-before[1]):0;
      rows.push({pings:n,pingCost:pings,total:pings+(any?returns*(early*plain+late*kept+Math.max(0,1-early-late)*cold):0)});
    }
    return rows;
  }
  return {arrive,after};
}

// -- one side of the comparison -----------------------------------------------

const block=(id,tokens,kind,turn,source=id)=>Object.freeze({id,tokens,kind,turn,source});
// The notice one archived part leaves. A turn's retrieved text may be several parts, a question or an answer is one.
const PART_NOTICE=28;
const noticeOf=(kind,c)=>kind==='retrieval'?c.archivedTokens:Math.min(PART_NOTICE,c.archivedTokens);
const LETTER={input:'i',retrieval:'r',output:'o'};
const sourceOf=d=>`${LETTER[d.kind]}${d.turn}`;
// The settings a comparison changes: the provider's long cache, or none.
function withCache(c,rules) {
  if(rules.cache==='long')return {...c,ttlMinutes:c.longTtlMinutes,model:{...c.model,writeMultiplier:c.longWriteMultiplier}};
  return c;
}
// Price every call one side makes on a workload. `strategy` is {kind: 'compache', policy} or {kind: 'rules', rules}.
export function runArm(workload, config, strategy, {collectDecisions=true,collectLedger=true}={}) {
  const base=configure(config), ours=strategy.kind==='compache'; if(!ours && strategy.kind!=='rules')throw new RangeError('invalid strategy');
  const rules=ours?null:configureRules(strategy.rules), policy=ours?configurePolicy(strategy.policy ?? {}):null, caching=ours || rules.cache!=='off', c=withCache(base,ours?{cache:policy.long && longAvailable(base)?'long':'standard'}:rules);
  const maxAge=Math.max(c.maxTurns,...workload.threads.map(t=>t.turns.length)), laws=needLaws(c,maxAge), future=forecast(c);
  const ledger=[], decisions=[], roles={}, cacheBreakdown=collectLedger?{cached:0,newContext:0,firstRequest:0,expiredPrefix:0,changedPrefix:0,missed:0}:null, total={inputTokens:0,cachedTokens:0,freshTokens:0,outputTokens:0,inputCost:0,cachedCost:0,outputCost:0,cost:0,calls:0};
  const freed=blocks=>blocks.reduce((n,b)=>n+b.tokens-noticeOf(b.kind,c),0);
  let completedThreads=0,completedRuns=0,attemptedRuns=0,compactions=0,clearings=0,recalls=0,required=0,covered=0,failedCalls=0,pings=0,totalRemovedTokens=0,totalRecallTokens=0;
  const threadCosts=[], pre=Object.fromEntries(KINDS.map(kind=>[kind,{expectedRequired:0,expectedRetained:0,observedRequired:0,observedRetained:0}]));
  for(const thread of workload.threads) {
    let history=[],snapshots=[],expiredSnapshots=[],seenBlocks=new Set(),gone=new Map(),callIndex=0,threadCost=0,summaryIndex=0,failed=false;
    // Compache: per kind, the earlier turns whose part is in the request, and those whose part is archived.
    const held={input:new Set(),retrieval:new Set(),output:new Set()};
    // The earlier turns that retrieved anything, oldest first.
    const retrievedBefore=[];
    const system=block('system',c.systemTokens,'system',-1);
    function expireSnapshots(time) {
      const stillLive=[]; for(const snapshot of snapshots) {if(time-snapshot.time<c.ttlMinutes)stillLive.push(snapshot);else if(collectLedger)expiredSnapshots.push(snapshot);}
      snapshots=stillLive; if(expiredSnapshots.length>16)expiredSnapshots=expiredSnapshots.slice(-16);
    }
    const livePrefix=blocks=>{let prefix=0;for(const s of snapshots)prefix=Math.max(prefix,s.blocks===blocks?sum(blocks):sharedPrefix(s.blocks,blocks));return prefix;};
    function admitted(input,output,role) {
      const reserve=ours?c.safetyReserve:role==='answer'?Math.min(rules.reserveTokens,.25*c.model.context):0;
      return input<=c.model.maxInput && input+Math.max(output,reserve)<=c.model.context;
    }
    // Price one call. `blocks` is kept as the cached request, so the caller must not change it afterwards.
    function request(blocks,role,time,output,turn) {
      const input=sum(blocks);
      if(!admitted(input,output,role)){failedCalls++;if(collectLedger)ledger.push({thread:thread.id,turn,role,time,inputTokens:input,outputTokens:output,admitted:false});return null;}
      expireSnapshots(time);
      const prefix=livePrefix(blocks), own=cacheable(prefix,c);
      // The call reads its own thread's live prefix when it hits it, and otherwise the system block when another
      // conversation has that cached. It is priced at the expected tokens read: the blend of those outcomes.
      const shared=caching && blocks[0]===system?sharedSystem(c):0, cached=expectedRead(own,shared,c);
      let attribution=null;
      if(collectLedger && role==='answer') {
        let best=prefix; if(callIndex)for(const snapshot of expiredSnapshots) best=Math.max(best,sharedPrefix(snapshot.blocks,blocks));
        const bestCached=cacheable(best,c);
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
      const costs=priceCall(input,cached,output,c.model,{cacheable:caching && input>=c.cacheMinTokens});
      const row=collectLedger?{thread:thread.id,turn,role,time,admitted:true,...costs,...(attribution?{cacheBreakdown:attribution}:{})}:costs; if(collectLedger)ledger.push(row); callIndex++;
      tally(role,costs);
      if(collectLedger)for(const b of blocks)seenBlocks.add(`${b.id}:${b.tokens}`);
      if(caching && input>=c.cacheMinTokens) snapshots.push({time,blocks});
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
      const count=c.callsPerTurn-1, input=sum(blocks), can=caching && input>=c.cacheMinTokens;
      const one=priceCall(input,caching?expectedRead(can?cacheable(input,c):0,blocks[0]===system?sharedSystem(c):0,c):0,0,c.model,{cacheable:can});
      const costs=Object.fromEntries(COST_KEYS.map(key=>[key,count*one[key]]));
      if(collectLedger)ledger.push({thread:thread.id,turn,role:'further',time,admitted:true,calls:count,...costs,long:one.long});
      tally('further',costs,count);
    }
    // The pings of one wait, until the user is back.
    function ping(blocks,count,turn,i) {
      const next=thread.turns[i+1]?.time ?? Infinity, interval=c.ttlMinutes-c.pingLeadMinutes; let sent=0;
      for(let n=1;n<=count && interval>0;n++) {const time=turn.time+n*interval;if(time>=next)break;if(!request(blocks,'ping',time,c.pingOutputTokens,i)){failed=true;break;}pings++;sent++;}
      return sent;
    }
    for(let i=0;i<thread.turns.length;i++) {
      const turn=thread.turns[i], retrievedTokens=ours?turn.retrievalTokens:Math.round(turn.retrievalTokens*(1-rules.compress));
      const input=block(`i${i}`,turn.inputTokens,'input',i), retrieval=block(`r${i}`,retrievedTokens,'retrieval',i), output=block(`o${i}`,turn.outputTokens,'output',i);
      attemptedRuns++;
      const needs=turn.dependencies.filter(d=>d.kind!=='retrieval'||thread.turns[d.turn]?.retrievalTokens!==0);
      let decision=null, added=[], plan=null, cutViews=null;
      if(ours) {
        expireSnapshots(turn.time);
        const keep=[system,...history,input,...(retrieval.tokens?[retrieval]:[])], warm=snapshots.some(s=>sharedPrefix(s.blocks,keep)>=Math.max(1,c.cacheMinTokens));
        // The cut the policy would make now: per kind, what lies outside its window and is larger than its notice.
        const from={input:policy.questions===null?0:i-policy.questions,output:policy.answers===null?0:i-policy.answers,
          retrieval:policy.retrieved===null?0:policy.retrieved===0?i:retrievedBefore.length>=policy.retrieved?retrievedBefore[retrievedBefore.length-policy.retrieved]:0};
        const goes=b=>from[b.kind]>b.turn && b.tokens>Math.max(policy.keepUnder,noticeOf(b.kind,c)), removed=[];
        let cut=null;
        if(history.some(goes)) {
          const kept=history.map(b=>{if(!goes(b))return b;removed.push(b);const left=noticeOf(b.kind,c);return left?block(`archived-${b.id}`,left,'archived',b.turn,'archived'):null;}).filter(Boolean);
          cut={history:kept,blocks:[system,...kept,input,...(retrieval.tokens?[retrieval]:[])]};
        }
        const views=[{size:sum(keep),shared:sum(keep)}]; if(cut)views.push({size:sum(cut.blocks),shared:sharedPrefix(keep,cut.blocks)});
        cutViews=views;
        const planner=lookahead(c,views,policy.maxPings,future), price=views.map((view,v)=>planner.arrive(i,LOOKAHEAD,v,0,livePrefix(v?cut.blocks:keep)));
        // A tie keeps the request as it is, and so does a cut that would not fit any better.
        const chosen=cut && price[1]<price[0]?1:0;
        plan={planner,chosen};
        const before=sum(keep);
        if(chosen) {
          history=cut.history; compactions++; totalRemovedTokens+=freed(removed);
          for(const b of removed)held[b.kind].delete(b.turn);
        }
        // What is in view before anything is fetched back. The expectation integrates over the age laws, not the needs drawn.
        if(i) for(const kind of KINDS) {
          const p=kind==='retrieval' && (!retrieves(c) || !retrievedBefore.length)?0:c[`${kind}DependencyProbability`];
          let share=1;
          if(p) {
            if(kind!=='retrieval' || retrievedBefore.length===i){share=0;for(const origin of held[kind])share+=laws[kind].pmf(i-origin,i);}
            else {let all=0,kept=0;for(const origin of retrievedBefore){const w=laws.retrieval.weight(i-origin);all+=w;if(held.retrieval.has(origin))kept+=w;}share=all>0?kept/all:held.retrieval.has(retrievedBefore[0])?1:0;}
          }
          pre[kind].expectedRequired+=p; pre[kind].expectedRetained+=p*share;
        }
        // A needed part that is not in view is fetched back for this turn. It is not kept afterwards.
        for(const d of needs) if(!held[d.kind].has(d.turn)) {
          const from=thread.turns[d.turn], tokens=d.kind==='input'?from.inputTokens:d.kind==='output'?from.outputTokens:from.retrievalTokens;
          added.push(block(`recall-${i}-${sourceOf(d)}`,tokens,'recall',i,sourceOf(d))); recalls++; totalRecallTokens+=tokens;
        }
        if(collectDecisions) {
          const state=kind=>Array.from({length:i},(_,t)=>kind==='retrieval' && !thread.turns[t].retrievalTokens?'-':added.some(b=>b.source===`${LETTER[kind]}${t}`)?'f':held[kind].has(t)?'k':'a').join('');
          decision={thread:thread.id,turn:i,time:turn.time,cache:i===0?'first':warm?'warm':'cold',action:chosen?'cut':'keep',beforeTokens:before,afterTokens:views[chosen].size,cutTokens:cut?views[1].size:null,keepCost:price[0],cutCost:cut?price[1]:null,
            removed:removed.length && chosen?removed.length:0,removedTokens:chosen?freed(removed):0,fetched:added.map(b=>b.source),recallTokens:sum(added),fullTokens:c.systemTokens+thread.turns.slice(0,i).reduce((n,t)=>n+t.inputTokens+t.retrievalTokens+t.outputTokens,0)+turn.inputTokens+turn.retrievalTokens,
            parts:{input:state('input'),retrieval:state('retrieval'),output:state('output')},byKind:null,planner:null,plannedPings:0,actualPings:0,admitted:null};
        }
        history=[...history,input,...(retrieval.tokens?[retrieval]:[])];
        held.input.add(i); if(retrieval.tokens)held.retrieval.add(i);
        for(const d of needs) {pre[d.kind].observedRequired++; if(held[d.kind].has(d.turn))pre[d.kind].observedRetained++;}
      } else {
        // Dropped for good, or until it is fetched again.
        const drop=keepBlock=>{history=history.filter(b=>{const stays=b.turn<0 || keepBlock(b);if(!stays && b.kind==='retrieval')gone.set(b.source,true);return stays;});};
        if(rules.keepTurns)drop(b=>b.turn>=i-rules.keepTurns);
        if(rules.dropRetrievedAfter>=0)drop(b=>b.kind!=='retrieval' || b.turn>=i-rules.dropRetrievedAfter);
        if(rules.clear) {
          // All but the newest tool results become a placeholder, above a size and if that frees enough.
          // A turn that retrieved nothing has no tool result to keep or to clear.
          const results=history.filter(b=>b.kind==='retrieval' && b.tokens>0), old=results.slice(0,Math.max(0,results.length-(rules.clearKeepResults-(retrieval.tokens>0?1:0))));
          if(old.length && c.systemTokens+sum(history)+input.tokens+retrieval.tokens>=Math.max(rules.clearTriggerFloor,Math.floor(c.model.context*rules.clearTriggerShare)) && sum(old)-old.length*rules.clearedTokens>=Math.max(rules.clearMinFreedFloor,Math.floor(c.model.context*rules.clearMinFreedShare))) {
            const now=new Set(old); for(const b of old)gone.set(b.source,true);
            history=history.map(b=>now.has(b)?block(`cleared-${b.id}`,rules.clearedTokens,'cleared',b.turn,'cleared'):b); clearings++;
          }
        }
        // A cleared or dropped result that this turn needs is fetched again, when the rules say so. A summarized one is not: the summary stands in for it.
        if(rules.refetch) for(const d of needs) {const source=sourceOf(d); if(gone.has(source)) {const tokens=Math.round(thread.turns[d.turn].retrievalTokens*(1-rules.compress));added.push(block(`refetch-${i}-${source}`,tokens,'retrieval',i,source)); gone.delete(source); recalls++; totalRecallTokens+=tokens;}}
        const now=[...(retrieval.tokens?[retrieval]:[])], view=[system,...history,input,...now,...added];
        const threshold=Math.min(rules.summaryTriggerShare*c.model.context,rules.summaryTriggerTokens || Infinity,c.model.context-Math.min(rules.reserveTokens,.25*c.model.context));
        if(rules.summarize && sum(view)>=threshold && history.length) {
          // The newest complete turns stay until the tail is full. The rest becomes the summary.
          const kept=[]; let size=0;
          if(rules.tailTokens) for(let j=i-1;j>=0;j--){const blocks=history.filter(b=>b.turn===j && b.kind!=='summary' && b.kind!=='record'); const n=sum(blocks); if(!blocks.length)continue; kept.unshift(...blocks);size+=n;if(size>=rules.tailTokens)break;}
          const keptIds=new Set(kept.map(b=>b.id)), head=history.filter(b=>!keptIds.has(b.id));
          if(rules.flushOutputTokens && !request([block('flush-system',c.systemTokens,'system',-1),...history,input,...now,...added],'flush',turn.time,rules.flushOutputTokens,i)){failed=true;break;}
          let asked;
          if(rules.summaryOfRequest)asked=[system,...history];
          else {
            // The summary reads only the older head, with bounded tool results and a system prompt of its own.
            const projected=[]; let budget=rules.summaryInputCap;
            for(const b of head) {
              const available=b.kind==='retrieval'?Math.min(b.tokens,rules.toolSummaryTokens):b.tokens, n=Math.min(available,budget);
              if(n>0)projected.push(block(`summarized-${b.id}`,n,b.kind,b.turn,b.source)); budget-=n;
              if(budget<=0)break;
            }
            asked=[block('summary-system',rules.summarySystemTokens,'system',-1),...projected];
          }
          if(!request(asked,'summary',turn.time,rules.summaryTokens,i)){failed=true;break;}
          const sequence=++summaryIndex;
          history=[block(`summary-${sequence}`,rules.summaryTokens,'summary',-1),...(rules.recordTokens?[block(`record-${sequence}`,rules.recordTokens,'record',-1)]:[]),...kept]; compactions++;
        }
        history=[...history,input,...now];
        for(const d of needs) {pre[d.kind].observedRequired++; if(history.some(b=>b.source===sourceOf(d)))pre[d.kind].observedRetained++;}
      }
      const asked=[system,...history];
      let view=asked;
      if(added.length) {
        // The model asks for the missing sources in one short reply; the answer call then carries them.
        if(!request(asked,'recovery',turn.time,c.recoveryOutputTokens,i)){if(decision){decision.admitted=false;decisions.push(decision);}failed=true;break;}
        const call=block(`reopen-${i}`,c.recoveryOutputTokens,'recovery',i,'recovery');
        view=[...asked,call,...added];
        // Compache answers the fetch itself, and the application never sees it. A tool call of the application's own stays in its history.
        if(!ours)history=[...history,call,...added];
      }
      required+=needs.length;
      for(const d of needs) if(view.some(b=>b.source===sourceOf(d))) covered++;
      // Reasoning is billed as output with the answer and is not sent again.
      const row=request(view,'answer',turn.time,turn.outputTokens+c.reasoningTokens,i);
      if(!row){if(decision){decision.admitted=false;decisions.push(decision);}failed=true;break;}
      if(c.callsPerTurn>1)furtherCalls(view,turn.time,i);
      completedRuns++; if(output.tokens)history=[...history,output]; held.output.add(i); if(turn.retrievalTokens>0)retrievedBefore.push(i);
      if(ours) {
        const full=[system,...history];
        // Planned on what the next request will hold: this view with its answer, without what was fetched back.
        const rows=plan.planner.after(i,LOOKAHEAD-1,plan.chosen,c.outputTokens,cutViews[plan.chosen].size);
        let best=rows[0]; for(const r of rows)if(r.total<best.total-1e-15)best=r;
        const sent=ping(full,best.pings,turn,i);
        if(decision) {
          const {returns}=future.at(i,0), kinds={system:c.systemTokens}; for(const b of asked)if(b.kind!=='system')kinds[b.kind]=(kinds[b.kind]||0)+b.tokens; if(added.length)kinds.recall=sum(added)+c.recoveryOutputTokens;
          const reason=!c.keepWarm?'disabled':!policy.maxPings?'no pings allowed':future.interval<=0?'ping interval is not positive':future.answer>=future.interval?'answer outlasts the ping interval':cutViews[plan.chosen].size<c.cacheMinTokens?'prefix is below cache minimum':returns<=0?'no modeled future return':rows.length<2?'the request leaves no room for a ping':best.pings?'positive expected value':'no positive expected value';
          Object.assign(decision,{admitted:true,byKind:kinds,sentTokens:row.inputTokens,cachedTokens:row.cachedTokens,cost:row.cost,plannedPings:best.pings,actualPings:sent,planner:{reason,returnProbability:returns,interval:future.interval,chosen:best.pings,rows:rows.map(r=>({pings:r.pings,expectedCost:r.pingCost,expectedGain:rows[0].total-r.total+r.pingCost,net:rows[0].total-r.total}))}});
          decisions.push(decision);
        }
      } else if(rules.pings && caching) ping([system,...history],rules.pings,turn,i);
      if(failed)break;
    }
    threadCosts.push(threadCost);
    if(!failed)completedThreads++;
  }
  const add=key=>Object.values(pre).reduce((n,x)=>n+x[key],0), expectedRequired=add('expectedRequired'), expectedRetained=add('expectedRetained'), observedRequired=add('observedRequired'), observedRetained=add('observedRetained');
  const byKindRetention=Object.fromEntries(Object.entries(pre).map(([kind,row])=>[kind,{...row,expected:row.expectedRequired?row.expectedRetained/row.expectedRequired:null,observed:row.observedRequired?row.observedRetained/row.observedRequired:null}]));
  const retention={expected:ours && expectedRequired?expectedRetained/expectedRequired:null,observed:observedRequired?observedRetained/observedRequired:null,expectedRequired,expectedRetained,observedRequired,observedRetained,byKind:byKindRetention};
  return {totals:total,byRole:roles,ledger,decisions,retention,cacheBreakdown,threadCosts,removedTokens:totalRemovedTokens,recallTokens:totalRecallTokens,completedThreads,plannedThreads:workload.threads.length,completedRuns,plannedRuns:workload.totalRuns,attemptedRuns,failedCalls,complete:completedThreads===workload.threads.length,compactions,clearings,recalls,pings,dependencyCoverage:required?covered/required:null,dependencies:{required,covered}};
}

// -- the fit --------------------------------------------------------------------

// Windows follow 1, 2, 5, 10 and so on below the longest thread, plus the look-backs that cover set shares of what a kind's turns need.
export function windows(c,kind,longest) {
  const list=new Set([0]); for(let scale=1;scale<longest;scale*=10)for(const n of [1,2,5])if(n*scale<longest)list.add(n*scale);
  const law=needLaw(c,kind); for(const q of [.8,.85,.9,.95,.975])if(law.quantile(q)<longest)list.add(law.quantile(q));
  return [null,...[...list].sort((a,b)=>b-a)];
}
// Every combination of one window per kind, scored without a replay as the package does: what an applied cut leaves of the
// history on an average turn, and of what turns need. Only the frontier is priced: each cut keeps at least a point more than every smaller one.
export function candidateCuts(workload,config,limit=6) {
  const c=configure(config), longest=Math.max(...workload.threads.map(t=>t.turns.length)), laws=needLaws(c,Math.max(c.maxTurns,longest)), scored={};
  for(const kind of KINDS) {
    const size=t=>kind==='input'?t.inputTokens:kind==='output'?t.outputTokens:t.retrievalTokens, largest=Math.max(0,...workload.threads.flatMap(t=>t.turns.map(size)));
    // A kind that nothing can be archived of has no window.
    const left=noticeOf(kind,c), list=largest>left?windows(c,kind,longest):[null];
    scored[kind]=list.map(window=>{
      let kept=0,held=0,needed=0;
      for(const thread of workload.threads) {
        const origins=[], before=[0];
        for(const [i,turn] of thread.turns.entries()) {
          const n=origins.length, first=window===null?0:Math.max(0,n-window), archived=before[first];
          kept+=before[n]-archived; for(let k=0;k<first;k++)kept+=Math.min(size(thread.turns[origins[k]]),left);
          const p=n?c[`${kind}DependencyProbability`]:0;
          if(p) {
            let share=1;
            if(window!==null && first>0) {
              if(n===i)share=laws[kind].within(window,i);
              else {let all=0,inside=0;origins.forEach((origin,k)=>{const w=laws[kind].weight(i-origin);all+=w;if(k>=first)inside+=w;});share=all>0?inside/all:0;}
            }
            needed+=p; held+=p*share;
          }
          if(kind!=='retrieval' || turn.retrievalTokens>0) {origins.push(i);before.push(before.at(-1)+size(turn));}
        }
      }
      return {window,kept,held,needed};
    });
  }
  const all=KINDS.reduce((n,kind)=>n+scored[kind][0].kept,0), needed=KINDS.reduce((n,kind)=>n+scored[kind][0].needed,0), rows=[];
  for(const a of scored.input)for(const b of scored.retrieval)for(const d of scored.output) {
    const kept=a.kept+b.kept+d.kept; if(kept>=all)continue;
    const coverage=needed?(a.held+b.held+d.held)/needed:null;
    if(coverage!==null && coverage+1e-12<c.retentionRequirement)continue;
    rows.push({policy:{questions:a.window,retrieved:b.window,answers:d.window},kept:all?kept/all:1,coverage});
  }
  rows.sort((x,y)=>x.kept-y.kept || (y.coverage??1)-(x.coverage??1));
  const frontier=[]; let best=null;
  for(const row of rows) if(best===null || (row.coverage??1)>=best+.01){frontier.push(row);best=row.coverage??1;}
  if(frontier.length>limit){const step=(frontier.length-1)/(limit-1);return Array.from({length:limit},(_,i)=>frontier[Math.round(i*step)]);}
  return frontier;
}
// Whether a policy can ask for the long cache, as the package can of Claude.
export const longAvailable=c=>c.anthropic && c.longTtlMinutes>c.ttlMinutes;
export const pingCounts=c=>c.keepWarm?[...new Set([0,1,3,c.maxPings])].filter(n=>n<=c.maxPings).sort((a,b)=>a-b):[0];
// The cheapest candidate on these threads. Keeping everything, with and without pings, is always one of them.
export function fitPolicy(workload,config,onProgress) {
  const c=configure(config), counts=pingCounts(c), cuts=candidateCuts(workload,c), candidates=[];
  const lifetimes=longAvailable(c)?[false,true]:[false], windows=[{policy:{},kept:1,coverage:null,everything:true},...cuts];
  const total=lifetimes.length*windows.length*counts.length;
  for(const long of lifetimes) for(const row of windows) for(const maxPings of counts) {
    onProgress?.(candidates.length+1,total);
    const policy=configurePolicy({...row.policy,maxPings,long}), result=runArm(workload,c,{kind:'compache',policy},{collectDecisions:false,collectLedger:false});
    candidates.push({policy,everything:!!row.everything,kept:row.kept,coverage:row.everything?1:row.coverage,retention:result.retention.expected,cost:result.totals.cost,requestTokens:(result.byRole.answer?.inputTokens??0)/Math.max(1,result.byRole.answer?.calls??0),complete:result.complete});
  }
  const usable=candidates.filter(x=>x.complete);
  usable.sort((a,b)=>a.cost-b.cost || (b.coverage??1)-(a.coverage??1) || a.policy.long-b.policy.long || a.policy.maxPings-b.policy.maxPings);
  return {candidates,chosen:usable[0] ?? null};
}
// A policy in the package's words. `longMinutes` is how long the long cache lives.
export function describePolicy(policy,longMinutes=60) {
  const lasting=longMinutes===60?'one-hour':longMinutes%60===0?`${longMinutes/60}-hour`:`${longMinutes}-minute`;
  const window=(n,noun)=>n===null?`all ${noun}`:n===0?`no earlier ${noun}`:`${noun} from the last ${n===1?'turn':`${n} turns`}`;
  const keep=policy.questions===null && policy.answers===null && policy.retrieved===null?'Keep the whole history in view.':`Keep ${window(policy.questions,'questions')}, ${window(policy.answers,'answers')} and ${window(policy.retrieved,'retrieved text')}.`;
  return [keep,policy.keepUnder?`Parts of at most ${policy.keepUnder} tokens always stay.`:'',policy.long?`Ask for ${lasting} cache entries.`:'',policy.maxPings?`Send up to ${policy.maxPings} keep-warm ping${policy.maxPings===1?'':'s'} while a user is away.`:'Send no keep-warm pings.'].filter(Boolean);
}
// The saving when the threads are drawn again from the ones at hand: the 5th and 95th percentile.
function savingRange(ours,theirs,seed) {
  const random=rng(seed^0x9E3779B9), n=ours.length, draws=[];
  for(let k=0;k<400;k++){let a=0,b=0;for(let i=0;i<n;i++){const j=Math.floor(random()*n);a+=ours[j];b+=theirs[j];}draws.push(b?1-a/b:0);}
  draws.sort((x,y)=>x-y); return {low:quantile(draws,.05),high:quantile(draws,.95)};
}
// Baseline selection, app rules, a measured bill and a pinned policy affect the replay only.
function fitKey(config) {
  const {compare,app,measuredCost,policy,...trainingInputs}=config;
  return JSON.stringify(trainingInputs);
}

export function fitScenario(config,onProgress) {
  const training=generateWorkload(config);
  return {...fitPolicy(training,config,onProgress),threads:training.threads.length,turns:training.totalRuns,key:fitKey(config)};
}

// Keep just the last custom fit in memory. The published starter comes from this same engine at build time.
export function createFitCache(starter=null) {
  let previous;
  return (config,onProgress)=>{
    const key=fitKey(config);
    if(starter?.key===key)return starter;
    if(previous?.key!==key)previous=fitScenario(config,onProgress);
    return previous;
  };
}

export function simulate(overrides={},getFit=fitScenario) {
  // The policy is fitted on one set of threads and every side is costed on a second set drawn the same way.
  const config=configure(overrides), fit=getFit(config), workload=generateWorkload({...config,seed:(config.seed+1)>>>0});
  const asked=config.policy ?? fit.chosen?.policy ?? configurePolicy({});
  // A policy made for a model with a long cache asks for none where there is none.
  const policy=asked.long && !longAvailable(config)?configurePolicy({...asked,long:false}):asked;
  const compache=runArm(workload,config,{kind:'compache',policy}), fittedHere=config.policy && fit.chosen?runArm(workload,config,{kind:'compache',policy:fit.chosen.policy},{collectDecisions:false,collectLedger:false}):null;
  const per=cost=>1000*cost/workload.totalRuns, fetches=arm=>(arm.byRole.recovery?.calls ?? 0)/workload.totalRuns;
  // Every comparison, and the calls of the one on screen. A measured cost has no calls of its own: your app's rules stand in for them.
  const list=comparisons(config);
  const measured=config.compare==='measured' && config.measuredCost>0;
  if(!measured && !list.some(b=>b.id===config.compare)) config.compare=DEFAULTS.compare;
  const shown=measured?'app':config.compare;
  let baseline=null; const sides=list.map(b=>{
    const arm=runArm(workload,config,{kind:'rules',rules:b.rules},{collectDecisions:false,collectLedger:b.id===shown});
    if(b.id===shown)baseline={id:b.id,name:b.name,rules:b.rules,arm};
    return {id:b.id,name:b.name,note:b.note,url:b.url,checked:b.checked,rules:b.rules,complete:arm.complete,per1000:arm.complete?per(arm.totals.cost):null,inView:arm.retention.observed,missing:arm.dependencyCoverage===null?null:1-arm.dependencyCoverage,fetches:fetches(arm),saving:arm.complete && compache.complete && arm.totals.cost?1-compache.totals.cost/arm.totals.cost:null};
  });
  if(config.measuredCost>0)sides.unshift({id:'measured',name:'What you pay now',note:'The cost you entered. Not modeled.',rules:null,complete:true,per1000:config.measuredCost,inView:null,missing:null,fetches:null,saving:compache.complete?1-per(compache.totals.cost)/config.measuredCost:null});
  const selected=sides.find(s=>s.id===config.compare), both=compache.complete && selected.complete;
  const savings=both && (fit.chosen || config.policy)?{absolute:selected.per1000-per(compache.totals.cost),percent:selected.per1000?100*(1-per(compache.totals.cost)/selected.per1000):null,range:selected.id==='measured' || !baseline.arm.complete || !selected.per1000?null:savingRange(compache.threadCosts,baseline.arm.threadCosts,config.seed)}:null;
  const warnings=['A model of token counts and list prices: no product or provider is run, and answer quality is not measured.','The policy is the cheapest of the priced candidates on the fitting threads. Costs are reported on a second set of threads drawn with the next seed.','A needed part that Compache archived costs one extra call. A comparison whose rules drop text for good does not pay for it and does not have it.','Comparisons follow documented defaults as listed in the methodology. None of the products is run.','Decisions look three turns ahead in expectation, as the package does, with the configured returns and the mean size of a turn. Query matching and cited sources are not modeled.'];
  return {config,workload:{threads:workload.threads,totalRuns:workload.totalRuns,plannedLengths:workload.plannedLengths,realizedLengths:workload.realizedLengths,gaps:workload.gaps},
    compache,baseline,sides,selected,savings,policy,
    fit:{candidates:fit.candidates,chosen:fit.chosen,threads:fit.threads,turns:fit.turns,fittedHere:fittedHere && fittedHere.complete?per(fittedHere.totals.cost):null,pinned:config.policy!==null},warnings};
}
