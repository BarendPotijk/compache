import test from 'node:test';
import assert from 'node:assert/strict';
import {DEFAULTS, ageLaw, configure, generateWorkload, runArm, sharedPrefix, priceCall, simulate, theoreticalSurvival, returnCdf, continueChance, warmReturnShare} from './engine.mjs';
import {applyPreset} from './models.mjs';
import {comparisons} from './baselines.mjs';
const OURS={kind:'compache',policy:{}};
const clawRules=(o={},anthropic=false)=>({kind:'rules',rules:{...comparisons(configure({anthropic})).find(b=>b.id==='openclaw').rules,recordTokens:0,summarySystemTokens:100,...o}});
const OPENCLAW=clawRules();

// Mechanics are tested without the archive notice, so that token counts are the parts' own.
const config = o => configure({threads:1,maxTurns:8,plannedTurns:8,plannedTurnsP95:8,lengthDistribution:'fixed',continuationProbability:1,keepWarm:false,archivedTokens:0, ...o});
// The same chance and age law for every kind of earlier item. One number for the age means exactly that many turns back.
const needs = (chance,back=1,p95=back) => Object.fromEntries(['input','retrieval','output'].flatMap(kind=>[[`${kind}DependencyProbability`,chance],[`${kind}DependencyTurnsBack`,back],[`${kind}DependencyTurnsBackP95`,p95]]));
const workload = turns => ({threads:[{id:'thread-1',turns}],totalRuns:turns.length});
const t = (index,time,retrievalTokens=300,dependencies=[]) => ({index,time,inputTokens:10,retrievalTokens,outputTokens:20,dependencies});
test('seeded workload is identical and has only earlier dependencies',()=>{
  const c=config({threads:3,...needs(1,3)});
  const a=generateWorkload(c); assert.deepEqual(a,generateWorkload(c));
  for(const thread of a.threads) for(const turn of thread.turns) {assert.equal(turn.dependencies.length,turn.index?3:0); for(const dep of turn.dependencies) assert.equal(turn.index-dep.turn,Math.min(3,turn.index));}
  assert.equal(a.realizedLengths.count,3); assert.equal(a.survival[0].realized,1);
});
test('prefix accounting respects identity, granularity, TTL and outputs are not in the original snapshot',()=>{
  const a=[{id:'s',tokens:2048},{id:'x',tokens:500}];
  assert.equal(sharedPrefix(a,[{id:'s',tokens:2048},{id:'y',tokens:500}]),2048);
  let c=config({ttlMinutes:5,systemTokens:2048,cacheMinTokens:1024,cacheBlockTokens:128});
  const w=workload([t(0,0),t(1,4),t(2,10)]), r=runArm(w,c,OURS);
  const answers=r.ledger.filter(x=>x.role==='answer');
  assert.equal(answers[0].cachedTokens,0);
  assert.equal(answers[1].cachedTokens,Math.floor((2048+10+300)/128)*128);
  assert.equal(answers[2].cachedTokens,0);
});
test('prices sum and long tiers apply to fresh, cache, and output',()=>{
  const m=configure().model, row=priceCall(300000,100000,1000,m);
  assert.equal(row.inputCost,200000*.1*1.25*2/1e6);
  assert.equal(row.cachedCost,100000*.01*2/1e6);
  assert.equal(row.outputCost,1000*.5*1.5/1e6);
  assert.equal(row.cost,row.inputCost+row.cachedCost+row.outputCost);
  assert.throws(()=>priceCall(1,2,0,m));
});
test('zero costs and zero dependencies have null denominators',()=>{
  const r=simulate({threads:1,maxTurns:2,plannedTurns:2,plannedTurnsP95:2,lengthDistribution:'fixed',...needs(0),keepWarm:false,model:{inputRate:0,cachedRate:0,outputRate:0}});
  assert.equal(r.savings.absolute,0); assert.equal(r.savings.percent,null); assert.equal(r.compache.dependencyCoverage,null);
});
test('zero long threshold disables long pricing tier',()=>{
  const model={...configure().model,longThreshold:0};
  const row=priceCall(300000,100000,1000,model);
  assert.equal(row.long,false); assert.equal(row.inputCost,200000*.1*1.25/1e6);
});
test('workload token median refers to retrieval and dependency probability is per run',()=>{
  const c=config({threads:1,maxTurns:4,plannedTurns:4,plannedTurnsP95:4,tokenMedian:50000,tokenP95:50000,inputTokens:200,...needs(1)});
  const turns=generateWorkload(c).threads[0].turns;
  for(const turn of turns) {assert.equal(turn.retrievalTokens,50000);assert.equal(turn.inputTokens,200);if(turn.index)assert.ok(turn.dependencies.length>=1 && turn.dependencies.length<=3);}
});
test('admission uses arm-specific headroom without double-counting output',()=>{
  const c=config({systemTokens:10,safetyReserve:32000,cacheMinTokens:1024,model:{context:100000,maxInput:100000}});
  const w=workload([t(0,0,75000)]);
  assert.equal(runArm(w,c,OURS).complete,false);
  assert.equal(runArm(w,c,OPENCLAW).complete,true);
  const close=config({systemTokens:10,safetyReserve:32000,model:{context:100000,maxInput:100000}});
  assert.equal(runArm(workload([t(0,0,67950)]),close,OURS).complete,true);
});
test('each kind has its own chance and its own distance back',()=>{
  const c=config({threads:1,maxTurns:12,plannedTurns:12,plannedTurnsP95:12,inputDependencyProbability:0,retrievalDependencyProbability:1,outputDependencyProbability:0,retrievalDependencyTurnsBack:3,retrievalDependencyTurnsBackP95:3});
  const turns=generateWorkload(c).threads[0].turns;
  for(const turn of turns.slice(1)) {assert.equal(turn.dependencies.length,1);assert.equal(turn.dependencies[0].kind,'retrieval');assert.equal(turn.index-turn.dependencies[0].turn,Math.min(3,turn.index));}
  const all=generateWorkload({...c,inputDependencyProbability:1,outputDependencyProbability:1,inputDependencyTurnsBack:1,inputDependencyTurnsBackP95:1,outputDependencyTurnsBack:5,outputDependencyTurnsBackP95:5}).threads[0].turns;
  for(const turn of all.slice(1)) {assert.deepEqual(turn.dependencies.map(d=>d.kind),['input','retrieval','output']);assert.deepEqual(turn.dependencies.map(d=>turn.index-d.turn),[1,Math.min(3,turn.index),Math.min(5,turn.index)]);}
});
test('how far back a need lies follows the configured median and 95th percentile',()=>{
  // A table long enough that no draw is cut off: on a 160-turn thread the oldest ages are folded back and the percentiles shift.
  const law=ageLaw(3,30,5000), ages=[]; let state=7;
  const random=()=>{state=(state*1664525+1013904223)>>>0;return state/4294967296;};
  for(let n=0;n<200000;n++)ages.push(law.sample(random(),5000));
  ages.sort((a,b)=>a-b);
  assert.equal(ages[0],1);assert.equal(ages[ages.length/2],3);assert.equal(ages[Math.ceil(.95*ages.length)-1],30);
  assert.equal(law.quantile(.5),3);assert.equal(law.quantile(.95),30);assert.equal(law.quantile(.25),1);
  // Its default for retrieved text: a quarter within one turn, three quarters within eight, nine in ten within eighteen.
  assert.deepEqual([.25,.5,.75,.9,.95,.99].map(q=>law.quantile(q)),[1,3,8,18,30,78]);
  for(const [median,p95] of [[1,20],[1,35],[12,24],[40,160]]) {const l=ageLaw(median,p95);assert.equal(l.quantile(.5),median);assert.equal(l.quantile(.95),p95);assert.ok(l.cdf(160)<=1&&l.cdf(0)===0);}
  // The default laws put most needs close by: 59% of needed questions and answers lie one turn back, 31% of needed retrieved text.
  assert.ok(Math.abs(ageLaw(1,20).cdf(1)-.588)<.002);assert.ok(Math.abs(ageLaw(3,30).cdf(1)-.310)<.002);
});
test('ages are cut to the turns that exist and their shares still sum to one',()=>{
  const law=ageLaw(3,30);
  for(const oldest of [1,2,5,40]) {
    let total=0;for(let age=1;age<=oldest;age++){assert.ok(law.pmf(age,oldest)>=0);total+=law.pmf(age,oldest);}
    assert.ok(Math.abs(total-1)<1e-12);
    for(const u of [0,.2,.5,.8,.999999])assert.ok(law.sample(u,oldest)>=1&&law.sample(u,oldest)<=oldest);
  }
  // With five earlier turns the weight of each age is its share of the first five.
  assert.ok(Math.abs(law.pmf(1,5)-law.cdf(1)/law.cdf(5))<1e-12);assert.equal(law.sample(0,5),1);
  assert.equal(law.sample(law.cdf(1)/law.cdf(5)+1e-9,5),2);
  // A law that reaches further back than the thread is old falls on the oldest turn.
  const far=ageLaw(40,40);assert.equal(far.sample(.3,3),3);assert.equal(far.pmf(3,3),1);assert.equal(far.pmf(1,3),0);
});
test('a 95th percentile equal to the median means exactly that many turns back',()=>{
  const law=ageLaw(4,4);
  for(const u of [0,.25,.5,.75,.999])assert.equal(law.sample(u,20),4);
  assert.equal(law.pmf(4,20),1);assert.equal(law.pmf(3,20),0);assert.equal(law.cdf(3),0);assert.equal(law.cdf(4),1);
  assert.equal(ageLaw(2.5,2.5).sample(.5,20),3);
});
test('below cache minimum fresh tokens use base input price',()=>{
  const c=config({systemTokens:10,cacheMinTokens:1024,cacheBlockTokens:128,model:{context:100000,maxInput:100000,inputRate:1,writeMultiplier:2},safetyReserve:0});
  const r=runArm(workload([t(0,0,100)]),c,OURS);
  assert.equal(r.ledger[0].inputCost,120/1e6);
  assert.equal(priceCall(120,0,0,c.model,{cacheable:false}).inputCost,120/1e6);
});

test('retrieval bounds, retention range and nonfinite values are validated',()=>{
  const c=config({tokenMedian:500000,tokenP95:500000,retentionRequirement:.75});assert.equal(generateWorkload(c).threads[0].turns[0].retrievalTokens,500000);
  assert.throws(()=>configure({tokenMedian:500001}));assert.throws(()=>configure({retentionRequirement:-.01}));assert.throws(()=>configure({retentionRequirement:1.01}));assert.equal(configure({retentionRequirement:.4}).retentionRequirement,.4);assert.throws(()=>configure({tokenMedian:NaN}));assert.throws(()=>configure({tokenP95:Infinity}));
});
test('answer cache breakdown sums exactly and distinguishes expired matches and new blocks',()=>{
  const c=config({systemTokens:100,cacheMinTokens:1,cacheBlockTokens:1,safetyReserve:0,model:{context:100000,maxInput:100000},ttlMinutes:5});
  const r=runArm(workload([t(0,0,100),t(1,4,100),t(2,10,100)]),c,OURS);
  const answers=r.ledger.filter(x=>x.role==='answer');
  assert.equal(answers[0].cacheBreakdown.firstRequest,answers[0].inputTokens);
  assert.ok(answers[1].cacheBreakdown.cached>0);assert.ok(answers[1].cacheBreakdown.newContext>0);
  assert.ok(answers[2].cacheBreakdown.expiredPrefix>0);
  for(const row of answers)assert.equal(Object.values(row.cacheBreakdown).reduce((a,b)=>a+b,0),row.inputTokens);
  assert.equal(Object.values(r.cacheBreakdown).reduce((a,b)=>a+b,0),r.byRole.answer.inputTokens);
});
test('Compache archives an expired prefix before a no-ping cold return',()=>{
  const c=config({systemTokens:100,cacheMinTokens:1,cacheBlockTokens:1,safetyReserve:0,model:{context:100000,maxInput:100000},ttlMinutes:5,keepWarm:false});
  const r=runArm(workload([t(0,0,100),t(1,6,100)]),c,OURS);
  const second=r.ledger.filter(row=>row.role==='answer')[1];
  assert.deepEqual(second.cacheBreakdown,{cached:0,newContext:130,firstRequest:0,expiredPrefix:210,changedPrefix:0,missed:0});
  assert.equal(second.inputTokens,340);assert.equal(second.inputCost,340*c.model.inputRate*c.model.writeMultiplier/1e6);
});
test('failed admission with decisions but no ledger remains safe',()=>{
  const c=config({systemTokens:100,safetyReserve:0,model:{context:1000,maxInput:100}});
  const r=runArm(workload([t(0,0,100)]),c,OURS,{collectLedger:false,collectDecisions:true});
  assert.equal(r.complete,false);assert.equal(r.ledger.length,0);assert.equal(r.decisions.length,1);assert.equal(r.decisions[0].admitted,false);
});
test('an unknown setting is an error, not a silent default',()=>{
  assert.throws(()=>configure({companies:1}),/companies is not a setting/);
  assert.throws(()=>configure({retrievalDependencyLookback:24}),/not a setting/);
  assert.throws(()=>configure({model:{price:1}}),/model\.price is not a setting/);
  assert.throws(()=>configure({model:7}),/model must be an object/);
  assert.deepEqual(configure(configure()),configure());
  assert.deepEqual(Object.keys(configure()).sort(),Object.keys(DEFAULTS).sort());
});
test('the new settings are checked like every other one',()=>{
  // By chance of another turn the median and 95th percentile are not used, so their order is not checked.
  const c=configure({lengthDistribution:'continuation',plannedTurns:9,plannedTurnsP95:2});
  assert.deepEqual([c.continueFirst,c.continueLong,c.continueHalfTurn,c.gapFirstFactor,c.gapHalfTurn,c.breakShare,c.breakMedianMinutes],[.6,.95,3,1,3,0,1440]);
  assert.deepEqual([c.answerSeconds,c.callsPerTurn,c.sharedPrefixWarm,c.retrievalShare,c.reasoningTokens,c.cacheHitShare,c.archivedTokens],[0,1,0,1,0,1,28]);
  for(const bad of [{continueFirst:1.1},{continueLong:-.1},{continueHalfTurn:0},{gapFirstFactor:0},{gapHalfTurn:200},{breakShare:1.5},{breakMedianMinutes:.5},{answerSeconds:-1},{callsPerTurn:.9},{callsPerTurn:101},{sharedPrefixWarm:2},{retrievalShare:-1},{reasoningTokens:1.5},{cacheHitShare:1.2},{archivedTokens:2.5}]) assert.throws(()=>configure(bad),RangeError,JSON.stringify(bad));
  assert.equal(configure({callsPerTurn:2.4}).callsPerTurn,2.4);
});
