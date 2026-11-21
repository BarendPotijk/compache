import test from 'node:test';
import assert from 'node:assert/strict';
import {DEFAULTS, ageLaw, configure, generateWorkload, runArm, sharedPrefix, priceCall, simulate, coldPreview, planPings, evaluatePingPlan, theoreticalSurvival, returnCdf, continueChance, warmReturnShare} from './engine.mjs';
import {applyPreset} from './models.mjs';

// Mechanics are tested without the archive notice, so that token counts are the parts' own.
const config = o => configure({threads:1,maxTurns:8,plannedTurns:8,plannedTurnsP95:8,lengthDistribution:'fixed',continuationProbability:1,keepWarm:false,recordTokens:0,summarySystemTokens:100,archivedTokens:0, ...o});
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
  const w=workload([t(0,0),t(1,4),t(2,10)]), r=runArm(w,c,'compache');
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
test('compaction preserves current dependencies and recalls previously dropped retrievals',()=>{
  const c=config({systemTokens:100,cacheMinTokens:0,cacheBlockTokens:1,safetyReserve:0,target:.2, model:{context:1000,maxInput:1000},ttlMinutes:1});
  const w=workload([t(0,0),t(1,2),t(2,4,100,[{kind:'retrieval',turn:0},{kind:'input',turn:0},{kind:'output',turn:0}])]);
  const r=runArm(w,c,'compache'); assert.ok(r.compactions>0); assert.equal(r.recalls,1); assert.equal(r.byRole.recovery.calls,1); assert.equal(r.byRole.answer.calls,3); assert.equal(r.retention.observed,2/3); assert.equal(r.dependencies.required,3);assert.equal(r.dependencies.covered,3); assert.equal(r.dependencyCoverage,1); assert.equal(r.complete,true);
});
test('summary and flush are separate charged calls with separate prefix identities',()=>{
  const c=config({systemTokens:100,cacheMinTokens:1,cacheBlockTokens:1,safetyReserve:0,tailTokens:100,summaryTokens:80,flushOutputTokens:5,model:{context:1000,maxInput:1000},ttlMinutes:10});
  const r=runArm(workload([t(0,0,200),t(1,1,200),t(2,2,200,[{kind:'retrieval',turn:1}])]),c,'openclaw');
  assert.ok(r.byRole.summary.calls>0);assert.ok(r.byRole.flush.calls>0); assert.equal(r.dependencyCoverage,1);
  assert.equal(r.byRole.summary.cachedTokens,0); assert.equal(r.byRole.flush.cachedTokens,0);
  assert.ok(Math.abs(r.totals.cost-r.ledger.reduce((s,x)=>s+(x.cost||0),0))<1e-12);
});
test('incomplete calls do not produce a full workload savings claim',()=>{
  const r=simulate({threads:1,maxTurns:1,plannedTurns:1,plannedTurnsP95:1,lengthDistribution:'fixed',tokenMedian:2000,tokenP95:2000,systemTokens:100,safetyReserve:0,keepWarm:false,model:{context:500,maxInput:500}});
  assert.equal(r.savings,null);assert.equal(r.compache.complete,false);assert.equal(r.openclaw.complete,false);assert.equal(r.compache.ledger[0].admitted,false);
});
test('zero costs and zero dependencies have null denominators',()=>{
  const r=simulate({threads:1,maxTurns:2,plannedTurns:2,plannedTurnsP95:2,lengthDistribution:'fixed',...needs(0),keepWarm:false,model:{inputRate:0,cachedRate:0,outputRate:0}});
  assert.equal(r.savings.absolute,0); assert.equal(r.savings.percent,null); assert.equal(r.compache.dependencyCoverage,null);
});
test('negative savings are retained and not clamped after optimization',()=>{
  const r=simulate({threads:1,plannedTurns:60,plannedTurnsP95:60,maxTurns:60,lengthDistribution:'fixed',continuationProbability:1,tokenMedian:25000,tokenP95:25000,...needs(.25,24),gapMedianMinutes:100,gapP95Minutes:100,keepWarm:false,retentionRequirement:1});
  assert.ok(r.optimization.feasible && r.compache.complete && r.openclaw.complete); assert.ok(r.savings.absolute<0);
});
test('configuration bounds reject impossible values',()=>{
  assert.throws(()=>configure({threads:201}));assert.throws(()=>configure({gapMedianMinutes:2,gapP95Minutes:1}));assert.throws(()=>configure({lengthDistribution:'other'}));
  assert.throws(()=>configure({retrievalDependencyTurnsBack:5,retrievalDependencyTurnsBackP95:4}),/at least/);assert.throws(()=>configure({inputDependencyTurnsBack:.5}));
  assert.throws(()=>configure({lengthDistribution:'fixed',plannedTurns:7.5,plannedTurnsP95:8}),/whole number/);assert.equal(configure({plannedTurns:7.5}).plannedTurns,7.5);
  assert.throws(()=>configure({clearKeepResults:0}));assert.throws(()=>configure({safetyReserve:600000}),/safetyReserve must be at most 525000/);
});
test('expected-value planner charges pings after abandonment and stops when the next turn arrives',()=>{
  const c=config({systemTokens:2048,keepWarm:true,continuationProbability:.8,gapMedianMinutes:10,gapP95Minutes:10,ttlMinutes:5, pingLeadMinutes:.75});
  const abandoned=runArm(workload([t(0,0,100000)]),c,'compache');
  assert.equal(abandoned.pings,2);assert.deepEqual(abandoned.ledger.filter(r=>r.role==='ping').map(r=>r.time),[4.25,8.5]);
  const resumed=runArm(workload([t(0,0,100000),t(1,6,100)]),c,'compache');
  assert.equal(resumed.ledger.filter(r=>r.role==='ping' && r.turn===0).length,1);
  assert.ok(resumed.ledger.find(r=>r.role==='answer' && r.turn===1).cachedTokens>0);
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
test('cold preview and planner account for eligible retrieval cuts without future knowledge',()=>{
  const c=config({keepWarm:true,systemTokens:2000,continuationProbability:.1,gapMedianMinutes:6,gapP95Minutes:6,ttlMinutes:5,pingLeadMinutes:.75,model:{context:1000000,maxInput:1000000}});
  const system={id:'system',tokens:2000,kind:'system',turn:-1};
  const retrieval={id:'r0',source:'r0',tokens:300000,kind:'retrieval',turn:0};
  const input={...retrieval,id:'i0',source:'i0',kind:'input'};
  const output={id:'o0',source:'o0',tokens:500,kind:'output',turn:0};
  assert.deepEqual(coldPreview([retrieval,output],c).map(b=>b.id),['system','o0']);
  assert.equal(planPings([system,retrieval],[system,retrieval,output],c),0);
  assert.ok(planPings([system,input],[system,input,output],c)>0);
});
test('admission uses arm-specific headroom without double-counting output',()=>{
  const c=config({systemTokens:10,safetyReserve:32000,cacheMinTokens:1024,model:{context:100000,maxInput:100000}});
  const w=workload([t(0,0,75000)]);
  assert.equal(runArm(w,c,'compache').complete,false);
  assert.equal(runArm(w,c,'openclaw').complete,true);
  const close=config({systemTokens:10,safetyReserve:32000,model:{context:100000,maxInput:100000}});
  assert.equal(runArm(workload([t(0,0,67950)]),close,'compache').complete,true);
});
test('summary sees bounded older head and deterministic record is charged only on replay',()=>{
  const base={systemTokens:100,safetyReserve:0,tailTokens:100,summaryTokens:80,summarySystemTokens:100,summaryInputCap:60,toolSummaryTokens:50,flushOutputTokens:5,model:{context:1000,maxInput:1000},ttlMinutes:10};
  const w=workload([t(0,0,200),t(1,1,200),t(2,2,200)]);
  const without=runArm(w,config({...base,recordTokens:0}),'openclaw');
  const withRecord=runArm(w,config({...base,recordTokens:25}),'openclaw');
  const summary=withRecord.ledger.find(x=>x.role==='summary');
  assert.equal(summary.inputTokens,160); // 100 system + cap of 60 from old head
  assert.equal(summary.outputTokens,80); // record generation is not billed
  assert.equal(withRecord.ledger.find(x=>x.role==='answer' && x.turn===2).inputTokens-without.ledger.find(x=>x.role==='answer' && x.turn===2).inputTokens,25);
});
test('decision snapshots reproduce the gate, protect current and dependent sources, and are immutable',()=>{
  const c=config({systemTokens:100,cacheMinTokens:0,cacheBlockTokens:1,safetyReserve:0,target:.2,model:{context:1000,maxInput:1000},ttlMinutes:1});
  const r=runArm(workload([t(0,0),t(1,2,300,[{kind:'retrieval',turn:0}]),t(2,4)]),c,'compache');
  assert.equal(r.decisions.length,r.attemptedRuns);
  for(const d of r.decisions) {
    assert.equal(d.beforeTokens-d.afterTokens,d.removedTokens);
    assert.equal(d.beforeTokens,d.beforeBlocks.reduce((a,b)=>a+b.tokens,0)+d.beforeOmittedTokens);
    assert.equal(d.afterTokens,d.afterBlocks.reduce((a,b)=>a+b.tokens,0)+d.afterOmittedTokens);
    assert.ok(d.afterBlocks.some(b=>b.id===`r${d.turn}`));
    assert.ok(d.removedSources.every(b=>!d.protectedSources.includes(b.source)));
    assert.equal(d.removedTokens,d.removedSources.reduce((a,b)=>a+b.tokens,0));
    const row=r.ledger.find(x=>x.role==='answer'&&x.turn===d.turn);
    assert.equal(row.inputTokens,d.afterTokens); assert.equal(row.removedTokens,d.removedTokens);
    assert.equal(row.plannedPings,d.plannedPings); assert.equal(row.actualPings,d.actualPings);
    assert.ok(Object.isFrozen(d) && Object.isFrozen(d.beforeBlocks) && Object.isFrozen(d.beforeBlocks[0]));
    assert.throws(()=>{d.beforeBlocks[0].tokens=999;},TypeError);
  }
  assert.equal(r.removedTokens,r.decisions.reduce((a,d)=>a+d.removedTokens,0));
  assert.equal(r.recallTokens,r.decisions.reduce((a,d)=>a+d.recallTokens,0));
});
test('planned survival and continuation determine age-dependent return probability',()=>{
  const fixed=config({plannedTurns:2,plannedTurnsP95:2,maxTurns:4,continuationProbability:.8});
  const s=theoreticalSurvival(fixed);
  assert.deepEqual(s.map(x=>x.planned),[1,1,0,0]);
  assert.deepEqual(s.map(x=>x.realized),[1,.8,0,0]);
  const held=[{id:'system',kind:'system',turn:-1,tokens:2048},{id:'i0',kind:'input',turn:0,tokens:100000}];
  const full=[...held,{id:'o0',kind:'output',turn:0,tokens:500}];
  assert.equal(evaluatePingPlan(held,full,fixed,0).returnProbability,.8);
  const terminal=evaluatePingPlan(held,full,{...fixed,keepWarm:true},1);
  assert.equal(terminal.chosen,0);assert.equal(terminal.returnProbability,0);
  const geo=config({lengthDistribution:'geometric',plannedTurns:4,plannedTurnsP95:4,maxTurns:4,continuationProbability:.8,keepWarm:true});
  assert.ok(Math.abs(evaluatePingPlan(held,full,geo,2).returnProbability-.6)<1e-12);
  assert.ok(returnCdf(geo.gapMedianMinutes,geo)>0 && returnCdf(geo.gapMedianMinutes,geo)<1);
});
test('planner returns expected costs and gains for each permitted plan',()=>{
  const c=config({keepWarm:true,continuationProbability:.8,gapMedianMinutes:10,gapP95Minutes:10});
  const held=[{id:'system',kind:'system',turn:-1,tokens:2048},{id:'i0',kind:'input',turn:0,tokens:100000}];
  const full=[...held,{id:'o0',kind:'output',turn:0,tokens:500}];
  const p=evaluatePingPlan(held,full,c,0);
  assert.equal(p.rows.length,c.maxPings+1);assert.equal(p.rows[0].pings,0);
  for(const r of p.rows)assert.equal(r.net,r.expectedGain-r.expectedCost);
  assert.equal(p.chosen,planPings(held,full,c,0));
});

test('decision previews are bounded while kind totals cover the full view',()=>{
  const c=config({systemTokens:100,cacheMinTokens:0,safetyReserve:0,model:{context:1000000,maxInput:1000000},ttlMinutes:100});
  const r=runArm(workload(Array.from({length:12},(_,i)=>t(i,i,30))),c,'compache');
  const d=r.decisions.at(-1);
  assert.equal(d.beforeBlocks.length,16);assert.ok(d.beforeOmittedCount>0);
  assert.equal(Object.values(d.beforeByKind).reduce((a,b)=>a+b,0),d.beforeTokens);
  assert.equal(Object.values(d.afterByKind).reduce((a,b)=>a+b,0),d.afterTokens);
  assert.equal(d.beforeTokens,d.beforeBlocks.reduce((a,b)=>a+b.tokens,0)+d.beforeOmittedTokens);
  assert.ok(Object.isFrozen(d.beforeByKind));
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
test('retention is measured before recall and separates observed and expected from post-recall',()=>{
  const c=config({systemTokens:100,cacheMinTokens:0,cacheBlockTokens:1,safetyReserve:0,target:.2,model:{context:1000,maxInput:1000},ttlMinutes:1,inputDependencyProbability:0,retrievalDependencyProbability:1,outputDependencyProbability:0,retrievalDependencyTurnsBack:1,retrievalDependencyTurnsBackP95:2,retrievalRetentionRuns:0});
  const r=runArm(workload([t(0,0),t(1,2),t(2,4,100,[{kind:'retrieval',turn:0}])]),c,'compache');
  assert.ok(r.retention.expected<1);assert.equal(r.retention.observed,0);assert.equal(r.rawSourceAvailability,1);assert.equal(r.recalls,1);
  assert.equal(r.retention.byKind.retrieval.observedRetained,0);
});
test('expected retention weights each retrieved text still in view by how often its age is needed',()=>{
  // Turn two keeps its only earlier text. By turn three the first text is archived and the second, one turn back, remains.
  const base={systemTokens:100,cacheMinTokens:0,cacheBlockTokens:1,safetyReserve:0,target:.2,model:{context:1000,maxInput:1000},ttlMinutes:1,inputDependencyProbability:.5,outputDependencyProbability:.25,retrievalDependencyProbability:.4,retrievalDependencyTurnsBack:1,retrievalDependencyTurnsBackP95:2,retrievalRetentionRuns:0};
  const law=ageLaw(1,2), turns=[t(0,0),t(1,2),t(2,4,100)];
  const r=runArm(workload(turns),config(base),'compache'), kinds=r.retention.byKind;
  assert.ok(Math.abs(kinds.retrieval.expectedRequired-.8)<1e-12);
  assert.ok(Math.abs(kinds.retrieval.expectedRetained-.4*(1+law.pmf(1,2)))<1e-12);
  assert.ok(law.pmf(1,2)>.8&&law.pmf(1,2)<.9);
  // Questions and answers are never archived, so all of their expected needs are met.
  assert.equal(kinds.input.expectedRetained,kinds.input.expectedRequired);assert.equal(kinds.output.expectedRetained,kinds.output.expectedRequired);
  assert.ok(Math.abs(r.retention.expected-(1+.5+.4*(1+law.pmf(1,2)))/(1+.5+.8))<1e-12);
  // Keeping one more turn of retrieved text leaves nothing out of view on a three-turn thread whose needs lie at most two back.
  const kept=runArm(workload(turns),config({...base,retrievalRetentionRuns:2}),'compache');
  assert.ok(Math.abs(kept.retention.byKind.retrieval.expectedRetained-.8)<1e-12);assert.ok(Math.abs(kept.retention.expected-1)<1e-12);
});
test('grid optimizer is deterministic and tightening retention cannot lower its minimum cost',()=>{
  const common={threads:3,plannedTurns:32,plannedTurnsP95:32,maxTurns:32,lengthDistribution:'fixed',continuationProbability:1,tokenMedian:10000,tokenP95:10000,keepWarm:false};
  const lo=simulate({...common,retentionRequirement:.9}),hi=simulate({...common,retentionRequirement:.95});
  assert.ok(lo.optimization.feasible && hi.optimization.feasible);
  // The floor binds on the threads the policy is fitted on, and so does the ordering of the two minimum costs.
  const cheapest=r=>Math.min(...r.optimization.candidates.filter(x=>x.feasible).map(x=>x.cost));
  assert.ok(lo.optimization.fittedRetention>=.9-1e-12 && hi.optimization.fittedRetention>=.95-1e-12);
  assert.ok(cheapest(hi)>=cheapest(lo)-1e-12);
  assert.deepEqual(hi.optimization.candidates.map(x=>x.cost),lo.optimization.candidates.map(x=>x.cost));
  assert.ok(lo.optimization.candidates.every(x=>x.feasible===!!(x.complete&&(x.retention===null||x.retention>=.9-1e-12))));
  assert.deepEqual(hi.optimization.selected,simulate({...common,retentionRequirement:.95}).optimization.selected);
});
test('infeasible retention or admission never claims recommended policy or savings',()=>{
  const x=simulate({threads:1,maxTurns:3,plannedTurns:3,plannedTurnsP95:3,lengthDistribution:'fixed',tokenMedian:2000,tokenP95:2000,systemTokens:100,safetyReserve:0,keepWarm:false,model:{context:500,maxInput:500}});
  assert.equal(x.optimization.feasible,false);assert.equal(x.optimization.selected,null);assert.equal(x.savings,null);
});
test('a later dependency cannot affect earlier policy decisions',()=>{
  const c=config({systemTokens:100,cacheMinTokens:0,cacheBlockTokens:1,safetyReserve:0,target:.2,model:{context:1000,maxInput:1000},ttlMinutes:1});
  const first=[t(0,0),t(1,2),t(2,4,100)], second=[t(0,0),t(1,2),t(2,4,100,[{kind:'retrieval',turn:0}])];
  const a=runArm(workload(first),c,'compache'),b=runArm(workload(second),c,'compache');
  assert.deepEqual(a.decisions.slice(0,2),b.decisions.slice(0,2));
});
test('below cache minimum fresh tokens use base input price',()=>{
  const c=config({systemTokens:10,cacheMinTokens:1024,cacheBlockTokens:128,model:{context:100000,maxInput:100000,inputRate:1,writeMultiplier:2},safetyReserve:0});
  const r=runArm(workload([t(0,0,100)]),c,'compache');
  assert.equal(r.ledger[0].inputCost,120/1e6);
  assert.equal(priceCall(120,0,0,c.model,{cacheable:false}).inputCost,120/1e6);
});

test('zero retrieval mode produces no retrieval dependencies or requirements',()=>{
  const c=config({threads:2,tokenMedian:0,tokenP95:0,retrievalDependencyProbability:1,inputDependencyProbability:0,outputDependencyProbability:0});
  const w=generateWorkload(c);
  for(const thread of w.threads)for(const turn of thread.turns){assert.equal(turn.retrievalTokens,0);assert.ok(turn.dependencies.every(d=>d.kind!=='retrieval'));}
  const r=runArm(w,c,'compache');assert.equal(r.retention.byKind.retrieval.expectedRequired,0);assert.equal(r.retention.expected,null);
  const sim=simulate({...c,retentionRequirement:.75});assert.equal(sim.optimization.feasible,true);assert.equal(sim.optimization.retention,null);
});
test('retrieval bounds, retention range and nonfinite values are validated',()=>{
  const c=config({tokenMedian:500000,tokenP95:500000,retentionRequirement:.75});assert.equal(generateWorkload(c).threads[0].turns[0].retrievalTokens,500000);
  assert.throws(()=>configure({tokenMedian:500001}));assert.throws(()=>configure({retentionRequirement:-.01}));assert.throws(()=>configure({retentionRequirement:1.01}));assert.equal(configure({retentionRequirement:.4}).retentionRequirement,.4);assert.throws(()=>configure({tokenMedian:NaN}));assert.throws(()=>configure({tokenP95:Infinity}));
});
test('answer cache breakdown sums exactly and distinguishes expired matches and new blocks',()=>{
  const c=config({systemTokens:100,cacheMinTokens:1,cacheBlockTokens:1,safetyReserve:0,model:{context:100000,maxInput:100000},ttlMinutes:5});
  const r=runArm(workload([t(0,0,100),t(1,4,100),t(2,10,100)]),c,'compache');
  const answers=r.ledger.filter(x=>x.role==='answer');
  assert.equal(answers[0].cacheBreakdown.firstRequest,answers[0].inputTokens);
  assert.ok(answers[1].cacheBreakdown.cached>0);assert.ok(answers[1].cacheBreakdown.newContext>0);
  assert.ok(answers[2].cacheBreakdown.expiredPrefix>0);
  for(const row of answers)assert.equal(Object.values(row.cacheBreakdown).reduce((a,b)=>a+b,0),row.inputTokens);
  assert.equal(Object.values(r.cacheBreakdown).reduce((a,b)=>a+b,0),r.byRole.answer.inputTokens);
});
test('a compaction rewrite leaves previously sent blocks in changed-prefix fresh tokens',()=>{
  const c=config({systemTokens:100,cacheMinTokens:1,cacheBlockTokens:1,safetyReserve:0,target:.2,warmTrigger:.8,model:{context:1200,maxInput:1200},ttlMinutes:10});
  const r=runArm(workload([t(0,0),t(1,1),t(2,2)]),c,'compache');
  assert.ok(r.compactions>0);assert.ok(r.ledger.find(x=>x.role==='answer'&&x.turn===2).cacheBreakdown.changedPrefix>0);
});
test('ping payoff includes zero and exposes timing even when disabled',()=>{
  const c=config({keepWarm:false}),held=[{id:'s',tokens:1000,kind:'system',turn:-1}],full=held;
  const p=evaluatePingPlan(held,full,c,0);assert.deepEqual(p.rows,[{pings:0,expectedGain:0,expectedCost:0,net:0}]);assert.equal(p.ttl,c.ttlMinutes);assert.equal(p.interval,c.ttlMinutes-c.pingLeadMinutes);assert.equal(p.reason,'disabled');
});

test('Compache archives an expired prefix before a no-ping cold return',()=>{
  const c=config({systemTokens:100,cacheMinTokens:1,cacheBlockTokens:1,safetyReserve:0,model:{context:100000,maxInput:100000},ttlMinutes:5,keepWarm:false});
  const r=runArm(workload([t(0,0,100),t(1,6,100)]),c,'compache');
  const second=r.ledger.filter(row=>row.role==='answer')[1];
  assert.deepEqual(second.cacheBreakdown,{cached:0,newContext:130,firstRequest:0,expiredPrefix:210,changedPrefix:0,missed:0});
  assert.equal(second.inputTokens,340);assert.equal(second.inputCost,340*c.model.inputRate*c.model.writeMultiplier/1e6);
});
test('failed admission with decisions but no ledger remains safe',()=>{
  const c=config({systemTokens:100,safetyReserve:0,model:{context:1000,maxInput:100}});
  const r=runArm(workload([t(0,0,100)]),c,'compache',{collectLedger:false,collectDecisions:true});
  assert.equal(r.complete,false);assert.equal(r.ledger.length,0);assert.equal(r.decisions.length,1);assert.equal(r.decisions[0].admitted,false);
});
test('a fetched-back source costs one extra call that reads the prefix and a short reply',()=>{
  const c=config({systemTokens:100,cacheMinTokens:0,safetyReserve:0,target:.2,recoveryOutputTokens:7,model:{context:1000,maxInput:1000},ttlMinutes:100});
  const turns=[t(0,0),t(1,2),t(2,4,100,[{kind:'retrieval',turn:0}])], plain=[t(0,0),t(1,2),t(2,4,100)];
  const r=runArm(workload(turns),c,'compache'), call=r.ledger.find(x=>x.role==='recovery'), answer=r.ledger.filter(x=>x.role==='answer').at(-1);
  assert.equal(call.outputTokens,7); assert.equal(call.turn,2); assert.ok(call.cachedTokens>0);
  assert.equal(answer.inputTokens,call.inputTokens+7+300); assert.equal(answer.cachedTokens,call.inputTokens);
  assert.ok(r.totals.cost>runArm(workload(plain),c,'compache').totals.cost);
});
test('OpenClaw default clearing keeps the three newest tool results and fetches a cleared one again when needed',()=>{
  const c=config({systemTokens:100,cacheMinTokens:0,safetyReserve:0,toolClearing:true,clearedTokens:5,model:{context:200000,maxInput:200000},ttlMinutes:100});
  const turns=[t(0,0,20000),t(1,1,20000),t(2,2,20000),t(3,3,20000),t(4,4,20000,[{kind:'retrieval',turn:0},{kind:'retrieval',turn:3}])];
  const off=runArm(workload(turns),{...c,toolClearing:false},'openclaw'), on=runArm(workload(turns),c,'openclaw');
  assert.equal(off.clearings,0); assert.equal(off.recalls,0); assert.equal(off.retention.observed,1);
  // Above 30% of the window (60,000 tokens) every result but the newest three is cleared: the first at run four, the second at run five.
  assert.equal(on.clearings,2); assert.equal(on.recalls,1); assert.equal(on.byRole.recovery.calls,1);
  assert.equal(on.retention.observed,.5); assert.equal(on.rawSourceAvailability,1);
  const fourth=on.ledger.find(x=>x.role==='answer'&&x.turn===3); assert.equal(fourth.inputTokens,100+4*10+3*20+5+3*20000);
  assert.equal(runArm(workload(turns),c,'compache').clearings,0);
});
test('the OpenClaw constants are settings, and their defaults are its documented values',()=>{
  const c=config({systemTokens:100,cacheMinTokens:0,safetyReserve:0,toolClearing:true,clearedTokens:5,model:{context:200000,maxInput:200000},ttlMinutes:100});
  const w=workload([0,1,2,3,4].map(i=>t(i,i,20000))), run=o=>runArm(w,{...c,...o},'openclaw');
  const base=run({}), spelled=run({clearTriggerShare:.3,clearTriggerFloor:50000,clearKeepResults:3,clearMinFreedShare:.05,clearMinFreedFloor:12500,summaryTriggerShare:.75,baselineReserve:20000});
  assert.equal(base.clearings,2);assert.equal(base.compactions,0);assert.deepEqual(spelled.totals,base.totals);assert.deepEqual(spelled.ledger,base.ledger);
  // Two newest results kept: the third turn already has one to clear. Each later turn clears one more.
  assert.equal(run({clearKeepResults:2}).clearings,3);
  // A trigger at the whole window is never reached by 100,000 tokens of history.
  assert.equal(run({clearTriggerShare:1}).clearings,0);
  // A 90,000-token floor on the trigger, or 30,000 on what must be freed, waits for the fifth turn and clears two results at once.
  assert.equal(run({clearTriggerFloor:90000}).clearings,1);assert.equal(run({clearMinFreedFloor:30000}).clearings,1);
  assert.equal(run({clearMinFreedShare:.5}).clearings,0);
  // Without clearing, the summary waits for 75% of the window. At 40% it runs on the fourth turn.
  assert.equal(run({toolClearing:false}).compactions,0);
  const early=run({toolClearing:false,summaryTriggerShare:.4});assert.equal(early.compactions,1);assert.equal(early.ledger.find(x=>x.role==='summary').turn,3);
  // The reserve an answer must leave free is capped at a quarter of the window.
  const tight=config({systemTokens:10,cacheMinTokens:1024,model:{context:100000,maxInput:100000}}), one=workload([t(0,0,75000)]);
  assert.equal(runArm(one,tight,'openclaw').complete,true);assert.equal(runArm(one,{...tight,baselineReserve:30000},'openclaw').complete,false);
  assert.equal(runArm(one,{...tight,baselineReserve:24000},'openclaw').complete,true);
});
test('an unknown setting is an error, not a silent default',()=>{
  assert.throws(()=>configure({companies:1}),/companies is not a setting/);
  assert.throws(()=>configure({retrievalDependencyLookback:24}),/not a setting/);
  assert.throws(()=>configure({model:{price:1}}),/model\.price is not a setting/);
  assert.throws(()=>configure({model:7}),/model must be an object/);
  assert.deepEqual(configure(configure()),configure());
  assert.deepEqual(Object.keys(configure()).sort(),Object.keys(DEFAULTS).sort());
});
test('candidate horizons cover set shares of the retrieved text later turns need',()=>{
  const common={threads:2,plannedTurns:12,plannedTurnsP95:12,maxTurns:12,lengthDistribution:'fixed',continuationProbability:1,keepWarm:false};
  const horizons=r=>[...new Set(r.optimization.candidates.map(x=>x.retrievalRetentionRuns))];
  assert.deepEqual(horizons(simulate(common)),[0,1,3,8,12]);  // 18, 30 and 78 turns back do not exist on a 12-turn thread
  assert.deepEqual(horizons(simulate({...common,maxTurns:160})),[0,1,3,8,18,30,78]);
  assert.deepEqual(horizons(simulate({...common,retrievalDependencyTurnsBack:4,retrievalDependencyTurnsBackP95:4})),[0,4]);
  assert.deepEqual(horizons(simulate({...common,tokenMedian:0,tokenP95:0})),[0]);
  // Six targets and five warm triggers give 21 pairs with the target at least two points below the trigger. Each comes with every horizon.
  assert.equal(simulate({...common,maxTurns:160}).optimization.candidateCount,21*7);
});
test('every candidate is priced as its own run, although runs the warm trigger never touched are shared',()=>{
  // Slow returns on a five-minute cache: the cache is cold at nearly every return, so most warm triggers never decide anything.
  const base={threads:3,plannedTurns:14,plannedTurnsP95:14,maxTurns:14,lengthDistribution:'fixed',continuationProbability:1,tokenMedian:30000,tokenP95:60000,gapMedianMinutes:20,gapP95Minutes:90,ttlMinutes:5,seed:5};
  for(const scenario of [base,{...base,gapMedianMinutes:1,gapP95Minutes:4,tokenMedian:60000,tokenP95:120000}]) {
    const r=simulate(scenario), fitted=generateWorkload(configure(scenario));
    assert.equal(r.optimization.candidateCount,21*r.optimization.candidates.filter(x=>x.target===.01&&x.warmTrigger===.6).length);
    for(const x of r.optimization.candidates) {
      const own=runArm(fitted,{...configure(scenario),target:x.target,warmTrigger:x.warmTrigger,retrievalRetentionRuns:x.retrievalRetentionRuns},'compache',{collectDecisions:false,collectLedger:false});
      assert.equal(x.cost,own.totals.cost);assert.equal(x.retention,own.retention.expected);assert.equal(x.complete,own.complete);
    }
  }
  // The second scenario has warm requests far above the lowest triggers, so those candidates really differ.
  const fast=simulate({...base,gapMedianMinutes:1,gapP95Minutes:4,tokenMedian:60000,tokenP95:120000}).optimization.candidates.filter(x=>x.target===.01&&x.retrievalRetentionRuns===0);
  assert.ok(new Set(fast.map(x=>x.cost)).size>1);
});
test('among equally cheap candidates the fit takes the one that removes least',()=>{
  // Two short turns never reach any trigger: every candidate costs the same and keeps everything.
  const r=simulate({threads:2,plannedTurns:2,plannedTurnsP95:2,maxTurns:2,lengthDistribution:'fixed',continuationProbability:1,tokenMedian:1000,tokenP95:1000});
  assert.equal(new Set(r.optimization.candidates.map(x=>x.cost)).size,1);
  assert.deepEqual(r.optimization.selected,{target:.45,warmTrigger:.6,retrievalRetentionRuns:Math.max(...r.optimization.candidates.map(x=>x.retrievalRetentionRuns))});
});
test('the policy is fitted on one set of threads and both arms are costed on another',()=>{
  const base={threads:4,plannedTurns:20,plannedTurnsP95:20,maxTurns:20,lengthDistribution:'fixed',continuationProbability:1,tokenMedian:60000,tokenP95:90000,seed:7};
  const r=simulate(base), fitted=generateWorkload(configure(base)), held=generateWorkload(configure({...base,seed:8}));
  assert.deepEqual(r.workload,held); assert.notDeepEqual(r.workload.threads,fitted.threads);
  assert.equal(r.optimization.fittedThreads,4); assert.equal(r.optimization.fittedRuns,fitted.totalRuns);
  const chosen=r.optimization.candidates.find(x=>x.feasible&&x.target===r.config.target&&x.warmTrigger===r.config.warmTrigger&&x.retrievalRetentionRuns===r.config.retrievalRetentionRuns);
  assert.equal(chosen.cost,runArm(fitted,r.config,'compache',{collectDecisions:false,collectLedger:false}).totals.cost);
  assert.equal(r.optimization.fittedRetention,chosen.retention);
  assert.equal(r.compache.totals.cost,runArm(held,r.config,'compache').totals.cost);
});

// What depends on how far a thread has come, and what one turn costs beyond its answer.
const held=[{id:'system',kind:'system',turn:-1,tokens:2048},{id:'i0',kind:'input',turn:0,tokens:100000}], full=[...held,{id:'o0',kind:'output',turn:0,tokens:500}];
const answersOf=r=>r.ledger.filter(x=>x.role==='answer');
test('the published Sonnet result is pinned, and the settings that are off by default change nothing',()=>{
  const r=simulate(applyPreset({},'sonnet')), per1000=arm=>(1000*arm.totals.cost/r.workload.totalRuns).toFixed(4);
  assert.equal(per1000(r.compache),'86.3986');assert.equal(per1000(r.openclaw),'207.7703');
  // The archive notice is the one setting that is on: without it Compache costs what it did before the notice was priced.
  const bare=simulate(applyPreset({archivedTokens:0},'sonnet'));
  assert.equal((1000*bare.compache.totals.cost/bare.workload.totalRuns).toFixed(4),'86.2176');
  assert.equal(r.workload.totalRuns,877);assert.equal(r.compache.pings,26);
  assert.deepEqual(r.optimization.selected,{target:.01,warmTrigger:.05,retrievalRetentionRuns:0});
  assert.equal(r.compache.byRole.further,undefined);assert.equal(r.compache.cacheBreakdown.missed,0);
  // A setting whose effect is switched off draws nothing: the threads are the same, number for number.
  const idle={continueFirst:.2,continueLong:.3,continueHalfTurn:9,gapHalfTurn:9,breakMedianMinutes:5000};
  assert.deepEqual(generateWorkload(configure(idle)),generateWorkload(configure()));
});
test('the new settings are checked like every other one',()=>{
  // By chance of another turn the median and 95th percentile are not used, so their order is not checked.
  const c=configure({lengthDistribution:'continuation',plannedTurns:9,plannedTurnsP95:2});
  assert.deepEqual([c.continueFirst,c.continueLong,c.continueHalfTurn,c.gapFirstFactor,c.gapHalfTurn,c.breakShare,c.breakMedianMinutes],[.6,.95,3,1,3,0,1440]);
  assert.deepEqual([c.answerSeconds,c.callsPerTurn,c.sharedPrefixWarm,c.retrievalShare,c.reasoningTokens,c.cacheHitShare,c.archivedTokens],[0,1,0,1,0,1,22]);
  for(const bad of [{continueFirst:1.1},{continueLong:-.1},{continueHalfTurn:0},{gapFirstFactor:0},{gapHalfTurn:200},{breakShare:1.5},{breakMedianMinutes:.5},{answerSeconds:-1},{callsPerTurn:.9},{callsPerTurn:101},{sharedPrefixWarm:2},{retrievalShare:-1},{reasoningTokens:1.5},{cacheHitShare:1.2},{archivedTokens:2.5}]) assert.throws(()=>configure(bad),RangeError,JSON.stringify(bad));
  assert.equal(configure({callsPerTurn:2.4}).callsPerTurn,2.4);
});
test('by chance of another turn, a thread that has come further is more likely to go on',()=>{
  const law={lengthDistribution:'continuation',continueFirst:.5,continueLong:.97,continueHalfTurn:4,threads:200}, c=configure(law);
  // p(n) = long - (long - first) x 0.5^((n - 1)/half): the first value after turn 1, halfway four turns later, the long value in the end.
  assert.equal(continueChance(1,c),.5);assert.ok(Math.abs(continueChance(5,c)-.735)<1e-12);assert.ok(Math.abs(continueChance(400,c)-.97)<1e-12);
  // The share of threads that reaches a turn is the running product of those chances.
  const curve=theoreticalSurvival(c); let alive=1;
  for(let n=1;n<=160;n++){assert.ok(Math.abs(curve[n-1].realized-alive)<1e-12);assert.equal(curve[n-1].planned,curve[n-1].realized);alive*=continueChance(n,c);}
  assert.equal(curve[1].realized,.5);
  // 20,000 sampled threads against that curve.
  const reached=new Array(161).fill(0); let threads=0;
  for(let seed=1;seed<=100;seed++)for(const thread of generateWorkload({...law,seed}).threads){threads++;for(let n=1;n<=thread.turns.length;n++)reached[n]++;}
  assert.equal(threads,20000);
  for(const turn of [2,3,5,10,30,80])assert.ok(Math.abs(reached[turn]/threads-curve[turn-1].realized)<.012,`turn ${turn}: ${reached[turn]/threads} against ${curve[turn-1].realized}`);
  // The median, the 95th percentile and the flat chance are not used.
  assert.deepEqual(generateWorkload({...law,plannedTurns:3,plannedTurnsP95:5,continuationProbability:.1}),generateWorkload(law));
  // No thread outlives the cap, and one that is sure to go on runs up to it.
  const capped=generateWorkload({...law,continueFirst:1,continueLong:1,maxTurns:5,threads:3});
  assert.deepEqual(capped.threads.map(x=>x.turns.length),[5,5,5]);assert.deepEqual(capped.threads.map(x=>x.plannedTurns),[5,5,5]);
  // The ping planner's chance of a return is the chance at that position, and none after the last turn allowed.
  const plan=runIndex=>evaluatePingPlan(held,full,{...c,keepWarm:true},runIndex);
  assert.equal(plan(0).returnProbability,.5);assert.equal(plan(4).returnProbability,continueChance(5,c));assert.ok(Math.abs(plan(14).returnProbability-.9285)<.0001);
  assert.equal(plan(159).returnProbability,0);assert.equal(plan(159).reason,'no modeled future return');
  // The other length laws keep their flat chance.
  assert.equal(evaluatePingPlan(held,full,config({continuationProbability:.8,continueFirst:.1,continueLong:.2}),0).returnProbability,.8);
});
test('the usual return delay depends on how far the thread has come',()=>{
  const c=config({threads:200,maxTurns:7,plannedTurns:7,plannedTurnsP95:7,gapMedianMinutes:4,gapP95Minutes:60,gapFirstFactor:3,gapHalfTurn:2}), factor=n=>3**(.5**((n-1)/2));
  // Median 4 x 3 minutes after turn 1, 4 x sqrt(3) two turns later, 4 in a long thread. The 95th percentile scales with it.
  assert.equal(factor(1),3);assert.ok(Math.abs(factor(3)-Math.sqrt(3))<1e-12);
  for(const n of [1,3,5,40]){assert.ok(Math.abs(returnCdf(4*factor(n),c,n)-.5)<1e-6);assert.ok(Math.abs(returnCdf(60*factor(n),c,n)-.95)<1e-6);}
  assert.ok(Math.abs(returnCdf(4,c)-.5)<1e-6);assert.ok(Math.abs(returnCdf(60,c)-.95)<1e-6);assert.ok(returnCdf(4,c,1)<.3);
  // 8,000 sampled delays after each turn.
  const delays=o=>{const after=Array.from({length:7},()=>[]);for(let seed=1;seed<=40;seed++)for(const thread of generateWorkload({...c,...o,seed}).threads)for(let i=1;i<thread.turns.length;i++)after[i].push(thread.turns[i].time-thread.turns[i-1].time);return after;};
  const median=values=>[...values].sort((a,b)=>a-b)[values.length>>1], after=delays({}), flat=delays({gapFirstFactor:1});
  for(const n of [1,2,3,5,6]){assert.equal(after[n].length,8000);assert.ok(Math.abs(median(after[n])/(4*factor(n))-1)<.1,`after turn ${n}: ${median(after[n])}`);}
  // The same draws as without the effect, each stretched by the factor of its position.
  for(const n of [1,4,6])for(const k of [0,17,7999])assert.ok(Math.abs(after[n][k]/flat[n][k]-factor(n))<1e-9);
  // The planner prices a wait with the delay of its own position: fewer returns inside the cache lifetime after turn 1.
  assert.ok(Math.abs(warmReturnShare(c,1)-returnCdf(c.ttlMinutes,c,1))<1e-15);assert.ok(warmReturnShare(c,1)<warmReturnShare(c,3)&&warmReturnShare(c,3)<warmReturnShare(c));
  const planner={...c,keepWarm:true,continuationProbability:.8,maxTurns:60,plannedTurns:60,plannedTurnsP95:60}, first=evaluatePingPlan(held,full,planner,0), late=evaluatePingPlan(held,full,planner,39), same=evaluatePingPlan(held,full,{...planner,gapFirstFactor:1},0);
  // One ping covers the returns between 30 and 59.25 minutes: more of them after turn 1, where the delay is three times as long.
  const bridged=turn=>returnCdf(2*c.ttlMinutes-c.pingLeadMinutes,c,turn)-returnCdf(c.ttlMinutes,c,turn);
  assert.ok(Math.abs(first.rows[1].expectedGain/late.rows[1].expectedGain-bridged(1)/bridged(40))<1e-9);
  assert.ok(Math.abs(late.rows[1].expectedGain/same.rows[1].expectedGain-bridged(40)/bridged(Infinity))<1e-9);
  assert.ok(first.rows[1].expectedGain>1.5*same.rows[1].expectedGain);
});
test('a share of returns comes after a long break, whatever the turn',()=>{
  const c=config({threads:200,maxTurns:5,plannedTurns:5,plannedTurnsP95:5,gapMedianMinutes:4,gapP95Minutes:20,breakShare:.3,breakMedianMinutes:1000});
  // 70% usual delays, all over long before 1,000 minutes, and 30% breaks with a median of 1,000 minutes and a 95th percentile of 4,000.
  assert.ok(Math.abs(returnCdf(1000,c)-(.7+.3*.5))<1e-5);assert.ok(Math.abs(returnCdf(4000,c)-(.7+.3*.95))<1e-5);assert.ok(Math.abs(returnCdf(100,c)-.7)<.002);
  // Breaks alone do not depend on the position, while the usual delay may.
  const only={...c,breakShare:1,gapFirstFactor:5};
  for(const minutes of [300,1000,2500])assert.equal(returnCdf(minutes,only,1),returnCdf(minutes,only,30));
  const gaps=o=>{const all=[];for(let seed=1;seed<=10;seed++)for(const thread of generateWorkload({...c,...o,seed}).threads)for(let i=1;i<thread.turns.length;i++)all.push(thread.turns[i].time-thread.turns[i-1].time);return all;};
  const drawn=gaps({}), over=limit=>drawn.filter(g=>g>limit).length/drawn.length;
  assert.equal(drawn.length,8000);assert.ok(Math.abs(over(100)-.3)<.02);assert.ok(Math.abs(over(1000)-.15)<.015);assert.ok(Math.abs(over(4000)-.015)<.006);
  // Without breaks nothing is drawn for them: the delays are those of the default law.
  assert.deepEqual(gaps({breakShare:0}),gaps({breakShare:0,breakMedianMinutes:77}));
  // The planner sees the mixture: of the returns that outlast the cache, the breaks are beyond what a few pings reach.
  const planner={...c,keepWarm:true,continuationProbability:.9,gapMedianMinutes:40,gapP95Minutes:60,maxPings:3}, plain=evaluatePingPlan(held,full,{...planner,breakShare:0},0), mixed=evaluatePingPlan(held,full,planner,0);
  const bridged=(config,n)=>returnCdf(n*(config.ttlMinutes-config.pingLeadMinutes)+config.ttlMinutes,config)-returnCdf(config.ttlMinutes,config);
  for(const n of [1,2,3])assert.ok(Math.abs(mixed.rows[n].expectedGain/plain.rows[n].expectedGain-bridged(planner,n)/bridged({...planner,breakShare:0},n))<1e-9);
  assert.ok(mixed.rows[3].expectedGain<.75*plain.rows[3].expectedGain);
});
test('the cache lifetime runs from when the request was sent, so the answer time comes off the wait',()=>{
  // Every user is back exactly 4.5 minutes after the answer, on a five-minute cache.
  const base={maxTurns:3,plannedTurns:3,plannedTurnsP95:3,gapMedianMinutes:4.5,gapP95Minutes:4.5,ttlMinutes:5,pingLeadMinutes:.75,systemTokens:2048,cacheMinTokens:1024,tokenMedian:300,tokenP95:300,...needs(0)};
  const quick=config(base), slow=config({...base,answerSeconds:60});
  assert.deepEqual(generateWorkload(quick).threads[0].turns.map(x=>x.time),[0,4.5,9]);
  assert.deepEqual(generateWorkload(slow).threads[0].turns.map(x=>x.time),[0,5.5,11]);
  // An answer that takes a minute puts the next question 5.5 minutes after the request: the entry is gone.
  for(const arm of ['compache','openclaw']) {
    const warm=answersOf(runArm(generateWorkload(quick),quick,arm)), cold=answersOf(runArm(generateWorkload(slow),slow,arm));
    assert.ok(warm[1].cachedTokens>0&&warm[2].cachedTokens>0);assert.equal(cold[1].cachedTokens,0);assert.equal(cold[2].cachedTokens,0);
  }
  assert.equal(warmReturnShare(quick),1);assert.equal(warmReturnShare(slow),0);
  // The planner: a ping is due 4.25 minutes after the request. With no answer time the return is warm without it.
  const idle=evaluatePingPlan(held,full,{...quick,keepWarm:true},0);
  assert.equal(idle.chosen,0);assert.equal(idle.reason,'no positive expected value');assert.equal(idle.rows[1].expectedGain,0);
  // With a minute of answer time no user is back before the first ping (F(4.25 - 1) = 0) and that one ping bridges every
  // return (F(4.25 + 5 - 1) - F(5 - 1) = 1): it is sent for certain and saves the whole difference between a cold and a warm request.
  const plan=evaluatePingPlan(held,full,{...slow,keepWarm:true},0), k=102548, m=slow.model;
  assert.equal(plan.chosen,1);assert.equal(plan.returnProbability,1);
  assert.equal(plan.rows[1].expectedGain,priceCall(k,0,0,m).cost-priceCall(k,k,0,m).cost);
  assert.equal(plan.rows[1].expectedCost,priceCall(k,102048,slow.pingOutputTokens,m).cost);
  // The second ping would be due at 8.5 minutes, after everyone is back at 5.5: it is never sent and adds nothing.
  assert.equal(plan.rows[2].expectedCost,plan.rows[1].expectedCost);assert.equal(plan.rows[2].expectedGain,plan.rows[1].expectedGain);
  // In the run, that ping goes out at 4.25 and 9.75 minutes and both returns read the cache.
  const pinged=runArm(generateWorkload(slow),{...slow,keepWarm:true},'compache');
  assert.deepEqual(pinged.ledger.filter(x=>x.role==='ping').map(x=>x.time),[4.25,9.75]);
  assert.ok(answersOf(pinged)[1].cachedTokens>0&&answersOf(pinged)[2].cachedTokens>0);
  // An answer still running when the first ping is due leaves nothing to keep warm.
  const late=evaluatePingPlan(held,full,{...slow,keepWarm:true,answerSeconds:255},0);
  assert.equal(late.chosen,0);assert.equal(late.reason,'answer outlasts the ping interval');assert.equal(late.rows.length,1);
  assert.notEqual(evaluatePingPlan(held,full,{...slow,keepWarm:true,answerSeconds:254},0).reason,'answer outlasts the ping interval');
});
test('each further model call of a turn sends its request again from the cache',()=>{
  const c=config({systemTokens:2048,cacheMinTokens:1024,callsPerTurn:3,model:{context:100000,maxInput:100000}}), w=workload([t(0,0),t(1,1)]);
  for(const arm of ['compache','openclaw']) {
    const r=runArm(w,c,arm), once=runArm(w,{...c,callsPerTurn:1},arm), further=r.ledger.filter(x=>x.role==='further');
    assert.equal(once.byRole.further,undefined);assert.deepEqual(r.byRole.answer,once.byRole.answer);
    // Two turns of three calls: four further calls, in a ledger row per turn.
    assert.equal(further.length,2);assert.equal(r.byRole.further.calls,4);assert.equal(r.totals.calls,once.totals.calls+4);
    answersOf(r).forEach((answer,i)=>{
      const one=priceCall(answer.inputTokens,answer.inputTokens,0,c.model);
      assert.equal(further[i].calls,2);assert.equal(further[i].cost,2*one.cost);assert.equal(further[i].inputTokens,2*answer.inputTokens);
      assert.equal(further[i].cachedTokens,2*answer.inputTokens);assert.equal(further[i].freshTokens,0);assert.equal(further[i].outputTokens,0);
    });
    assert.ok(Math.abs(r.totals.cost-once.totals.cost-r.byRole.further.cost)<1e-15);
    assert.ok(Math.abs(r.totals.cost-r.ledger.reduce((s,x)=>s+x.cost,0))<1e-15);
    // An average of 1.5 calls a turn is half a further call each.
    const half=runArm(w,{...c,callsPerTurn:1.5},arm);
    assert.equal(half.byRole.further.calls,1);assert.ok(Math.abs(half.byRole.further.cost-r.byRole.further.cost/4)<1e-15);
  }
  // A request too short to cache is sent again at the full input price, without the surcharge for writing it.
  const small=config({systemTokens:10,cacheMinTokens:1024,callsPerTurn:2,model:{context:100000,maxInput:100000,inputRate:1,writeMultiplier:2}});
  const r=runArm(workload([t(0,0,100)]),small,'compache');
  assert.equal(r.byRole.further.cost,120/1e6);assert.equal(r.byRole.further.cachedTokens,0);assert.equal(r.byRole.further.calls,1);
  // The fit sees the charge: every candidate policy costs more with it.
  const scenario={threads:2,plannedTurns:10,plannedTurnsP95:10,maxTurns:10,lengthDistribution:'fixed',continuationProbability:1,keepWarm:false};
  const single=simulate(scenario), triple=simulate({...scenario,callsPerTurn:3});
  for(const [i,x] of triple.optimization.candidates.entries())assert.ok(x.cost>single.optimization.candidates[i].cost);
});
test('a system prompt that another conversation has cached is read by expectation',()=>{
  const base={systemTokens:2048,cacheMinTokens:1024,ttlMinutes:5,model:{context:100000,maxInput:100000}}, w=workload([t(0,0),t(1,1),t(2,20)]), m=config(base).model;
  for(const arm of ['compache','openclaw']) {
    const alone=answersOf(runArm(w,config(base),arm)), shared=answersOf(runArm(w,config({...base,sharedPrefixWarm:.4}),arm));
    // The first request and the return after 19 minutes find nothing of their own thread: the 2,048 tokens of the
    // system block are read with chance 0.4. Tokens and cost are the blend of the two outcomes.
    for(const i of [0,2]) {
      const input=alone[i].inputTokens, miss=priceCall(input,0,20,m), hit=priceCall(input,2048,20,m);
      assert.equal(alone[i].cachedTokens,0);assert.ok(Math.abs(shared[i].cachedTokens-.4*2048)<1e-9);assert.ok(Math.abs(shared[i].freshTokens-(input-.4*2048))<1e-9);
      assert.ok(Math.abs(shared[i].cost-(.6*miss.cost+.4*hit.cost))<1e-15);assert.ok(shared[i].cost<alone[i].cost);
      assert.ok(Math.abs(Object.values(shared[i].cacheBreakdown).reduce((a,b)=>a+b,0)-input)<1e-9);assert.ok(Math.abs(shared[i].cacheBreakdown.cached-.4*2048)<1e-9);
    }
    assert.ok(Math.abs(shared[0].cacheBreakdown.firstRequest-(shared[0].inputTokens-.4*2048))<1e-9);
    assert.ok(Math.abs(alone[2].cacheBreakdown.expiredPrefix-shared[2].cacheBreakdown.expiredPrefix-.4*2048)<1e-9);
    // A minute after the first turn the thread's own entry is longer than the system block: nothing changes.
    assert.deepEqual(shared[1],alone[1]);
    // A system block too short to cache cannot be shared.
    const short={...base,systemTokens:1000};
    assert.deepEqual(runArm(w,config({...short,sharedPrefixWarm:.4}),arm).ledger,runArm(w,config(short),arm).ledger);
  }
  // OpenClaw's flush and summary prompts are its own: no other conversation has them cached.
  const summarizing=config({systemTokens:100,cacheMinTokens:1,cacheBlockTokens:1,safetyReserve:0,tailTokens:100,summaryTokens:80,flushOutputTokens:5,model:{context:1000,maxInput:1000},ttlMinutes:10,sharedPrefixWarm:.5});
  const r=runArm(workload([t(0,0,200),t(1,1,200),t(2,2,200)]),summarizing,'openclaw');
  assert.ok(r.byRole.summary.calls>0);assert.equal(r.byRole.summary.cachedTokens,0);assert.equal(r.byRole.flush.cachedTokens,0);assert.equal(answersOf(r)[0].cachedTokens,50);
  // The planner's cold return carries the same expectation, so a warm return saves less.
  const planner=config({systemTokens:2048,cacheMinTokens:1024,ttlMinutes:5,keepWarm:true,continuationProbability:.8,gapMedianMinutes:6,gapP95Minutes:9}), k=102548;
  const without=evaluatePingPlan(held,full,planner,0), withShared=evaluatePingPlan(held,full,{...planner,sharedPrefixWarm:.4},0);
  const saving=cached=>priceCall(k,cached,0,m).cost-priceCall(k,k,0,m).cost;
  assert.ok(without.rows[1].expectedGain>0);assert.ok(Math.abs(withShared.rows[1].expectedGain/without.rows[1].expectedGain-saving(.4*2048)/saving(0))<1e-9);
  assert.deepEqual(withShared.rows.map(x=>x.expectedCost),without.rows.map(x=>x.expectedCost));
});
test('a turn that retrieves nothing adds no retrieved text, and none can be needed from it',()=>{
  const c=config({threads:200,maxTurns:6,plannedTurns:6,plannedTurnsP95:6,retrievalShare:.5,tokenMedian:300,tokenP95:300,inputDependencyProbability:0,outputDependencyProbability:0,retrievalDependencyProbability:1,retrievalDependencyTurnsBack:1,retrievalDependencyTurnsBackP95:4});
  const sample=o=>{const threads=[];for(let seed=1;seed<=50;seed++)threads.push(...generateWorkload({...c,...o,seed}).threads);return threads;};
  const threads=sample({}), turns=threads.flatMap(x=>x.turns), law=ageLaw(1,4);
  // Half of 60,000 turns retrieve, and those that do retrieve what the median and 95th percentile say.
  assert.equal(turns.length,60000);assert.ok(Math.abs(turns.filter(x=>x.retrievalTokens>0).length/turns.length-.5)<.02);assert.ok(turns.every(x=>x.retrievalTokens===0||x.retrievalTokens===300));
  let cases=0, near=0;
  for(const thread of threads)for(const turn of thread.turns) {
    const earlier=thread.turns.slice(0,turn.index).filter(x=>x.retrievalTokens>0);
    // With the chance at 1, a turn needs retrieved text exactly when an earlier turn retrieved any, and from such a turn.
    assert.equal(turn.dependencies.length,earlier.length?1:0);
    for(const d of turn.dependencies)assert.ok(thread.turns[d.turn].retrievalTokens>0);
    // Turn 4 after turns 1 and 3 retrieved and turn 2 did not: the age law's weights on one and on three turns back, renormalised.
    if(turn.index===3&&earlier.length===2&&thread.turns[1].retrievalTokens===0){cases++;if(turn.dependencies[0].turn===2)near++;}
  }
  assert.ok(cases>1000);assert.ok(Math.abs(near/cases-law.weight(1)/(law.weight(1)+law.weight(3)))<.04,`${near/cases}`);
  // A law with all its weight one turn back: the turn before if it retrieved, otherwise the oldest turn that did.
  for(const thread of sample({retrievalDependencyTurnsBackP95:1}))for(const turn of thread.turns)for(const d of turn.dependencies)
    assert.equal(d.turn,thread.turns[turn.index-1].retrievalTokens>0?turn.index-1:thread.turns.findIndex(x=>x.retrievalTokens>0));
  // No turn retrieves: no retrieved text, no need for any, and nothing for the fit to keep.
  const none=generateWorkload({...c,retrievalShare:0});
  assert.ok(none.threads.every(thread=>thread.turns.every(turn=>turn.retrievalTokens===0&&!turn.dependencies.length)));
  assert.deepEqual([...new Set(simulate({...c,threads:2,retrievalShare:0}).optimization.candidates.map(x=>x.retrievalRetentionRuns))],[0]);
  // Every turn retrieves: nothing is drawn to decide it, so the threads are those of the setting left alone.
  assert.deepEqual(generateWorkload({...c,retrievalShare:1}),generateWorkload({...c,retrievalShare:1,seed:c.seed}));
});
test('the expected share in view counts only turns that retrieved',()=>{
  const base={systemTokens:100,cacheMinTokens:0,cacheBlockTokens:1,safetyReserve:0,target:.2,model:{context:1000,maxInput:1000},ttlMinutes:1,inputDependencyProbability:0,outputDependencyProbability:0,retrievalDependencyProbability:.4,retrievalDependencyTurnsBack:1,retrievalDependencyTurnsBackP95:4,retrievalRetentionRuns:0};
  const law=ageLaw(1,4), kind=turns=>runArm(workload(turns),config(base),'compache').retention.byKind.retrieval;
  // Turns 1 and 3 retrieve. Turn 2 has turn 1's text in view. By turn 3 it is archived and nothing else was retrieved.
  // Turn 4 has turn 3's text, one turn back, and misses turn 1's, three back: the law's weights on those two ages, renormalised.
  const mixed=kind([t(0,0,300),t(1,2,0),t(2,4,300),t(3,6,0)]);
  assert.ok(Math.abs(mixed.expectedRequired-1.2)<1e-12);
  assert.ok(Math.abs(mixed.expectedRetained-.4*(1+0+law.weight(1)/(law.weight(1)+law.weight(3))))<1e-12);
  // Before any turn has retrieved, no retrieved text can be needed.
  const late=kind([t(0,0,0),t(1,2,300),t(2,4,0)]);
  assert.ok(Math.abs(late.expectedRequired-.4)<1e-12);assert.ok(Math.abs(late.expectedRetained-.4)<1e-12);
  // A turn without retrieved text is never archived: it would free nothing and only move what follows it.
  const r=runArm(workload([t(0,0,0),t(1,2,0),t(2,4,0)]),config(base),'compache');
  assert.equal(r.compactions,0);assert.equal(r.removedTokens,0);
});
test('OpenClaw keeps its three newest tool results, and a turn without one is not among them',()=>{
  const c=config({systemTokens:100,cacheMinTokens:0,safetyReserve:0,toolClearing:true,clearedTokens:5,model:{context:200000,maxInput:200000},ttlMinutes:100});
  const r=runArm(workload([t(0,0,20000),t(1,1,20000),t(2,2,20000),t(3,3,0),t(4,4,20000)]),c,'openclaw'), answers=answersOf(r);
  // The fourth turn retrieves nothing: its three results are the newest three and all stay. The fifth turn brings a fourth result and clears the first.
  assert.equal(answers[3].inputTokens,100+3*(10+20000+20)+10);
  assert.equal(answers[4].inputTokens,100+(10+5+20)+2*(10+20000+20)+(10+20)+10+20000);
  assert.equal(r.clearings,1);
});
test('reasoning is billed as output with each answer and never sent again',()=>{
  const c=config({systemTokens:100,cacheMinTokens:0,model:{context:100000,maxInput:100000}}), w=workload([t(0,0),t(1,1)]);
  for(const arm of ['compache','openclaw']) {
    const plain=runArm(w,c,arm), thinking=runArm(w,{...c,reasoningTokens:700},arm);
    assert.deepEqual(thinking.ledger.map(x=>x.inputTokens),plain.ledger.map(x=>x.inputTokens));
    assert.deepEqual(plain.ledger.map(x=>x.outputTokens),[20,20]);assert.deepEqual(thinking.ledger.map(x=>x.outputTokens),[720,720]);
    assert.ok(Math.abs(thinking.totals.cost-plain.totals.cost-2*700*c.model.outputRate/1e6)<1e-15);
    assert.equal(thinking.totals.inputCost,plain.totals.inputCost);assert.equal(thinking.totals.cachedCost,plain.totals.cachedCost);
  }
  // The window has to hold the reasoning while it is written: 210 tokens of input and 820 of output do not fit 1,000.
  const tight=config({systemTokens:100,safetyReserve:0,reasoningTokens:800,model:{context:1000,maxInput:1000}}), one=workload([t(0,0,100)]);
  assert.equal(runArm(one,tight,'compache').complete,false);assert.equal(runArm(one,{...tight,reasoningTokens:0},'compache').complete,true);
});
test('a live cache entry is read for its configured share, priced by expectation',()=>{
  const base={systemTokens:2048,cacheMinTokens:1024,ttlMinutes:5,model:{context:100000,maxInput:100000}}, w=workload([t(0,0),t(1,1)]), m=config(base).model;
  for(const arm of ['compache','openclaw']) {
    const sure=answersOf(runArm(w,config(base),arm)), partial=answersOf(runArm(w,config({...base,cacheHitShare:.8}),arm));
    // Nothing is live at the first request. The second finds its thread's entry and reads four fifths of it.
    assert.deepEqual(partial[0],sure[0]);
    const input=sure[1].inputTokens, prefix=sure[1].cachedTokens;
    assert.ok(prefix>2048);assert.ok(Math.abs(partial[1].cachedTokens-.8*prefix)<1e-9);
    assert.ok(Math.abs(partial[1].cost-(.8*priceCall(input,prefix,20,m).cost+.2*priceCall(input,0,20,m).cost))<1e-15);
    assert.ok(Math.abs(partial[1].cacheBreakdown.missed-.2*prefix)<1e-9);assert.equal(sure[1].cacheBreakdown.missed,0);
    assert.ok(Math.abs(Object.values(partial[1].cacheBreakdown).reduce((a,b)=>a+b,0)-input)<1e-9);
    assert.ok(Math.abs(partial[1].cacheBreakdown.newContext-sure[1].cacheBreakdown.newContext)<1e-9);
    // With a shared system prompt, a request that misses its own entry still finds that block with its chance.
    const both=answersOf(runArm(w,config({...base,cacheHitShare:.8,sharedPrefixWarm:.5}),arm));
    assert.ok(Math.abs(both[1].cachedTokens-(.8*prefix+.2*.5*2048))<1e-9);assert.ok(Math.abs(both[0].cachedTokens-.5*2048)<1e-9);
    assert.ok(Math.abs(both[1].cacheBreakdown.missed-(.2*.5*prefix+.2*.5*(prefix-2048)))<1e-9);
    // Further calls of the turn read the same share.
    const again=runArm(w,config({...base,cacheHitShare:.8,callsPerTurn:2}),arm).ledger.filter(x=>x.role==='further');
    assert.ok(Math.abs(again[1].cachedTokens-.8*input)<1e-9);
  }
  // Keeping an entry alive is worth its share: the saving from a warm return shrinks with it, and a ping reads less.
  const planner=config({systemTokens:2048,cacheMinTokens:1024,ttlMinutes:5,keepWarm:true,continuationProbability:.8,gapMedianMinutes:6,gapP95Minutes:9}), k=102548;
  const sure=evaluatePingPlan(held,full,planner,0), partial=evaluatePingPlan(held,full,{...planner,cacheHitShare:.8},0);
  assert.ok(sure.rows[1].expectedGain>0);assert.ok(Math.abs(partial.rows[1].expectedGain/sure.rows[1].expectedGain-.8)<1e-9);
  assert.ok(Math.abs(partial.rows[1].expectedCost/sure.rows[1].expectedCost-priceCall(k,.8*102048,16,m).cost/priceCall(k,102048,16,m).cost)<1e-9);
});
test('an archived source leaves a notice of the configured size',()=>{
  const base={systemTokens:100,cacheMinTokens:0,cacheBlockTokens:1,safetyReserve:0,target:.2,model:{context:1000,maxInput:1000},ttlMinutes:1};
  const turns=[t(0,0),t(1,2),t(2,4,100,[{kind:'retrieval',turn:0}])];
  const bare=runArm(workload(turns),config(base),'compache'), noted=runArm(workload(turns),config({...base,archivedTokens:7}),'compache');
  // The second turn archives the first source and the third the second: every later request carries seven tokens for each.
  assert.deepEqual(answersOf(bare).map(x=>x.inputTokens),[410,440,602]);assert.deepEqual(answersOf(noted).map(x=>x.inputTokens),[410,447,616]);
  assert.deepEqual(noted.decisions.map(d=>d.removedTokens),[0,300,300]);assert.deepEqual(noted.decisions.map(d=>d.beforeTokens-d.afterTokens),[0,293,293]);
  assert.equal(noted.decisions[1].afterByKind.archived,7);
  // The notice is not the source: a turn that needs it fetches it back, as without a notice.
  assert.equal(noted.recalls,1);assert.equal(noted.retention.observed,bare.retention.observed);assert.equal(noted.retention.expected,bare.retention.expected);
  // A source no larger than its notice stays where it is.
  const kept=runArm(workload(turns),config({...base,archivedTokens:300}),'compache');
  assert.equal(kept.compactions,0);assert.equal(kept.recalls,0);
  // The planner's cold request carries the notice too.
  const c=config({keepWarm:true,systemTokens:2000,continuationProbability:.1,gapMedianMinutes:6,gapP95Minutes:6,ttlMinutes:5,pingLeadMinutes:.75,archivedTokens:7,model:{context:1000000,maxInput:1000000}});
  const system={id:'system',tokens:2000,kind:'system',turn:-1}, retrieval={id:'r0',source:'r0',tokens:300000,kind:'retrieval',turn:0}, output={id:'o0',source:'o0',tokens:500,kind:'output',turn:0};
  assert.deepEqual(coldPreview([retrieval,output],c).map(b=>[b.id,b.tokens]),[['system',2000],['archived-r0',7],['o0',500]]);
  assert.equal(evaluatePingPlan([system,retrieval],[system,retrieval,output],c,0).coldTokens,2507);
  assert.equal(evaluatePingPlan([system,retrieval],[system,retrieval,output],{...c,archivedTokens:0},0).coldTokens,2500);
});
