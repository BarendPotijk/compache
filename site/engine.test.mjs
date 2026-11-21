import test from 'node:test';
import assert from 'node:assert/strict';
import {DEFAULTS, ageLaw, configure, generateWorkload, runArm, sharedPrefix, priceCall, simulate, coldPreview, planPings, evaluatePingPlan, theoreticalSurvival, returnCdf} from './engine.mjs';

const config = o => configure({threads:1,maxTurns:8,plannedTurns:8,plannedTurnsP95:8,lengthDistribution:'fixed',continuationProbability:1,keepWarm:false,recordTokens:0,summarySystemTokens:100, ...o});
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
  assert.deepEqual(second.cacheBreakdown,{cached:0,newContext:130,firstRequest:0,expiredPrefix:210,changedPrefix:0});
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
