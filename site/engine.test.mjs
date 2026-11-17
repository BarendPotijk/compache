import test from 'node:test';
import assert from 'node:assert/strict';
import {configure, generateWorkload, runArm, sharedPrefix, priceCall, simulate, coldPreview, planPings, evaluatePingPlan, theoreticalSurvival, returnCdf} from './engine.mjs';

const config = o => configure({threads:1,maxTurns:8,plannedTurns:8,plannedTurnsP95:8,lengthDistribution:'fixed',continuationProbability:1,keepWarm:false,companyVariation:0,recordTokens:0,summarySystemTokens:100, ...o});
const workload = turns => ({threads:[{id:'thread-1',turns}],totalRuns:turns.length});
const t = (index,time,retrievalTokens=300,dependencies=[]) => ({index,time,inputTokens:10,retrievalTokens,outputTokens:20,dependencies});
test('seeded workload is identical and has only earlier dependencies',()=>{
  const c=config({threads:3,dependencyProbability:1,dependencyLookback:3});
  const a=generateWorkload(c); assert.deepEqual(a,generateWorkload(c));
  for(const thread of a.threads) for(const turn of thread.turns) for(const dep of turn.dependencies) assert.ok(dep.turn<turn.index && turn.index-dep.turn<=3);
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
  const r=runArm(w,c,'compache'); assert.ok(r.compactions>0); assert.equal(r.recalls,1); assert.equal(r.dependencies.required,3);assert.equal(r.dependencies.covered,3); assert.equal(r.dependencyCoverage,1); assert.equal(r.complete,true);
});
test('summary and flush are separate charged calls with separate prefix identities',()=>{
  const c=config({systemTokens:100,cacheMinTokens:1,cacheBlockTokens:1,safetyReserve:0,tailTokens:100,summaryTokens:80,flushOutputTokens:5,model:{context:1000,maxInput:1000},ttlMinutes:10});
  const r=runArm(workload([t(0,0,200),t(1,1,200),t(2,2,200,[{kind:'retrieval',turn:1}])]),c,'openclaw');
  assert.ok(r.byRole.summary.calls>0);assert.ok(r.byRole.flush.calls>0); assert.equal(r.dependencyCoverage,1);
  assert.equal(r.byRole.summary.cachedTokens,0); assert.equal(r.byRole.flush.cachedTokens,0);
  assert.ok(Math.abs(r.totals.cost-r.ledger.reduce((s,x)=>s+(x.cost||0),0))<1e-12);
});
test('incomplete calls do not produce a full workload savings claim',()=>{
  const r=simulate({threads:1,maxTurns:1,plannedTurns:1,plannedTurnsP95:1,lengthDistribution:'fixed',tokenMedian:2000,tokenP95:2000,companyVariation:0,systemTokens:100,safetyReserve:0,keepWarm:false,model:{context:500,maxInput:500}});
  assert.equal(r.savings,null);assert.equal(r.compache.complete,false);assert.equal(r.openclaw.complete,false);assert.equal(r.compache.ledger[0].admitted,false);
});
test('zero costs and zero dependencies have null denominators',()=>{
  const r=simulate({threads:1,maxTurns:2,plannedTurns:2,plannedTurnsP95:2,lengthDistribution:'fixed',dependencyProbability:0,keepWarm:false,model:{inputRate:0,cachedRate:0,outputRate:0}});
  assert.equal(r.savings.absolute,0); assert.equal(r.savings.percent,null); assert.equal(r.compache.dependencyCoverage,null);
});
test('negative savings are retained and not clamped after optimization',()=>{
  const r=simulate({threads:1,plannedTurns:60,plannedTurnsP95:60,maxTurns:60,lengthDistribution:'fixed',continuationProbability:1,tokenMedian:25000,tokenP95:25000,companyVariation:0,dependencyProbability:.25,gapMedianMinutes:100,gapP95Minutes:100,keepWarm:false});
  assert.ok(r.optimization.feasible && r.compache.complete && r.openclaw.complete); assert.ok(r.savings.absolute<0);
});
test('configuration bounds reject impossible values',()=>{
  assert.throws(()=>configure({threads:201}));assert.throws(()=>configure({gapMedianMinutes:2,gapP95Minutes:1}));assert.throws(()=>configure({lengthDistribution:'other'}));
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
  const c=config({threads:1,maxTurns:4,plannedTurns:4,plannedTurnsP95:4,tokenMedian:50000,tokenP95:50000,inputTokens:200,dependencyProbability:1});
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
test('company heterogeneity is shared by threads assigned to the same company',()=>{
  const a=generateWorkload(config({companies:2,threads:5,companyVariation:.5,outputTokens:0}));
  assert.equal(a.threads[0].companyId,a.threads[2].companyId);
  assert.equal(a.threads[0].factor,a.threads[2].factor);
  assert.notEqual(a.threads[0].factor,a.threads[1].factor);
  assert.equal(a.threads[0].turns[0].outputTokens,0);
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
  assert.equal(p.rows.length,c.maxPings);
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

test('independent per-kind dependency controls and lookbacks are causal',()=>{
  const c=config({threads:1,maxTurns:12,plannedTurns:12,plannedTurnsP95:12,inputDependencyProbability:0,retrievalDependencyProbability:1,outputDependencyProbability:0,retrievalDependencyLookback:3});
  const turns=generateWorkload(c).threads[0].turns;
  for(const turn of turns.slice(1)) {assert.equal(turn.dependencies.length,1);assert.equal(turn.dependencies[0].kind,'retrieval');assert.ok(turn.index-turn.dependencies[0].turn<=3);}
  const all=generateWorkload({...c,inputDependencyProbability:1,outputDependencyProbability:1}).threads[0].turns;
  for(const turn of all.slice(1))assert.deepEqual(turn.dependencies.map(d=>d.kind),['input','retrieval','output']);
});
test('retention is measured before recall and separates observed and expected from post-recall',()=>{
  const c=config({systemTokens:100,cacheMinTokens:0,cacheBlockTokens:1,safetyReserve:0,target:.2,model:{context:1000,maxInput:1000},ttlMinutes:1,inputDependencyProbability:0,retrievalDependencyProbability:1,outputDependencyProbability:0,retrievalDependencyLookback:2,retrievalRetentionRuns:0});
  const r=runArm(workload([t(0,0),t(1,2),t(2,4,100,[{kind:'retrieval',turn:0}])]),c,'compache');
  assert.ok(r.retention.expected<1);assert.equal(r.retention.observed,0);assert.equal(r.rawSourceAvailability,1);assert.equal(r.recalls,1);
  assert.equal(r.retention.byKind.retrieval.observedRetained,0);
});
test('grid optimizer is deterministic and tightening retention cannot lower its minimum cost',()=>{
  const common={threads:3,companies:1,plannedTurns:32,plannedTurnsP95:32,maxTurns:32,lengthDistribution:'fixed',continuationProbability:1,tokenMedian:10000,tokenP95:10000,keepWarm:false};
  const lo=simulate({...common,retentionRequirement:.9}),hi=simulate({...common,retentionRequirement:.95});
  assert.ok(lo.optimization.feasible && hi.optimization.feasible);
  assert.ok(hi.optimization.retention>=.95-1e-12);
  assert.ok(hi.compache.totals.cost>=lo.compache.totals.cost-1e-12);
  assert.ok(lo.optimization.candidates.every(x=>x.feasible===!!(x.complete&&(x.retention===null||x.retention>=.9-1e-12))));
  assert.deepEqual(hi.optimization.selected,simulate({...common,retentionRequirement:.95}).optimization.selected);
});
test('infeasible retention or admission never claims recommended policy or savings',()=>{
  const x=simulate({threads:1,maxTurns:3,plannedTurns:3,plannedTurnsP95:3,lengthDistribution:'fixed',tokenMedian:2000,tokenP95:2000,companyVariation:0,systemTokens:100,safetyReserve:0,keepWarm:false,model:{context:500,maxInput:500}});
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
