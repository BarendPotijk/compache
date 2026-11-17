/* Deterministic hypothetical cost model. This does not execute either product. */
export const DEFAULTS = Object.freeze({
  seed: 42, companies: 1, threads: 100, maxTurns: 160, lengthDistribution: 'lognormal', plannedTurns: 80, plannedTurnsP95: 140,
  continuationProbability: 0.99, tokenMedian: 25000, tokenP95: 45000,
  companyVariation: 0, gapMedianMinutes: 4, gapP95Minutes: 60,
  dependencyProbability: 0.25, dependencyLookback: 24, inputTokens: 200, outputTokens: 500, ttlMinutes: 5,
  target: 0.20, warmTrigger: 0.5, maxPings: 3, keepWarm: true, pingLeadMinutes: 0.75,
  pingOutputTokens: 16, safetyReserve: 32000, systemTokens: 2048,
  cacheMinTokens: 1024, cacheBlockTokens: 128, tailTokens: 20000,
  summaryTokens: 4000, summaryInputCap: 40000, toolSummaryTokens: 500, summarySystemTokens: 2048, recordTokens: 1500, flushOutputTokens: 256,
  model: Object.freeze({context: 1050000, maxInput: 922000, inputRate: 0.1,
    cachedRate: 0.01, outputRate: 0.5, writeMultiplier: 1.25,
    longThreshold: 272000, longInputMultiplier: 2, longOutputMultiplier: 1.5})
});
const z95 = 1.6448536269514722;
const sum = blocks => blocks.reduce((n, b) => n + b.tokens, 0);
const byKind = blocks => Object.freeze(blocks.reduce((totals,b)=>{totals[b.kind]=(totals[b.kind]||0)+b.tokens;return totals;},{}));
const ceil = n => Math.ceil(n);
function finite(n, name, min, max) {
  if (!Number.isFinite(n) || n < min || n > max) throw new RangeError(`${name} must be between ${min} and ${max}`);
}
export function configure(overrides = {}) {
  const c = {...DEFAULTS, ...overrides, model: {...DEFAULTS.model, ...(overrides.model || {})}};
  for (const [k,min,max] of [['companies',1,200],['threads',1,200],['maxTurns',1,160],['plannedTurns',1,160],['plannedTurnsP95',1,320],['dependencyLookback',1,160],['maxPings',0,10],['target',0.01,0.45],['seed',0,4294967295],['tokenMedian',1,500000],['tokenP95',1,1000000],['gapMedianMinutes',0.001,100000],['gapP95Minutes',0.001,1000000],['companyVariation',0,2],['continuationProbability',0,1],['dependencyProbability',0,1],['ttlMinutes',0.01,100000],['warmTrigger',0.23,1],['pingLeadMinutes',0,100000],['inputTokens',1,100000],['outputTokens',0,200000],['pingOutputTokens',0,10000],['safetyReserve',0,1000000],['systemTokens',0,1000000],['cacheMinTokens',0,1000000],['cacheBlockTokens',1,100000],['tailTokens',0,1000000],['summaryTokens',1,100000],['summaryInputCap',1,1000000],['toolSummaryTokens',1,100000],['summarySystemTokens',0,100000],['recordTokens',0,100000],['flushOutputTokens',0,100000]]) finite(c[k],k,min,max);
  for (const k of ['companies','threads','maxTurns','plannedTurns','plannedTurnsP95','dependencyLookback','maxPings','seed','inputTokens','outputTokens','pingOutputTokens','safetyReserve','systemTokens','cacheMinTokens','cacheBlockTokens','tailTokens','summaryTokens','summaryInputCap','toolSummaryTokens','summarySystemTokens','recordTokens','flushOutputTokens']) if (!Number.isInteger(c[k])) throw new RangeError(`${k} must be an integer`);
  if (!['fixed','geometric','lognormal'].includes(c.lengthDistribution)) throw new RangeError('invalid lengthDistribution');
  if ((c.lengthDistribution==='lognormal' && c.plannedTurnsP95 < c.plannedTurns) || c.tokenP95 < c.tokenMedian || c.gapP95Minutes < c.gapMedianMinutes) throw new RangeError('p95 must be at least the median');
  if (c.target + 0.02 >= c.warmTrigger || c.warmTrigger > 1 - c.safetyReserve/c.model.context) throw new RangeError('require cold trigger below warm trigger and warm trigger within reserve');
  if (typeof c.keepWarm !== 'boolean') throw new TypeError('keepWarm must be boolean');
  for (const [k,min,max] of [['context',1,10000000],['maxInput',1,10000000],['inputRate',0,100000],['cachedRate',0,100000],['outputRate',0,100000],['writeMultiplier',0,100],['longThreshold',0,10000000],['longInputMultiplier',0,100],['longOutputMultiplier',0,100]]) finite(c.model[k],`model.${k}`,min,max);
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
export function generateWorkload(config) {
  const c=configure(config), random=rng(c.seed), threads=[], gaps=[], planned=[], realized=[];
  const companies=Array.from({length:c.companies},(_,i)=>({id:`company-${i+1}`,factor:Math.exp(c.companyVariation*normal(random)-c.companyVariation**2/2)}));
  for(let ti=0;ti<c.threads;ti++) {
    const company=companies[ti%companies.length], factor=company.factor;
    let length=c.lengthDistribution==='fixed' ? c.plannedTurns : c.lengthDistribution==='geometric' ? Math.ceil(Math.log(Math.max(random(),1e-12))/Math.log(1-1/Math.max(1.00000001,c.plannedTurns))) : Math.round(lognormal(random,c.plannedTurns,c.plannedTurnsP95));
    length=Math.max(1,Math.min(c.maxTurns,length)); planned.push(length);
    const turns=[]; let time=0;
    for(let i=0;i<length;i++) {
      if(i>0) {if(random()>=c.continuationProbability) break; const g=lognormal(random,c.gapMedianMinutes,c.gapP95Minutes); gaps.push(g); time+=g;}
      const total=Math.max(2,Math.min(1000000,Math.round(lognormal(random,c.tokenMedian,c.tokenP95)*factor)));
      const input=c.inputTokens, retrieval=total, deps=[];
      if(i && random()<c.dependencyProbability) {
        const kinds=['input','retrieval','output'], count=1+Math.floor(random()*3);
        for(let j=0;j<count;j++) { const k=Math.floor(random()*kinds.length), kind=kinds.splice(k,1)[0]; deps.push({kind,turn:i-1-Math.floor(random()*Math.min(i,c.dependencyLookback))}); }
      }
      turns.push({index:i,time,inputTokens:input,retrievalTokens:retrieval,outputTokens:c.outputTokens,dependencies:deps});
    }
    realized.push(turns.length); threads.push({id:`thread-${ti+1}`,companyId:company.id,factor,plannedTurns:length,turns});
  }
  const survival=Array.from({length:c.maxTurns},(_,i)=>({turn:i+1,planned:planned.filter(x=>x>i).length/c.threads,realized:realized.filter(x=>x>i).length/c.threads})).filter(x=>x.planned||x.realized);
  return {companies,threads,plannedLengths:distribution(planned),realizedLengths:distribution(realized),gaps:distribution(gaps),survival,totalRuns:realized.reduce((a,b)=>a+b,0)};
}
export function sharedPrefix(a,b) {
  let total=0;
  for(let i=0;i<Math.min(a.length,b.length);i++) {if(a[i].id!==b[i].id || a[i].tokens!==b[i].tokens) break; total+=a[i].tokens;}
  return total;
}
export function priceCall(input,cached,output,model) {
  if (![input,cached,output].every(Number.isFinite) || input<0 || output<0 || cached<0 || cached>input) throw new RangeError('invalid call token counts');
  const long=model.longThreshold>0 && input>model.longThreshold, im=long?model.longInputMultiplier:1, om=long?model.longOutputMultiplier:1;
  const fresh=input-cached, inputCost=fresh*model.inputRate*model.writeMultiplier*im/1e6, cachedCost=cached*model.cachedRate*im/1e6, outputCost=output*model.outputRate*om/1e6;
  return {inputTokens:input,cachedTokens:cached,freshTokens:fresh,outputTokens:output,inputCost,cachedCost,outputCost,cost:inputCost+cachedCost+outputCost,long};
}
// Normal CDF approximation used only by the explicitly modeled planner.
function normalCdf(x) {const sign=x<0?-1:1, t=1/(1+.3275911*Math.abs(x)/Math.SQRT2); const e=1-(((((1.061405429*t-1.453152027)*t+1.421413741)*t-.284496736)*t+.254829592)*t)*Math.exp(-x*x/2);return .5*(1+sign*e);}
function gapCdf(t,c) {if(t<=0)return 0; const s=Math.log(c.gapP95Minutes/c.gapMedianMinutes)/z95; return s===0 ? (t>=c.gapMedianMinutes?1:0) : normalCdf(Math.log(t/c.gapMedianMinutes)/s);}
export function coldPreview(history, c) {
  // No future question or dependency is known. Only prior retrieval blocks
  // are candidates; user input and assistant output stay in their original order.
  const system=block('system',c.systemTokens,'system',-1), view=[...history];
  let size=c.systemTokens+sum(view);
  if(size>(c.target+.02)*c.model.context) {
    for(let j=0;j<view.length && size>c.target*c.model.context;) {
      if(view[j].kind==='retrieval') {size-=view[j].tokens;view.splice(j,1);} else j++;
    }
  }
  return [system,...view];
}
// Planned length is rounded and capped in generateWorkload; return probability
// is conditional on reaching this turn, without inspecting a realized future.
function plannedSurvival(turn,c) {
  if(turn<=1)return 1;
  if(turn>c.maxTurns)return 0;
  if(c.lengthDistribution==='fixed')return turn<=Math.min(c.plannedTurns,c.maxTurns)?1:0;
  if(c.lengthDistribution==='geometric')return (1-1/Math.max(1.00000001,c.plannedTurns))**(turn-1);
  const sigma=Math.log(c.plannedTurnsP95/c.plannedTurns)/z95;
  if(sigma===0)return turn<=Math.round(c.plannedTurns)?1:0;
  return 1-normalCdf(Math.log((turn-.5)/c.plannedTurns)/sigma);
}
export function theoreticalSurvival(config={}) {
  const c=configure(config);
  return Array.from({length:c.maxTurns},(_,i)=>({turn:i+1,planned:plannedSurvival(i+1,c),realized:plannedSurvival(i+1,c)*c.continuationProbability**i}));
}
// CDF of return delay conditional on a continuation, not an unconditional return probability.
export function returnCdf(minutes,config={}) {return gapCdf(minutes,configure(config));}
export function evaluatePingPlan(held, full, c, runIndex=0) {
  if(!Number.isInteger(runIndex)||runIndex<0)throw new RangeError('runIndex must be nonnegative');
  const interval=c.ttlMinutes-c.pingLeadMinutes, heldInput=sum(held), k=sum(full);
  const currentSurvival=plannedSurvival(runIndex+1,c), nextSurvival=plannedSurvival(runIndex+2,c);
  const returnProbability=currentSurvival>0?c.continuationProbability*nextSurvival/currentSurvival:0;
  const cold=coldPreview(full.slice(1),c), coldInput=sum(cold);
  let reason=!c.keepWarm?'disabled':!c.maxPings?'no pings allowed':interval<=0?'ping interval is not positive':heldInput<c.cacheMinTokens?'prefix is below cache minimum':k>(c.warmTrigger-c.safetyReserve/c.model.context)*c.model.context?'prefix exceeds keep-warm ceiling':returnProbability<=0?'no modeled future return':null;
  const rows=[];
  if(reason)return {chosen:0,reason,coldTokens:coldInput,heldTokens:heldInput,returnProbability,rows};
  const heldShared=sharedPrefix(held,full);
  const cachedHeld=heldShared>=c.cacheMinTokens ? Math.floor(heldShared/c.cacheBlockTokens)*c.cacheBlockTokens : 0;
  const fullCached=k>=c.cacheMinTokens ? Math.floor(k/c.cacheBlockTokens)*c.cacheBlockTokens : 0;
  // Expired structural prefixes receive no cold cache credit. One later call
  // approximates the cost of carrying a larger warm prefix.
  const coldCost=priceCall(coldInput,0,0,c.model).cost;
  const warmCost=priceCall(k,fullCached,0,c.model).cost;
  const laterExtra=Math.max(0,warmCost-priceCall(coldInput,coldInput>=c.cacheMinTokens?Math.floor(coldInput/c.cacheBlockTokens)*c.cacheBlockTokens:0,0,c.model).cost);
  const saving=coldCost-warmCost-laterExtra;
  let best=0, value=0, cost=0;
  for(let n=1;n<=c.maxPings;n++) {
    const sent=1-returnProbability*gapCdf(n*interval,c);
    cost+=sent*priceCall(k,n===1?cachedHeld:fullCached,c.pingOutputTokens,c.model).cost;
    const bridged=returnProbability*Math.max(0,gapCdf(n*interval+c.ttlMinutes,c)-gapCdf(c.ttlMinutes,c));
    const expectedGain=bridged*saving, net=expectedGain-cost;
    rows.push({pings:n,expectedGain,expectedCost:cost,net});
    if(net>value){value=net;best=n;}
  }
  return {chosen:best,reason:best?'positive expected value':'no positive expected value',coldTokens:coldInput,heldTokens:heldInput,returnProbability,rows};
}
export function planPings(held,full,c,runIndex=0) {return evaluatePingPlan(held,full,c,runIndex).chosen;}

const block=(id,tokens,kind,turn,source=id)=>Object.freeze({id,tokens,kind,turn,source});
function admitted(input,output,c,arm,role){const reserve=arm==='compache'?c.safetyReserve:role==='answer'?Math.min(20000,.25*c.model.context):0;return input<=c.model.maxInput && input+Math.max(output,reserve)<=c.model.context;}
export function runArm(workload, config, arm, {collectDecisions=true}={}) {
  const c=configure(config); if(!['compache','openclaw'].includes(arm)) throw new RangeError('invalid arm');
  const ledger=[], curves=[], decisions=[], roles={}, total={inputTokens:0,cachedTokens:0,freshTokens:0,outputTokens:0,inputCost:0,cachedCost:0,outputCost:0,cost:0,calls:0};
  let completedThreads=0,completedRuns=0,attemptedRuns=0,compactions=0,recalls=0,required=0,covered=0,failedCalls=0,pings=0,totalRemovedTokens=0,totalRecallTokens=0;
  for(const thread of workload.threads) {
    let history=[],snapshots=[],callIndex=0,threadCost=0,summaryIndex=0,failed=false;
    const system=block('system',c.systemTokens,'system',-1);
    function request(blocks,role,time,output,turn) {
      const input=sum(blocks);
      if(!admitted(input,output,c,arm,role)){failedCalls++;ledger.push({arm,thread:thread.id,turn,role,time,inputTokens:input,outputTokens:output,admitted:false});return null;}
      snapshots=snapshots.filter(s=>time-s.time<c.ttlMinutes);
      let prefix=0; for(const s of snapshots) prefix=Math.max(prefix,sharedPrefix(s.blocks,blocks));
      const cached=prefix>=c.cacheMinTokens ? Math.floor(prefix/c.cacheBlockTokens)*c.cacheBlockTokens : 0;
      const costs=priceCall(input,cached,output,c.model);
      const row={arm,thread:thread.id,turn,role,time,admitted:true,...costs}; ledger.push(row); callIndex++;
      const roleTotal=roles[role] ||= {calls:0,inputTokens:0,cachedTokens:0,freshTokens:0,outputTokens:0,inputCost:0,cachedCost:0,outputCost:0,cost:0};
      for(const target of [total,roleTotal]) {target.calls++; for(const k of ['inputTokens','cachedTokens','freshTokens','outputTokens','inputCost','cachedCost','outputCost','cost']) target[k]+=costs[k];}
      threadCost+=costs.cost; if(input>=c.cacheMinTokens) snapshots.push({time,blocks:blocks.slice()});
      return row;
    }
    for(let i=0;i<thread.turns.length;i++) {
      const turn=thread.turns[i], input=block(`i${i}`,turn.inputTokens,'input',i), retrieval=block(`r${i}`,turn.retrievalTokens,'retrieval',i), output=block(`o${i}`,turn.outputTokens,'output',i);
      attemptedRuns++;
      const protectedSources=new Set(turn.dependencies.filter(d=>d.kind==='retrieval').map(d=>`r${d.turn}`));
      let decision=null;
      if(arm==='compache') {
        let added=[];
        for(const source of protectedSources) if(!history.some(b=>b.source===source)) {
          const sourceTurn=thread.turns[Number(source.slice(1))];
          added.push(block(`recall-${i}-${source}`,sourceTurn.retrievalTokens,'retrieval',i,source)); recalls++;
        }
        let view=[system,...history,input,retrieval,...added];
        const before=view.slice(), removed=[];
        const recallTokens=sum(added); totalRecallTokens+=recallTokens;
        snapshots=snapshots.filter(s=>turn.time-s.time<c.ttlMinutes);
        const warm=snapshots.some(s=>sharedPrefix(s.blocks,view)>=c.cacheMinTokens);
        const threshold=(warm?c.warmTrigger:c.target+.02)*c.model.context;
        if(sum(view)>threshold) {
          let changed=false;
          for(let j=0;j<history.length && sum(view)>c.target*c.model.context;) {
            const b=history[j];
            if(b.kind==='retrieval' && !protectedSources.has(b.source)){history.splice(j,1);removed.push(b);changed=true;view=[system,...history,input,retrieval,...added];} else j++;
          }
          if(changed)compactions++;
        }
        const removedTokens=sum(removed); totalRemovedTokens+=removedTokens;
        if(collectDecisions) {const beforeBlocks=before.slice(-16), afterBlocks=view.slice(-16); decision={thread:thread.id,turn:i,beforeTokens:sum(before),afterTokens:sum(view),cacheState:warm?'warm':'cold',triggerTokens:threshold,targetTokens:c.target*c.model.context,removedTokens,removedSources:Object.freeze(removed.slice()),protectedSources:Object.freeze([...protectedSources,retrieval.source]),recallTokens,plannedPings:0,actualPings:0,beforeBlocks:Object.freeze(beforeBlocks),afterBlocks:Object.freeze(afterBlocks),beforeByKind:byKind(before),afterByKind:byKind(view),beforeOmittedCount:before.length-beforeBlocks.length,afterOmittedCount:view.length-afterBlocks.length,beforeOmittedTokens:sum(before)-sum(beforeBlocks),afterOmittedTokens:sum(view)-sum(afterBlocks),planner:null,admitted:null};}
        history.push(input,retrieval,...added);
      } else {
        let view=[system,...history,input,retrieval];
        const threshold=Math.min(.75*c.model.context,c.model.context-Math.min(20000,.25*c.model.context));
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
          const flush=[block('openclaw-flush-system',c.systemTokens,'system',-1),...history,input,retrieval];
          const summary=[block('openclaw-summary-system',c.summarySystemTokens,'system',-1),...projected];
          if(!request(flush,'flush',turn.time,c.flushOutputTokens,i) || !request(summary,'summary',turn.time,c.summaryTokens,i)){failed=true;break;}
          const sequence=++summaryIndex;
          history=[block(`summary-${sequence}`,c.summaryTokens,'summary',-1),...(c.recordTokens?[block(`record-${sequence}`,c.recordTokens,'record',-1)]:[]),...kept]; compactions++;
        }
        history.push(input,retrieval);
      }
      const view=[system,...history];
      required+=turn.dependencies.length;
      for(const d of turn.dependencies) if(view.some(b=>b.source===`${d.kind==='input'?'i':d.kind==='retrieval'?'r':'o'}${d.turn}`)) covered++;
      const row=request(view,'answer',turn.time,turn.outputTokens,i);
      if(!row){if(decision){decision.admitted=false;Object.assign(ledger.at(-1),{beforeTokens:decision.beforeTokens,afterTokens:decision.afterTokens,cacheState:decision.cacheState,triggerTokens:decision.triggerTokens,targetTokens:decision.targetTokens,removedTokens:decision.removedTokens,recallTokens:decision.recallTokens});decisions.push(Object.freeze(decision));}failed=true;break;}
      if(decision)decision.admitted=true;
      completedRuns++; history.push(output);
      if(thread===workload.threads[0]) curves.push({turn:i+1,inputTokens:row.inputTokens,cachedTokens:row.cachedTokens,cost:row.cost,cumulativeCost:threadCost});
      if(arm==='compache') {
        const plan=evaluatePingPlan(view,[system,...history],c,i), count=plan.chosen, next=thread.turns[i+1]?.time ?? Infinity, interval=c.ttlMinutes-c.pingLeadMinutes;
        if(decision){decision.planner=Object.freeze({...plan,rows:Object.freeze(plan.rows.map(r=>Object.freeze({...r})))});decision.plannedPings=count;}
        for(let n=1;n<=count;n++) {const time=turn.time+n*interval;if(time>=next)break;const ping=request([system,...history],'ping',time,c.pingOutputTokens,i);if(!ping){failed=true;break;}pings++;if(decision)decision.actualPings++;}
        if(decision){Object.assign(row,{beforeTokens:decision.beforeTokens,afterTokens:decision.afterTokens,cacheState:decision.cacheState,triggerTokens:decision.triggerTokens,targetTokens:decision.targetTokens,removedTokens:decision.removedTokens,recallTokens:decision.recallTokens,plannedPings:decision.plannedPings,actualPings:decision.actualPings});decisions.push(Object.freeze(decision));}
        if(failed)break;
      }
    }
    if(!failed)completedThreads++;
  }
  return {totals:total,byRole:roles,ledger,decisions,removedTokens:totalRemovedTokens,recallTokens:totalRecallTokens,completedThreads,plannedThreads:workload.threads.length,completedRuns,plannedRuns:workload.totalRuns,attemptedRuns,failedCalls,complete:completedThreads===workload.threads.length,compactions,recalls,pings,rawSourceAvailability:required?covered/required:null,dependencyCoverage:required?covered/required:null,dependencies:{required,covered},curve:curves};
}
export function simulate(overrides={}) {
  const config=configure(overrides), workload=generateWorkload(config);
  const compache=runArm(workload,config,'compache'), openclaw=runArm(workload,config,'openclaw');
  const savings=compache.complete&&openclaw.complete ? {absolute:openclaw.totals.cost-compache.totals.cost,percent:openclaw.totals.cost?100*(openclaw.totals.cost-compache.totals.cost)/openclaw.totals.cost:null} : null;
  const ttlScenarios=[5,30].map(ttlMinutes=>{const a=ttlMinutes===config.ttlMinutes?compache:runArm(workload,{...config,ttlMinutes},'compache',{collectDecisions:false}); const b=ttlMinutes===config.ttlMinutes?openclaw:runArm(workload,{...config,ttlMinutes},'openclaw',{collectDecisions:false});return {ttlMinutes,compache:a.totals.cost,openclaw:b.totals.cost,complete:a.complete&&b.complete,savings:a.complete&&b.complete?b.totals.cost-a.totals.cost:null};});
  const warnings=['Synthetic scenario projection: no actual Compache, OpenClaw or provider execution.', 'Pricing and token counts are assumptions; cache and long-context rules vary by provider.', 'OpenClaw client-summary Chat Completions is approximate: summary input uses a token proxy for character limits; one flush per compaction is modeled, but the independent 2 MB flush trigger is omitted. Native Responses compaction, retries, quality and other hosts are excluded.', 'Raw source availability tracks whether modeled dependency blocks remain verbatim; summaries may preserve facts and answer quality is not measured.', 'Keep-warm uses the configured gap distribution and an expected-value approximation, not a fitted Python policy. The cold preview assumes no future query or dependencies; call duration is modeled as zero and pings can occur after abandonment.', 'Incomplete arms suppress full-workload savings; observed partial costs are not comparable totals.'];
  return {config,workload,compache,openclaw,savings,ttlScenarios,curves:{compache:compache.curve,openclaw:openclaw.curve},warnings};
}
