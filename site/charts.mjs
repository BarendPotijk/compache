import {WINDOW,configure,needLaw,theoreticalSurvival} from './engine.mjs';

// Plot the same surviving cohort in both arms at each turn. Ended or failed
// threads are not padded with zero tokens, and late-turn sample sizes are exposed.
export function contextSeries(result,threadId='average') {
  const answerRows=arm=>arm.ledger.filter(r=>r.role==='answer'&&r.admitted&&(threadId==='average'||r.thread===threadId));
  const a=answerRows(result.compache),b=answerRows(result.baseline.arm),key=r=>`${r.thread}:${r.turn}`;
  const bKeys=new Set(b.map(key)),aKeys=new Set(a.map(key));
  const matched=[a.filter(r=>bKeys.has(key(r))),b.filter(r=>aKeys.has(key(r)))];
  return matched.map(rows=>{
    const grouped=new Map();
    for(const row of rows){const group=grouped.get(row.turn)||[];group.push(row);grouped.set(row.turn,group);}
    return [...grouped.entries()].sort(([a],[b])=>a-b).map(([turn,group])=>({turn,count:group.length,inputTokens:group.reduce((n,r)=>n+r.inputTokens,0)/group.length,cachedTokens:group.reduce((n,r)=>n+r.cachedTokens,0)/group.length}));
  });
}

// What one kind's window keeps of what later turns need of it, over the turns of the priced threads.
export function coverage(result,kind) {
  const window=result.policy[WINDOW[kind]],law=needLaw(result.config,kind,Math.max(result.config.maxTurns,...result.workload.threads.map(t=>t.turns.length)));
  let held=0,all=0;
  for(const t of result.workload.threads)for(let i=1;i<t.turns.length;i++){all++;held+=window===null?1:law.within(window,i);}
  return all?held/all:1;
}

export function survivalView(config) {
  const c=configure(config),all=theoreticalSurvival(c),oneTurn=1-(all[1]?.realized??0);
  if(c.lengthDrawn || c.lengthDistribution==='continuation') {
    // By chance of another turn, the median and the 95th percentile of the length are read off the curve.
    const within=share=>{let turn=1;while(turn<c.maxTurns&&all[turn].realized>1-share)turn++;return turn;};
    const max=Math.min(c.maxTurns,Math.max(3,Math.ceil(within(.95)*1.15)+1));
    return {max,reference:within(.5),referenceLabel:'median',points:all.filter(p=>p.turn<=max),tail:all[max-1]?.realized??0,oneTurn};
  }
  const length95=c.lengthDistribution==='fixed'?c.plannedTurns:c.lengthDistribution==='geometric'?(c.plannedTurns===1?1:Math.ceil(Math.log(.05)/Math.log(1-1/c.plannedTurns))):c.plannedTurnsP95;
  const max=Math.min(c.maxTurns,Math.max(3,Math.ceil(length95*1.15)+1));
  return {max,reference:Math.min(c.maxTurns,c.plannedTurns),referenceLabel:c.lengthDistribution==='geometric'?'mean':c.lengthDistribution==='fixed'?'length':'median',points:all.filter(p=>p.turn<=max),tail:all[max-1]?.realized??0,oneTurn};
}

// Averages of eligible per-prefix plans after the selected run, not a promise
// that each thread receives this many pings. Individual plans remain inspectable.
export function pingProfile(result,threadId,turn) {
  const decisions=result.compache.decisions.filter(d=>d.turn===turn&&(threadId==='average'||d.thread===threadId)&&d.admitted);
  const eligible=decisions.filter(d=>d.planner?.rows.some(r=>r.pings>0));
  const rows=[];
  for(let n=0;n<=result.config.maxPings;n++) {
    if(n>0&&!eligible.length)break;
    const source=eligible.map(d=>d.planner.rows.find(r=>r.pings===n)).filter(Boolean);
    const expectedGain=source.length?source.reduce((s,r)=>s+r.expectedGain,0)/source.length:0;
    const expectedCost=source.length?source.reduce((s,r)=>s+r.expectedCost,0)/source.length:0;
    rows.push({pings:n,expectedGain,expectedCost,net:expectedGain-expectedCost});
  }
  let best=rows[0];for(const row of rows)if(row.net>best.net)best=row;
  const mean=(items,read)=>items.length?items.reduce((n,d)=>n+read(d),0)/items.length:null;
  const reasons={};for(const d of decisions)if(!eligible.includes(d)){const reason=d.planner?.reason||'no plan';reasons[reason]=(reasons[reason]||0)+1;}
  return {rows,best,decisions:decisions.length,eligible:eligible.length,reasons,returnProbability:mean(decisions,d=>d.planner?.returnProbability||0),plannedPings:mean(decisions,d=>d.plannedPings),actualPings:mean(decisions,d=>d.actualPings),interval:result.config.ttlMinutes-result.config.pingLeadMinutes,ttl:result.config.ttlMinutes};
}
