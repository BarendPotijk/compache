import test from 'node:test';
import assert from 'node:assert/strict';
import {presets,applyPreset} from './models.mjs';
import {configure,priceCall,runArm} from './engine.mjs';
test('provider presets bind cache defaults and remove legacy model',()=>{
 assert.equal(presets.gpt41,undefined);
 for(const id of ['luna','astra','terra']){const c=configure(applyPreset({},id));assert.equal(c.ttlMinutes,30);assert.equal(c.cacheMinTokens,1024);assert.equal(c.cacheBlockTokens,1);}
 for(const id of ['opus','sonnet','haiku','fable']){const c=configure(applyPreset({},id));assert.equal(c.ttlMinutes,5);assert.equal(c.cacheMinTokens,512);assert.equal(c.cacheBlockTokens,1);}
 assert.equal(applyPreset({ttlMinutes:5},'astra').ttlMinutes,30);
});
test('Haiku tiers include cached tokens and apply to the entire request',()=>{
 const m=presets.haiku.model;
 const low=priceCall(100000,90000,1000,m),high=priceCall(100001,90000,1000,m);
 assert.equal(low.long,false);assert.equal(high.long,true);
 assert.equal(high.inputCost,10001*.625/1e6);assert.equal(high.cachedCost,90000*.05/1e6);assert.equal(high.outputCost,1000*2.5/1e6);
 for(const id of ['opus','sonnet','fable'])assert.equal(priceCall(900000,0,1000,presets[id].model).long,false);
});
test('identical six-minute gap is warm for Luna and cold for Claude',()=>{
 const workload={totalRuns:2,threads:[{id:'x',turns:[0,1].map(i=>({index:i,time:i*6,inputTokens:100,retrievalTokens:2000,outputTokens:100,dependencies:[]}))}]};
 const a=runArm(workload,applyPreset({keepWarm:false},'luna'),'compache'),b=runArm(workload,applyPreset({keepWarm:false},'opus'),'compache');
 assert.ok(a.ledger[1].cachedTokens>0);assert.equal(b.ledger[1].cachedTokens,0);
});
