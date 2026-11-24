import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {presets,applyPreset,presetKey} from './models.mjs';
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
 const a=runArm(workload,applyPreset({keepWarm:false},'luna'),{kind:'compache'}),b=runArm(workload,applyPreset({keepWarm:false},'opus'),{kind:'compache'});
 assert.ok(a.ledger[1].cachedTokens>0);assert.equal(b.ledger[1].cachedTokens,0);
});
test('the calculator presets and the package price sheet agree on every number',()=>{
 const sheet=JSON.parse(readFileSync(new URL('../src/compache/models.json',import.meta.url),'utf8')), seen=new Set();
 assert.equal(sheet.checked,'2026-11-20');
 for(const [id,m] of Object.entries(sheet.models)) {
  const p=presets[m.preset];assert.ok(p,`${id} names the preset ${m.preset}, which the calculator does not have`);
  assert.ok(!seen.has(m.preset),`two models name the preset ${m.preset}`);seen.add(m.preset);
  assert.equal(p.id,id);assert.equal(presetKey(id),m.preset);assert.equal(presetKey(m.preset),m.preset);
  assert.deepEqual({name:p.name,url:p.url,input:p.model.inputRate,cacheRead:p.model.cachedRate,output:p.model.outputRate,write:p.model.writeMultiplier,context:p.model.context,maxInput:p.model.maxInput,
    ttlSeconds:p.ttlMinutes*60,cacheMin:p.cacheMinTokens,cacheBlock:p.cacheBlockTokens,longThreshold:p.model.longThreshold,longInput:p.model.longInputMultiplier,longOutput:p.model.longOutputMultiplier},
   {name:m.name,url:m.source,input:m.input_price,cacheRead:m.cache_read_price,output:m.output_price,write:m.cache_write_multiplier,context:m.context_tokens,maxInput:m.max_input_tokens,
    ttlSeconds:m.cache_ttl_seconds,cacheMin:m.cache_min_tokens,cacheBlock:m.cache_block_tokens,longThreshold:m.long_input_threshold??0,longInput:m.long_input_factor,longOutput:m.long_output_factor},id);
  // The longer cache that a policy can ask for. OpenAI's is there to compare with, the package does not ask for it.
  if(p.anthropic)assert.deepEqual(m.cache_lifetimes,[{seconds:p.longTtlMinutes*60,write_multiplier:p.longWriteMultiplier}],id);else assert.equal(m.cache_lifetimes,undefined,id);
  // What depends on the provider: the comparisons it has.
  assert.equal(p.anthropic,m.provider==='anthropic',id);
 }
 assert.deepEqual([...seen].sort(),Object.keys(presets).sort());
 assert.equal(presetKey('custom'),undefined);assert.equal(presetKey('claude-sonnet'),undefined);
});
