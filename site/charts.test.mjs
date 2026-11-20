import test from 'node:test';
import assert from 'node:assert/strict';
import {contextSeries,survivalView,pingProfile} from './charts.mjs';
import {DEFAULTS} from './engine.mjs';
const row=(thread,turn,inputTokens,admitted=true)=>({thread,turn,inputTokens,cachedTokens:0,admitted,role:'answer'});
test('context average uses the same surviving cohort, without zero-padding ended threads',()=>{
 const r={compache:{ledger:[row('a',0,10),row('b',0,30),row('a',1,50),row('b',1,99,false)]},openclaw:{ledger:[row('a',0,20),row('b',0,40),row('a',1,70),row('b',1,90)]}};
 const [a,b]=contextSeries(r);assert.deepEqual(a.map(x=>[x.turn,x.count,x.inputTokens]),[[0,2,20],[1,1,50]]);assert.deepEqual(b.map(x=>[x.count,x.inputTokens]),[[2,30],[1,70]]);
 assert.equal(contextSeries(r,'b')[0].length,1);
});
test('survival axis responds to short median and tail rather than a fixed 160 runs',()=>{
 const short=survivalView({plannedTurns:8,plannedTurnsP95:20}),long=survivalView({plannedTurns:80,plannedTurnsP95:140});
 assert.ok(short.max<30);assert.equal(short.reference,8);assert.equal(long.max,160);assert.equal(short.points[0].realized,1);
 const fixed=survivalView({lengthDistribution:'fixed',plannedTurns:5});assert.ok(fixed.max<10);assert.equal(fixed.points[5].realized,0);
});
test('ping average excludes ineligible prefixes and exposes both eligible and total counts',()=>{
 const rows=[{pings:0,expectedGain:0,expectedCost:0,net:0},{pings:1,expectedGain:5,expectedCost:2,net:3},{pings:2,expectedGain:6,expectedCost:4,net:2}];
 const d=(thread,eligible)=>({thread,turn:0,admitted:true,plannedPings:eligible?1:0,actualPings:0,planner:{heldTokens:100,coldTokens:50,returnProbability:.5,reason:eligible?'positive expected value':'disabled',rows:eligible?rows:rows.slice(0,1)}});
 const r={config:{...DEFAULTS,maxPings:2},compache:{decisions:[d('a',true),d('b',false)]}};
 const p=pingProfile(r,'average',0);assert.equal(p.decisions,2);assert.equal(p.eligible,1);assert.equal(p.best.pings,1);assert.equal(p.best.net,3);assert.equal(p.plannedPings,.5);
 assert.equal(pingProfile(r,'b',0).eligible,0);assert.deepEqual(pingProfile(r,'b',0).rows,[rows[0]]);
});
