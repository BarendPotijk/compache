import test from 'node:test';
import assert from 'node:assert/strict';
import {contextSeries,survivalView,pingProfile,ticks} from './charts.mjs';
import {DEFAULTS} from './engine.mjs';
const row=(thread,turn,inputTokens,admitted=true)=>({thread,turn,inputTokens,cachedTokens:0,admitted,role:'answer'});
test('chart axes terminate for unavailable forecasts and have a usable range at zero cost', () => {
  for (const bound of [NaN, Infinity, -Infinity]) assert.deepEqual(ticks(0, bound).values, [0, 1]);
  assert.deepEqual(ticks(0, 0).values, [0, .25, .5, .75, 1]);
  assert.deepEqual(ticks(-.31, .45, 4).values, [-.4, -.2, 0, .2, .4, .6]);
});
test('context average uses the same surviving cohort, without zero-padding ended threads',()=>{
 const r={compache:{ledger:[row('a',0,10),row('b',0,30),row('a',1,50),row('b',1,99,false)]},baseline:{arm:{ledger:[row('a',0,20),row('b',0,40),row('a',1,70),row('b',1,90)]}}};
 const [a,b]=contextSeries(r);assert.deepEqual(a.map(x=>[x.turn,x.count,x.inputTokens]),[[0,2,20],[1,1,50]]);assert.deepEqual(b.map(x=>[x.count,x.inputTokens]),[[2,30],[1,70]]);
 assert.equal(contextSeries(r,'b')[0].length,1);
});
test('survival axis responds to short median and tail rather than a fixed 160 runs',()=>{
 const short=survivalView({plannedTurns:8,plannedTurnsP95:20}),long=survivalView({plannedTurns:80,plannedTurnsP95:140});
 assert.ok(short.max<30);assert.equal(short.reference,8);assert.equal(long.max,160);assert.equal(short.points[0].realized,1);
 const fixed=survivalView({lengthDistribution:'fixed',plannedTurns:5});assert.ok(fixed.max<10);assert.equal(fixed.points[5].realized,0);
});
test('by chance of another turn the survival view reads its median and its reach off the curve',()=>{
 // Half the threads end after one turn. 95% are over by turn 8, so the axis runs to turn 11, which 3.6% reach.
 const v=survivalView({lengthDistribution:'continuation',continueFirst:.5,continueLong:.97,continueHalfTurn:4});
 assert.equal(v.oneTurn,.5);assert.equal(v.reference,1);assert.equal(v.referenceLabel,'median');assert.equal(v.max,11);assert.equal(v.points.length,11);
 assert.ok(Math.abs(v.tail-.0356)<.0005);assert.equal(v.points[0].realized,1);assert.equal(v.points[1].realized,.5);
 // A steady nine in ten: half are over by turn 7 (0.9^7 = 0.48) and 95% by turn 29.
 const steady=survivalView({lengthDistribution:'continuation',continueFirst:.9,continueLong:.9,continueHalfTurn:3});
 assert.equal(steady.reference,7);assert.equal(steady.max,35);assert.ok(Math.abs(steady.oneTurn-.1)<1e-12);assert.ok(Math.abs(steady.tail-.9**34)<1e-12);
 // Threads that always go on reach the cap.
 const endless=survivalView({lengthDistribution:'continuation',continueFirst:1,continueLong:1,maxTurns:40});
 assert.equal(endless.max,40);assert.equal(endless.reference,40);assert.equal(endless.tail,1);assert.equal(endless.oneTurn,0);
 // The other laws report the share of one-turn threads as well.
 assert.ok(Math.abs(survivalView({lengthDistribution:'geometric',plannedTurns:4,continuationProbability:1}).oneTurn-.25)<1e-12);
});
test('ping average excludes ineligible prefixes and exposes both eligible and total counts',()=>{
 const rows=[{pings:0,expectedGain:0,expectedCost:0,net:0},{pings:1,expectedGain:5,expectedCost:2,net:3},{pings:2,expectedGain:6,expectedCost:4,net:2}];
 const d=(thread,eligible)=>({thread,turn:0,admitted:true,plannedPings:eligible?1:0,actualPings:0,planner:{returnProbability:.5,reason:eligible?'positive expected value':'disabled',rows:eligible?rows:rows.slice(0,1)}});
 const r={config:{...DEFAULTS,maxPings:2},compache:{decisions:[d('a',true),d('b',false)]}};
 const p=pingProfile(r,'average',0);assert.equal(p.decisions,2);assert.equal(p.eligible,1);assert.equal(p.best.pings,1);assert.equal(p.best.net,3);assert.equal(p.plannedPings,.5);
 assert.equal(pingProfile(r,'b',0).eligible,0);assert.deepEqual(pingProfile(r,'b',0).rows,[rows[0]]);
});
