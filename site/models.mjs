// List prices in US dollars per million tokens and default cache rules, checked against the provider pages on 2026-11-20.
// The package's price sheet, src/compache/models.json, carries the same numbers: models.test.mjs fails when the two drift apart.
// Cache lifetime is a modeled cutoff at the documented default, not a hit guarantee.
const openai={context:1050000,maxInput:922000,writeMultiplier:1.25,longThreshold:272000,longInputMultiplier:2,longOutputMultiplier:1.5};
const claude={context:1000000,maxInput:1000000,writeMultiplier:1.25,longThreshold:0,longInputMultiplier:1,longOutputMultiplier:1};
const oaCache={ttlMinutes:30,cacheMinTokens:1024,cacheBlockTokens:1,longTtlMinutes:720,longWriteMultiplier:2.5,anthropic:false};
// Each provider's long cache and its write price: Anthropic's one-hour lifetime, OpenAI's 12-hour option (read 2026-11-24).
// OpenClaw's Anthropic defaults also clear old tool results on the server; its other providers do not.
const clCache={ttlMinutes:5,cacheMinTokens:512,cacheBlockTokens:1,longTtlMinutes:60,longWriteMultiplier:2,anthropic:true};
export const presets=Object.freeze({
 luna:{id:'gpt-6-luna',name:'GPT-6 Luna',url:'https://developers.openai.com/api/docs/models/gpt-6-luna',cacheUrl:'https://developers.openai.com/api/docs/guides/prompt-caching',...oaCache,model:{...openai,inputRate:.1,cachedRate:.01,outputRate:.5}},
 astra:{id:'gpt-6-astra',name:'GPT-6 Astra',url:'https://developers.openai.com/api/docs/models/gpt-6-astra',cacheUrl:'https://developers.openai.com/api/docs/guides/prompt-caching',...oaCache,model:{...openai,inputRate:10,cachedRate:1,outputRate:50}},
 terra:{id:'gpt-5.6-terra',name:'GPT-5.6 Terra',url:'https://developers.openai.com/api/docs/models/gpt-5.6-terra',cacheUrl:'https://developers.openai.com/api/docs/guides/prompt-caching',...oaCache,model:{...openai,inputRate:2,cachedRate:.2,outputRate:12}},
 opus:{id:'claude-opus-5-5',name:'Claude Opus 5.5',url:'https://platform.claude.com/docs/en/about-claude/pricing',cacheUrl:'https://platform.claude.com/docs/en/build-with-claude/prompt-caching',...clCache,model:{...claude,inputRate:4,cachedRate:.2,outputRate:20}},
 sonnet:{id:'claude-sonnet-5-5',name:'Claude Sonnet 5.5',url:'https://platform.claude.com/docs/en/about-claude/pricing',cacheUrl:'https://platform.claude.com/docs/en/build-with-claude/prompt-caching',...clCache,model:{...claude,inputRate:2,cachedRate:.1,outputRate:10}},
 haiku:{id:'claude-haiku-5-5',name:'Claude Haiku 5.5',url:'https://platform.claude.com/docs/en/about-claude/pricing',cacheUrl:'https://platform.claude.com/docs/en/build-with-claude/prompt-caching',...clCache,model:{...claude,inputRate:.1,cachedRate:.01,outputRate:.5,longThreshold:100000,longInputMultiplier:5,longOutputMultiplier:5}},
 fable:{id:'claude-fable-5-2',name:'Claude Fable 5.2',url:'https://platform.claude.com/docs/en/about-claude/pricing',cacheUrl:'https://platform.claude.com/docs/en/build-with-claude/prompt-caching',...clCache,model:{...claude,inputRate:8,cachedRate:.2,outputRate:40}}
});
// The preset for a calculator key ('sonnet') or for the package's model ID ('claude-sonnet-5-5'). Undefined for anything else.
export const presetKey=name=>Object.hasOwn(presets,name)?name:Object.keys(presets).find(key=>presets[key].id===name);
export function applyPreset(config,id){const p=presets[id];return p?{...config,model:{...p.model},ttlMinutes:p.ttlMinutes,cacheMinTokens:p.cacheMinTokens,cacheBlockTokens:p.cacheBlockTokens,longTtlMinutes:p.longTtlMinutes,longWriteMultiplier:p.longWriteMultiplier,anthropic:p.anthropic}:config;}
