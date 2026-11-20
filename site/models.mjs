// Standard API prices checked against provider documentation on 2026-11-17.
// Cache lifetime is a modeled cutoff at the documented default, not a hit guarantee.
const openai={context:1050000,maxInput:922000,writeMultiplier:1.25,longThreshold:272000,longInputMultiplier:2,longOutputMultiplier:1.5};
const claude={context:1000000,maxInput:1000000,writeMultiplier:1.25,longThreshold:0,longInputMultiplier:1,longOutputMultiplier:1};
const oaCache={ttlMinutes:30,cacheMinTokens:1024,cacheBlockTokens:1};
const clCache={ttlMinutes:5,cacheMinTokens:512,cacheBlockTokens:1};
export const presets=Object.freeze({
 luna:{name:'GPT-6 Luna',url:'https://developers.openai.com/api/docs/models/gpt-6-luna',cacheUrl:'https://developers.openai.com/api/docs/guides/prompt-caching',...oaCache,model:{...openai,inputRate:.1,cachedRate:.01,outputRate:.5}},
 astra:{name:'GPT-6 Astra',url:'https://developers.openai.com/api/docs/models/gpt-6-astra',cacheUrl:'https://developers.openai.com/api/docs/guides/prompt-caching',...oaCache,model:{...openai,inputRate:10,cachedRate:1,outputRate:50}},
 terra:{name:'GPT-5.6 Terra',url:'https://developers.openai.com/api/docs/models/gpt-5.6-terra',cacheUrl:'https://developers.openai.com/api/docs/guides/prompt-caching',...oaCache,model:{...openai,inputRate:2,cachedRate:.2,outputRate:12}},
 opus:{name:'Claude Opus 5.5',url:'https://platform.claude.com/docs/en/about-claude/pricing',cacheUrl:'https://platform.claude.com/docs/en/build-with-claude/prompt-caching',...clCache,model:{...claude,inputRate:4,cachedRate:.2,outputRate:20}},
 sonnet:{name:'Claude Sonnet 5.5',url:'https://platform.claude.com/docs/en/about-claude/pricing',cacheUrl:'https://platform.claude.com/docs/en/build-with-claude/prompt-caching',...clCache,model:{...claude,inputRate:2,cachedRate:.1,outputRate:10}},
 haiku:{name:'Claude Haiku 5.5',url:'https://platform.claude.com/docs/en/about-claude/pricing',cacheUrl:'https://platform.claude.com/docs/en/build-with-claude/prompt-caching',...clCache,model:{...claude,inputRate:.1,cachedRate:.01,outputRate:.5,longThreshold:100000,longInputMultiplier:5,longOutputMultiplier:5}},
 fable:{name:'Claude Fable 5.2',url:'https://platform.claude.com/docs/en/about-claude/pricing',cacheUrl:'https://platform.claude.com/docs/en/build-with-claude/prompt-caching',...clCache,model:{...claude,inputRate:8,cachedRate:.2,outputRate:40}}
});
export function applyPreset(config,id){const p=presets[id];return p?{...config,model:{...p.model},ttlMinutes:p.ttlMinutes,cacheMinTokens:p.cacheMinTokens,cacheBlockTokens:p.cacheBlockTokens}:config;}
