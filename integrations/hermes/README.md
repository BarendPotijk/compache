# compache for Hermes

Long sessions get cheaper. This context engine leaves out of each request what later turns rarely need, by a policy fitted to conversations like yours, and only once the prompt cache has expired: then the history has to be paid for in full anyway. The conversation Hermes keeps is untouched, and the model fetches a trimmed part back with a `reopen` tool.

## Set up

```sh
hermes plugins install BarendPotijk/compache/integrations/hermes
compache default --model claude-sonnet-5-5 -o ~/.hermes/compache.json
echo "COMPACHE_POLICY=$HOME/.hermes/compache.json" >> ~/.hermes/.env
```

```yaml
# ~/.hermes/config.yaml
context:
  engine: compache
prompt_caching:
  cache_ttl: 5m        # or 1h, where the policy was fitted for one-hour entries
```

`compache default` is a start. [Fit your own](https://github.com/BarendPotijk/compache#get-started) to save more.

## What it touches

| | |
|---|---|
| Reads | The policy file in `COMPACHE_POLICY` |
| Writes | Nothing, unless `COMPACHE_RECORD` names a folder. Then sizes, times and token counts of every turn, never text |
| Network, model calls, telemetry, credentials | None |
| Background threads or processes | None |
| Tools | `reopen`, once something has been trimmed. It returns that text from the conversation |
| When a conversation outgrows the model | Hermes' own compressor |
| Installs | `compache` from PyPI, which has no dependencies |

Needs Hermes 0.23 or later. [Other ways in](https://github.com/BarendPotijk/compache/blob/main/docs/integrations.md), with keep-warm pings and a policy learned from your own sessions: `compache proxy`.
