# Compare

What other tools make of [DocChat](../docchat/README.md)'s held-out conversations, and what their requests would cost. Results on [Other tools compared](../../docs/comparisons.md).

![Cost per 1,000 turns of document chats, by tool](../../docs/assets/compare-cost.svg)

| Arm | How it is priced |
|---|---|
| Whole history, default cache | Every request whole, five-minute entries, no pings |
| Headroom | `compress()` was run on every turn's retrieved text. What it left is sent and cached like the whole history |
| Kompressor | Its `compress` command on the same text, under its rule for the cache: five-minute entries until a conversation first pauses past five minutes, one-hour entries from then on |
| Whole history, cache kept warm | The cheapest of ten settings: five-minute and one-hour entries with up to 0, 1, 3, 6 or 8 pings, sent when compache's planner finds that one pays |
| compache | Fitted on the 15 training threads, cache lifetime included, and replayed on the 15 held-out ones, fetching back included |

No model is called. Prices are Claude Sonnet 5.5's on 2026-11-20, a one-hour entry costs twice the input price to write. The whole history is priced twice, here and by compache's replay, and the two have to agree.

```sh
export PYTHONPATH=src

# In an environment with headroom-ai[proxy]. Without --text-model it is the base install.
python benchmarks/compare/run.py shrink --tool headroom --text-model --target-ratio 0.5 \
  --name "Headroom, target ratio 0.5" --data docchat-mixed --output headroom.json

python benchmarks/compare/run.py shrink --tool kompressor --binary ./kompressor \
  --name Kompressor --data docchat-mixed --output kompressor.json

python benchmarks/compare/run.py price --data docchat-fast docchat-mixed docchat-slow \
  --sizes headroom.json kompressor.json --output docs/assets/data/compare.json
python scripts/figures.py compare
```

## Live

![Bill per 1,000 turns and right answers, by tool](../../docs/assets/compare-live.svg)

The held-out conversations of the [live runs](../live/README.md), through the tool loop of Anthropic's guide on Claude Haiku 5.5, at their users' pace. Everything a tool sends is recorded on its way to the provider.

| Arm | What goes out |
|---|---|
| Whole history | With the cache mark of the guide, and with one that asks for an hour |
| Headroom | `compress()` in process before every request, at `target_ratio=0.5` |
| Kompressor | Its proxy, with the settings its package ships for Claude. It holds no key: the harness puts it in behind it |
| compache | Its proxy, under the policy that `benchmarks/live/run.py fit` wrote |
| Headroom and compache | Both |

```sh
# In an environment with headroom-ai[proxy], anthropic and openai. Everything after --policy is as in benchmarks/live.
python benchmarks/compare/live.py play --work runs --data docchat-mixed --env .env \
  --policy runs/policy-claude-haiku-5-5-tool-backward.json --kompressor ./kompressor
python benchmarks/compare/live.py report --work runs --figure docs/assets/data/compare-live.json
python scripts/figures.py tools
```

Five arms took an hour and $0.55, grading included. The whole history on its default cache is the one of the live runs.

## Notes

`--shape json` hands the passages over as a JSON list. Headroom's text model took some 35 minutes for the 625 turns. The output holds sizes only, never text.
