# Run Compache locally

These examples use the sample conversations in this checkout. Start at the repository root with Python 3.11 or later. Installation may download packages. Once installed, every command below runs without a provider account or network connection.

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
compache demo -o policy.json
```

The demo fits a policy on four conversations (71 turns) and writes it to `policy.json` in the current directory. Read the first line of the decision: it keeps questions from the last four turns, answers from the last nine and retrieved text from the last turn. The reported costs are modeled on the same sample used for fitting. The held-out example below evaluates a separate conversation.

## Put a policy in a chatbot request

Keep the generated `policy.json` in the current directory. Install the SDK version used by the offline example, then run the chatbot.

```sh
python -m pip install 'anthropic==1.17.0'
python examples/quickstart/wrapped_chatbot.py
```

```text
Application history: 12 user turns
Hinge number in original message: True
Hinge number in recall request before reopen: False
Reopen calls: 1
Provider requests: 13 observed, 13 counted by Compache
Answer: Ticket PKG-104 needs hinge H-772.
```

The first message records the replacement hinge for ticket PKG-104. Ten inspections later, the user asks for the hinge number. The application still holds the first message, but the request reaching the model contains only an archive notice in its place. The scripted model calls `reopen("PKG-104")`. Compache sends the original text back in a tool result, and the model reads the answer from that text. This accounts for the thirteenth request. [Wrapped chatbot source](quickstart/wrapped_chatbot.py)

In your app, wrap the SDK client once, pass your conversation ID with `compache.thread`, and send your usual history. The example uses long inspection notes and a two-hour clock step to trigger a cut. The endpoint scripts tool use and returns synthetic usage. Pings are disabled. Measuring cache hits, billing or answer quality requires a real provider.

## Fit and check an unseen conversation

```sh
python examples/quickstart/fit_replay.py
```

```text
Fit: 3 threads, 46 turns
Training retention: 85.7% (minimum 80.0%)
Held out: 1 thread, 25 turns
Historical requirements retained: 5/8
Modeled USD per 1,000 turns, policy: 14.54
Modeled USD per 1,000 turns, full history: 50.17
```

The script splits the four sample conversations by thread and assigns each label to the same split as its conversation. It fits a fresh policy in memory and replays the held-out thread against both that policy and full history. Five of the eight assessed historical requirements stay in view before recovery (62.5%), below the 80% training target. The lower modeled cost includes modeled recovery. Both cost figures scale the same 25 turns to 1,000. For your evaluation, reserve whole conversations and check quality on answers your users actually receive. [Fit and replay source](quickstart/fit_replay.py)

## See each turn or own the tool loop

Both scripts below read the `policy.json` produced by the demo.

| Run from the repository root | What to look for |
| --- | --- |
| `python examples/quickstart/apply.py` | Replays a 26-turn recorded conversation. The last prepared view has 18,643 estimated tokens against 89,285 for full history, with 164 archived parts. The script takes each answer from the recording. [Source](quickstart/apply.py) |
| `python examples/quickstart/app_managed.py` | The host sends a lookup request and an answer request per user turn, passes synthetic usage receipts to the session, and records which retrieved part the answer cited. Expect “It ships Friday.”, “Changes close Thursday.” and four calls recorded by the application. [Source](quickstart/app_managed.py) |

Use the [developer guide](../docs/guide.md) to take the next step with your own traffic.
