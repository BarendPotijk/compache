# Retention: the keep-warm planner

When a run ends with an answer, decide from token counts and a return curve whether to ping, lapse, or force the next start cold. Nothing of the conversation is persisted for this.

## The rule

| Symbol | Meaning |
|---|---|
| `C` | What the provider holds: the last request's input less its request-scoped share |
| `A` | The final answer |
| `K` | `C + A`: what pings keep warm and a warm return reads |
| `S` | The shared part. A cold start reads it too, so it cancels |
| `V` | The view the gate's cold plan would show |
| `read`, `write` | Cache prices as multiples of base input |
| `F(t)` | P(next run starts within `t` s of the last request's start) |
| `interval` | TTL − ping lead |

```text
warm_saving = write × max(0, V − S)
            − read  × max(0, K − S)
            − read  × max(0, K − V) × later_calls

bridged(n)  = F(n × interval + TTL) − F(TTL)
sent(i)     = 1 − F(i × interval)

value(n)    = bridged(n) × warm_saving
            − sent(1) × (read × C + write × A)
            − read × K × Σ[i=2..n] sent(i)

if interval > 0 and K ≤ ceiling:
    n* = argmax value(n) for n in 1..max_pings
    if value(n*) > 0: keep_warm(n*)
if cold plan reduces turns and warm_saving < 0: compact
otherwise: lapse
```

A warm return saves the write of the cold view, less the read of the kept prefix, less the extra re-reads of a prefix a cold plan would have shrunk. Pings stop once the user returns, so each is weighted by the chance it's sent. The first ping writes the answer behind the prefix; later ones read it.

Above the ceiling (warm trigger less reserve), a warm start would compact anyway, so a ping buys nothing. `compact` only changes a return inside the TTL, and its band is narrow; an implementation may drop it and lapse.

## Test vectors

Synthetic curve, `last_call_seconds` 30, `later_calls` 3, TTL 300, lead 45, write 1.25:

| Idle seconds | 0–300 | –600 | –1,800 | –3,600 | –86,400 | –604,800 | Never |
|---|---|---|---|---|---|---|---|
| Probability | 0.40 | 0.10 | 0.10 | 0.05 | 0.10 | 0.05 | 0.20 |

**A:** C = 100,000, S = 10,000, A = 0, V = C.

| Pings | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| read 0.05 | 8,920 | 10,540 | 10,391 | 10,349 | 10,412 | **10,582** |
| read 0.10 | **4,903** | 3,643 | 954 | −1,521 | −3,784 | −5,835 |

**B, a long answer:** C = 50,000, A = 32,000, S = 10,000, V = 82,000, read 0.05. One ping is worth −19,814, and every value is negative, so it lapses.

**C, barely above static:** C = 15,000, A = 0, S = 10,000, V = C. At read 0.10 one ping is worth −389, so it lapses. At read 0.05 it's worth +165, so one ping.

## The return curve

A histogram of idle seconds per bin, plus a "no return" mass, interpolated linearly within a bin, with `idle = t − last_call_seconds`. Use rows by the run's index in its conversation: later runs tend to return sooner and more often.

It describes one product's users, so compache ships none. Fit it from your own timings, along with `last_call_seconds` and `later_calls` (calls per later run, minus one). A client "user is still here" signal would beat any richer model.

## Input traps

All three are easy to reintroduce, and all come from pricing the inputs wrongly rather than from the rule:

1. Pricing the whole bill as held. Request-scoped context sits in no prefix, so the first ping reads as a miss.
2. Ignoring the answer. A long answer behind a short prefix must be priced out.
3. Crediting the whole standing part as shared. Dynamic instructions aren't.

Take `V` from the gate's own cold plan, sized the way the gate sizes a view, with no request sent. Check for a newer run after reading the inputs, not before: only then does the digest still name this run's last request.

## Pings

**Capture.** Keep the last completed main-agent request per run in memory, exactly as sent, with its start time: bounded to 64 entries and one hour. Every other exit releases it. A worker restart drops the plan, by design. A response ending in a tool call or in thinking alone plans nothing.

**Send.** Replay the history without request-scoped messages, then the answer, then a request holding a cache point and a one-character prompt. Use the run's own settings, thinking included, since changing them voids the cache, with `max_tokens: 0`.

**Loop.** Each ping is due `interval` after the previous refresh, the first after the last request's start. Planning runs in the background and never blocks or fails a run.

| Outcome | Meaning | Continues? |
|---|---|---|
| `sent` | Read at least half the held prefix, no output billed, refresh recorded | Yes |
| `conversation_moved_on` | A newer run exists, or the refresh couldn't be recorded against the plan's digest | No |
| `expired` | Couldn't start the freshness margin before expiry | No |
| `prefix_missed` | Reads under half of held: the bytes differ or the cache is gone | No |
| `output_billed` | The gateway ignored `max_tokens: 0` | No |
| `failed` | Provider or transport error | No |

**Gate read-back.** The plan is tied to the request it followed by view digest. A sent ping records `refreshed_at`, which the window classifier uses as last use. `compact` makes the next start cold. A newer request's usage record clears the plan, so it never outlives its gap, and a late write against an old digest is refused.
