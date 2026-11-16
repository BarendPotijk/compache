# Native snapshot comparison

Three fresh Compache 0.4.0 runs are compared with three unchanged historical OpenCode recordings. Runtime tool declarations differ; see baseline-provenance.json and README.md. Credit equivalents use standard rates for reported usage; they are not a subscription billing statement.

| Arm | Input | Physical cached input | Output | Reported usage credit equivalent |
|---|---:|---:|---:|---:|
| opencode | 78,982,362 | 2,654,464 | 15,693 | 191.6795 |
| compache | 56,513,547 | 5,412,864 | 15,192 | 129.2948 |

Additional Codex invocations within successful logical calls: 22 (included in measured totals and modeled scenarios). Their correction tails are not visible; modeled cache reuse assumes the same snapshot blocks.

Discarded transport attempts are retry overhead, separate from package recovery. Combined totals include successful calls and retries; setup and controller activity are incomplete.

| Arm | Retry model calls | Retry input | Retry cached | Retry output | Retry credits | Combined credits |
|---|---:|---:|---:|---:|---:|---:|
| opencode | 2 | 1,286,566 | 652,416 | 107 | 1.7498 | 193.4293 |
| compache | 0 | 0 | 0 | 0 | 0.0000 | 129.2948 |
| Both | 2 | 1,286,566 | 652,416 | 107 | 1.7498 | 322.7242 |

Modeled prefix-cache savings are counterfactual; physical cache behavior differs because each snapshot uses a fresh ephemeral thread.

| TTL | OpenCode credits | Compache credits | Savings | Savings % |
|---|---:|---:|---:|---:|
| 300 seconds | 36.5457 | 28.4093 | 8.1364 | 22.3% |
| 300s, replicate 1 | 12.1098 | 9.5392 | 2.5707 | 21.2% |
| 300s, replicate 2 | 12.3208 | 9.4296 | 2.8912 | 23.5% |
| 300s, replicate 3 | 12.1151 | 9.4405 | 2.6746 | 22.1% |
| 1800 seconds | 32.2125 | 28.4093 | 3.8032 | 11.8% |
| 1800s, replicate 1 | 10.6632 | 9.5392 | 1.1240 | 10.5% |
| 1800s, replicate 2 | 10.8739 | 9.4296 | 1.4443 | 13.3% |
| 1800s, replicate 3 | 10.6753 | 9.4405 | 1.2349 | 11.6% |

| Run | Answer / summary / ping / recovery calls | Max input | Wall seconds | Missing requirements | Rewrites | Plain text / unknown format |
|---|---:|---:|---:|---:|---:|---:|
| opencode-1 | 72/1/0/0 | 799,519 | 512.7 | not assessed | 0 | 14 / 0 |
| opencode-2 | 72/1/0/0 | 799,405 | 493.0 | not assessed | 0 | 12 / 8 |
| opencode-3 | 72/1/0/0 | 814,596 | 457.7 | not assessed | 0 | 10 / 0 |
| compache-1 | 72/0/0/0 | 408,563 | 457.0 | 0 | 5 | 4 / 0 |
| compache-2 | 72/0/0/0 | 408,451 | 811.3 | 0 | 5 | 12 / 0 |
| compache-3 | 72/0/0/0 | 408,439 | 442.7 | 0 | 5 | 11 / 0 |

Recovery is included in the primary totals. The separate recovery exclusion removes direct calls only and does not model later carryover.

- Physical cached counters are observed for fresh ephemeral snapshots; modeled prefix reuse is not observed cache behavior.
- Runtime input overhead is included in usage but cannot be split exactly.
- Additional provider invocations use the same modeled snapshot blocks and logical time; their native correction tails are not observable.
- The direct recovery exclusion does not remove downstream carryover.
- OpenCode is a source-derived adapter, not the native executable.
