# Headroom comparison fixture

The frozen results replay Headroom 0.44.0's native lossless library against the recommended Compache policy on the existing RAG development fixture. They do not evaluate Headroom's full proxy or generate new answers.

- `results/summary.json`: scope, provenance, cohort exclusions and aggregate results.
- `results/heldout-all.json`: all 12 held-out threads, including one with unavailable text; savings are unavailable.
- `results/heldout-complete-text.json`: 11 complete-text threads / 49 runs, with per-run accounting.
- `results/policy.json`, `results/profile.json`, `results/fit-report.json`: the training-derived policy, workload profile and candidate evaluation.
- `results/positive-control.json`: a separate structured JSON compression check, excluded from workload results.
- `results/environment.txt`: the executed environment's pinned packages.

See [comparison scopes and reproduction](../../docs/comparisons.md). Recreate the artifacts with `python scripts/compare_headroom.py --output ./work/headroom-comparison` after installing the documented isolated environment. Costs are modeled; compression is executed locally with no provider calls.
