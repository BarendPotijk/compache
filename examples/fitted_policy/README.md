# Run the fitted policies offline

These two compressed JSON files are the original five- and 30-minute policy artifacts fitted on 200 fictional RAG threads from 20 synthetic companies, totaling 15,840 turns. They include aggregate demand distributions, header statistics, recipe scores and hashed training thread IDs. Each artifact screened two discretionary compaction recipes. Both failed its admission criteria. The example retains that outcome.

From the root of this checkout, with Python 3.11 or newer:

```sh
PYTHONPATH=src python3 -B examples/fitted_policy/run.py
PYTHONPATH=src python3 -B -m compache planner --help
```

The first command checks the SHA-256 of each decompressed artifact, constructs real `WorkloadPolicy` and `WorkloadSession` objects and prepares two requests that use a fictional harbor notice. The second question arrives six minutes after the first completes. Both policies report 200 training threads, 15,840 training turns, two candidate recipes and zero admitted recipes. The tiny input falls below the model's cacheable prefix minimum. The printed token count uses the package's local estimate. Every answer and time in the demonstration is authored, and no provider receipt or charge is generated.

The artifacts preserve the alternative prices and context limit of the original modeled training scenario. The public long-conversation evaluation adjusted the model specification and runtime options to its live harness. This two-turn example validates the policy format and serving entry points. See the [workload planner guide](../../docs/workload-planner.md) for receipt accounting, cache-window assumptions, zero-ping controls and full-trace integration.
