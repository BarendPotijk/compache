"""Run the installed Compache planner on invented research and support workloads."""

from compache.optimizer import Request, RetentionRecipe, Source
from compache.optimizer.replay import assess
from compache.optimizer.retention import render_recipe
from compache.setup_metadata import evaluate_metadata, fit_metadata
from compache.workflow_presets import example_metadata, example_model_spec


def fit(workflow, spec):
    rows, cutoff = example_metadata(workflow)
    return fit_metadata(rows, spec, observation_end=cutoff)


def compare_costs():
    spec = example_model_spec()
    mixed, _ = fit("general", spec)
    for workflow in ("research", "support"):
        own, profile = fit(workflow, spec)
        heldout, cutoff = example_metadata(workflow, split="heldout")
        print(f"\n{workflow.capitalize()} ({len(heldout)} held-out turns)")
        print(f"Training median: {profile['median_added_tokens']:,} tokens added per turn, "
              f"{profile['median_return_gap_seconds']:.0f} seconds until the next turn.")
        for name, policy in (("Mixed synthetic fit", mixed), ("Workflow-specific fit", own)):
            replay = evaluate_metadata(heldout, policy, observation_end=cutoff)
            if replay["savings_fraction"] is None:
                print(f"{name}: {replay['status']}")
                continue
            print(f"{name}: {replay['baseline_cost']:,.0f} no-ping units, "
                  f"{replay['candidate_cost']:,.0f} fitted units, "
                  f"{replay['ping_calls']} pings, {replay['savings_fraction']:.2%} savings.")


def compare_evidence():
    # Three historical requirements per question, hand-authored for the example.
    spec = example_model_spec()
    examples = {
        "research": (
            "Limit the review to the 2022 agreements.",
            "Signed 2022 addendum A, section 8: deliver the notice by courier.",
            "I marked the renewal clause for review.",
            "Updated document index, version 3: six documents received.",
            "Which agreements did I select, what did you flag, and how must notice be delivered?",
            "r1",
        ),
        "support": (
            "Send the replacement to the Utrecht pickup point.",
            "The original order is under review.",
            "I opened replacement case R41.",
            "Current dispatch estimate: three working days.",
            "Where did I ask you to send it, which case did you open, and what is the current estimate?",
            "r3",
        ),
    }
    print("\nHistorical source availability with only the latest retrieval turn in full")
    recipe = RetentionRecipe(retrieval_runs=1)
    for workflow, (initial, early_doc, answer, recent_doc, question, document_id) in examples.items():
        request = Request(workflow, "q4", 4, spec.model_id, (
            Source("i1", "input", 1, initial),
            Source("r1", "retrieval", 1, early_doc, "first-document", "1", "1"),
            Source("o2", "output", 2, answer),
            Source("r3", "retrieval", 3, recent_doc, "latest-document", "1", "1"),
            Source("i4", "input", 4, question),
        ))
        label = {"status": "complete", "task_requirements": [
            {"requirement_id": source, "historical_context_needed": "yes",
             "alternatives": [{"sources": [{"part_id": source}]}]}
            for source in ("i1", "o2", document_id)
        ]}
        view = render_recipe(request, spec, recipe)
        result = assess(label, view.full_part_ids)
        print(f"{workflow.capitalize()}: {result['historical_requirements_retained']}/"
              f"{result['historical_requirements_assessed']} dependencies retained.")


if __name__ == "__main__":
    compare_costs()
    compare_evidence()
