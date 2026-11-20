"""Public workflow descriptions and reproducible synthetic metadata for an offline trial."""

from datetime import datetime, timedelta, timezone
from random import Random

from compache.optimizer.contracts import ModelSpec
from compache.traces import ContractError


WORKFLOW_NAMES = ("general", "research", "support")

_DESCRIPTIONS = {
    "general": (
        "General assistant",
        "Start here for mixed or unfamiliar workloads.",
        "Compare the time between turns with the cache lifetime.",
        "Measure the tokens added by each turn and how many model calls reuse the same history.",
        "A follow-up may rely on an earlier question, answer or retrieved source.",
    ),
    "research": (
        "Document research",
        "A researcher reads retrieved documents and asks follow-up questions about evidence.",
        "Long model calls and time spent reading can bring the next request past cache expiry.",
        "Each retrieval can add substantial source text to the thread.",
        "A later question may need the exact passage, version or citation retrieved several turns earlier.",
    ),
    "support": (
        "Customer support",
        "A customer and a support assistant exchange short messages about a case.",
        "Replies within an active exchange may arrive while the cache is still warm.",
        "A short lookup can add little context. Tool follow-ups can still reread the existing prefix.",
        "Order details and corrected instructions may remain necessary after earlier lookup results expire.",
    ),
}


def describe_workflow(name="general"):
    """Describe what to examine. Selecting a workflow grants no data access."""
    if name not in _DESCRIPTIONS:
        raise ContractError("workflow must be general, research or support")
    title, description, timing, growth, dependency = _DESCRIPTIONS[name]
    return {
        "name": name,
        "display_name": title,
        "description": description,
        "timing": timing,
        "context_growth": growth,
        "dependency_example": dependency,
        "default_actions": {"compaction": "host", "maintenance_pings": False},
    }


def example_model_spec():
    """Shared illustrative model and prices for the three synthetic examples."""
    return ModelSpec(
        model_id="example-model",
        context_tokens=1_000_000,
        cache_ttl_seconds=300,
        cache_min_tokens=1024,
        cache_block_tokens=1,
        read_multiplier=0.1,
        write_multiplier=1.25,
        output_multiplier=5,
        ping_output_tokens=8,
        assumption_note="Illustrative five-minute cache priced in ordinary input-token equivalents.",
    )


def example_metadata(workflow="general", *, split="train"):
    """Return invented count rows and an ISO observation cutoff for offline fitting."""
    describe_workflow(workflow)
    if split not in ("train", "heldout"):
        raise ContractError("example split must be train or heldout")
    heldout = split == "heldout"
    seed = 7321 + 100 * WORKFLOW_NAMES.index(workflow) + int(heldout)
    rng = Random(seed)
    epoch = datetime(2025, 6 if heldout else 1, 1, tzinfo=timezone.utc)
    rows = []
    for index in range(12 if heldout else 24):
        family = rng.choice(("research", "support")) if workflow == "general" else workflow
        research = family == "research"
        length = rng.choice((4, 6, 8, 10, 12) if research else (1, 2, 3, 4, 5, 7))
        start = epoch + timedelta(days=index)
        for position in range(1, length + 1):
            duration = rng.randint(35, 105) if research else rng.randint(1, 8)
            added = rng.randint(16_000, 42_000) if research else rng.randint(120, 900)
            # The first input includes 100 tokens of static framing. Retrieved
            # text is a subset of the input delta, never an additional charge.
            dynamic = added - (100 if position == 1 else 0)
            retrieved = dynamic * 9 // 10 if research else (dynamic // 2 if position % 2 else 0)
            visible = rng.randint(350, 1100) if research else rng.randint(25, 130)
            end = start + timedelta(seconds=duration)
            rows.append({
                "thread_id": f"{workflow}-{split}-{index:02d}",
                "model_id": "example-model",
                "route_id": "example-route",
                "run_index": position,
                "started_at": start.isoformat(),
                "ended_at": end.isoformat(),
                "added_tokens": added,
                "retrieved_tokens": retrieved,
                "visible_output_tokens": visible,
                "output_tokens": visible + rng.randint(0, visible // 5),
                "calls_per_run": rng.choice((1, 2, 3) if research else (1, 1, 2)),
            })
            if research:
                gap = rng.choice((100, 320, 350, 375, 420, 2400)) + rng.randint(0, 25)
            else:
                gap = rng.choice((10, 20, 32, 45, 65, 700)) + rng.randint(0, 8)
            start = end + timedelta(seconds=gap)
    # Every final run has a full day of observable follow-up.
    cutoff = end + timedelta(days=2)
    return rows, cutoff.isoformat()
