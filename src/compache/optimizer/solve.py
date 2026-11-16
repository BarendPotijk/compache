"""Deterministic constrained training search, then one untouched holdout report."""

from __future__ import annotations

import math
from dataclasses import replace

from compache.traces import ContractError, Thread
from .contracts import Constraints, ModelSpec, Policy, digest, number
from .fit import fit_profile, label_index
from .replay import replay


def split_threads(
    threads: list[Thread],
    *,
    heldout_company=None,
    heldout_fraction=0.25,
    salt="compache-v1",
):
    if len(threads) < 2:
        raise ContractError("solve requires at least two complete threads")
    if heldout_company is not None:
        test = [t for t in threads if t.company_id == heldout_company]
    else:
        number(heldout_fraction, "heldout_fraction", 0.01, 0.99)
        ordered = sorted(threads, key=lambda t: digest([salt, t.thread_id]))
        test = ordered[
            : max(1, min(len(threads) - 1, math.ceil(len(threads) * heldout_fraction)))
        ]
    test_ids = {t.thread_id for t in test}
    train = [t for t in threads if t.thread_id not in test_ids]
    if not test or not train:
        raise ContractError("split must contain training and held-out threads")
    return sorted(train, key=lambda t: t.thread_id), sorted(
        test, key=lambda t: t.thread_id
    )


def feasible(summary, constraints):
    return (
        not summary["admission_failures"]
        and not summary["skipped_runs"]
        and (summary["miss_rate"] or 0) <= constraints.max_miss_rate
        and summary["unknown_change_rate"] <= constraints.max_unknown_change_rate
    )


def candidates(profile, spec, max_pings=6):
    ph = digest(profile)
    distances = sorted(
        {
            x["distance"]
            for x in profile["dependency"]["occurrences_in_history_needed_alternatives"]
            if x["kind"] == "retrieval"
        }
    )
    grace = sorted(
        {
            0,
            1,
            *(
                distances
                if len(distances) <= 3
                else (distances[0], distances[len(distances) // 2], distances[-1])
            ),
        }
    )
    ping_counts = sorted({0, min(1, max_pings), min(3, max_pings), max_pings})
    for n in ping_counts:
        yield Policy(spec.to_dict(), ph, max_pings=n)
    if not profile["dependency"]["assessable_runs"]:
        return
    for g in grace:
        for target in (0.15, 0.30):
            for warm in (0.65, 0.85):
                for n in ping_counts:
                    yield Policy(
                        spec.to_dict(),
                        ph,
                        retrieval_grace_runs=g,
                        cold_trigger=target + 0.02,
                        target=target,
                        warm_trigger=warm,
                        max_pings=n,
                    )


def optimize(
    threads: list[Thread],
    labels: list[dict],
    spec: ModelSpec,
    *,
    observation_end: str,
    constraints: Constraints | None = None,
    heldout_company=None,
    heldout_fraction=0.25,
    timing_source="supplied_timestamps",
    max_pings=6,
) -> dict:
    constraints = constraints or Constraints()
    number(max_pings, "max_pings", 0, 100, integer=True)
    index = label_index(threads, labels)
    train, test = split_threads(
        threads, heldout_company=heldout_company, heldout_fraction=heldout_fraction
    )

    def selected(ts):
        return [index[r.run_id] for t in ts for r in t.runs if r.run_id in index]

    train_labels, test_labels = selected(train), selected(test)
    profile = fit_profile(
        train,
        train_labels,
        spec,
        observation_end=observation_end,
        timing_source=timing_source,
    )
    evaluated = []
    for policy in candidates(profile, spec, max_pings):
        result = replay(
            train,
            train_labels,
            policy,
            profile,
            observation_end=observation_end,
            include_ledger=False,
        )["summary"]
        evaluated.append(
            {
                "policy": policy.to_dict(),
                "training": result,
                "feasible": feasible(result, constraints),
            }
        )
    eligible = [e for e in evaluated if e["feasible"]]
    if not eligible:
        raise ContractError(
            "no feasible policy: even the preserve-all baseline cannot satisfy constraints/admission"
        )

    def key(e):
        p = e["policy"]
        # On exact ties preserve more history, then fewer pings, then larger target.
        grace = p["retrieval_grace_runs"]
        return (
            round(e["training"]["cost_input_equivalents"], 9),
            -(grace if grace is not None else 10**9),
            p["max_pings"],
            -p["target"],
            digest(p),
        )

    winner = min(eligible, key=key)
    policy = Policy.from_dict(winner["policy"])
    heldout = replay(
        test, test_labels, policy, profile, observation_end=observation_end
    )
    baseline = Policy(spec.to_dict(), digest(profile))
    no_cache = replay(
        test,
        test_labels,
        baseline,
        profile,
        observation_end=observation_end,
        caching=False,
    )
    caching_only = replay(
        test, test_labels, baseline, profile, observation_end=observation_end
    )
    compaction_only = replay(
        test,
        test_labels,
        replace(policy, max_pings=0),
        profile,
        observation_end=observation_end,
    )
    original = caching_only["summary"]["cost_input_equivalents"]
    optimized = heldout["summary"]["cost_input_equivalents"]
    checks = {
        "dependency_and_admission_constraints": feasible(
            heldout["summary"], constraints
        ),
        "complete_cost_coverage": heldout["summary"]["runs_with_unavailable_text"] == 0,
        "lower_modeled_cost_than_baseline": optimized < original - 1e-9,
    }
    promoted = all(checks.values())
    recommended = (
        policy
        if promoted
        else baseline
        if feasible(caching_only["summary"], constraints)
        else None
    )
    report = {
        "schema_version": "compache-optimization-v1",
        "profile": profile,
        "policy": policy.to_dict(),
        "constraints": constraints.to_dict(),
        "split": {
            "method": "heldout_company"
            if heldout_company is not None
            else "stable_thread_hash",
            "heldout_company": heldout_company,
            "training_threads": [t.thread_id for t in train],
            "heldout_threads": [t.thread_id for t in test],
        },
        "selection": {
            "candidate_count": len(evaluated),
            "feasible_count": len(eligible),
            "training": winner["training"],
            "candidates": evaluated,
            "heldout_used_for_selection": False,
        },
        "heldout": heldout,
        "baselines": {
            "no_cache": no_cache,
            "cache_only": caching_only,
            "compaction_only": compaction_only,
        },
        "result": {
            "heldout_constraints_satisfied": feasible(heldout["summary"], constraints),
            "modeled_saving_vs_cache_only": original - optimized,
            "modeled_saving_fraction_vs_cache_only": (original - optimized) / original
            if original
            else None,
            "status": "heldout_constraints_passed"
            if feasible(heldout["summary"], constraints)
            else "heldout_constraints_failed_do_not_promote",
            "waterfall": {
                "no_cache": no_cache["summary"]["cost_input_equivalents"],
                "cache_only": original,
                "compaction_only": compaction_only["summary"]["cost_input_equivalents"],
                "compaction_and_retention": optimized,
            },
            "scope": "Synthetic/offline estimate under explicit model spec and frozen labels. No monetary or quality-preservation claim. Heldout result does not refit or change the selected policy.",
        },
    }
    report["recommended_policy"] = recommended.to_dict() if recommended else None
    report["result"]["promotion_checks"] = checks
    report["result"]["candidate_promoted"] = promoted
    report["result"]["status"] = (
        "candidate_passed_offline_validation"
        if promoted
        else "baseline_retained"
        if recommended
        else "no_admissible_policy"
    )
    report["result"]["recommended_modeled_saving_vs_cache_only"] = (
        original - optimized if promoted else 0 if recommended else None
    )
    report["result"]["scope"] += (
        " Validation can veto promotion; it never selects another fitted candidate. Missing body costs block promotion. A retained baseline is not proof of provider compatibility."
    )
    return report
