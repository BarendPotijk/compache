"""Content-free workload fitting with right-censored return observations."""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from functools import cached_property

from compache.dependencies import build_packet, validate_label
from compache.traces import ContractError, Thread
from .contracts import FrozenDict, ModelSpec, digest, freeze_json, number, seconds
from .gate import Request
from .position import STATUSES, fit_position_growth, validate_growth


def label_index(threads: list[Thread], labels: list[dict]) -> dict:
    # Python callers can construct individually valid threads without the
    # cross-thread identity checks made by load_threads(). Labels and peaks
    # are keyed by run ID, so a collision could leak a held-out observation.
    thread_ids = [t.thread_id for t in threads]
    run_ids = [r.run_id for t in threads for r in t.runs]
    if len(set(thread_ids)) != len(thread_ids) or len(set(run_ids)) != len(run_ids):
        raise ContractError("training/evaluation requires globally unique thread and run IDs")
    packets = {r.run_id: build_packet(t, r.run_id) for t in threads for r in t.runs}
    result = {}
    for label in labels:
        rid = label.get("target_run_id") if isinstance(label, dict) else None
        if rid not in packets or rid in result:
            raise ContractError(
                "label targets must be unique runs in the supplied traces"
            )
        errors = validate_label(label, packets[rid])
        if errors:
            raise ContractError(f"invalid label for {rid}: " + "; ".join(errors[:3]))
        result[rid] = label
    return result


def survival_curve(observations: list[tuple[float, bool]], horizon: float) -> dict:
    """Kaplan-Meier; tied returns happen before censoring at the same instant."""
    counts = defaultdict(lambda: [0, 0])
    for duration, returned in observations:
        number(duration, "duration", 0, horizon)
        if type(returned) is not bool:
            raise ContractError("return event must be boolean")
        counts[duration][0 if returned else 1] += 1
    at_risk = len(observations)
    survival = 1.0
    points = []
    for time, (events, censored) in sorted(counts.items()):
        if events:
            survival *= 1 - events / at_risk
            points.append([time, round(1 - survival, 12)])
        at_risk -= events + censored
    return {
        "observations": len(observations),
        "returns": sum(e for _, e in observations),
        "censored": sum(not e for _, e in observations),
        "horizon_seconds": horizon,
        "cdf_points": points,
        "residual_survival": round(survival, 12),
        "interpolation": "step",
    }


def cdf(curve: dict, idle_seconds: float) -> float:
    if idle_seconds < 0:
        return 0.0
    value = 0.0
    previous_time = 0.0
    for time, probability in curve["cdf_points"]:
        if time > min(idle_seconds, curve["horizon_seconds"]):
            if curve.get("interpolation", "step") == "linear" and time > previous_time:
                value += (
                    (probability - value)
                    * (min(idle_seconds, curve["horizon_seconds"]) - previous_time)
                    / (time - previous_time)
                )
            break
        value = probability
        previous_time = time
    return value


def index_run_peaks(peak_observations, run_ids):
    """A peak covers initial input through the largest input-plus-output occupancy."""
    if peak_observations is None:
        return None
    indexed = {}
    for row in peak_observations:
        if not isinstance(row, dict) or set(row) != {
            "run_id", "start_input_tokens", "peak_context_tokens", "status"
        }:
            raise ContractError("invalid run peak observation fields")
        rid = row["run_id"]
        if not isinstance(rid, str) or rid not in run_ids or rid in indexed:
            raise ContractError("run peaks must identify unique runs from the supplied traces")
        if row["status"] not in STATUSES:
            raise ContractError("invalid run peak observation status")
        if row["status"] == "missing":
            if row["start_input_tokens"] is not None or row["peak_context_tokens"] is not None:
                raise ContractError("missing run peaks require null counts")
        else:
            number(row["start_input_tokens"], "start_input_tokens", integer=True)
            number(row["peak_context_tokens"], "peak_context_tokens", row["start_input_tokens"], integer=True)
        indexed[rid] = row
    return indexed


def run_remaining_growth(run, spec, peaks):
    if peaks is None:
        if run.generation.usage is not None:
            return "single_generation_reported_output", "complete", dict(run.generation.usage)["output"]
        remaining = spec.text_tokens(run.output.text) if run.output.text is not None else None
        return "single_generation_proxy", "complete" if remaining is not None else "missing", remaining
    peak = peaks.get(run.run_id)
    status = peak["status"] if peak else "missing"
    remaining = peak["peak_context_tokens"] - peak["start_input_tokens"] if status != "missing" else None
    return "observed_peak", status, remaining


def fit_profile(
    threads: list[Thread],
    labels: list[dict],
    spec: ModelSpec,
    *,
    observation_end: str,
    horizon_seconds=604_800,
    min_index_samples=5,
    timing_source="supplied_timestamps",
    peak_observations=None,
) -> dict:
    if not threads:
        raise ContractError("fit requires at least one thread")
    number(horizon_seconds, "horizon_seconds", 1)
    number(min_index_samples, "min_index_samples", 1, integer=True)
    if not isinstance(timing_source, str) or not timing_source.strip():
        raise ContractError("timing_source is required")
    end = seconds(observation_end)
    all_runs = [r for t in threads for r in t.runs]
    if any(r.generation.model != spec.model_id for r in all_runs):
        raise ContractError("fit requires one model matching the supplied model spec")
    if any(seconds(r.ended_at) > end for r in all_runs):
        raise ContractError("observation_end precedes a run end")
    labels_by_id = label_index(threads, labels)
    peaks = index_run_peaks(peak_observations, {r.run_id for r in all_runs})
    growth_rows = []
    observations = defaultdict(list)
    durations = defaultdict(list)
    growth = []
    distances = Counter()
    unknown = 0
    assessed = 0
    for t in sorted(threads, key=lambda t: t.thread_id):
        for index, r in enumerate(t.runs):
            duration = seconds(r.ended_at) - seconds(r.generation.started_at)
            durations[r.run_index].append(duration)
            if index + 1 < len(t.runs):
                gap = seconds(t.runs[index + 1].started_at) - seconds(r.ended_at)
                event = gap <= horizon_seconds
                obs = (min(gap, horizon_seconds), event)
            else:
                obs = (min(end - seconds(r.ended_at), horizon_seconds), False)
            observations[r.run_index].append(obs)
            request = Request.from_thread(t, r.run_id)
            growth.append(sum(spec.part_tokens(p.text) for p in request.parts))
            arriving = [p for p in request.parts if p.run_index == r.run_index]
            arrival_tokens = (sum(spec.part_tokens(p.text) for p in arriving)
                              if all(p.text is not None for p in arriving) else None)
            source, status, remaining = run_remaining_growth(r, spec, peaks)
            growth_rows.append({"thread_id": t.thread_id, "run_index": r.run_index,
                                "arrival_tokens": arrival_tokens, "remaining_tokens": remaining,
                                "status": status, "source": source})
            label = labels_by_id.get(r.run_id)
            if (
                label is None
                or label["status"] != "complete"
                or any(
                    q["historical_context_needed"] == "unknown" or not q["alternatives"]
                    for q in label["task_requirements"]
                )
            ):
                unknown += 1
            else:
                assessed += 1
            if label:
                parts = {p.part_id: p for p in request.parts}
                # Count occurrences in sufficient alternatives, not token-weighted recall.
                seen = set()
                for req in label["task_requirements"]:
                    if req["historical_context_needed"] != "yes":
                        continue
                    for alternative in req["alternatives"]:
                        for source in alternative["sources"]:
                            p = parts[source["part_id"]]
                            if p.run_index < r.run_index:
                                seen.add((p.part_id, p.kind, r.run_index - p.run_index))
                for _, kind, distance in seen:
                    distances[kind, distance] += 1
    pooled = survival_curve(
        [x for values in observations.values() for x in values], horizon_seconds
    )
    rows = {}
    for i, values in sorted(observations.items()):
        use_pooled = len(values) < min_index_samples
        rows[str(i)] = {
            "curve": pooled if use_pooled else survival_curve(values, horizon_seconds),
            "fallback_to_pooled": use_pooled,
            "last_call_seconds": statistics.median(durations[i]),
            "later_calls": 0,
        }
    profile = {
        "schema_version": "compache-profile-v1",
        "trace_hash": digest(
            [t.to_dict() for t in sorted(threads, key=lambda t: t.thread_id)]
        ),
        "labels_hash": digest(sorted(labels, key=lambda x: x["target_run_id"])),
        "training_thread_hashes": sorted(digest(t.thread_id) for t in threads),
        "model_spec_hash": digest(spec.to_dict()),
        "timing_source": timing_source,
        "observation_end": observation_end,
        "threads": len(threads),
        "runs": len(all_runs),
        "ruler": {
            "kind": "utf8_byte_estimate",
            "bytes_per_token": spec.bytes_per_token,
            "part_overhead_tokens": spec.part_overhead_tokens,
            "measured_provider_tokens_used": False,
        },
        "cadence": {
            "pooled": pooled,
            "by_run_index": rows,
            "last_call_seconds": statistics.median(
                d for values in durations.values() for d in values
            ),
            "later_calls": 0,
            "residual_meaning": "No observed return by horizon under right censoring; not a measured never-return probability.",
        },
        "dependency": {
            "labeled_runs": len(labels),
            "assessable_runs": assessed,
            "unknown_runs": unknown,
            "occurrences_in_history_needed_alternatives": [
                {"kind": k, "distance": d, "count": n}
                for (k, d), n in sorted(distances.items())
            ],
            "evidence_kind": "frozen observational task requirements; OR alternatives, AND sources; no causal or token-weighted recall claim",
        },
        "growth": {
            "estimated_input_tokens_min": min(growth),
            "estimated_input_tokens_median": statistics.median(growth),
            "estimated_input_tokens_max": max(growth),
            "runs_per_thread": [
                len(t.runs) for t in sorted(threads, key=lambda t: t.thread_id)
            ],
        },
        "position_growth": fit_position_growth(growth_rows, min_threads=min_index_samples),
    }
    return profile


class _ValidatedProfile(FrozenDict):
    @cached_property
    def fingerprint(self):
        return digest(self)


def freeze_profile(profile):
    """Validate and freeze one profile for reuse by a runtime/replay loop."""
    if isinstance(profile, _ValidatedProfile):
        return profile
    validate_profile(profile)
    return _ValidatedProfile((k, freeze_json(v)) for k, v in profile.items())


def validate_profile(profile: dict) -> None:
    if isinstance(profile, _ValidatedProfile):
        return
    try:
        if profile["schema_version"] != "compache-profile-v1":
            raise ValueError()
        if "position_growth" in profile:
            validate_growth(profile["position_growth"])
        curves = [profile["cadence"]["pooled"]] + [
            v["curve"] for v in profile["cadence"]["by_run_index"].values()
        ]
        for curve in curves:
            if curve.get("interpolation", "step") not in ("step", "linear"):
                raise ValueError()
            previous_time, previous_p = -1, 0
            number(curve["horizon_seconds"], "horizon", 1)
            for time, p in curve["cdf_points"]:
                number(time, "cdf time", 0, curve["horizon_seconds"])
                number(p, "cdf probability", 0, 1)
                if time <= previous_time or p < previous_p:
                    raise ValueError()
                previous_time, previous_p = time, p
        for row in profile["cadence"]["by_run_index"].values():
            number(row["last_call_seconds"], "last call duration")
            number(row["later_calls"], "later calls", integer=True)
        number(profile["cadence"]["last_call_seconds"], "last call duration")
        number(profile["cadence"]["later_calls"], "later calls", integer=True)
        for curve in curves:
            for field in ("observations", "returns", "censored"):
                number(curve[field], field, integer=True)
            if curve["returns"] + curve["censored"] != curve["observations"]:
                raise ValueError()
            number(curve["residual_survival"], "residual survival", 0, 1)
            if (
                abs(
                    curve["residual_survival"]
                    + (curve["cdf_points"][-1][1] if curve["cdf_points"] else 0)
                    - 1
                )
                > 1e-9
            ):
                raise ValueError()
    except (KeyError, TypeError, ValueError, AttributeError, IndexError) as exc:
        raise ContractError("invalid workload profile") from exc
