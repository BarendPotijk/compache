"""Trace-fitted, expected-cost RAG planning and a serial host integration."""

from copy import deepcopy
from dataclasses import dataclass, replace

from compache.errors import ContractError

from .cache_breaks import CacheBreakChecker, CacheScope
from .contracts import DEFAULT_MIN_RETENTION, ModelSpec, digest, number
from .distribution import (
    fit_distribution,
    freeze_distribution,
    measure_cache_reads,
    plan_distribution,
    plan_distribution_idle,
    repeat_ping_charge,
)
from .edit_accounting import EditErrorHistory, InputAnchor, PartMeter, estimate_input
from .header_traffic import fit_shared_header_traffic, header_traffic_signature
from .parts import Request, Source, wire_part
from .retention import (
    RecipeView,
    RetentionRecipe,
    evidence_windows,
    fit_recipes,
    matched_prefix_tokens,
    recipe_grid,
    render_recipe,
    size_ladder,
)
from .shared_prefix import SharedPrefix, SharedPrefixCatalog, SharedReuse, _wire
from .usage import response_usage


def fit_workload(
    threads,
    labels,
    spec,
    *,
    observation_end,
    recipes=None,
    retention_floor=DEFAULT_MIN_RETENTION,
    tradeoff_floor=None,
    max_unknown_change_rate=None,
    unknowns="excluded",
    select=None,
    horizon_seconds=86400,
    calls_per_run=None,
    next_endpoint_reuse_fraction=None,
    header=None,
    tokenize=None,
    count_tokens=None,
):
    """Fit external demand and assess checkpoint recipes on training threads only.

    Without `recipes` the candidates are `recipe_grid` up to the longest training
    thread, over the kinds and sizes of part those threads contain, and the
    cheapest passing one is admitted. A supplied list is admitted in its own order.
    `select` overrides either default. `tradeoff_floor` lets a recipe between it
    and `retention_floor` compete once its lost requirements are priced.
    """
    threads, labels = list(threads), list(labels)
    if recipes is None:
        kinds = {"input", "output"}
        if any(
            p.exposed_to_model for t in threads for r in t.runs for p in r.retrievals
        ):
            kinds.add("retrieval")
        recipes = recipe_grid(
            max((len(t.runs) for t in threads), default=1),
            kinds=kinds,
            keep_under_tokens=size_ladder(threads, spec),
            include_citations=any(r.cited_part_ids for t in threads for r in t.runs),
            extra_windows=evidence_windows(threads, labels),
        )
        select = select or "cheapest"
    select = select or "supplied"
    forecasts = fit_distribution(
        threads,
        spec,
        observation_end=observation_end,
        horizon_seconds=horizon_seconds,
        calls_per_run=calls_per_run,
        next_endpoint_reuse_fraction=next_endpoint_reuse_fraction,
    )
    scores = fit_recipes(
        threads,
        labels,
        spec,
        recipes,
        retention_floor=retention_floor,
        tradeoff_floor=tradeoff_floor,
        max_unknown_change_rate=max_unknown_change_rate,
        unknowns=unknowns,
        select=select,
    )
    traffic = None
    if header is not None:
        traffic = fit_shared_header_traffic(
            threads, spec, **header, tokenize=tokenize, count_tokens=count_tokens
        )
    return {
        "header_traffic": traffic,
        "schema_version": "compache-workload-planner-v1",
        "model_spec": spec.to_dict(),
        "demand": forecasts,
        "cache_reads": measure_cache_reads(threads, spec, next_endpoint_reuse_fraction),
        "recipes": scores,
        "retention_floor": retention_floor,
        "recipe_selection": {
            "select": select,
            "tradeoff_floor": tradeoff_floor,
            "unknowns": unknowns,
            "max_unknown_change_rate": max_unknown_change_rate,
        },
        "training_threads": sorted(digest(t.thread_id) for t in threads),
    }


class WorkloadPolicy:
    """Validate and compile the same fitted profile once for multiple threads."""

    def __init__(
        self,
        artifact,
        *,
        lookahead=3,
        branches=3,
        seed=0,
        sampler="brownian",
        max_pings=None,
        ping_lead_seconds=45,
        reserve_tokens=0,
        overhead_tokens=0,
        ambient_hit_probability=None,
    ):
        if (
            not isinstance(artifact, dict)
            or artifact.get("schema_version") != "compache-workload-planner-v1"
        ):
            raise ContractError("expected a fitted workload policy artifact")
        self.spec = ModelSpec.from_dict(artifact["model_spec"])
        self.forecast = freeze_distribution(artifact["demand"], self.spec)
        self.recipes = {
            f"cut-{i}": RetentionRecipe(**row["recipe"])
            for i, row in enumerate(artifact["recipes"])
            if row["eligible"]
        }
        self.training_threads = frozenset(artifact["training_threads"])
        number(artifact["retention_floor"], "retention_floor", 0, 1)
        self.retention_floor = artifact["retention_floor"]
        self.catalog = SharedPrefixCatalog()
        self.header_traffic = artifact.get("header_traffic") or {}
        self.ambient_override = ambient_hit_probability
        if ambient_hit_probability is None:
            ambient_hit_probability = self.header_traffic.get(
                "ambient_hit_probability", 0.0
            )
        number(ambient_hit_probability, "ambient_hit_probability", 0, 1)
        self.ambient_hit_probability = ambient_hit_probability
        number(overhead_tokens, "overhead_tokens", integer=True)
        self.overhead_tokens = overhead_tokens
        if max_pings is None:
            # A solved policy carries the ping count that was cheapest on its threads.
            max_pings = (artifact.get("planner_options") or {}).get("max_pings", 1)
        share = artifact.get("cache_reads", {}).get("share")
        self.cache_read_share = 1.0 if share is None else share
        number(self.cache_read_share, "cache read share", 0, 1)
        self.options = {
            "cache_read_share": self.cache_read_share,
            "lookahead": lookahead,
            "branches": branches,
            "seed": seed,
            "sampler": sampler,
            "max_pings": max_pings,
            "ping_lead_seconds": ping_lead_seconds,
            "reserve_tokens": reserve_tokens,
        }


@dataclass(frozen=True)
class PlannedRun:
    request: Request
    started_at: float
    view: RecipeView
    decision: dict
    arrival_tokens: int
    header_changes: tuple
    header: SharedPrefix
    reuse: SharedReuse


class WorkloadAdmissionError(ContractError):
    """All available views were infeasible. Accounting excludes raw prompt content."""

    def __init__(self, accounting, input_cap_tokens, forecast_status):
        super().__init__("no admissible workload view for this request")
        self.candidate_input_accounting = accounting
        self.input_cap_tokens = input_cap_tokens
        self.forecast_status = forecast_status


def _reported_usage(response, **explicit):
    """Merge host-supplied counters with the exact provider response."""
    meter = response_usage(response) if response is not None else {}
    for name, value in explicit.items():
        reported = meter.get(name)
        if reported is not None and value is not None and value != reported:
            raise ContractError(f"conflicting response and explicit {name}")
        if value is not None:
            meter[name] = value
    response_id = meter.get("response_id")
    if response_id is not None and (
        not isinstance(response_id, str) or not response_id.strip()
    ):
        raise ContractError("response_id must be a nonempty string")
    return meter


def _reusable_capacity(spec, input_tokens, reads, writes, bound, mixed_ttls):
    """Return the eligible request size and the prefix available to its successor."""
    for name, count in (
        ("input_tokens", input_tokens),
        ("cached_input_tokens", reads),
        ("cache_write_input_tokens", writes),
        ("reusable_input_tokens", bound),
    ):
        if count is not None:
            number(count, name, integer=True)
    if input_tokens is None:
        return bound, bound
    if (reads or 0) + (writes or 0) > input_tokens:
        raise ContractError("cache read and write counters exceed input_tokens")

    def eligible(count):
        rounded = count // spec.cache_block_tokens * spec.cache_block_tokens
        return (
            rounded
            if rounded >= spec.cache_min_tokens and spec.cache_ttl_seconds > 0
            else 0
        )

    limit = eligible(input_tokens)
    if bound is not None:
        number(bound, "reusable_input_tokens", 0, limit, integer=True)
        limit = bound
    elif mixed_ttls:
        # Aggregate writes at different retention periods do not identify which
        # contiguous prefix is retained for this scope's configured lifetime.
        return 0, 0
    if reads is None or writes is None:
        return limit, limit
    return limit, min(limit, eligible(reads + writes))


class WorkloadSession:
    """One chronological thread. Keep this object with the host's thread state.

    CacheBreakChecker can be shared across sessions and backed by a database.
    prepare() cancels old timers; complete() commits only a successful request.
    Failed sends can be retried by preparing the same run again. The host owns
    the actual provider request, the thread's canonical archive and timers.
    """

    def __init__(
        self,
        policy: WorkloadPolicy,
        scope: CacheScope,
        checker: CacheBreakChecker,
        *,
        catalog=None,
        tokenize=None,
        count_tokens=None,
        count_part_tokens=None,
        share_headers=True,
        component_order=("system_prompt", "default_tools"),
        tokenizer_id=None,
        verified_contiguous_prefix_reads=False,
        ping_has_private_suffix=None,
    ):
        if scope.model != policy.spec.model_id:
            raise ContractError("cache scope and fitted model differ")
        if type(verified_contiguous_prefix_reads) is not bool or (
            verified_contiguous_prefix_reads and tokenize is None
        ):
            raise ContractError(
                "verified contiguous prefix attribution requires a native wire-token callback"
            )
        self.verified_contiguous_prefix_reads = verified_contiguous_prefix_reads
        if (
            ping_has_private_suffix is not None
            and type(ping_has_private_suffix) is not bool
        ):
            raise ContractError("ping_has_private_suffix must be boolean or None")
        self._private_ping_forecast = (
            scope.provider.casefold() == "openai" and scope.model == "gpt-6-luna"
            if ping_has_private_suffix is None
            else ping_has_private_suffix
        )
        self._private_ping_override = ping_has_private_suffix
        self.policy, self.scope, self.checker = policy, scope, checker
        self.catalog = catalog if catalog is not None else policy.catalog
        self.share_headers = share_headers
        self.header_options = {
            "tokenize": tokenize,
            "count_tokens": count_tokens,
            "share": share_headers,
            "component_order": tuple(component_order),
            "tokenizer_id": tokenizer_id,
        }
        self._count_part_tokens = count_part_tokens
        self._part_meter = PartMeter(policy.spec, count_part_tokens)
        self._anchor = self._anchor_header_key = None
        self._errors = EditErrorHistory()
        self._measured_header_key = self._measured_header = None
        self._committed_response_ids = set()
        self._last_header = self._held_capacity = None
        self.archived_ids = frozenset()
        self._pending = self._cached = self._ping_view = None
        self.last_ping_receipt = None
        self._refreshed = self._last_run = self._last_start = self._last_finish = None
        self._last_index = 0
        self._due = ()
        self._idle_arguments = None

    def invalidate_input_anchor(self):
        """Forget the local size baseline after an untracked serializer/context change.

        Normal cache expiry does not call this. Start a new scoped session for
        a different provider/model. The durable cache-break latch is independent.
        """
        if self._pending is not None:
            raise ContractError("cannot invalidate accounting during a pending request")
        self._anchor = self._anchor_header_key = None
        self._errors = EditErrorHistory()
        self._cached = self._refreshed = self._held_capacity = None
        self._measured_header_key = self._measured_header = None
        self._part_meter = PartMeter(self.policy.spec, self._count_part_tokens)
        self._due = ()
        self._idle_arguments = None

    def _ambient_rate(self, header, system, tools):
        """Price other threads only in a matching cache namespace and header family."""
        if not self.share_headers:
            return 0.0, "thread_isolated"
        if self.policy.ambient_override is not None:
            return self.policy.ambient_override, "scenario_override"
        fit = self.policy.header_traffic
        if not fit or fit["provider"] != self.scope.provider:
            return 0.0, "no_matching_traffic_fit"
        if (
            fit["header_signature"]
            != header_traffic_signature(system, tools, header.component_order)
            or fit["count_source"] != header.source
            or fit["static_tokens"] != header.static_tokens
        ):
            return 0.0, "header_or_ruler_changed"
        key = digest(
            [
                self.scope.installation,
                fit["provider"],
                self.scope.model,
                self.scope.cache_key,
            ]
        )
        cohort = fit["by_scope"].get(key)
        if cohort is None:
            return 0.0, "unseen_cache_scope"
        return cohort["ambient_hit_probability"], cohort["cold_returns"]["support"]

    def _views(self, request, archived, overhead):
        views, estimates = {}, {}
        for name, recipe in (("keep", RetentionRecipe()), *self.policy.recipes.items()):
            view = render_recipe(
                request,
                self.policy.spec,
                recipe,
                archived_ids=archived,
                cut=name != "keep",
                overhead_tokens=overhead,
                part_meter=self._part_meter,
            )
            if any(view.archived_ids == prior.archived_ids for prior in views.values()):
                continue
            estimate = estimate_input(view, self._anchor)
            if name != "keep" and estimate.tokens >= views["keep"].tokens:
                continue
            views[name] = (
                replace(view, tokens=estimate.tokens)
                if estimate.tokens != view.tokens
                else view
            )
            estimates[name] = estimate
        return views, estimates

    def _accounting_report(self, estimate):
        changed = (
            self._anchor_header_key is not None
            and self._measured_header_key != self._anchor_header_key
        )
        kind = self._errors.classify(estimate, header_changed=changed)
        samples, margin = self._errors.allowance(kind)
        return {
            **estimate.to_dict(),
            "edit_class": kind,
            "calibration_observations": samples,
            "admission_margin_tokens": margin,
            "admission_input_tokens": estimate.tokens + margin,
        }

    def _measure_header(self, system, tools, dynamic):
        # Ordered wire equality handles mutable host dictionaries and reordered schemas.
        # Itemized and opaque tool inputs with identical bytes have different reporting.
        key = (
            _wire(system),
            _wire(tools),
            _wire(dynamic),
            isinstance(tools, (list, tuple)),
        )
        if key != self._measured_header_key:
            header = self.catalog.measure(
                self.scope,
                system_prompt=system,
                default_tools=tools,
                dynamic_tools=dynamic,
                spec=self.policy.spec,
                **self.header_options,
            )
            self._measured_header_key, self._measured_header = key, header
        return self._measured_header

    def _choices(
        self, views, header, *, header_changes=(), shared_prefix_upper_bound=None
    ):
        keep = next(iter(views.values()))
        overhead = keep.overhead_tokens
        choices = []
        for name, view in views.items():
            stable = min(overhead, self._cached.overhead_tokens) if self._cached else 0
            prior_shared = matched_prefix_tokens(
                self._cached, view, stable_overhead_tokens=stable
            )
            if self._held_capacity is not None:
                prior_shared = min(prior_shared, self._held_capacity)
            if self._cached is not None:
                prior_shared = min(prior_shared, self._cached.tokens)
            if header_changes:
                prior_shared = min(prior_shared, shared_prefix_upper_bound or 0)
            elif shared_prefix_upper_bound is not None:
                prior_shared = min(prior_shared, shared_prefix_upper_bound)
            choices.append(
                {
                    "name": name,
                    "input_tokens": view.tokens,
                    "shared_prefix_tokens": min(prior_shared, view.tokens),
                    # The unchanged candidate shares its entire input with itself.
                    # For a cut, residual provider overhead has no known position.
                    "rewrite_prefix_tokens": keep.tokens
                    if name == "keep"
                    else min(
                        keep.tokens,
                        view.tokens,
                        matched_prefix_tokens(
                            keep, view, stable_overhead_tokens=overhead
                        ),
                    ),
                    "admitted": view.wire_bytes + header.wire_bytes
                    <= self.policy.spec.byte_limit,
                }
            )
        return choices

    def prepare(
        self,
        request,
        *,
        started_at,
        system_prompt,
        tools,
        shared_prefix_upper_bound=None,
        dynamic_tools=None,
        warm_chance=None,
    ):
        """Pass the actual provider-visible system and tool prefixes, in wire order.

        `warm_chance` is how likely the provider still holds the request before, where
        that is not known, as for a conversation first met in the middle: every view is
        priced both ways and weighed by it.
        """
        if (
            request.thread_id != self.scope.thread_id
            or request.model != self.scope.model
        ):
            raise ContractError("request and cache scope differ")
        number(started_at, "started_at", -1e20)
        if shared_prefix_upper_bound is not None:
            number(shared_prefix_upper_bound, "shared_prefix_upper_bound", integer=True)
        if request.run_index <= self._last_index or (
            self._last_finish is not None and started_at < self._last_finish
        ):
            raise ContractError("expected a subsequent run in chronological order")
        if not self.checker.begin_return(
            self.scope, run_id=request.run_id, started_at=started_at
        ):
            raise ContractError("request superseded by newer cache observation")
        self._pending, self._due = None, ()
        self._idle_arguments = None
        spec = self.policy.spec
        header = self._measure_header(system_prompt, tools, dynamic_tools)
        # Fingerprint dynamic tools too. The priced static prefix stops before this suffix.
        guard_tools = (
            tools
            if dynamic_tools is None
            else self._tool_fingerprint(tools, dynamic_tools)
        )
        changed = self.checker.prefix_changed(
            self.scope, system_prompt=system_prompt, tools=guard_tools
        )
        views, estimates = self._views(
            request,
            self.archived_ids,
            self.policy.overhead_tokens + header.total_tokens,
        )
        keep = views["keep"]

        # The verified limit applies to this thread. A separate thread can
        # independently have published a matching header in this namespace.
        def lookup(view):
            return self.catalog.lookup(
                header,
                at=started_at,
                request_tokens=view.tokens,
                same_thread_upper_bound=shared_prefix_upper_bound,
            )

        reuse = lookup(keep)
        warm = (
            self._refreshed is not None
            and 0 <= started_at - self._refreshed < spec.cache_ttl_seconds
        )
        candidates = self._choices(
            views,
            header,
            header_changes=changed,
            shared_prefix_upper_bound=shared_prefix_upper_bound,
        )
        accounting = {name: self._accounting_report(e) for name, e in estimates.items()}
        for candidate in candidates:
            candidate["admission_margin_tokens"] = accounting[candidate["name"]][
                "admission_margin_tokens"
            ]
        # Demand was fitted with ModelSpec.part_tokens; only fresh arrivals enter this forecast.
        arrival = sum(
            spec.part_tokens(p.text)
            for p in request.parts
            if p.run_index == request.run_index
        )
        ambient_rate, ambient_source = self._ambient_rate(header, system_prompt, tools)
        options = dict(
            self.policy.options,
            ping_has_private_suffix=self._private_ping_forecast,
            shared_header_tokens=reuse.eligible_tokens,
            ambient_header_tokens=header.static_tokens,
            ambient_hit_probability=ambient_rate,
        )

        if warm_chance is not None:
            number(warm_chance, "warm_chance", 0, 1)

        def forecast(hot):
            return plan_distribution(
                self.policy.forecast,
                spec,
                run_index=request.run_index,
                candidates=candidates,
                warm=hot,
                current_arrival_tokens=arrival,
                ping_allowed=self.checker.can_ping(self.scope),
                **options,
            )

        def decide():
            if warm_chance is None:
                return forecast(warm)
            hot, cold = forecast(True), forecast(False)
            weighed = {}
            for name, cost in hot["costs"].items():
                other = cold["costs"].get(name)
                if cost is not None and other is not None:
                    weighed[name] = warm_chance * cost + (1 - warm_chance) * other
            made = dict(hot if warm_chance >= 0.5 else cold)
            if weighed:
                # Ties go to the first view, which keeps the conversation as it is.
                best = min(weighed, key=weighed.get)
                made.update(
                    action=best, expected_cost=weighed[best], costs=weighed, status="ok"
                )
            made["warm_chance"] = warm_chance
            return made

        decision = decide()
        if decision["action"] is None and options["lookahead"] > 1:
            # A hypothetical later arrival should not deny a feasible request today.
            fallback_reason = decision["status"]
            options["lookahead"] = 1
            decision = decide()
            decision["fallback"] = fallback_reason
        if warm_chance is not None:
            warm = warm_chance >= 0.5
        name = decision["action"]
        if name not in views:
            raise WorkloadAdmissionError(
                accounting,
                spec.context_tokens - self.policy.options["reserve_tokens"],
                decision["status"],
            )
        reuse = lookup(views[name])
        chosen = views[name]
        thread_match = (
            next(c["shared_prefix_tokens"] for c in candidates if c["name"] == name)
            if warm
            else 0
        )
        decision["input_accounting"] = accounting[name]
        decision["candidate_input_accounting"] = accounting
        decision["ambient_source"] = ambient_source
        decision["header"] = header_usage(header, reuse)
        decision["header"]["planned_input_cost"] = price_header(
            spec,
            header,
            chosen.tokens,
            max(thread_match, reuse.eligible_tokens),
            self.policy.cache_read_share,
        )
        plan = PlannedRun(
            request,
            started_at,
            views[name],
            decision,
            arrival,
            tuple(changed),
            header,
            reuse,
        )
        self._pending = (
            plan,
            deepcopy(system_prompt),
            deepcopy(guard_tools),
            shared_prefix_upper_bound,
            estimates[name],
            accounting[name]["edit_class"],
        )
        return plan

    def extend(self, plan, parts, *, at=None):
        """Add parts that arrived while the pending run was being answered.

        A tool loop learns what it retrieved only after its first call. The
        choice `prepare` made stands: the history goes out as chosen and the
        new parts follow it verbatim. Pass the returned plan to `complete`.
        """
        if self._pending is None or self._pending[0] is not plan:
            raise ContractError("expected the current pending request")
        parts, req = tuple(parts), plan.request
        if any(
            not isinstance(p, Source)
            or p.kind != "retrieval"
            or p.run_index != req.run_index
            or p.text is None
            for p in parts
        ):
            raise ContractError("expected retrieved parts of the pending run")
        if not parts:
            return plan
        request = replace(req, parts=req.parts + parts)
        view = render_recipe(
            request,
            self.policy.spec,
            RetentionRecipe(),
            archived_ids=plan.view.archived_ids,
            cut=False,
            overhead_tokens=plan.view.overhead_tokens,
            part_meter=self._part_meter,
        )
        estimate = estimate_input(view, self._anchor)
        if estimate.tokens != view.tokens:
            view = replace(view, tokens=estimate.tokens)
        extended = replace(
            plan,
            request=request,
            view=view,
            arrival_tokens=plan.arrival_tokens
            + sum(self.policy.spec.part_tokens(p.text) for p in parts),
        )
        _, system, tools, upper_bound, _, edit_class = self._pending
        self._pending = (extended, system, tools, upper_bound, estimate, edit_class)
        return extended

    @staticmethod
    def _tool_fingerprint(default_tools, dynamic_tools):
        from .cache_breaks import _fingerprint

        return [_fingerprint(default_tools), _fingerprint(dynamic_tools)]

    def complete(
        self,
        plan,
        answer,
        *,
        ended_at,
        answer_history_tokens=None,
        cached_input_tokens=None,
        cache_write_input_tokens=None,
        input_tokens=None,
        reusable_input_tokens=None,
        shared_prefix_tokens=None,
        verified_shared_static_tokens=None,
        intentional_change=False,
        response=None,
        response_id=None,
    ):
        """Acknowledge a successful foreground answer, observe reuse, then choose idle timers.

        Cache counts for the checker must describe this provider request. If
        capacity is unavailable, omit it and keep the reuse diagnosis unknown.
        The cost model has its own estimated cache state and token ruler.
        """
        if self._pending is None or self._pending[0] is not plan:
            raise ContractError("expected the current pending request")
        _, system, tools, prepared_upper_bound, prior_estimate, edit_class = (
            self._pending
        )
        req, spec = plan.request, self.policy.spec
        number(ended_at, "ended_at", plan.started_at)
        if (
            not isinstance(answer, Source)
            or answer.kind != "output"
            or answer.run_index != req.run_index
            or answer.text is None
        ):
            raise ContractError("expected the completed output of the pending request")
        meter = _reported_usage(
            response,
            input_tokens=input_tokens,
            cached_input_tokens=cached_input_tokens,
            cache_write_input_tokens=cache_write_input_tokens,
            response_id=response_id,
        )
        if response is not None and meter["status"] != "completed":
            raise ContractError("expected a completed response for the submitted view")
        input_tokens = meter.get("input_tokens")
        cached_input_tokens = meter.get("cached_input_tokens")
        response_id = meter.get("response_id")
        request_eligible, reusable_input_tokens = _reusable_capacity(
            spec,
            input_tokens,
            cached_input_tokens,
            meter.get("cache_write_input_tokens"),
            reusable_input_tokens,
            meter.get("mixed_cache_write_ttls", False),
        )
        if response_id in self._committed_response_ids:
            raise ContractError(
                "response_id already acknowledged for a different request"
            )
        explicit_answer_count = answer_history_tokens is not None
        if answer_history_tokens is None:
            answer_history_tokens = self._part_meter.count(answer)
        number(answer_history_tokens, "answer_history_tokens", integer=True)
        if type(intentional_change) is not bool:
            raise ContractError("intentional_change must be boolean")
        # A default refresh sends the selected foreground view plus exactly
        # the rendered answer. Its historical blocks have already been sized.
        view = plan.view
        if answer.part_id in view.full_part_ids or answer.part_id in view.archived_ids:
            raise ContractError("answer duplicates an existing source ID")
        answer_wire = wire_part(answer)
        answer_bytes = len(answer_wire.encode("utf-8"))
        ping_view = replace(
            view,
            run_id=req.run_id + ":ping",
            parts=view.parts + (answer,),
            wire=view.wire + (answer_wire,),
            part_tokens=view.part_tokens + (answer_history_tokens,),
            part_wire_bytes=view.part_wire_bytes + (answer_bytes,),
            wire_bytes=view.wire_bytes + answer_bytes,
            full_part_ids=view.full_part_ids | {answer.part_id},
            tokens=view.tokens + answer_history_tokens,
        )
        # Prepare the arithmetic without committing a failed or superseded response.
        next_anchor = (
            self._anchor
            if input_tokens is None
            else InputAnchor(plan.view, input_tokens, response_id=response_id)
        )
        ping_estimate = estimate_input(ping_view, next_anchor)
        ping_view = replace(ping_view, tokens=ping_estimate.tokens)
        if shared_prefix_tokens is None:
            shared_prefix_tokens = prepared_upper_bound
        if (
            verified_shared_static_tokens is None
            and self.verified_contiguous_prefix_reads
            and self._last_header is not None
        ):
            # An observed contiguous read ending inside this many leading tokens
            # cannot have reached the thread body, even after a tool-schema change.
            verified_shared_static_tokens = min(
                self._last_header.static_tokens, plan.header.static_tokens
            )
        event = self.checker.observe_return(
            self.scope,
            run_id=req.run_id,
            started_at=plan.started_at,
            system_prompt=system,
            tools=tools,
            ttl_seconds=spec.cache_ttl_seconds,
            reusable_input_tokens=0
            if reusable_input_tokens is None
            else reusable_input_tokens,
            request_eligible_input_tokens=request_eligible,
            cached_input_tokens=cached_input_tokens,
            shared_prefix_tokens=shared_prefix_tokens,
            verified_shared_static_tokens=verified_shared_static_tokens,
            intentional_change=intentional_change
            or (
                not plan.header_changes and plan.view.archived_ids != self.archived_ids
            ),
        )
        if event["outcome"] == "out_of_order_ignored":
            raise ContractError("response superseded by newer cache observation")
        # The catalog needs total input capacity. A provider's read counter may
        # exceed our local size estimate and cannot identify a component by itself.
        self.catalog.record(
            plan.header,
            at=plan.started_at,
            request_tokens=plan.view.tokens
            if reusable_input_tokens is None
            else reusable_input_tokens,
        )
        if response_id is not None:
            self._committed_response_ids.add(response_id)
        if explicit_answer_count:
            self._part_meter.observe_count(answer, answer_history_tokens)
        self._errors.observe(edit_class, prior_estimate, input_tokens)
        self._anchor = next_anchor
        if input_tokens is not None:
            self._anchor_header_key = self._measured_header_key
        self._part_meter.retain(plan.view)
        self._held_capacity = reusable_input_tokens
        self._last_header = plan.header
        self.archived_ids = plan.view.archived_ids
        self._cached, self._refreshed, self._ping_view = (
            plan.view
            if input_tokens is None
            else replace(plan.view, tokens=input_tokens),
            plan.started_at,
            ping_view,
        )
        self._last_index, self._last_start, self._last_finish, self._last_run = (
            req.run_index,
            plan.started_at,
            ended_at,
            req.run_id,
        )
        self._pending = None
        views, _ = self._views(req, self.archived_ids, plan.view.overhead_tokens)
        choices = self._choices(views, plan.header)
        ping_report = self._accounting_report(ping_estimate)
        ping_bound = ping_report["admission_input_tokens"]
        self._ping_accounting = (ping_estimate, ping_report["edit_class"])
        ping_allowed = (
            ping_bound <= spec.context_tokens - self.policy.options["reserve_tokens"]
            and ping_bound + spec.ping_output_tokens <= spec.context_tokens
            and ping_view.wire_bytes + plan.header.wire_bytes <= spec.byte_limit
            and self.checker.can_ping(self.scope, run_id=req.run_id, at=ended_at)
        )
        idle_arguments = dict(
            run_index=req.run_index,
            candidates=choices,
            answer_tokens=answer_history_tokens,
            held_cache_tokens=self._held_capacity,
            duration_seconds=ended_at - plan.started_at,
            current_arrival_tokens=plan.arrival_tokens,
            ping_allowed=ping_allowed,
            ping_has_private_suffix=self._private_ping_forecast,
            ambient_header_tokens=plan.header.static_tokens,
            ambient_hit_probability=plan.decision["ambient_hit_probability"],
            **self.policy.options,
        )
        idle = plan_distribution_idle(self.policy.forecast, spec, **idle_arguments)
        self._due = tuple(
            plan.started_at + offset for offset in idle["due_after_start_seconds"]
        )
        # Only the numeric planning inputs are needed to reconsider later timers.
        self._idle_arguments = idle_arguments if self._due else None
        return {
            "cache_event": event,
            "idle": idle,
            "due_times": self._due,
            "input_reconciliation": {
                "estimated_input_tokens": plan.view.tokens,
                "reported_input_tokens": input_tokens,
                "response_id": response_id,
                "actual_minus_estimated": None
                if input_tokens is None
                else input_tokens - plan.view.tokens,
            },
            "ping_input_accounting": ping_report,
        }

    @property
    def next_ping_at(self):
        """Earliest planned absolute deadline. Recheck ping_view immediately before send."""
        return self._due[0] if self._due else None

    def ping_view(self, *, run_id, at):
        """Fetch a due, still-valid view immediately before sending. None means cancel."""
        if (
            self._pending is not None
            or run_id != self._last_run
            or not self._due
            or at < self._due[0]
            or not self.checker.can_ping(self.scope, run_id=run_id, at=at)
        ):
            return None
        return self._ping_view

    def _remaining_ping_decision(
        self,
        *,
        dispatched_at,
        completed_at,
        capacity,
        reads,
        writes,
        output,
        foreground_bound=None,
    ):
        spec = self.policy.spec
        # A late receipt cannot retroactively dispatch another ping. At equal
        # time the completion is visible to a timer, and a foreground return wins.
        remaining = tuple(due for due in self._due[1:] if due >= completed_at)
        decision = {"decision": "stop", "due_times": []}
        if not remaining:
            decision["reason"] = "schedule_exhausted"
            return decision
        if remaining[0] - dispatched_at >= spec.cache_ttl_seconds:
            decision["reason"] = "cache_expired_before_next_ping"
            return decision
        repeat, kind = self._ping_accounting
        _, margin = self._errors.allowance(kind)
        if repeat.tokens + margin > min(
            spec.context_tokens - self.policy.options["reserve_tokens"],
            spec.context_tokens - output,
        ):
            decision["reason"] = "repeat_not_admissible"
            return decision
        started_at = self._last_start
        original_due = {due - started_at: due for due in remaining}
        uncached = self._ping_view.tokens - reads
        use_receipt_mix = writes is not None and uncached > 0
        uncached_rate = (
            1 + (writes / uncached) * (spec.write_multiplier - 1)
            if use_receipt_mix
            else spec.write_multiplier
        )
        result = plan_distribution_idle(
            self.policy.forecast,
            spec,
            **self._idle_arguments,
            after_ping={
                "elapsed_seconds": dispatched_at - started_at,
                "observed_elapsed_seconds": completed_at - started_at,
                "remaining_due_after_start_seconds": list(original_due),
                "input_tokens": self._ping_view.tokens,
                "reusable_tokens": capacity,
                "foreground_reusable_tokens": foreground_bound,
                "output_tokens": output,
                "uncached_input_rate": uncached_rate,
            },
        )
        costs = result["costs"]
        continued = [value for n, value in costs.items() if n and value is not None]
        decision.update(
            estimated_ping_output_tokens=output,
            uncached_input_rate=uncached_rate,
            uncached_input_rate_source="receipt_write_fraction"
            if use_receipt_mix
            else "model_spec",
            estimated_next_ping_cost_input_equivalents=repeat_ping_charge(
                spec,
                self._ping_view.tokens,
                capacity,
                output,
                uncached_input_rate=uncached_rate,
                read_share=self.policy.cache_read_share,
            ),
            expected_cost_if_stopped=costs.get(0),
            expected_cost_if_continued=min(continued, default=None),
            conditional_support=result["support"],
            planner_status=result["status"],
            ping_economics=result["ping_economics"],
        )
        if result["pings"]:
            decision.update(
                decision="continue",
                reason="marginal_cost_savings",
                due_times=[original_due[t] for t in result["due_after_start_seconds"]],
            )
        else:
            decision["reason"] = (
                "no_marginal_savings" if result["status"] == "ok" else result["status"]
            )
        return decision

    def acknowledge_ping(
        self,
        *,
        run_id,
        at,
        completed_at=None,
        ping_has_private_suffix=None,
        input_tokens=None,
        cached_input_tokens=None,
        cache_write_input_tokens=None,
        reusable_input_tokens=None,
        output_tokens=None,
        response=None,
        response_id=None,
        verified_shared_static_tokens=None,
        modeled_cached_input_tokens=None,
        modeled_cache_write_input_tokens=None,
        modeled_output_tokens=None,
    ):
        """Validate the receipt from the exact dispatched ping using ordinary usage.

        True means the completed ping read the eligible prior prefix and
        supplied its input and output totals. Inspect last_ping_receipt for
        the cost and next decision. Every confirmed receipt reprices later timers.
        Use the dispatch timestamp for at. Supply completed_at if the receipt
        arrives later. Both timestamps use the foreground clock. Explicit
        counters attest to the completed ping. Offline replay uses the modeled counters, which
        leave the provider input anchor untouched.
        """
        if (
            ping_has_private_suffix is not None
            and type(ping_has_private_suffix) is not bool
        ):
            raise ContractError("ping_has_private_suffix must be a boolean or None")
        number(at, "at", -1e20)
        completed_at = at if completed_at is None else completed_at
        number(completed_at, "completed_at", at)
        view = self.ping_view(run_id=run_id, at=at)
        if view is None:
            self.last_ping_receipt = {
                "event": "compache.ping_receipt",
                "outcome": "stale_or_not_due",
                "refresh_confirmed": False,
                "continuation": {"decision": "ignore", "reason": "stale_or_not_due"},
            }
            return False
        spec = self.policy.spec
        modeled = any(
            v is not None
            for v in (
                modeled_cached_input_tokens,
                modeled_cache_write_input_tokens,
                modeled_output_tokens,
            )
        )
        if modeled and any(
            v is not None
            for v in (
                response,
                input_tokens,
                cached_input_tokens,
                cache_write_input_tokens,
                output_tokens,
                response_id,
            )
        ):
            raise ContractError(
                "modeled ping and provider counters must be supplied separately"
            )
        meter = _reported_usage(
            response,
            input_tokens=input_tokens,
            cached_input_tokens=cached_input_tokens,
            cache_write_input_tokens=cache_write_input_tokens,
            output_tokens=output_tokens,
            response_id=response_id,
        )
        if ping_has_private_suffix is None:
            # An OpenAI Luna maintenance call can have a sibling user return or
            # prewarm only its seed. Its receipt alone cannot put its full input
            # on the foreground branch.
            ping_has_private_suffix = (
                self._private_ping_override
                if self._private_ping_override is not None
                else response is not None
                and self.scope.provider.casefold() == "openai"
                and meter.get("model") == spec.model_id == "gpt-6-luna"
            )
        status = meter.get("status", "completed")
        if (
            response is not None
            and meter["model"] is not None
            and meter["model"] != spec.model_id
        ):
            status = "model_mismatch"
        response_id = meter.get("response_id")
        if response_id in self._committed_response_ids:
            raise ContractError(
                "response_id already acknowledged for a different request"
            )
        input_tokens = view.tokens if modeled else meter.get("input_tokens")
        reads = (
            modeled_cached_input_tokens if modeled else meter.get("cached_input_tokens")
        )
        output = modeled_output_tokens if modeled else meter.get("output_tokens")
        writes = (
            modeled_cache_write_input_tokens
            if modeled
            else meter.get("cache_write_input_tokens")
        )
        if (
            verified_shared_static_tokens is None
            and self.verified_contiguous_prefix_reads
        ):
            verified_shared_static_tokens = self._last_header.static_tokens
        if output is not None:
            number(output, "output_tokens", integer=True)
        if verified_shared_static_tokens is not None:
            number(
                verified_shared_static_tokens,
                "verified_shared_static_tokens",
                integer=True,
            )
        request_eligible, capacity = _reusable_capacity(
            spec,
            input_tokens,
            reads,
            writes,
            reusable_input_tokens,
            meter.get("mixed_cache_write_ttls", False),
        )
        if input_tokens is None:
            capacity = None
        event = self.checker._observe_ping(
            self.scope,
            run_id=run_id,
            at=at,
            completed_at=completed_at,
            previous_refresh=self._refreshed,
            status=status,
            request_eligible=request_eligible,
            unverified_layout=meter.get("mixed_cache_write_ttls", False)
            and reusable_input_tokens is None,
            capacity=capacity,
            reads=reads,
            input_tokens=input_tokens,
            output_tokens=output,
            spec=spec,
            verified_static=verified_shared_static_tokens,
            modeled=modeled,
            modeled_read_share=self.policy.cache_read_share,
            write_tokens=writes,
            ping_has_private_suffix=ping_has_private_suffix,
        )
        if event is None:
            self._due = ()
            self._idle_arguments = None
            self.last_ping_receipt = {
                "event": "compache.ping_receipt",
                "outcome": "superseded",
                "refresh_confirmed": False,
                "continuation": {"decision": "ignore", "reason": "superseded"},
            }
            return False
        self.last_ping_receipt = event
        if not event["refresh_confirmed"]:
            self._due = ()
            self._idle_arguments = None
        if status != "completed":
            return False
        private = event["ping_has_private_suffix"]
        if response_id is not None:
            self._committed_response_ids.add(response_id)
        if input_tokens is not None and not modeled:
            ping_anchor = InputAnchor(view, input_tokens, response_id=response_id)
            if not private:
                estimate, kind = self._ping_accounting
                self._errors.observe(kind, estimate, input_tokens)
                self._anchor = ping_anchor
                self._anchor_header_key = self._measured_header_key
            repeat = estimate_input(view, ping_anchor)
            self._ping_accounting = (repeat, "unchanged")
            view = replace(view, tokens=input_tokens)
            self._ping_view = view
        self._part_meter.retain(view)
        if not event["refresh_confirmed"]:
            surviving = event["remaining_reuse_limit_tokens"]
            if surviving is not None:
                self._held_capacity = event["foreground_reusable_tokens"]
                if not self._held_capacity:
                    self._refreshed = None
                self.catalog.limit_publisher_reuse(
                    self._last_header,
                    at=at,
                    read_tokens=self._held_capacity,
                )
            return False
        self._held_capacity = event["foreground_reusable_tokens"]
        self.catalog.record(
            self._last_header, at=at, request_tokens=self._held_capacity
        )
        if not private:
            self._cached = view
        self._refreshed = at
        continuation = self._remaining_ping_decision(
            dispatched_at=at,
            completed_at=completed_at,
            capacity=capacity,
            reads=reads,
            writes=writes,
            output=output,
            foreground_bound=self._held_capacity if private else None,
        )
        if (
            event["output_budget_exceeded"]
            and continuation["due_times"]
            and not self.checker._resume_repriced_pings(
                self.scope, run_id=run_id, refreshed_at=at
            )
        ):
            continuation.update(decision="stop", reason="superseded", due_times=[])
        event["continuation"] = continuation
        self._due = tuple(continuation["due_times"])
        if not self._due:
            self._idle_arguments = None
        return True


def header_usage(header, reuse):
    """Serializable, content-free component accounting for one actual view."""
    return {
        "system_prompt_tokens": header.system_tokens,
        "default_tool_tokens": header.default_tool_tokens,
        "default_tool_breakdown": list(header.tool_breakdown),
        "dynamic_tool_tokens": header.dynamic_tool_tokens,
        "total_header_tokens": header.total_tokens,
        "static_tokens": header.static_tokens,
        "modeled_reusable_header_tokens": reuse.eligible_tokens,
        "reusable_system_tokens": reuse.system_tokens,
        "reusable_default_tool_tokens": reuse.default_tool_tokens,
        "unattributed_default_tool_tokens": header.unattributed_tool_tokens,
        "reuse_status": reuse.status,
        "cross_thread": reuse.cross_thread,
        "age_seconds": reuse.age_seconds,
        "count_source": header.source,
        "reuse_source": reuse.source,
    }


def price_header(spec, header, request_tokens, matching_tokens, read_share=1.0):
    """Model the header portion of this request's input charge once, in wire order."""
    eligible = spec.cache_ttl_seconds > 0 and request_tokens >= spec.cache_min_tokens
    factor = spec.input_price_factor(request_tokens)
    cold_rate = factor * (spec.write_multiplier if eligible else 1.0)
    reads = (
        min(request_tokens, matching_tokens)
        // spec.cache_block_tokens
        * spec.cache_block_tokens
    )
    reads = (
        int(reads * read_share) if eligible and reads >= spec.cache_min_tokens else 0
    )
    sizes = {
        "system_prompt": header.system_tokens,
        "default_tools": header.default_tool_tokens,
        "dynamic_tools": header.dynamic_tool_tokens,
    }
    components = {}
    for name in (*header.component_order, "dynamic_tools"):
        size = sizes[name]
        reused = min(size, reads)
        reads -= reused
        cost = reused * spec.read_multiplier * factor + (size - reused) * cold_rate
        components[name] = {
            "tokens": size,
            "read_tokens": reused,
            "input_equivalents": cost,
        }
    cost = sum(v["input_equivalents"] for v in components.values())
    return {
        "components": components,
        "header_input_equivalents": cost,
        "same_header_fully_cold_input_equivalents": header.total_tokens * cold_rate,
    }
