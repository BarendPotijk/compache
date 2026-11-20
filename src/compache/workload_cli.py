"""Command-line entry points for workload policy fitting, replay and header accounting."""

from __future__ import annotations

import json
from pathlib import Path

from .cli import _read_jsonl, _write_json
from .optimizer.cache_breaks import CacheScope
from .optimizer.contracts import ModelSpec, digest, number
from .optimizer.distribution import request_charge
from .optimizer.host_ledger import count_foreground_calls
from .optimizer.retention import RetentionRecipe
from .optimizer.shared_prefix import COMPONENTS, SharedPrefixCatalog
from .optimizer.workload import fit_workload, header_usage
from .optimizer.workload_replay import replay_workload
from .traces import ContractError, load_threads


def register_planner(groups):
    commands = groups.add_parser(
        "planner",
        help="Fit a workload policy and estimate its cost on held-out conversations",
    ).add_subparsers(dest="action", required=True)
    fit = commands.add_parser(
        "fit",
        help="Fit demand, retention, and optional header reuse on training traces",
    )
    replay = commands.add_parser(
        "replay", help="Compare a frozen planner with the preserve-all reference"
    )
    headers = commands.add_parser(
        "headers", help="Price the first request of a new thread with a shared header"
    )

    for command in (fit, replay):
        command.add_argument(
            "--calls-per-run",
            type=Path,
            help="JSON run-ID to physical foreground count, or {complete_run_ids, events} call ledger for this split",
        )
        command.add_argument(
            "--threads",
            type=Path,
            required=True,
            help="Normalized Compache threads (JSONL)",
        )
        command.add_argument(
            "--labels", type=Path, help="Dependency labels for these threads (JSONL)"
        )
        command.add_argument(
            "--observation-end",
            required=True,
            help="Timezone-aware ISO observation cutoff",
        )
        command.add_argument("--output", type=Path, required=True)
        command.add_argument(
            "--provider",
            default="declared",
            help="Provider/routing namespace for shared-header accounting",
        )
        command.add_argument(
            "--cache-key",
            default="",
            help="Shared cache routing key for this trace cohort",
        )
    for command in (fit, headers):
        command.add_argument("--model-spec", type=Path, required=True)
    for command in (fit, replay, headers):
        command.add_argument(
            "--header",
            type=Path,
            required=command is not fit,
            help="JSON file containing system_prompt, default_tools and optional component_order",
        )
        command.add_argument(
            "--encoding",
            metavar="model|NAME",
            help="Tokenize header with tiktoken's model mapping or a named encoding. "
            "Default: estimate from the model spec's bytes_per_token.",
        )

    fit.add_argument("--retention-floor", type=float, default=0.95)
    fit.add_argument(
        "--next-endpoint-reuse-fraction",
        type=Path,
        help="JSON object mapping nonterminal training run IDs to the reusable fraction of their published prefix on the next foreground input",
    )
    fit.add_argument("--max-unknown-change-rate", type=float, default=0.0)
    fit.add_argument(
        "--horizon-seconds",
        type=float,
        default=86400,
        help="Return observation horizon",
    )
    fit.add_argument(
        "--recipes",
        type=Path,
        help="JSON list of per-part historical retention recipes",
    )
    replay.add_argument(
        "--policy", type=Path, required=True, help="Output of planner fit"
    )
    replay.add_argument("--lookahead", type=int, default=3)
    replay.add_argument(
        "--branches", type=int, default=3, help="Quadrature strata per distribution"
    )
    replay.add_argument("--seed", type=int, default=0)
    replay.add_argument(
        "--sampler", choices=("brownian", "empirical"), default="brownian"
    )
    replay.add_argument("--max-pings", type=int, default=1)
    replay.add_argument(
        "--modeled-ping-output-tokens",
        type=int,
        help="Simulate billed output for each dispatched ping. The initial forecast comes from the model spec.",
    )
    replay.add_argument("--ping-lead-seconds", type=float, default=45.0)
    replay.add_argument("--reserve-tokens", type=int, default=0)
    replay.add_argument("--runtime-overhead-tokens", type=int, default=0)
    replay.add_argument(
        "--ambient-hit-probability",
        type=float,
        help="Override fitted cross-thread reuse probability for future returns",
    )
    replay.add_argument(
        "--no-share-headers",
        dest="share_headers",
        action="store_false",
        help="Isolate shared prefixes per thread in both arms",
    )
    headers.add_argument(
        "--additional-input-tokens",
        type=int,
        default=0,
        help="User or retrieval input following the static header",
    )
    headers.add_argument(
        "--gap-seconds",
        type=float,
        default=0.0,
        help="Age of the other thread's cached prefix at the new request",
    )
    headers.add_argument(
        "--output", type=Path, help="Optional JSON report; default: stdout"
    )


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _calls_per_run(path, *, model, provider, cache_key):
    if path is None:
        return None
    data = _read(path)
    if isinstance(data, dict) and set(data) == {"complete_run_ids", "events"}:
        if not isinstance(data["complete_run_ids"], list) or not isinstance(
            data["events"], list
        ):
            raise ContractError(
                "call ledger requires arrays of complete_run_ids and events"
            )
        return count_foreground_calls(
            data["events"],
            complete_run_ids=data["complete_run_ids"],
            model=model,
            provider=None if provider == "declared" else provider,
            cache_key=cache_key or None,
        )
    return data


def _header(path):
    value = _read(path)
    if (
        not isinstance(value, dict)
        or not set(COMPONENTS).issubset(value)
        or set(value) - {*COMPONENTS, "component_order"}
    ):
        raise ContractError(
            "header JSON requires system_prompt and default_tools; optional key: component_order"
        )
    order = value.get("component_order", COMPONENTS)
    if not isinstance(order, (tuple, list)) or tuple(order) not in (
        COMPONENTS,
        COMPONENTS[::-1],
    ):
        raise ContractError(
            "component_order must list system_prompt and default_tools exactly once"
        )
    return {**value, "component_order": tuple(order)}


def _header_ruler(name, spec):
    if name is None:
        return None, {
            "source": "estimated_from_bytes",
            "bytes_per_token": spec.bytes_per_token,
        }
    try:
        import tiktoken
    except ImportError as exc:
        raise ContractError(
            "--encoding requires tiktoken. Install tiktoken or omit --encoding for a byte estimate."
        ) from exc
    try:
        encoding = (
            tiktoken.encoding_for_model(spec.model_id)
            if name == "model"
            else tiktoken.get_encoding(name)
        )
    except KeyError as exc:
        raise ContractError(
            f"No tiktoken encoding mapping for {spec.model_id!r}. "
            "Pass --encoding with the host's encoding name or omit it for a byte estimate."
        ) from exc
    except ValueError as exc:
        raise ContractError(f"Cannot load tiktoken encoding {name!r}: {exc}") from exc

    def tokenize(data):
        return encoding.encode_ordinary(data.decode("utf-8"))

    return tokenize, {
        "source": "tiktoken",
        "encoding": encoding.name,
        "selection": name,
    }


def _read_recipes(path):
    values = _read(path)
    if (
        not isinstance(values, list)
        or not values
        or not all(isinstance(row, dict) for row in values)
    ):
        raise ContractError(
            "recipes must be a nonempty JSON list of retention recipe objects"
        )
    try:
        return [RetentionRecipe(**row) for row in values]
    except TypeError as exc:
        raise ContractError(f"invalid retention recipe: {exc}") from exc


def _header_report(args, spec, header, tokenize, ruler):
    number(args.additional_input_tokens, "additional_input_tokens", integer=True)
    number(args.gap_seconds, "gap_seconds")
    catalog = SharedPrefixCatalog()
    options = dict(spec=spec, tokenize=tokenize, **header)
    prior = catalog.measure(
        CacheScope("example", "modeled", spec.model_id, "prior"), **options
    )
    new = catalog.measure(
        CacheScope("example", "modeled", spec.model_id, "new"), **options
    )
    size = new.total_tokens + args.additional_input_tokens
    if size > spec.context_tokens or new.wire_bytes > spec.byte_limit:
        raise ContractError(
            "header example exceeds the model's input context or wire byte limit"
        )
    catalog.record(prior, at=0.0, request_tokens=size)
    reuse = catalog.lookup(new, at=args.gap_seconds, request_tokens=size)
    cold_cost = request_charge(spec, size, 0, 0)
    warm_cost = request_charge(spec, size, reuse.eligible_tokens, 0)
    return {
        "schema_version": "compache-header-example-v1",
        "model_id": spec.model_id,
        "header_token_ruler": ruler,
        "component_order": list(new.component_order),
        **header_usage(new, reuse),
        "unattributed_default_tool_tokens": new.unattributed_tool_tokens,
        "request_input_tokens": size,
        "additional_input_tokens": args.additional_input_tokens,
        "gap_seconds": args.gap_seconds,
        "cache_ttl_seconds": spec.cache_ttl_seconds,
        "cache_min_tokens": spec.cache_min_tokens,
        "cache_block_tokens": spec.cache_block_tokens,
        "cost_unit": "fresh input token equivalents",
        "cold_input_cost": cold_cost,
        "new_thread_input_cost": warm_cost,
        "input_savings": cold_cost - warm_cost,
        "input_savings_fraction": (cold_cost - warm_cost) / cold_cost
        if cold_cost
        else None,
        "new_thread_prefix_status": reuse.status,
        "new_thread_used_other_thread": reuse.cross_thread,
    }


def run_planner(args):
    if args.action == "replay":
        artifact = _read(args.policy)
        if (
            not isinstance(artifact, dict)
            or artifact.get("schema_version") != "compache-workload-planner-v1"
        ):
            raise ContractError("expected a fitted workload policy artifact")
        spec = ModelSpec.from_dict(artifact.get("model_spec"))
    else:
        spec = ModelSpec.from_dict(_read(args.model_spec))
    if args.encoding and not args.header:
        raise ContractError("--encoding on planner fit requires --header")
    header = _header(args.header) if args.header else None
    tokenize, ruler = _header_ruler(args.encoding, spec)

    if args.action == "headers":
        result = _header_report(args, spec, header, tokenize, ruler)
        if args.output is not None:
            _write_json(args.output, result)
        return result

    threads = load_threads(args.threads)
    counts = _calls_per_run(
        args.calls_per_run,
        model=spec.model_id,
        provider=args.provider,
        cache_key=args.cache_key,
    )
    labels = _read_jsonl(args.labels) if args.labels is not None else []
    if args.action == "fit":
        options = {"recipes": _read_recipes(args.recipes)} if args.recipes else {}
        if args.next_endpoint_reuse_fraction:
            options["next_endpoint_reuse_fraction"] = _read(
                args.next_endpoint_reuse_fraction
            )
        if header is not None:
            options.update(
                header={
                    **header,
                    "provider": args.provider,
                    "cache_key": args.cache_key,
                },
                tokenize=tokenize,
            )
        elif counts is None and (args.provider != "declared" or args.cache_key):
            raise ContractError(
                "--provider and --cache-key on planner fit apply to --header or --calls-per-run"
            )
        result = fit_workload(
            threads,
            labels,
            spec,
            observation_end=args.observation_end,
            horizon_seconds=args.horizon_seconds,
            retention_floor=args.retention_floor,
            max_unknown_change_rate=args.max_unknown_change_rate,
            calls_per_run=counts,
            **options,
        )
        if header is not None:
            result["header_token_ruler"] = ruler
        _write_json(args.output, result)
        return {
            "output": str(args.output),
            "training_threads": len(threads),
            "eligible_recipes": sum(row["eligible"] for row in result["recipes"]),
            "retention_floor": result["retention_floor"],
        }

    result = replay_workload(
        threads,
        labels,
        artifact,
        observation_end=args.observation_end,
        calls_per_run=counts,
        **header,
        tokenize=tokenize,
        provider=args.provider,
        cache_key=args.cache_key,
        lookahead=args.lookahead,
        branches=args.branches,
        seed=args.seed,
        sampler=args.sampler,
        max_pings=args.max_pings,
        modeled_ping_output_tokens=args.modeled_ping_output_tokens,
        ping_lead_seconds=args.ping_lead_seconds,
        reserve_tokens=args.reserve_tokens,
        overhead_tokens=args.runtime_overhead_tokens,
        share_headers=args.share_headers,
        ambient_hit_probability=args.ambient_hit_probability,
    )
    result["header_token_ruler"] = ruler
    result["shared_cache_scope"] = {
        "provider": args.provider,
        "cache_key_digest": digest(args.cache_key),
    }
    _write_json(args.output, result)
    return {
        "output": str(args.output),
        "reference": result["reference"],
        "savings_input_equivalents": result["savings_input_equivalents"],
        "savings_fraction": result["savings_fraction"],
        "heldout_threads": result["heldout_threads"],
        "candidate_complete": result["candidate"]["complete"],
        "baseline_complete": result["baseline"]["complete"],
    }
