"""Threads from OpenTelemetry spans.

Spans exported as OTLP JSON become the rows `compache.langfuse` reads, and the
conversations are rebuilt there. A span is read as a model call, a retrieval
or a tool run by the OpenTelemetry GenAI conventions or by OpenInference, and
its conversation is its `gen_ai.conversation.id` or `session.id`.

    threads, report = pull("spans.jsonl")
    policy = compache.fit(threads)

`rows_from_spans` maps the spans, `threads_from_spans` rebuilds the
conversations, and `pull` reads an export file and does both.
"""

from __future__ import annotations

import gzip
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

from .errors import ContractError
from .langfuse import _loads, threads_from_observations

__all__ = ["pull", "rows_from_spans", "threads_from_spans"]

_CHAT = {"chat", "text_completion", "generate_content"}


# -- OTLP JSON -------------------------------------------------------------


def _value(value: Any) -> Any:
    """A plain value from an OTLP AnyValue, spelled in camelCase or snake_case."""
    if not isinstance(value, dict):
        return value
    for key, inner in value.items():
        kind = key.replace("_", "").lower()
        if kind == "intvalue":
            # A 64-bit integer is a decimal string in JSON.
            try:
                return int(inner)
            except (TypeError, ValueError):
                return inner
        if kind == "arrayvalue":
            return [_value(v) for v in (inner or {}).get("values") or ()]
        if kind == "kvlistvalue":
            return {
                pair.get("key"): _value(pair.get("value"))
                for pair in (inner or {}).get("values") or ()
            }
        if kind in ("stringvalue", "boolvalue", "doublevalue", "bytesvalue"):
            return inner
    return None


def _attributes(attributes: Any) -> dict:
    """Attributes by key, from OTLP's list of keys and typed values."""
    if isinstance(attributes, dict):
        return attributes
    return {
        pair.get("key"): _value(pair.get("value"))
        for pair in attributes or ()
        if isinstance(pair, dict)
    }


def _time(nanoseconds: Any) -> str | None:
    """An ISO time from nanoseconds since the epoch, which OTLP JSON writes as text.

    Always with six decimals, so that the order of the texts is that of the times.
    """
    try:
        nanoseconds = int(nanoseconds)
    except (TypeError, ValueError):
        return None
    if nanoseconds <= 0:
        return None
    moment = datetime.fromtimestamp(nanoseconds // 10**9, timezone.utc)
    return f"{moment:%Y-%m-%dT%H:%M:%S}.{nanoseconds % 10**9 // 1000:06d}Z"


def _spans(source: Any) -> Iterator[dict]:
    """Every span of an OTLP export, of several, or of a plain list of spans."""
    if isinstance(source, dict):
        batches = source.get("resourceSpans", source.get("resource_spans"))
        if batches is None:
            yield source
            return
        for batch in batches or ():
            scopes = batch.get("scopeSpans", batch.get("scope_spans"))
            for scope in scopes or ():
                yield from scope.get("spans") or ()
    elif isinstance(source, Iterable) and not isinstance(source, (str, bytes)):
        for item in source:
            yield from _spans(item)


# -- conventions -----------------------------------------------------------


def _first(attributes: dict, *keys: str) -> Any:
    return next((attributes[k] for k in keys if attributes.get(k) is not None), None)


def _with_system(system: Any, sent: Any) -> Any:
    """Messages with a system prompt that was recorded apart from them in front."""
    if not system:
        return sent
    if isinstance(sent, str):
        sent = [{"role": "user", "content": sent}]
    elif not isinstance(sent, list):
        sent = [sent] if sent else []
    return [{"role": "system", "content": system}, *sent]


def _buckets(given: Any, output: Any, read: Any = None, written: Any = None):
    """Langfuse's usage details, from an input count that includes cached tokens.

    A cache counter is kept only where one was recorded, so that a call without
    one does not read as a call that found nothing cached. Some instrumentations
    leave a zero out: a write counter alone still shows that nothing was read.
    """

    def count(value):
        return value if type(value) is int and value >= 0 else None

    given, output, read, written = map(count, (given, output, read, written))
    if given is None and output is None:
        return None
    given, cached = given or 0, (read or 0) + (written or 0)
    # An input smaller than its cached part was counted without it, as Anthropic does.
    details = {
        "input": given - cached if cached <= given else given,
        "output": output or 0,
    }
    if read is not None or written is not None:
        details["input_cached_tokens"] = read or 0
    if written is not None:
        details["input_cache_creation"] = written
    return details


def _part(part: Any) -> Any:
    """A part of a GenAI message, with a tool's answer where the schema has it.

    The schema names the answer `response`. An older example names it `result`.
    """
    if isinstance(part, dict) and "result" in part:
        return {"response": part["result"], **part}
    return part


def _spoken(messages: Any) -> Any:
    """GenAI messages, given as JSON text or as a value."""
    messages = _loads(messages)
    if not isinstance(messages, list):
        return messages
    return [
        {**message, "parts": [_part(part) for part in message["parts"]]}
        if isinstance(message, dict) and isinstance(message.get("parts"), list)
        else message
        for message in messages
    ]


def _genai(attributes: dict) -> dict:
    def count(*names: str) -> Any:
        return _first(attributes, *("gen_ai.usage." + name for name in names))

    operation = attributes.get("gen_ai.operation.name")
    sent = _spoken(attributes.get("gen_ai.input.messages"))
    said = _spoken(attributes.get("gen_ai.output.messages"))
    # Before the conventions named the operation, a span with a model was a call.
    if operation in _CHAT or (
        operation is None and "gen_ai.request.model" in attributes
    ):
        sent = _with_system(_loads(attributes.get("gen_ai.system_instructions")), sent)
        tools = _loads(attributes.get("gen_ai.tool.definitions"))
        model = _first(attributes, "gen_ai.response.model", "gen_ai.request.model")
        return {
            "type": "GENERATION",
            "input": {"tools": tools, "messages": sent} if tools else sent,
            "output": said,
            "model": model,
            # Each counter under its name of today, then the one it had before.
            "usageDetails": _buckets(
                count("input_tokens", "prompt_tokens"),
                count("output_tokens", "completion_tokens"),
                count("cache_read.input_tokens"),
                count("cache_write.input_tokens", "cache_creation.input_tokens"),
            ),
        }
    if operation == "execute_tool":
        return {
            "type": "TOOL",
            "input": attributes.get("gen_ai.tool.call.arguments"),
            "output": attributes.get("gen_ai.tool.call.result"),
        }
    if operation == "retrieval":
        return {
            "type": "RETRIEVER",
            "input": attributes.get("gen_ai.retrieval.query.text"),
            "output": attributes.get("gen_ai.retrieval.documents"),
        }
    return {"type": "SPAN", "input": sent, "output": said}


def _listed(attributes: dict, prefix: str) -> list[dict]:
    """The items of a list that OpenInference spreads over numbered keys."""
    items: dict[int, dict] = {}
    for key, value in attributes.items():
        if isinstance(key, str) and key.startswith(prefix):
            number, _, rest = key[len(prefix) :].partition(".")
            if number.isdigit():
                items.setdefault(int(number), {})[rest] = value
    return [items[number] for number in sorted(items)]


def _flat_message(item: dict) -> dict:
    """An OpenAI chat message from the keys of one OpenInference message."""
    message = {"role": item.get("message.role"), "content": item.get("message.content")}
    parts = _listed(item, "message.contents.")
    if parts and not message["content"]:
        message["content"] = [
            {
                "type": p.get("message_content.type"),
                "text": p.get("message_content.text"),
            }
            for p in parts
        ]
    calls = _listed(item, "message.tool_calls.")
    if calls:
        message["tool_calls"] = [
            {
                "id": call.get("tool_call.id"),
                "type": "function",
                "function": {
                    "name": call.get("tool_call.function.name"),
                    "arguments": call.get("tool_call.function.arguments"),
                },
            }
            for call in calls
        ]
    if item.get("message.tool_call_id"):
        message["tool_call_id"] = item["message.tool_call_id"]
    return message


def _openinference(attributes: dict) -> dict:
    kind = attributes.get("openinference.span.kind")
    sent, said = attributes.get("input.value"), attributes.get("output.value")
    if kind == "LLM":
        asked = _listed(attributes, "llm.input_messages.")
        answered = _listed(attributes, "llm.output_messages.")
        tools = [
            tool
            for item in _listed(attributes, "llm.tools.")
            if (tool := _loads(item.get("tool.json_schema")))
        ]
        if asked:
            sent = [_flat_message(item) for item in asked]
        return {
            "type": "GENERATION",
            "input": {"tools": tools, "messages": sent} if tools else sent,
            "output": [_flat_message(item) for item in answered] if answered else said,
            "model": _first(
                attributes,
                "llm.model_name",
                "llm.response.model_name",
                "llm.request.model_name",
            ),
            "usageDetails": _buckets(
                attributes.get("llm.token_count.prompt"),
                attributes.get("llm.token_count.completion"),
                attributes.get("llm.token_count.prompt_details.cache_read"),
                attributes.get("llm.token_count.prompt_details.cache_write"),
            ),
        }
    if kind == "RETRIEVER":
        found = [
            {"id": item.get("document.id"), "content": item.get("document.content")}
            for item in _listed(attributes, "retrieval.documents.")
        ]
        return {"type": "RETRIEVER", "input": sent, "output": found or said}
    return {"type": "TOOL" if kind == "TOOL" else "SPAN", "input": sent, "output": said}


def _fields(attributes: dict) -> dict:
    """What a span's attributes say it was: the type of row and what it holds."""
    if "openinference.span.kind" in attributes:
        fields = _openinference(attributes)
    else:
        fields = _genai(attributes)
    session = _first(attributes, "gen_ai.conversation.id", "session.id")
    fields["sessionId"] = None if session is None else str(session)
    return fields


# -- rows and threads ------------------------------------------------------


def _either(span: dict, camel: str, snake: str) -> Any:
    return span.get(camel, span.get(snake))


def _id(value: Any) -> str | None:
    """A span's ID as it is written. An empty or all-zero one is no span."""
    return value if isinstance(value, str) and value.strip("0") else None


def rows_from_spans(spans: Any) -> list[dict]:
    """Rows as `threads_from_observations` takes them, from OTLP JSON spans.

    `spans` is a parsed OTLP export (`resourceSpans`), an iterable of them as
    a JSON Lines export holds, or a plain iterable of spans. A span without a
    conversation of its own belongs to that of its trace.
    """
    rows = []
    sessions: dict[Any, str] = {}
    for span in _spans(spans):
        if not isinstance(span, dict):
            continue
        start = _time(_either(span, "startTimeUnixNano", "start_time_unix_nano"))
        if start is None:
            continue
        trace = _either(span, "traceId", "trace_id")
        row = {
            "id": _either(span, "spanId", "span_id"),
            "traceId": trace,
            "parentObservationId": _id(_either(span, "parentSpanId", "parent_span_id")),
            "name": span.get("name"),
            "startTime": start,
            "endTime": _time(_either(span, "endTimeUnixNano", "end_time_unix_nano")),
            **_fields(_attributes(span.get("attributes"))),
        }
        if row["sessionId"]:
            sessions.setdefault(trace, row["sessionId"])
        rows.append(row)
    for row in rows:
        row["sessionId"] = row["sessionId"] or sessions.get(row["traceId"])
    return rows


def threads_from_spans(
    spans: Any, *, observed_until: str | None = None
) -> tuple[list[dict], dict]:
    """Chat-log threads from OTLP JSON spans, and a report of what was left out.

    Every conversation that can be read becomes one thread that `fit`,
    `replay` and `profile` take as it is.
    """
    return threads_from_observations(
        rows_from_spans(spans), observed_until=observed_until
    )


def _exported(path: Path) -> list:
    """The exports of a file: one JSON document, or one per line."""
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as stream:
        text = stream.read()
    try:
        return [json.loads(text)]
    except json.JSONDecodeError:
        pass
    found = []
    for number, line in enumerate(text.splitlines(), 1):
        if line.strip():
            try:
                found.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ContractError(
                    f"{path}:{number}: invalid JSON: {exc.msg}"
                ) from exc
    return found


def pull(path: str | Path, *, until: str | None = None) -> tuple[list[dict], dict]:
    """Threads of an OTLP export file, ready for `fit`, and a report of the read.

    The file holds OTLP JSON, or JSON Lines with one export a line as a
    collector's file exporter writes, gzipped or not. `until` is when the
    export was made, for one that was not made just now.
    """
    now = datetime.now(timezone.utc).replace(microsecond=0)
    observed = until or now.isoformat().replace("+00:00", "Z")
    threads, report = threads_from_spans(_exported(Path(path)), observed_until=observed)
    return threads, {**report, "since": None, "until": observed}
