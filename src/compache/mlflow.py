"""Threads from MLflow traces.

An MLflow trace is a turn, and the session it was logged with is its thread.
The spans of a trace become the rows `compache.langfuse` reads, and the
conversations are rebuilt there: a `CHAT_MODEL` or `LLM` span is a model call,
read in the shapes MLflow's OpenAI, Anthropic and LangChain autologging record.

    threads, report = pull(tracking_uri="http://localhost:5000", experiment_ids=["1"])
    policy = compache.fit(threads)

`fetch` reads the traces of a tracking server, `rows_from_traces` maps their
spans, `threads_from_traces` rebuilds the conversations, and `pull` does it
all. Nothing here needs the MLflow package.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Iterator

from . import _tls
from .errors import ContractError
from .langfuse import threads_from_observations
from .otel import _attributes, _buckets, _fields, _time, _with_system
from .threads import canonical_usage

__all__ = ["fetch", "pull", "rows_from_traces", "threads_from_traces"]

_TYPES = {
    "CHAT_MODEL": "GENERATION",
    "LLM": "GENERATION",
    "RETRIEVER": "RETRIEVER",
    "TOOL": "TOOL",
}
_SESSION = "mlflow.trace.session"
# Where the tracking store itself serves the spans of a trace. Elsewhere they
# come from the trace's artifact.
_SERVED = {"TRACKING_STORE", "ARCHIVE_REPO"}


# -- reading ---------------------------------------------------------------


def _send(url: str, headers: dict, timeout: float, body: bytes | None = None):
    """One request, a POST when it has a body: the status, the headers and the reply."""
    request = urllib.request.Request(url, data=body, headers=headers)
    try:
        with urllib.request.urlopen(
            request, timeout=timeout, context=_tls.FOUND["context"]
        ) as response:
            return response.status, dict(response.headers), response.read()
    except urllib.error.HTTPError as error:
        return error.code, dict(error.headers), error.read()
    except urllib.error.URLError as error:
        if _tls.unverified(error) and _tls.fall_back():
            return _send(url, headers, timeout, body)
        raise


def _millis(moment: str) -> int:
    try:
        parsed = datetime.fromisoformat(moment.strip().replace("Z", "+00:00"))
    except (AttributeError, ValueError) as exc:
        raise ContractError(f"{moment!r}: expected an ISO timestamp") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.timestamp() * 1000)


def _refusal(reply: bytes) -> str | None:
    """The message of an MLflow error reply. None for a reply that is not MLflow's."""
    try:
        return str(json.loads(reply)["message"])[:300]
    except (ValueError, KeyError, TypeError):
        return None


def _detail(reply: bytes) -> str:
    return _refusal(reply) or reply.decode("utf-8", "replace")[:300]


def fetch(
    *,
    tracking_uri: str | None = None,
    experiment_ids: Iterable[str] = ("0",),
    since: str | None = None,
    until: str | None = None,
    filter: str | None = None,
    token: str | None = None,
    page: int = 100,
    timeout: float = 120,
    pause=time.sleep,
) -> Iterator[dict]:
    """Traces of MLflow experiments that began between two times, each with its spans.

    The server and the token default to MLFLOW_TRACKING_URI and
    MLFLOW_TRACKING_TOKEN. `filter` is a further condition in MLflow's own
    search syntax. Traces come over the REST API of MLflow 3, oldest first,
    as `{"trace_info": ..., "spans": [...]}`.
    """
    host = (tracking_uri or os.environ.get("MLFLOW_TRACKING_URI") or "").rstrip("/")
    if not host:
        raise ContractError("MLflow server missing: set MLFLOW_TRACKING_URI")
    if not host.startswith(("http://", "https://")):
        raise ContractError(
            f"{host}: traces are read from the http address of a tracking server"
        )
    token = token or os.environ.get("MLFLOW_TRACKING_TOKEN")
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "User-Agent": "compache",
    }
    if token:
        headers["Authorization"] = "Bearer " + token
    if isinstance(experiment_ids, (str, int)):
        experiment_ids = [experiment_ids]
    conditions = [filter] if filter else []
    if since:
        conditions.append(f"trace.timestamp_ms >= {_millis(since)}")
    if until:
        conditions.append(f"trace.timestamp_ms < {_millis(until)}")
    query: dict[str, Any] = {
        "locations": [
            {
                "type": "MLFLOW_EXPERIMENT",
                "mlflow_experiment": {"experiment_id": str(experiment)},
            }
            for experiment in experiment_ids
        ],
        # Oldest first: traces logged during the read land after the pages already read.
        "order_by": ["timestamp_ms ASC"],
        "max_results": min(500, max(1, page)),
    }
    if conditions:
        query["filter"] = " AND ".join(conditions)

    def ask(url: str, body: bytes | None = None, patient: bool = True):
        """The status and reply of one call, tried again while the server is busy."""
        failures = 0
        while True:
            try:
                status, reply, raw = _send(url, headers, timeout, body)
            except (urllib.error.URLError, TimeoutError) as error:
                reason = str(getattr(error, "reason", error))
                if "CERTIFICATE_VERIFY_FAILED" in reason:
                    raise ContractError(
                        f"{host}: this Python has no root certificates to verify the host. "
                        "Install them, or export the traces and pass them."
                    ) from error
                status, reply, raw = 599, {}, reason.encode()
            if status in (401, 403):
                raise ContractError(f"{host} rejected the MLflow token ({status})")
            failures += 1
            if status == 429 and failures <= 8:
                waits = {k.lower(): v for k, v in reply.items()}.get("retry-after", "")
                pause(min(120.0, float(waits) if waits.isdigit() else 5.0 * failures))
            elif status >= 500 and patient and failures <= 8:
                pause(1.0)
            else:
                return status, raw

    def spans(info: dict) -> list:
        trace = urllib.parse.quote(str(info.get("trace_id")), safe="")
        if (info.get("tags") or {}).get("mlflow.trace.spansLocation") in _SERVED:
            # Not tried again: a store without the spans answers with a server error.
            status, raw = ask(
                f"{host}/api/3.0/mlflow/traces/get?trace_id={trace}&allow_partial=true",
                patient=False,
            )
            if status == 200:
                return (json.loads(raw).get("trace") or {}).get("spans") or []
        status, raw = ask(
            f"{host}/ajax-api/3.0/mlflow/get-trace-artifact?request_id={trace}"
        )
        if status == 200:
            return json.loads(raw).get("spans") or []
        if status == 404:
            # A trace that has logged no spans yet.
            return []
        trace = info.get("trace_id")
        raise ContractError(
            f"MLflow keeps failing on trace {trace} ({status}): {_detail(raw)}"
        )

    seen: set = set()
    while True:
        status, raw = ask(
            f"{host}/api/3.0/mlflow/traces/search", json.dumps(query).encode()
        )
        if status == 404 and _refusal(raw) is None:
            raise ContractError(
                f"{host} has no trace search API ({status}). It needs MLflow 3."
            )
        if status in (400, 404):
            # An unknown experiment is a 404 of MLflow's own on some stores.
            raise ContractError(f"MLflow refused the query: {_detail(raw)}")
        if status != 200:
            raise ContractError(f"MLflow keeps failing ({status}): {_detail(raw)}")
        data = json.loads(raw)
        for info in data.get("traces") or []:
            if info.get("trace_id") not in seen:
                seen.add(info.get("trace_id"))
                yield {"trace_info": info, "spans": spans(info)}
        if not data.get("next_page_token"):
            return
        query["page_token"] = data["next_page_token"]


# -- one span --------------------------------------------------------------


def _unquoted(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


def _request(sent: Any, tools: Any) -> Any:
    """What a model call was sent, as messages and tools the Langfuse reader takes."""
    if isinstance(sent, dict):
        tools = sent.get("tools") or tools
        asked = next(
            (
                sent[key]
                for key in ("messages", "input", "prompt")
                if isinstance(sent.get(key), (list, str))
            ),
            None,
        )
        if asked is not None:
            # Anthropic's system prompt and the instructions of OpenAI's Responses
            # API go in front.
            sent = _with_system(sent.get("system") or sent.get("instructions"), asked)
    elif isinstance(sent, list) and all(isinstance(p, (list, str)) for p in sent):
        # LangChain's own record: a prompt for each call of a batch, its messages
        # named by `type`.
        prompt = sent[0] if len(sent) == 1 else None
        if isinstance(prompt, list):
            prompt = [
                {"role": m.get("type"), **m} if isinstance(m, dict) else m
                for m in prompt
            ]
        sent = prompt
    return {"tools": tools, "messages": sent} if tools else sent


def _reply(said: Any) -> Any:
    """What a model call answered, as a message the Langfuse reader takes."""
    if not isinstance(said, dict):
        return said
    choices = said.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        # OpenAI's chat completions, and LangChain's answers as MLflow rewrites them.
        return choices[0].get("message") or choices[0].get("text")
    if isinstance(said.get("output"), list):
        # OpenAI's Responses API.
        return said["output"]
    batches = said.get("generations")
    if isinstance(batches, list) and len(batches) == 1:
        # LangChain's own record of one call: the text of its first generation.
        first = batches[0][0] if isinstance(batches[0], list) and batches[0] else None
        if isinstance(first, dict):
            return first.get("text")
    # Anthropic's message holds its content blocks, which the Langfuse reader takes.
    return said


def _result(said: Any) -> Any:
    """What LangChain's own record keeps beside the answers: the model and the usage."""
    return said.get("llm_output") if isinstance(said, dict) else None


def _model(sent: Any, said: Any, attributes: dict) -> str | None:
    """The model that answered: the name in the response before the one asked for."""
    asked = attributes.get("invocation_params")
    for holder, key in (
        (said, "model"),
        (_result(said), "model_name"),
        (attributes, "mlflow.llm.model"),
        (sent, "model"),
        (asked, "model"),
        (asked, "model_name"),
    ):
        if (
            isinstance(holder, dict)
            and isinstance(holder.get(key), str)
            and holder[key]
        ):
            return holder[key]
    return None


def _spent(said: Any, counted: Any) -> dict | None:
    """Usage of a model call: the provider's own counts where the response holds them.

    Otherwise MLflow's count, whose input includes the cached tokens. For
    Anthropic it did not in MLflow 3.10 and 3.11, which shows only where the
    cached tokens exceed the input.
    """
    for holder, key in ((said, "usage"), (_result(said), "token_usage")):
        raw = holder.get(key) if isinstance(holder, dict) else None
        if canonical_usage(raw) is not None:
            return raw
    if not isinstance(counted, dict):
        return None
    return _buckets(
        counted.get("input_tokens"),
        counted.get("output_tokens"),
        counted.get("cache_read_input_tokens"),
        counted.get("cache_creation_input_tokens"),
    )


def _row(span: Any, trace: Any, session: Any) -> dict | None:
    """One span, in either of MLflow's two encodings, as a Langfuse observation row."""
    if not isinstance(span, dict):
        return None
    attributes = span.get("attributes")
    if isinstance(attributes, dict):
        # As stored and as `Trace.to_dict()` gives it: every value is JSON text.
        attributes = {key: _unquoted(value) for key, value in attributes.items()}
    else:
        # As `traces/get` gives it: OTLP's typed values.
        attributes = _attributes(attributes)
    start = _time(span.get("start_time_unix_nano"))
    if start is None:
        return None
    sent, said = (
        attributes.get("mlflow.spanInputs"),
        attributes.get("mlflow.spanOutputs"),
    )
    row = {
        "id": span.get("span_id"),
        "traceId": trace
        or attributes.get("mlflow.traceRequestId")
        or span.get("trace_id"),
        "parentObservationId": span.get("parent_span_id") or None,
        "type": _TYPES.get(attributes.get("mlflow.spanType"), "SPAN"),
        "name": span.get("name"),
        "startTime": start,
        "endTime": _time(span.get("end_time_unix_nano")),
        "input": sent,
        "output": said,
    }
    if "openinference.span.kind" in attributes or "gen_ai.operation.name" in attributes:
        # A span that came in over OpenTelemetry still holds what it was sent with.
        row.update(_fields(attributes))
    elif row["type"] == "GENERATION":
        row.update(
            input=_request(sent, attributes.get("mlflow.chat.tools")),
            output=_reply(said),
            model=_model(sent, said, attributes),
            usageDetails=_spent(said, attributes.get("mlflow.chat.tokenUsage")),
        )
    row["sessionId"] = session
    return row


# -- traces and threads ----------------------------------------------------


def rows_from_traces(traces: Any) -> list[dict]:
    """Rows as `threads_from_observations` takes them, from MLflow traces.

    A trace is its info and its spans, as `fetch` yields it, as the `trace` of
    a `traces/get` reply, or as `Trace.to_dict()` and `mlflow traces search
    --output json` write it. `traces` is an iterable of them, or one reply
    that holds them under `traces`.
    """
    if isinstance(traces, dict):
        traces = traces.get("traces") or [traces]
    rows = []
    for trace in traces or ():
        if not isinstance(trace, dict):
            continue
        trace = trace.get("trace") or trace
        info = trace.get("trace_info") or trace.get("info") or {}
        session = (info.get("trace_metadata") or {}).get(_SESSION)
        found = trace.get("spans") or (trace.get("data") or {}).get("spans") or ()
        for span in found:
            row = _row(span, info.get("trace_id"), session)
            if row is not None:
                rows.append(row)
    return rows


def threads_from_traces(
    traces: Any, *, observed_until: str | None = None
) -> tuple[list[dict], dict]:
    """Chat-log threads from MLflow traces, and a report of what was left out.

    Every session with a readable conversation becomes one thread that `fit`,
    `replay` and `profile` take as it is.
    """
    return threads_from_observations(
        rows_from_traces(traces), observed_until=observed_until
    )


def pull(
    *,
    tracking_uri: str | None = None,
    experiment_ids: Iterable[str] = ("0",),
    days: float = 30,
    since: str | None = None,
    until: str | None = None,
    traces: Iterable[dict] | None = None,
    **options,
) -> tuple[list[dict], dict]:
    """Threads of MLflow experiments, ready for `fit`, and a report of the read.

    Reads the last `days` days unless `since` says otherwise. `options` go to
    `fetch`: `filter`, `token`, the page size and the timeout. `traces` takes
    traces already exported, and nothing is fetched.
    """
    now = datetime.now(timezone.utc).replace(microsecond=0)
    if traces is None:
        if since is None:
            since = (now - timedelta(days=days)).isoformat().replace("+00:00", "Z")
        traces = fetch(
            tracking_uri=tracking_uri,
            experiment_ids=experiment_ids,
            since=since,
            until=until,
            **options,
        )
    observed = until or now.isoformat().replace("+00:00", "Z")
    threads, report = threads_from_traces(traces, observed_until=observed)
    return threads, {**report, "since": since, "until": observed}
