"""Threads from a Langfuse project.

A Langfuse session is a thread. Each model call is recorded with the messages
it was sent, so a conversation is rebuilt from what every call added: the
messages after the last assistant message of its input, and then its answer.

    threads, report = pull(since="2026-11-01T00:00:00Z", tags=["chat"])
    policy = compache.fit(threads)

`fetch` reads the observations, `threads_from_observations` rebuilds the
conversations, and `pull` does both. Nothing here needs the Langfuse SDK.
"""

from __future__ import annotations

import base64
import json
import os
import re
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Iterator

from .errors import ContractError
from .threads import Thread, cache_read_reported, canonical_usage, normalize_thread

__all__ = ["fetch", "pull", "threads_from_observations"]

CLOUD = "https://cloud.langfuse.com"
_FIELDS = "core,basic,io,model,usage"
_ROLES = {
    "system": "system",
    "developer": "system",
    "user": "user",
    "human": "user",
    "assistant": "assistant",
    "ai": "assistant",
    "model": "assistant",
    "tool": "tool",
    "function": "tool",
}
_RESULT_BLOCKS = {"tool_result", "tool_call_response", "function_call_output"}
_TEXT_BLOCKS = {None, "text", "input_text", "output_text"}
_PIECE_KEYS = ("page_content", "content", "text", "document", "body")
# A request that resends less than this share of the conversation has trimmed it.
_RESENT_SHARE = 0.8


# -- reading ---------------------------------------------------------------


_CERTIFICATES: dict[str, Any] = {"context": None}


def _other_certificates() -> ssl.SSLContext | None:
    """Root certificates from certifi or the system bundle, for a Python that ships none."""
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        pass
    for bundle in (
        "/etc/ssl/cert.pem",
        "/etc/ssl/certs/ca-certificates.crt",
        "/etc/pki/tls/certs/ca-bundle.crt",
    ):
        if os.path.exists(bundle):
            return ssl.create_default_context(cafile=bundle)
    return None


def _get(url: str, headers: dict, timeout: float):
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(
            request, timeout=timeout, context=_CERTIFICATES["context"]
        ) as response:
            return response.status, dict(response.headers), response.read()
    except urllib.error.HTTPError as error:
        return error.code, dict(error.headers), error.read()
    except urllib.error.URLError as error:
        unverified = "CERTIFICATE_VERIFY_FAILED" in str(error.reason)
        if unverified and _CERTIFICATES["context"] is None:
            _CERTIFICATES["context"] = _other_certificates()
            if _CERTIFICATES["context"] is not None:
                return _get(url, headers, timeout)
        raise


def fetch(
    *,
    since: str | None = None,
    until: str | None = None,
    environment: str | None = None,
    tags: Iterable[str] = (),
    filters: Iterable[dict] = (),
    public_key: str | None = None,
    secret_key: str | None = None,
    base_url: str | None = None,
    page: int = 100,
    timeout: float = 120,
    pause=time.sleep,
) -> Iterator[dict]:
    """Observation rows of a Langfuse project between two start times.

    Keys and host default to LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY and
    LANGFUSE_BASE_URL. `tags` keeps observations of traces with any of them,
    and `filters` are further conditions in Langfuse's own filter form. Rows
    come from the observations API of Langfuse 4, newest first.
    """
    public_key = public_key or os.environ.get("LANGFUSE_PUBLIC_KEY")
    secret_key = secret_key or os.environ.get("LANGFUSE_SECRET_KEY")
    if not public_key or not secret_key:
        raise ContractError(
            "Langfuse keys missing: set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY"
        )
    host = (
        base_url
        or os.environ.get("LANGFUSE_BASE_URL")
        or os.environ.get("LANGFUSE_HOST")
        or CLOUD
    ).rstrip("/")
    query: dict[str, Any] = {"fields": _FIELDS}
    if since:
        query["fromStartTime"] = since
    if until:
        query["toStartTime"] = until
    if environment:
        query["environment"] = environment
    conditions = list(filters)
    tags = list(tags)
    if tags:
        conditions.append(
            {
                "type": "arrayOptions",
                "column": "tags",
                "operator": "any of",
                "value": tags,
            }
        )
    if conditions:
        query["filter"] = json.dumps(conditions, separators=(",", ":"))
    token = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()
    headers = {
        "Authorization": "Basic " + token,
        "Accept": "application/json",
        "User-Agent": "compache",
    }
    cursor, limit, failures = None, max(1, page), 0
    while True:
        params = {**query, "limit": limit}
        if cursor:
            params["cursor"] = cursor
        url = f"{host}/api/public/v2/observations?{urllib.parse.urlencode(params)}"
        try:
            status, reply, body = _get(url, headers, timeout)
        except (urllib.error.URLError, TimeoutError) as error:
            reason = str(getattr(error, "reason", error))
            if "CERTIFICATE_VERIFY_FAILED" in reason:
                raise ContractError(
                    f"{host}: this Python has no root certificates to verify the host. "
                    "Install them, or export the observations and pass the file."
                ) from error
            status, reply, body = 599, {}, reason.encode()
        if status == 200:
            data = json.loads(body)
            failures = 0
            yield from data.get("data") or []
            cursor = (data.get("meta") or {}).get("cursor")
            if not cursor:
                return
            continue
        detail = body.decode("utf-8", "replace")[:300]
        if status in (401, 403):
            raise ContractError(f"{host} rejected the Langfuse keys ({status})")
        if status in (404, 410):
            raise ContractError(
                f"{host} has no observations API v2 ({status}). It needs Langfuse 4."
            )
        if status == 400:
            raise ContractError(f"Langfuse refused the query: {detail}")
        failures += 1
        if failures > 8:
            raise ContractError(f"Langfuse keeps failing ({status}): {detail}")
        if status == 429:
            waits = {k.lower(): v for k, v in reply.items()}.get("retry-after", "")
            pause(min(120.0, float(waits) if waits.isdigit() else 5.0 * failures))
        else:
            # A page too large to serve: ask for fewer rows.
            limit = max(1, limit // 2)
            pause(1.0)


# -- one row ---------------------------------------------------------------


def _loads(value: Any) -> Any:
    """Langfuse returns inputs and outputs as text. JSON where it parses."""
    if isinstance(value, str) and value.lstrip()[:1] in ("[", "{"):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


def _plain(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        text = _blocks(value)[0]
        if text:
            return text
    return json.dumps(value, ensure_ascii=False)


def _blocks(content: Any) -> tuple[str, list[str]]:
    """The text of a message's content, and the tool results it carries."""
    if content is None:
        return "", []
    if isinstance(content, str):
        return content, []
    if isinstance(content, dict):
        content = [content]
    if not isinstance(content, list):
        return "", []
    texts, results = [], []
    for block in content:
        if isinstance(block, str):
            texts.append(block)
        elif isinstance(block, dict):
            kind = block.get("type")
            if kind in _RESULT_BLOCKS:
                inner = block.get("content", block.get("response", block.get("output")))
                results.append(_plain(inner))
            elif kind in _TEXT_BLOCKS:
                # OpenTelemetry message parts keep their text under `content`.
                text = block.get("text", block.get("content"))
                if isinstance(text, str):
                    texts.append(text)
            # Tool calls, reasoning and images add no text.
    return "\n\n".join(t for t in texts if t), [r for r in results if r]


def _message(raw: Any) -> dict | None:
    if isinstance(raw, str):
        return {"role": "user", "text": raw, "results": []}
    if not isinstance(raw, dict):
        return None
    kind = raw.get("type")
    if kind == "function_call_output":
        return {"role": "tool", "text": "", "results": [_plain(raw.get("output"))]}
    role = _ROLES.get(str(raw.get("role") or "").lower())
    if role is None:
        if kind in ("function_call", "reasoning"):
            return {"role": "assistant", "text": "", "results": []}
        return None
    text, results = _blocks(raw["parts"] if "parts" in raw else raw.get("content"))
    if role == "tool":
        results, text = ([text] if text else []) + results, ""
    return {"role": role, "text": text, "results": results}


def _messages(value: Any) -> list[dict]:
    """The messages a model call was sent, from any of the shapes integrations record."""
    value = _loads(value)
    if isinstance(value, dict):
        for key in ("messages", "input"):
            if isinstance(value.get(key), (list, str)):
                return _messages(value[key])
        if "role" in value:
            message = _message(value)
            return [message] if message else []
        return []
    if isinstance(value, list):
        return [m for m in map(_message, value) if m is not None]
    if isinstance(value, str) and value.strip():
        return [{"role": "user", "text": value, "results": []}]
    return []


def _answer(value: Any) -> str:
    value = _loads(value)
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        message = _message({**value, "role": "assistant"})
        return message["text"] if message else ""
    if isinstance(value, list):
        found = [_message(v) if isinstance(v, dict) else None for v in value]
        return "\n\n".join(
            m["text"] for m in found if m and m["role"] == "assistant" and m["text"]
        )
    return ""


def _asked(value: Any) -> str:
    """The user's message in the input of a trace's root, where it can be told."""
    value = _loads(value)
    if isinstance(value, str):
        return value.strip()
    users = [m["text"] for m in _messages(value) if m["role"] == "user" and m["text"]]
    if users:
        return users[-1]
    if isinstance(value, dict):
        named = next(
            (
                value[k]
                for k in ("question", "query", "message", "prompt", "user_input")
                if isinstance(value.get(k), str) and value[k].strip()
            ),
            None,
        )
        if named:
            return named.strip()
    # The arguments of a traced function: a single text among them is the message.
    leaves: list[str] = []

    def walk(item, depth):
        if isinstance(item, str):
            if item.strip():
                leaves.append(item.strip())
        elif depth < 3 and isinstance(item, (list, dict)):
            for inner in item.values() if isinstance(item, dict) else item:
                walk(inner, depth + 1)

    walk(value, 0)
    return leaves[0] if len(leaves) == 1 else ""


def _tools(value: Any) -> list | None:
    value = _loads(value)
    if isinstance(value, dict) and isinstance(value.get("tools"), list):
        return value["tools"] or None
    return None


def _pieces(value: Any) -> list[str]:
    """The texts a retrieval returned."""
    value = _loads(value)
    if isinstance(value, str):
        value = [value]
    elif isinstance(value, dict):
        nested = next(
            (
                value[k]
                for k in ("documents", "results", "chunks", "nodes")
                if isinstance(value.get(k), list)
            ),
            None,
        )
        value = nested if nested is not None else [value]
    found = []
    for item in value if isinstance(value, list) else []:
        if isinstance(item, str):
            found.append(item)
        elif isinstance(item, dict):
            text = next(
                (item[k] for k in _PIECE_KEYS if isinstance(item.get(k), str)), None
            )
            if text:
                found.append(text)
    return [p for p in found if len(p.strip()) >= 20]


def _usage(details: Any) -> dict | None:
    """Canonical token counters from Langfuse's usage details.

    Langfuse stores exclusive buckets: `input` is what was neither read from
    the cache nor written to it, and `output` leaves reasoning out. A `total`
    that the buckets do not add up to marks a shape this does not know.
    """
    usage = canonical_usage(details)
    if usage is not None or not isinstance(details, dict):
        return usage
    counts = {k: v for k, v in details.items() if type(v) is int and v >= 0}
    total = counts.pop("total", None)

    def some(test):
        return sum(v for k, v in counts.items() if test(k))

    read = some(lambda k: "cache" in k and ("read" in k or "cached" in k))
    written = some(lambda k: "cache" in k and ("creation" in k or "write" in k))
    given = some(lambda k: k.startswith("input")) + some(
        lambda k: "cache" in k and not k.startswith(("input", "output"))
    )
    output = some(lambda k: k.startswith("output"))
    if not given and not output:
        return None
    if total is not None and given + output != total:
        return None
    if read + written > given:
        return None
    return {
        "input": given,
        "output": output,
        "cache_read": read,
        "cache_write": written,
    }


def _slim(row: dict) -> dict:
    """What the rebuild needs from a row, without the whole input it came with."""
    kind = row.get("type")
    slim = {
        "id": row.get("id"),
        "trace": row.get("traceId"),
        "parent": row.get("parentObservationId"),
        "session": row.get("sessionId"),
        "type": kind,
        "start": row.get("startTime"),
        "end": row.get("endTime") or row.get("startTime"),
    }
    if kind == "GENERATION":
        sent = _messages(row.get("input"))
        cut = max(
            (i for i, m in enumerate(sent) if m["role"] == "assistant"), default=-1
        )
        spoken = [m for m in sent if m["role"] != "system"]
        slim.update(
            prior=sum(m["role"] == "assistant" for m in sent),
            last=sent[cut]["text"] if cut >= 0 else None,
            new=[m for m in sent[cut + 1 :] if m["role"] != "system"],
            size=sum(len(m["text"]) + sum(map(len, m["results"])) for m in spoken),
            system=next(
                (m["text"] for m in sent if m["role"] == "system" and m["text"]), None
            ),
            tools=_tools(row.get("input")),
            answer=_answer(row.get("output")),
            model=row.get("model") or None,
            usage=_usage(row.get("usageDetails")),
            read_reported=cache_read_reported(row.get("usageDetails")),
            text=bool(sent),
        )
    elif kind in ("RETRIEVER", "TOOL"):
        slim["pieces"] = _pieces(row.get("output"))
    elif slim["parent"] is None:
        # The root of a trace: what the user asked and what came back.
        slim["asked"] = _asked(row.get("input"))
        slim["said"] = _answer(row.get("output"))
    return slim


# -- one session -----------------------------------------------------------


def _same(a: str | None, b: str | None) -> bool:
    """The same answer, allowing for one side being trimmed or cut short."""
    a, b = (a or "").strip(), (b or "").strip()
    if a == b:
        return True
    short, long = sorted((a, b), key=len)
    return len(short) >= 40 and long.startswith(short)


def _one_per_call(calls: list[dict]) -> tuple[list[dict], int]:
    """One record per model call, where two instrumentations recorded the same one."""
    by_id = {c["id"]: c for c in calls}
    dropped: set = set()
    for call in calls:
        outer = by_id.get(call["parent"])
        if outer is not None and outer["id"] not in dropped:
            # A model call recorded inside another one. Keep the one with the text.
            dropped.add(
                call["id"] if outer["text"] or not call["text"] else outer["id"]
            )
    kept: list[dict] = []
    for call in calls:
        if call["id"] in dropped:
            continue
        twin = kept[-1] if kept else None
        if twin and (twin["trace"], twin["start"], twin["end"]) == (
            call["trace"],
            call["start"],
            call["end"],
        ):
            # Recorded twice side by side: same trace, same start and end.
            if call["text"] and not twin["text"]:
                kept[-1] = call
            dropped.add(call["id"])
            continue
        kept.append(call)
    return kept, len(calls) - len(kept)


def _conversation(calls: list[dict]) -> tuple[list[dict], bool]:
    """The calls that carry the conversation, and whether it began before the record.

    A call continues a conversation when the last answer in its input is that
    conversation's last answer. Helper calls, such as a query rewriter, start
    chains of their own and are left out.
    """
    # Most recently continued last.
    chains: list[dict] = []
    for call in calls:
        chain = None
        if call["prior"]:
            said = (call["last"] or "").strip()
            for candidate in reversed(chains):
                ended = (candidate["last"] or "").strip()
                # After a tool call that said nothing, the latest chain waiting on one.
                if _same(ended, said) if said else not ended:
                    chain = candidate
                    break
        if chain is None:
            chain = {"calls": [], "turns": 0, "last": None, "cut": bool(call["prior"])}
        else:
            chains.remove(chain)
        chains.append(chain)
        chain["calls"].append(call)
        chain["turns"] += any(m["role"] == "user" and m["text"] for m in call["new"])
        chain["last"] = call["answer"]
    best = max(chains, key=lambda c: (c["turns"], len(c["calls"])))
    return best["calls"], best["cut"]


def _unpaste(text: str, pieces: Iterable[str]) -> tuple[str, list[str]]:
    """`text` without the retrieved pieces pasted into it, and those pieces."""
    taken: list[str] = []
    for piece in pieces:
        if piece in text:
            text, taken = text.replace(piece, "", 1), taken + [piece]
    if not taken:
        return text, taken
    return re.sub(r"\n{3,}", "\n\n", text).strip(), taken


def _retrieved_before(call: dict, found) -> list[str]:
    return [
        piece
        for row in found.get(call["trace"], ())
        if row["end"] <= call["start"]
        for piece in row["pieces"]
    ]


def _chat_log(calls, found, trace_start, trace_first) -> tuple[list[dict], dict]:
    """Messages of one thread in the chat-log form, and what was noticed on the way.

    Retrieved text that the app pasted into the prompt, in the user message or
    the system prompt, is taken out of it and recorded as retrieved text.
    """
    prompts: Counter = Counter()
    messages: list[dict] = []
    seen = {"turns": 0, "pasted": 0, "trimmed": False, "overlap": 0}
    held = 0
    last_end = None
    for call in calls:
        at = call["start"]
        if trace_first.get(call["trace"]) == call["id"]:
            at = min(at, trace_start[call["trace"]])
        if last_end is not None and at < last_end:
            at, seen["overlap"] = last_end, seen["overlap"] + 1
        near = _retrieved_before(call, found)
        system, in_system = _unpaste(call["system"] or "", near)
        if system:
            prompts[system] += 1
        added = 0
        for message in call["new"]:
            if message["role"] == "user" and message["text"]:
                rest, pasted = _unpaste(message["text"], near)
                if not rest:
                    rest, pasted = message["text"], []
                pasted, in_system = pasted + in_system, []
                seen["pasted"] += bool(pasted)
                messages.append({"role": "user", "content": rest, "at": at})
                messages += [{"role": "tool", "content": p} for p in pasted]
                seen["turns"] += 1
                added += len(message["text"]) + sum(map(len, pasted))
            for result in message["results"]:
                messages.append({"role": "tool", "content": result})
                added += len(result)
        if held and call["size"] < _RESENT_SHARE * (held + added):
            seen["trimmed"] = True
        held += added + len(call["answer"])
        entry = {"role": "assistant", "content": call["answer"], "at": call["end"]}
        if call["usage"]:
            entry["usage"] = call["usage"]
            entry["cache_read_reported"] = call["read_reported"]
        if call["model"]:
            entry["model"] = call["model"]
        messages.append(entry)
        last_end = call["end"]
    if prompts:
        messages.insert(0, {"role": "system", "content": prompts.most_common(1)[0][0]})
    return messages, seen


def _trace_log(calls, roots, found) -> tuple[list[dict], dict]:
    """Messages of a thread whose turns are its traces: the root's input and output.

    For an app whose model calls do not carry the conversation, such as one
    that rewrites each question to stand alone and answers it from retrieved
    text only.
    """
    made = defaultdict(list)
    for call in calls:
        made[call["trace"]].append(call)
    prompts: Counter = Counter()
    messages: list[dict] = []
    seen = {"turns": 0, "pasted": 0, "trimmed": True, "overlap": 0}
    last_end = None
    for root in sorted(roots.values(), key=lambda r: r["start"]):
        at = root["start"]
        if last_end is not None and at < last_end:
            at, seen["overlap"] = last_end, seen["overlap"] + 1
        messages.append({"role": "user", "content": root["asked"], "at": at})
        pieces = [p for row in found.get(root["trace"], ()) for p in row["pieces"]]
        messages += [{"role": "tool", "content": piece} for piece in pieces]
        turn = made.get(root["trace"], [])
        for call in turn:
            system = _unpaste(call["system"] or "", pieces)[0]
            if system:
                prompts[system] += 1
            entry = {
                "role": "assistant",
                "content": root["said"] if call is turn[-1] else "",
                "at": max(call["end"], at),
            }
            if call["usage"]:
                entry["usage"] = call["usage"]
                entry["cache_read_reported"] = call["read_reported"]
            if call["model"]:
                entry["model"] = call["model"]
            messages.append(entry)
        if not turn:
            messages.append(
                {
                    "role": "assistant",
                    "content": root["said"],
                    "at": max(root["end"], at),
                }
            )
        last_end = messages[-1]["at"]
        seen["turns"] += 1
    if prompts:
        messages.insert(0, {"role": "system", "content": prompts.most_common(1)[0][0]})
    return messages, seen


def threads_from_observations(
    rows: Iterable[dict], *, observed_until: str | None = None
) -> tuple[list[dict], dict]:
    """Chat-log threads from Langfuse observation rows, and a report of what was left out.

    Rows are what `GET /api/public/v2/observations` returns with the fields
    `core,basic,io,model,usage`. Every session with a readable conversation
    becomes one thread that `fit`, `replay` and `profile` take as it is.
    """
    sessions: dict[str, dict] = defaultdict(
        lambda: {"calls": [], "found": defaultdict(list), "roots": {}}
    )
    trace_start: dict[str, str] = {}
    loose: set = set()
    report = Counter()
    for row in rows:
        if not isinstance(row, dict) or not row.get("startTime"):
            continue
        report["observations"] += 1
        slim = _slim(row)
        if not slim["session"]:
            loose.add(slim["trace"])
            continue
        trace = slim["trace"]
        if trace not in trace_start or slim["start"] < trace_start[trace]:
            trace_start[trace] = slim["start"]
        session = sessions[slim["session"]]
        if slim["type"] == "GENERATION":
            session["calls"].append(slim)
        elif slim.get("pieces"):
            session["found"][trace].append(slim)
        elif slim.get("asked") and slim.get("said"):
            session["roots"][trace] = slim
    threads, models = [], Counter()
    for session_id in sorted(
        sessions,
        key=lambda s: min((c["start"] for c in sessions[s]["calls"]), default=""),
    ):
        session = sessions[session_id]
        calls = sorted(session["calls"], key=lambda c: (c["start"], c["end"]))
        if not calls:
            continue
        report["sessions"] += 1
        calls, twice = _one_per_call(calls)
        report["calls_recorded_twice"] += twice
        report["model_calls"] += len(calls)
        if not any(c["text"] for c in calls):
            report["sessions_without_text"] += 1
            continue
        chosen, cut = _conversation(calls)
        told = sum(
            any(m["role"] == "user" and m["text"] for m in c["new"]) for c in chosen
        )
        by_trace = len(session["roots"]) > told
        if by_trace:
            # More traces with a question and an answer than turns the calls carry.
            messages, seen = _trace_log(calls, session["roots"], session["found"])
            chosen = calls
        elif cut:
            report["sessions_begun_earlier"] += 1
            continue
        else:
            trace_first: dict[str, str] = {}
            for call in calls:
                trace_first.setdefault(call["trace"], call["id"])
            messages, seen = _chat_log(
                chosen, session["found"], trace_start, trace_first
            )
        thread: dict[str, Any] = {"thread_id": session_id, "messages": messages}
        tools = next((c["tools"] for c in reversed(chosen) if c["tools"]), None)
        if tools:
            thread["tools"] = tools
        if observed_until:
            thread["observed_until"] = observed_until
        try:
            Thread(normalize_thread(thread))
        except ContractError:
            report["sessions_unreadable"] += 1
            continue
        threads.append(thread)
        report["threads"] += 1
        report["threads_read_trace_by_trace"] += by_trace
        report["turns"] += seen["turns"]
        report["helper_calls"] += len(calls) - len(chosen)
        report["turns_with_pasted_retrieval"] += seen["pasted"]
        report["threads_with_trimmed_history"] += seen["trimmed"]
        report["overlapping_turns"] += seen["overlap"]
        models.update(c["model"] for c in chosen if c["model"])
    report["traces_without_session"] = len(loose)
    return threads, {**dict(report), "models": dict(models.most_common())}


def pull(
    *,
    since: str | None = None,
    until: str | None = None,
    days: float = 30,
    rows: Iterable[dict] | None = None,
    **options,
) -> tuple[list[dict], dict]:
    """Threads of a Langfuse project, ready for `fit`, and a report of the read.

    Reads the last `days` days unless `since` says otherwise. `options` go to
    `fetch`: `environment`, `tags`, `filters`, the keys and the host. `rows`
    takes observations already exported, and nothing is fetched.
    """
    now = datetime.now(timezone.utc).replace(microsecond=0)
    if rows is None:
        if since is None:
            since = (now - timedelta(days=days)).isoformat().replace("+00:00", "Z")
        rows = fetch(since=since, until=until, **options)
    observed = until or now.isoformat().replace("+00:00", "Z")
    threads, report = threads_from_observations(rows, observed_until=observed)
    return threads, {**report, "since": since, "until": observed}
