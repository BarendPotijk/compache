"""Provider request and response shapes, read and rewritten as plain JSON.

compache decides per part of a conversation: a question, a retrieved text, an
answer. A provider's API speaks in messages, blocks and items. A dialect maps
one onto the other, in both directions, for the body of one endpoint:

    OpenAI Chat Completions   POST …/chat/completions
    Anthropic Messages        POST …/messages
    OpenAI Responses          POST …/responses
    Google Gemini             POST …/models/<model>:generateContent

Only text is ever replaced, and only in a copy: tool calls, images, thinking
blocks and everything a dialect does not know pass through as they came.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator

__all__ = [
    "DIALECTS",
    "Reply",
    "Scan",
    "Slot",
    "TTLS",
    "TextStream",
    "cache_mark",
    "dialect_for",
    "has_cache_mark",
    "sse_events",
    "with_default_lifetime",
    "with_lifetime",
    "without_lifetime",
]

_CACHE = {"type": "ephemeral"}
# How a cache mark spells the lifetimes a provider can be asked for, by their seconds.
TTLS = {300: "5m", 3600: "1h"}


@dataclass(frozen=True)
class Slot:
    """One piece of text in a request: its kind, where it sits and which message it is in.

    Retrieved pieces with the same `group` are one result of one tool call, and
    so one part: a tool that answers in several text blocks still retrieved once.
    `closes` marks an answer that says nothing and calls no tool: it has no
    text to decide on, and still ends its turn.
    """

    kind: str
    path: tuple
    text: str
    message: int
    group: Any = None
    closes: bool = False


@dataclass
class Scan:
    """What a request holds, in the terms compache decides in."""

    model: str | None = None
    system: str = ""
    tools: list = field(default_factory=list)
    slots: list = field(default_factory=list)
    stream: bool = False
    # False when the request does not end on something for the model to answer.
    awaiting: bool = True


@dataclass
class Reply:
    """What a response holds: its text, the tools it calls and the provider's usage."""

    text: str = ""
    calls: list = field(default_factory=list)
    usage: dict | None = None
    model: str | None = None
    cost: float | None = None
    # Where each piece of the text sits in the response body, in order.
    paths: list = field(default_factory=list)
    # Whether the output limit ended it.
    cut: bool = False


def get(body: Any, path: tuple) -> Any:
    for key in path:
        body = body[key]
    return body


def put(body: Any, path: tuple, value: Any) -> Any:
    """`body` with `value` at `path`. Containers along the path are copied, the rest is shared."""
    if not path:
        return value
    head = path[0]
    inner = body[head] if isinstance(body, list) else (body or {}).get(head)
    changed = put(inner, path[1:], value)
    if isinstance(body, list):
        return [*body[:head], changed, *body[head + 1 :]]
    return {**(body or {}), head: changed}


def _text_parts(content: Any, base: tuple, types=("text",)) -> list[tuple[tuple, str]]:
    """Where the text of a content value sits: a string, or a list of typed parts."""
    if isinstance(content, str):
        return [(base, content)]
    found = []
    if isinstance(content, list):
        for j, part in enumerate(content):
            if (
                isinstance(part, dict)
                and part.get("type", "text") in types
                and isinstance(part.get("text"), str)
            ):
                found.append(((*base, j, "text"), part["text"]))
    return found


def cache_mark(lifetime: float | None = None) -> dict:
    """A cache mark that asks for a lifetime in seconds, or for the provider's default."""
    if lifetime is None:
        return dict(_CACHE)
    if lifetime not in TTLS:
        raise ValueError(f"no cache mark asks for {lifetime:g} seconds")
    return {**_CACHE, "ttl": TTLS[lifetime]}


def _with_marks(body: Any, change, where=("system", "tools", "messages", "input")):
    """A request with `change` applied to every cache mark in it. The same object where none changed.

    Marks are looked for where one can stand: on the request, its instructions, tools and messages,
    and the pieces of those. What an application or a model wrote itself, such as a tool's schema
    or a call's arguments, may use the same word and is left alone.
    """
    pieces = ("content", "output", "function", "source")

    def within(value, keys):
        if isinstance(value, list):
            made = [within(inner, pieces) for inner in value]
            return made if any(a is not b for a, b in zip(made, value)) else value
        if not isinstance(value, dict):
            return value
        changed = {}
        mark = value.get("cache_control")
        if isinstance(mark, dict) and mark.get("type") == _CACHE["type"]:
            asked = change(mark)
            if asked != mark:
                changed["cache_control"] = asked
        for key in keys:
            inner = value.get(key)
            if isinstance(inner, (dict, list)):
                made = within(inner, pieces)
                if made is not inner:
                    changed[key] = made
        return {**value, **changed} if changed else value

    return within(body, where)


def _plain_mark(mark: dict) -> dict:
    return {key: value for key, value in mark.items() if key != "ttl"}


def with_lifetime(body: Any, lifetime: float) -> Any:
    """A request with every cache mark in it asking for this lifetime. What has none comes back as it is.

    All of them change: an entry with a longer lifetime may not follow one with
    a shorter, so one mark alone would be refused.
    """
    asked = cache_mark(lifetime)
    return _with_marks(body, lambda mark: {**mark, **asked})


def without_lifetime(body: Any) -> Any:
    """A request with every cache mark in it asking for the provider's default lifetime."""
    return _with_marks(body, _plain_mark)


def with_default_lifetime(body: Any) -> Any:
    """A request whose conversation is cached for the provider's default lifetime.

    What an application marks ahead of it, such as instructions that all its
    users share, keeps the lifetime the application chose.
    """
    return _with_marks(body, _plain_mark, ("messages", "input"))


def has_cache_mark(body: Any) -> bool:
    found: list = []
    _with_marks(body, lambda mark: found.append(mark) or mark)
    return bool(found)


def _cost(usage: Any) -> float | None:
    """What the provider says a call cost, where it says so, as OpenRouter does."""
    if isinstance(usage, dict) and isinstance(usage.get("cost"), (int, float)):
        return float(usage["cost"])
    return None


def _arguments(value: Any) -> dict:
    """A tool call's arguments as an object, from an object or its JSON text."""
    if isinstance(value, str):
        try:
            value = json.loads(value or "{}")
        except ValueError:
            return {}
    return value if isinstance(value, dict) else {}


def sse_events(chunks: Iterable[bytes]) -> Iterator[tuple[bytes, str | None, Any]]:
    """Server-sent events from a byte stream: each raw event, its name and its JSON data.

    Data that is not JSON, such as `[DONE]`, and comment lines come back with
    None for the data, so a caller can pass them on untouched.
    """
    pending = b""
    for chunk in chunks:
        pending += chunk
        while True:
            ends = [
                i for i in (pending.find(b"\n\n"), pending.find(b"\r\n\r\n")) if i >= 0
            ]
            if not ends:
                break
            cut = min(ends)
            size = 4 if pending[cut : cut + 4] == b"\r\n\r\n" else 2
            raw, pending = pending[: cut + size], pending[cut + size :]
            yield _event(raw)
    if pending:
        yield _event(pending)


def _event(raw: bytes) -> tuple[bytes, str | None, Any]:
    name, data = None, []
    for line in raw.decode("utf-8", "replace").splitlines():
        if line.startswith("event:"):
            name = line[6:].strip()
        elif line.startswith("data:"):
            data.append(line[5:].lstrip(" "))
    if not data:
        return raw, name, None
    try:
        return raw, name, json.loads("\n".join(data))
    except ValueError:
        return raw, name, None


def sse_bytes(name: str | None, data: Any) -> bytes:
    """One server-sent event carrying JSON."""
    head = f"event: {name}\n" if name else ""
    return (head + "data: " + json.dumps(data, separators=(",", ":")) + "\n\n").encode()


class TextStream:
    """A reply as it streams: passed on as it comes, its text and usage read off.

    `feed` takes one server-sent event and returns the events to pass on now.
    `end` returns what is still owed when the provider's stream is over.
    `text` is the reply as the application has seen it. A dialect that can take
    a tool call out of a stream builds on this and sets `again` when the
    provider stopped only to call that tool: the caller then answers the calls
    in `ours`, sends `said` back and feeds the next stream into the same object.
    """

    def __init__(self, dialect, tool: str | None = None):
        self.dialect, self.tool = dialect, tool
        self.usage: dict | None = None
        self.cost: float | None = None
        self.model: str | None = None
        # Tool calls the application sees, and calls of our tool answered here.
        self.calls: list = []
        self.ours: list = []
        self.again = False
        self.finished = False
        self._shown = ""

    @property
    def text(self) -> str:
        return self._shown

    def _meter(self, data: dict) -> None:
        usage = self.dialect.stream_usage(data)
        if isinstance(usage, dict):
            self.usage = {**(self.usage or {}), **usage}
            self.cost = _cost(usage) if _cost(usage) is not None else self.cost

    def feed(self, raw: bytes, name: str | None, data: Any) -> list[bytes]:
        if not isinstance(data, dict):
            return [raw]
        self._meter(data)
        self.calls += self.dialect.stream_calls(data)
        _, text = self.dialect.delta(data)
        self._shown += text or ""
        if self.dialect.stream_end(data) is not None:
            self.finished = True
        return [raw]

    def end(self) -> list[bytes]:
        return []

    def next_round(self) -> None:
        self.again, self.ours = False, []


class OpenAIChat:
    """`POST /chat/completions`, as OpenAI and every compatible gateway serve it."""

    name = "openai-chat"
    suffixes = ("/chat/completions",)

    # -- request -----------------------------------------------------------

    @staticmethod
    def scan(body: dict) -> Scan:
        scan = Scan(model=body.get("model"), stream=body.get("stream") is True)
        scan.tools = list(body.get("tools") or ())
        system, messages = [], body["messages"]
        for i, message in enumerate(messages):
            role, base = message.get("role"), ("messages", i, "content")
            found = _text_parts(message.get("content"), base, ("text", "refusal"))
            if role in ("system", "developer"):
                system += [text for _, text in found]
                continue
            kind = {"user": "input", "assistant": "output"}.get(role, "retrieval")
            scan.slots += [Slot(kind, path, text, i, i) for path, text in found]
            calls = message.get("tool_calls") or message.get("function_call")
            if role == "assistant" and not calls and not any(text for _, text in found):
                scan.slots.append(Slot("output", base, "", i, i, closes=True))
        scan.system = "\n\n".join(system)
        scan.awaiting = bool(messages) and messages[-1].get("role") != "assistant"
        return scan

    @staticmethod
    def waits_for_tools(body: dict) -> bool:
        """Whether the last thing said is a tool call whose result has not come in."""
        said = [
            m
            for m in body.get("messages") or ()
            if m.get("role") not in ("system", "developer")
        ]
        return bool(said) and bool(
            said[-1].get("tool_calls") or said[-1].get("function_call")
        )

    @staticmethod
    def tool_names(body: dict) -> set:
        return {(t.get("function") or {}).get("name") for t in body.get("tools") or ()}

    @staticmethod
    def _mark_last(body: dict, mark: dict) -> dict:
        """The request with a cache mark on the last piece of its last message."""
        if not body.get("messages"):
            return body
        at = len(body["messages"]) - 1
        content = body["messages"][at].get("content")
        if isinstance(content, str) and content:
            parts = [{"type": "text", "text": content, "cache_control": mark}]
        elif isinstance(content, list) and content and isinstance(content[-1], dict):
            parts = [*content[:-1], {**content[-1], "cache_control": mark}]
        else:
            return body
        return put(body, ("messages", at, "content"), parts)

    @classmethod
    def pinned(cls, body: dict) -> dict:
        """The request with a cache mark for the whole request put on its last piece instead.

        What is added after it then stays out of what is cached: a note must
        not be cached, because the next request will not carry it.
        """
        if "cache_control" not in body:
            return body
        rest = {k: v for k, v in body.items() if k != "cache_control"}
        return cls._mark_last(rest, body["cache_control"])

    @classmethod
    def with_note(cls, body: dict, text: str) -> dict:
        body = cls.pinned(body)
        return {
            **body,
            "messages": [*body["messages"], {"role": "user", "content": text}],
        }

    @classmethod
    def with_cache_mark(cls, body: dict) -> dict:
        """Mark the end of the conversation as cacheable, for a model that has to be told."""
        model = str(body.get("model") or "")
        if "claude" not in model and "anthropic" not in model:
            return body
        return body if has_cache_mark(body) else cls._mark_last(body, _CACHE)

    @staticmethod
    def with_tool(body: dict, tool: dict, instructions: str) -> dict:
        """The request with one more tool, and how to use it at the end of the system prompt."""
        messages = list(body["messages"])
        for i, message in enumerate(messages):
            if message.get("role") in ("system", "developer"):
                content = message.get("content")
                if isinstance(content, str):
                    said = f"{content}\n\n{instructions}"
                else:
                    said = [*(content or ()), {"type": "text", "text": instructions}]
                messages[i] = {**message, "content": said}
                break
        else:
            messages.insert(0, {"role": "system", "content": instructions})
        return {
            **body,
            "tools": [*(body.get("tools") or ()), tool],
            "messages": messages,
        }

    @staticmethod
    def schema(body: dict) -> dict | None:
        fmt = body.get("response_format")
        if isinstance(fmt, dict) and fmt.get("type") == "json_schema":
            found = (fmt.get("json_schema") or {}).get("schema")
            return found if isinstance(found, dict) else None
        return None

    @staticmethod
    def with_schema(body: dict, schema: dict) -> dict:
        return put(body, ("response_format", "json_schema", "schema"), schema)

    @staticmethod
    def without_schema(body: dict) -> dict:
        """The request free to answer in plain text."""
        return {k: v for k, v in body.items() if k != "response_format"}

    @staticmethod
    def after_tools(body: dict, said: dict, results: dict) -> dict:
        """The request that answers the tool calls of the reply `said`, with `results` by call ID."""
        message = said["choices"][0]["message"]
        turn = {
            "role": "assistant",
            "content": message.get("content"),
            "tool_calls": message["tool_calls"],
        }
        answers = [
            {"role": "tool", "tool_call_id": call_id, "content": text}
            for call_id, text in results.items()
        ]
        return {**body, "messages": [*body["messages"], turn, *answers]}

    @classmethod
    def follow_up(cls, body: dict, answer: str, text: str, limit: int) -> dict:
        """The conversation continued past its answer by one short user message.

        A keep-warm ping is this with a full stop and one token of output: it
        reads what the answer's request left in the cache and adds the answer.
        """
        out = {
            k: v
            for k, v in body.items()
            if k not in ("stream", "stream_options", "cache_control")
        }
        out["messages"] = [*body["messages"], {"role": "assistant", "content": answer}]
        model = str(body.get("model") or "")
        if "claude" in model or "anthropic" in model:
            out = cls._mark_last(out, body.get("cache_control") or _CACHE)
        out["messages"] = [*out["messages"], {"role": "user", "content": text}]
        key = (
            "max_completion_tokens" if "max_completion_tokens" in out else "max_tokens"
        )
        out[key] = limit
        return out

    # -- response ----------------------------------------------------------

    @staticmethod
    def reply(body: dict) -> Reply:
        message = body["choices"][0]["message"]
        reply = Reply(
            usage=body.get("usage"),
            model=body.get("model"),
            cost=_cost(body.get("usage")),
            cut=body["choices"][0].get("finish_reason") == "length",
        )
        if isinstance(message.get("content"), str):
            reply.text = message["content"]
            reply.paths = [("choices", 0, "message", "content")]
        for call in message.get("tool_calls") or ():
            function = call.get("function") or {}
            reply.calls.append(
                {
                    "id": call.get("id"),
                    "name": function.get("name"),
                    "arguments": _arguments(function.get("arguments")),
                }
            )
        return reply

    @staticmethod
    def without_tool(body: dict, name: str) -> dict:
        """A response without its calls of the tool `name`."""
        choice = body["choices"][0]
        message = dict(choice["message"])
        kept = [
            c
            for c in message.pop("tool_calls", None) or ()
            if (c.get("function") or {}).get("name") != name
        ]
        if kept:
            message["tool_calls"] = kept
        elif choice.get("finish_reason") == "tool_calls":
            choice = {**choice, "finish_reason": "stop"}
        return {
            **body,
            "choices": [{**choice, "message": message}, *body["choices"][1:]],
        }

    @staticmethod
    def prune(body: dict) -> dict:
        return body

    # -- streams -----------------------------------------------------------

    @staticmethod
    def delta(data: dict) -> tuple[tuple | None, str | None]:
        choices = data.get("choices") or ()
        if choices and isinstance((choices[0].get("delta") or {}).get("content"), str):
            return ("choices", 0, "delta", "content"), choices[0]["delta"]["content"]
        return None, None

    @staticmethod
    def stream_end(data: dict) -> str | None:
        choices = data.get("choices") or ()
        return choices[0].get("finish_reason") if choices else None

    @staticmethod
    def stream_usage(data: dict) -> dict | None:
        return data.get("usage")

    class Stream(TextStream):
        """A chat-completions stream, with calls of one tool taken out and answered unseen."""

        def __init__(self, dialect, tool=None):
            super().__init__(dialect, tool)
            self._calls: dict[int, dict] = {}
            self._mine: set[int] = set()
            self._round = ""
            self._last: dict | None = None

        def feed(self, raw, name, data):
            if self.again:
                # The provider stopped for our tool. What ends this stream is not passed on.
                if isinstance(data, dict):
                    self._meter(data)
                return []
            if not isinstance(data, dict):
                return [raw]
            self._meter(data)
            self.model = data.get("model") or self.model
            choices = data.get("choices") or ()
            if not choices:
                return [raw]
            self._last = data
            choice = choices[0]
            delta = dict(choice.get("delta") or {})
            changed = False
            fragments = delta.get("tool_calls") or ()
            for call in fragments:
                index, function = call.get("index", 0), call.get("function") or {}
                seen = self._calls.setdefault(
                    index, {"id": None, "name": None, "arguments": ""}
                )
                seen["id"] = call.get("id") or seen["id"]
                seen["name"] = function.get("name") or seen["name"]
                seen["arguments"] += function.get("arguments") or ""
                if self.tool is not None and seen["name"] == self.tool:
                    self._mine.add(index)
            if fragments and self._mine:
                kept = [c for c in fragments if c.get("index", 0) not in self._mine]
                changed = len(kept) != len(fragments)
                delta.pop("tool_calls")
                if kept:
                    delta["tool_calls"] = kept
            if isinstance(delta.get("content"), str):
                self._shown += delta["content"]
                self._round += delta["content"]
            reason = choice.get("finish_reason")
            if reason is not None:
                self.finished = True
                calls = sorted(self._calls.items())
                self.calls = [c for i, c in calls if i not in self._mine]
                self.ours = [
                    {"id": c["id"], "arguments": _arguments(c["arguments"])}
                    for i, c in calls
                    if i in self._mine
                ]
                if self.ours and not self.calls:
                    self.again = True
                    return []
            if not changed:
                return [raw]
            if not delta and reason is None and "usage" not in data:
                return []
            event = {**data, "choices": [{**choice, "delta": delta}, *choices[1:]]}
            return [sse_bytes(name, event)]

        def end(self):
            if not self.again or self._last is None:
                return []
            # Our tool was called and no further round could follow: close the reply as it stands.
            choice = {"index": 0, "delta": {}, "finish_reason": "stop"}
            closing = {k: v for k, v in self._last.items() if k != "usage"}
            return [
                sse_bytes(None, {**closing, "choices": [choice]}),
                b"data: [DONE]\n\n",
            ]

        def next_round(self):
            super().next_round()
            self._calls, self._mine, self._round = {}, set(), ""

        @property
        def said(self) -> dict:
            calls = [
                {
                    "id": c["id"],
                    "type": "function",
                    "function": {"name": c["name"], "arguments": c["arguments"]},
                }
                for _, c in sorted(self._calls.items())
            ]
            message: dict = {"role": "assistant", "content": self._round or None}
            if calls:
                message["tool_calls"] = calls
            return {"choices": [{"message": message}]}


class Anthropic:
    """`POST /messages`, the Anthropic Messages API."""

    name = "anthropic"
    suffixes = ("/messages",)

    @staticmethod
    def scan(body: dict) -> Scan:
        scan = Scan(model=body.get("model"), stream=body.get("stream") is True)
        scan.tools = list(body.get("tools") or ())
        system = [t for _, t in _text_parts(body.get("system"), ("system",))]
        messages = body["messages"]
        for i, message in enumerate(messages):
            content, base = message.get("content"), ("messages", i, "content")
            if message.get("role") == "system":
                # An instruction given in the middle of the conversation. Nobody said it.
                system += [t for _, t in _text_parts(content, base)]
                continue
            mine = "input" if message.get("role") == "user" else "output"
            if isinstance(content, str):
                scan.slots.append(
                    Slot(
                        mine, base, content, i, closes=mine == "output" and not content
                    )
                )
                continue
            said = called = False
            for j, block in enumerate(content or ()):
                if not isinstance(block, dict):
                    continue
                kind = str(block.get("type"))
                called = called or kind.endswith("tool_use")
                if kind == "text" and isinstance(block.get("text"), str):
                    scan.slots.append(Slot(mine, (*base, j, "text"), block["text"], i))
                    said = said or bool(block["text"])
                elif kind == "tool_result":
                    for path, text in _text_parts(
                        block.get("content"), (*base, j, "content")
                    ):
                        scan.slots.append(Slot("retrieval", path, text, i, (i, j)))
            if mine == "output" and not said and not called:
                scan.slots.append(Slot("output", base, "", i, closes=True))
        scan.system = "\n\n".join(system)
        spoken = [m for m in messages if m.get("role") != "system"]
        scan.awaiting = bool(spoken) and spoken[-1].get("role") == "user"
        return scan

    @classmethod
    def waits_for_tools(cls, body: dict) -> bool:
        at = cls._last_said(body)
        content = None if at is None else body["messages"][at].get("content")
        return isinstance(content, list) and any(
            str(b.get("type")).endswith("tool_use")
            for b in content
            if isinstance(b, dict)
        )

    @staticmethod
    def tool_names(body: dict) -> set:
        return {t.get("name") for t in body.get("tools") or ()}

    @staticmethod
    def _blocks(content: Any) -> list:
        if isinstance(content, str):
            return [{"type": "text", "text": content}] if content else []
        return list(content or ())

    @staticmethod
    def _last_said(body: dict) -> int | None:
        """Where the last message that somebody said is: an instruction may follow it."""
        said = [
            i
            for i, m in enumerate(body.get("messages") or ())
            if m.get("role") != "system"
        ]
        return said[-1] if said else None

    @classmethod
    def _mark_last(cls, body: dict, mark: dict) -> dict:
        at = cls._last_said(body)
        if at is None:
            return body
        blocks = cls._blocks(body["messages"][at].get("content"))
        if (
            not blocks
            or not isinstance(blocks[-1], dict)
            or "thinking" in blocks[-1].get("type", "")
        ):
            return body
        blocks[-1] = {**blocks[-1], "cache_control": mark}
        return put(body, ("messages", at, "content"), blocks)

    @classmethod
    def pinned(cls, body: dict) -> dict:
        """The request with a cache mark for the whole request put on its last block instead."""
        if "cache_control" not in body:
            return body
        rest = {k: v for k, v in body.items() if k != "cache_control"}
        return cls._mark_last(rest, body["cache_control"])

    @classmethod
    def with_note(cls, body: dict, text: str) -> dict:
        body = cls.pinned(body)
        at = cls._last_said(body)
        blocks = [
            *cls._blocks(body["messages"][at].get("content")),
            {"type": "text", "text": text},
        ]
        return put(body, ("messages", at, "content"), blocks)

    @classmethod
    def with_cache_mark(cls, body: dict) -> dict:
        return body if has_cache_mark(body) else cls._mark_last(body, _CACHE)

    @staticmethod
    def with_tool(body: dict, tool: dict, instructions: str) -> dict:
        function = tool["function"]
        named = {
            "name": function["name"],
            "description": function["description"],
            "input_schema": function["parameters"],
        }
        system = body.get("system")
        if isinstance(system, list):
            said: Any = [*system, {"type": "text", "text": instructions}]
        else:
            said = f"{system}\n\n{instructions}" if system else instructions
        return {**body, "tools": [*(body.get("tools") or ()), named], "system": said}

    @staticmethod
    def schema(body: dict) -> dict | None:
        fmt = (body.get("output_config") or {}).get("format")
        if isinstance(fmt, dict) and fmt.get("type") == "json_schema":
            return fmt.get("schema") if isinstance(fmt.get("schema"), dict) else None
        return None

    @staticmethod
    def with_schema(body: dict, schema: dict) -> dict:
        return put(body, ("output_config", "format", "schema"), schema)

    @staticmethod
    def without_schema(body: dict) -> dict:
        config = body.get("output_config")
        if not isinstance(config, dict) or "format" not in config:
            return body
        rest = {k: v for k, v in config.items() if k != "format"}
        out = {k: v for k, v in body.items() if k != "output_config"}
        return {**out, "output_config": rest} if rest else out

    @staticmethod
    def after_tools(body: dict, said: dict, results: dict) -> dict:
        answers = [
            {"type": "tool_result", "tool_use_id": call_id, "content": text}
            for call_id, text in results.items()
        ]
        turn = {"role": "assistant", "content": said["content"]}
        return {
            **body,
            "messages": [*body["messages"], turn, {"role": "user", "content": answers}],
        }

    @classmethod
    def follow_up(cls, body: dict, answer: str, text: str, limit: int) -> dict | None:
        thinking = body.get("thinking")
        if isinstance(thinking, dict) and thinking.get("budget_tokens"):
            # A thinking budget must fit under the output limit, so no short request is possible.
            return None
        out = {k: v for k, v in body.items() if k not in ("stream", "cache_control")}
        out["messages"] = [*body["messages"], {"role": "assistant", "content": answer}]
        # The answer is the last thing worth caching: the short message after it is not.
        out = cls._mark_last(out, body.get("cache_control") or _CACHE)
        out["messages"] = [*out["messages"], {"role": "user", "content": text}]
        out["max_tokens"] = limit
        return out

    @staticmethod
    def reply(body: dict) -> Reply:
        reply = Reply(
            usage=body.get("usage"),
            model=body.get("model"),
            cost=_cost(body.get("usage")),
            cut=body.get("stop_reason") == "max_tokens",
        )
        texts = []
        for j, block in enumerate(body.get("content") or ()):
            if block.get("type") == "text":
                texts.append(block.get("text") or "")
                reply.paths.append(("content", j, "text"))
            elif block.get("type") == "tool_use":
                reply.calls.append(
                    {
                        "id": block.get("id"),
                        "name": block.get("name"),
                        "arguments": _arguments(block.get("input")),
                    }
                )
        reply.text = "\n\n".join(texts)
        return reply

    @staticmethod
    def without_tool(body: dict, name: str) -> dict:
        kept = [
            b
            for b in body.get("content") or ()
            if not (b.get("type") == "tool_use" and b.get("name") == name)
        ]
        out = {**body, "content": kept}
        if body.get("stop_reason") == "tool_use" and not any(
            b.get("type") == "tool_use" for b in kept
        ):
            out["stop_reason"] = "end_turn"
        return out

    @staticmethod
    def prune(body: dict) -> dict:
        """A response without text blocks that are empty: a later request may not carry one."""
        content = body.get("content") or ()
        kept = [
            b for b in content if not (b.get("type") == "text" and not b.get("text"))
        ]
        return (
            body if len(kept) == len(content) or not kept else {**body, "content": kept}
        )

    @staticmethod
    def delta(data: dict) -> tuple[tuple | None, str | None]:
        if data.get("type") == "content_block_delta":
            delta = data.get("delta") or {}
            if delta.get("type") == "text_delta" and isinstance(delta.get("text"), str):
                return ("delta", "text"), delta["text"]
        return None, None

    @staticmethod
    def stream_usage(data: dict) -> dict | None:
        if data.get("type") == "message_start":
            return (data.get("message") or {}).get("usage")
        return data.get("usage") if data.get("type") == "message_delta" else None

    @staticmethod
    def stream_end(data: dict) -> str | None:
        if data.get("type") == "message_delta":
            return (data.get("delta") or {}).get("stop_reason")
        return None

    class Stream(TextStream):
        """A Messages stream. Blocks of one tool are taken out, and empty text blocks never start."""

        def __init__(self, dialect, tool=None):
            super().__init__(dialect, tool)
            self._blocks: dict[int, dict] = {}
            self._mine: set[int] = set()
            self._out: dict[int, int] = {}
            self._next = 0
            self._texts: list[str] = []
            self._rounds = 0

        @property
        def text(self) -> str:
            return "\n\n".join(self._texts)

        def _index(self, index: int) -> int:
            if index not in self._out:
                self._out[index] = self._next
                self._next += 1
            return self._out[index]

        def _text(self, index: int, shown: str) -> list[bytes]:
            """Events that show `shown` as text of block `index`, opening the block if this is its first."""
            if not shown:
                return []
            block, out = self._blocks[index], []
            if "at" not in block:
                block["at"] = len(self._texts)
                self._texts.append("")
                start = {
                    "type": "content_block_start",
                    "index": self._index(index),
                    "content_block": {"type": "text", "text": ""},
                }
                out.append(sse_bytes("content_block_start", start))
            self._texts[block["at"]] += shown
            block["shown"] = block.get("shown", "") + shown
            event = {
                "type": "content_block_delta",
                "index": self._index(index),
                "delta": {"type": "text_delta", "text": shown},
            }
            out.append(sse_bytes("content_block_delta", event))
            return out

        def feed(self, raw, name, data):
            if not isinstance(data, dict):
                return [raw]
            self._meter(data)
            kind = data.get("type")
            if self.again:
                return []
            if kind == "message_start":
                self.model = (data.get("message") or {}).get("model") or self.model
                return [] if self._rounds else [raw]
            index = data.get("index")
            if kind == "content_block_start":
                block = dict(data.get("content_block") or {})
                self._blocks[index] = {
                    "block": block,
                    "json": "",
                    "type": block.get("type"),
                }
                if (
                    self.tool is not None
                    and block.get("type") == "tool_use"
                    and block.get("name") == self.tool
                ):
                    self._mine.add(index)
                    return []
                if block.get("type") == "text":
                    # Opened once it has something to show.
                    return self._text(index, block.get("text") or "")
                at = self._index(index)
                return (
                    [raw] if at == index else [sse_bytes(name, {**data, "index": at})]
                )
            if kind == "content_block_delta":
                block, delta = self._blocks.get(index), data.get("delta") or {}
                if block is None:
                    return [raw]
                if delta.get("type") == "input_json_delta":
                    block["json"] += delta.get("partial_json") or ""
                elif delta.get("type") == "thinking_delta":
                    block["block"]["thinking"] = block["block"].get("thinking", "") + (
                        delta.get("thinking") or ""
                    )
                elif delta.get("type") == "signature_delta":
                    block["block"]["signature"] = delta.get("signature")
                if index in self._mine:
                    return []
                if delta.get("type") == "text_delta":
                    return self._text(index, delta.get("text") or "")
                at = self._index(index)
                return (
                    [raw] if at == index else [sse_bytes(name, {**data, "index": at})]
                )
            if kind == "content_block_stop":
                block = self._blocks.get(index)
                if block is None:
                    return [raw]
                if index in self._mine:
                    return []
                if block["type"] == "text":
                    if "at" not in block:
                        # Nothing of it was shown, so it never opened.
                        return []
                    return [sse_bytes(name, {**data, "index": self._index(index)})]
                at = self._index(index)
                return (
                    [raw] if at == index else [sse_bytes(name, {**data, "index": at})]
                )
            if kind == "message_delta":
                self.finished = True
                tools = [
                    (i, b)
                    for i, b in sorted(self._blocks.items())
                    if b["type"] == "tool_use"
                ]
                self.calls = [
                    {
                        "id": b["block"].get("id"),
                        "name": b["block"].get("name"),
                        "arguments": _arguments(b["json"]),
                    }
                    for i, b in tools
                    if i not in self._mine
                ]
                self.ours = [
                    {"id": b["block"].get("id"), "arguments": _arguments(b["json"])}
                    for i, b in tools
                    if i in self._mine
                ]
                if self.ours and not self.calls:
                    self.again = True
                    self._ending = data
                    return []
                return [raw]
            return [raw]

        def end(self):
            if not self.again:
                return []
            closing = {
                **self._ending,
                "delta": {
                    **(self._ending.get("delta") or {}),
                    "stop_reason": "end_turn",
                },
            }
            return [
                sse_bytes("message_delta", closing),
                sse_bytes("message_stop", {"type": "message_stop"}),
            ]

        def next_round(self):
            super().next_round()
            self._blocks, self._mine, self._out = {}, set(), {}
            self._rounds += 1

        @property
        def said(self) -> dict:
            """What the model said in this round, as the content of an assistant message."""
            content = []
            for _, entry in sorted(self._blocks.items()):
                block = dict(entry["block"])
                if entry["type"] == "text":
                    if not entry.get("shown"):
                        continue
                    block["text"] = entry["shown"]
                elif entry["type"] == "tool_use":
                    block["input"] = _arguments(entry["json"])
                content.append(block)
            return {"content": content}


class OpenAIResponses:
    """`POST /responses`, with the conversation sent in full on every request."""

    name = "openai-responses"
    suffixes = ("/responses",)
    # A streamed reply repeats its whole text in closing events, so a note cannot be filtered out.
    Stream = TextStream

    @staticmethod
    def _items(body: dict) -> list:
        items = body.get("input")
        return (
            [{"role": "user", "content": items}]
            if isinstance(items, str)
            else list(items or ())
        )

    @staticmethod
    def scan(body: dict) -> Scan:
        if body.get("previous_response_id") or body.get("conversation"):
            # The provider holds the history. There is nothing here to decide on.
            raise ValueError("the conversation is kept by the provider")
        scan = Scan(model=body.get("model"), stream=body.get("stream") is True)
        scan.tools = list(body.get("tools") or ())
        system = (
            [body["instructions"]] if isinstance(body.get("instructions"), str) else []
        )
        items = body.get("input")
        if isinstance(items, str):
            scan.slots.append(Slot("input", ("input",), items, 0))
            scan.system = "\n\n".join(system)
            return scan
        last = None
        for i, item in enumerate(items or ()):
            kind = item.get("type", "message")
            if kind == "message":
                role, base = item.get("role"), ("input", i, "content")
                found = _text_parts(
                    item.get("content"), base, ("input_text", "output_text", "text")
                )
                if role in ("system", "developer"):
                    system += [text for _, text in found]
                    continue
                mine = "input" if role == "user" else "output"
                scan.slots += [Slot(mine, path, text, i) for path, text in found]
                if mine == "output" and not any(text for _, text in found):
                    # Tool calls are items of their own here, so an empty message is an empty answer.
                    scan.slots.append(Slot("output", base, "", i, closes=True))
                last = role
            elif kind == "function_call_output":
                base = ("input", i, "output")
                found = _text_parts(
                    item.get("output"), base, ("input_text", "output_text", "text")
                )
                scan.slots += [
                    Slot("retrieval", path, text, i, i) for path, text in found
                ]
                last = "tool"
            elif kind == "function_call":
                last = "call"
            elif kind == "item_reference":
                # Part of the history is at the provider, and only its name is here.
                raise ValueError("the conversation is kept by the provider")
        scan.system = "\n\n".join(system)
        scan.awaiting = last in ("user", "tool")
        return scan

    @classmethod
    def waits_for_tools(cls, body: dict) -> bool:
        items = cls._items(body)
        return bool(items) and items[-1].get("type") == "function_call"

    @staticmethod
    def tool_names(body: dict) -> set:
        return {t.get("name") for t in body.get("tools") or ()}

    @classmethod
    def with_note(cls, body: dict, text: str) -> dict:
        return {**body, "input": [*cls._items(body), {"role": "user", "content": text}]}

    @staticmethod
    def with_cache_mark(body: dict) -> dict:
        return body

    @staticmethod
    def with_tool(body: dict, tool: dict, instructions: str) -> dict:
        said = body.get("instructions")
        return {
            **body,
            "tools": [
                *(body.get("tools") or ()),
                {"type": "function", **tool["function"]},
            ],
            "instructions": f"{said}\n\n{instructions}" if said else instructions,
        }

    @staticmethod
    def schema(body: dict) -> dict | None:
        fmt = (body.get("text") or {}).get("format")
        if isinstance(fmt, dict) and fmt.get("type") == "json_schema":
            return fmt.get("schema") if isinstance(fmt.get("schema"), dict) else None
        return None

    @staticmethod
    def with_schema(body: dict, schema: dict) -> dict:
        return put(body, ("text", "format", "schema"), schema)

    @staticmethod
    def without_schema(body: dict) -> dict:
        text = body.get("text")
        if not isinstance(text, dict) or "format" not in text:
            return body
        rest = {k: v for k, v in text.items() if k != "format"}
        out = {k: v for k, v in body.items() if k != "text"}
        return {**out, "text": rest} if rest else out

    @classmethod
    def after_tools(cls, body: dict, said: dict, results: dict) -> dict:
        answers = [
            {"type": "function_call_output", "call_id": call_id, "output": text}
            for call_id, text in results.items()
        ]
        return {
            **body,
            "input": [*cls._items(body), *(said.get("output") or ()), *answers],
        }

    @classmethod
    def follow_up(cls, body: dict, answer: str, text: str, limit: int) -> dict:
        out = {k: v for k, v in body.items() if k != "stream"}
        out["input"] = [
            *cls._items(body),
            {"role": "assistant", "content": answer},
            {"role": "user", "content": text},
        ]
        # The API takes no limit under sixteen tokens.
        out["max_output_tokens"] = max(16, limit)
        return out

    @staticmethod
    def reply(body: dict) -> Reply:
        reply = Reply(
            usage=body.get("usage"),
            model=body.get("model"),
            cost=_cost(body.get("usage")),
            cut=(body.get("incomplete_details") or {}).get("reason")
            == "max_output_tokens",
        )
        texts = []
        for i, item in enumerate(body.get("output") or ()):
            if item.get("type") == "message":
                for j, part in enumerate(item.get("content") or ()):
                    if part.get("type") == "output_text":
                        texts.append(part.get("text") or "")
                        reply.paths.append(("output", i, "content", j, "text"))
            elif item.get("type") == "function_call":
                reply.calls.append(
                    {
                        "id": item.get("call_id"),
                        "name": item.get("name"),
                        "arguments": _arguments(item.get("arguments")),
                    }
                )
        reply.text = "\n\n".join(texts)
        return reply

    @staticmethod
    def without_tool(body: dict, name: str) -> dict:
        kept = [
            i
            for i in body.get("output") or ()
            if not (i.get("type") == "function_call" and i.get("name") == name)
        ]
        return {**body, "output": kept}

    @staticmethod
    def prune(body: dict) -> dict:
        return body

    @staticmethod
    def delta(data: dict) -> tuple[tuple | None, str | None]:
        if data.get("type") == "response.output_text.delta" and isinstance(
            data.get("delta"), str
        ):
            return ("delta",), data["delta"]
        return None, None

    @staticmethod
    def stream_usage(data: dict) -> dict | None:
        if data.get("type") == "response.completed":
            return (data.get("response") or {}).get("usage")
        return None

    @staticmethod
    def stream_end(data: dict) -> str | None:
        return "completed" if data.get("type") == "response.completed" else None

    @staticmethod
    def stream_calls(data: dict) -> list:
        item = data.get("item") or {}
        if (
            data.get("type") == "response.output_item.done"
            and item.get("type") == "function_call"
        ):
            return [
                {
                    "id": item.get("call_id"),
                    "name": item.get("name"),
                    "arguments": _arguments(item.get("arguments")),
                }
            ]
        return []


class Gemini:
    """`POST …/models/<model>:generateContent` and its streaming twin."""

    name = "gemini"
    suffixes = (":generateContent", ":streamGenerateContent")
    Stream = TextStream

    @staticmethod
    def scan(body: dict) -> Scan:
        scan = Scan()
        instruction = (
            body.get("systemInstruction") or body.get("system_instruction") or {}
        )
        scan.system = "\n\n".join(
            p["text"]
            for p in instruction.get("parts") or ()
            if isinstance(p.get("text"), str)
        )
        scan.tools = list(body.get("tools") or ())
        contents = body["contents"]
        for i, content in enumerate(contents):
            mine = "output" if content.get("role") == "model" else "input"
            parts = content.get("parts") or ()
            if mine == "output" and not any(
                p.get("functionCall") or (p.get("text") and not p.get("thought"))
                for p in parts
            ):
                scan.slots.append(
                    Slot("output", ("contents", i, "parts"), "", i, closes=True)
                )
            for j, part in enumerate(parts):
                base = ("contents", i, "parts", j)
                if isinstance(part.get("text"), str) and not part.get("thought"):
                    scan.slots.append(Slot(mine, (*base, "text"), part["text"], i))
                elif isinstance(part.get("functionResponse"), dict):
                    answer = part["functionResponse"].get("response")
                    text = (
                        answer
                        if isinstance(answer, str)
                        else json.dumps(answer, ensure_ascii=False)
                    )
                    scan.slots.append(
                        Slot(
                            "retrieval",
                            (*base, "functionResponse", "response"),
                            text,
                            i,
                        )
                    )
        scan.awaiting = bool(contents) and contents[-1].get("role") != "model"
        return scan

    @staticmethod
    def waits_for_tools(body: dict) -> bool:
        contents = body.get("contents") or ()
        parts = (contents[-1].get("parts") or ()) if contents else ()
        return any("functionCall" in p or "function_call" in p for p in parts)

    @staticmethod
    def tool_names(body: dict) -> set:
        return {
            d.get("name")
            for t in body.get("tools") or ()
            for d in t.get("functionDeclarations")
            or t.get("function_declarations")
            or ()
        }

    @staticmethod
    def text_value(path: tuple, text: str) -> Any:
        """What to write at `path` for replacement text: a tool's response stays an object."""
        return {"output": text} if path[-1] == "response" else text

    @staticmethod
    def with_note(body: dict, text: str) -> dict:
        at = len(body["contents"]) - 1
        parts = [*(body["contents"][at].get("parts") or ()), {"text": text}]
        return put(body, ("contents", at, "parts"), parts)

    @staticmethod
    def with_cache_mark(body: dict) -> dict:
        return body

    @staticmethod
    def with_tool(body: dict, tool: dict, instructions: str) -> dict:
        function = tool["function"]

        def plain(schema):
            # Gemini's schema dialect has no additionalProperties and no length limits.
            if isinstance(schema, dict):
                dropped = ("additionalProperties", "minLength", "maxLength", "minimum")
                return {k: plain(v) for k, v in schema.items() if k not in dropped}
            return schema

        declared = {
            "name": function["name"],
            "description": function["description"],
            "parameters": plain(function["parameters"]),
        }
        key = (
            "system_instruction"
            if "system_instruction" in body
            else "systemInstruction"
        )
        parts = [*((body.get(key) or {}).get("parts") or ()), {"text": instructions}]
        return {
            **body,
            "tools": [*(body.get("tools") or ()), {"functionDeclarations": [declared]}],
            key: {**(body.get(key) or {}), "parts": parts},
        }

    @staticmethod
    def _config(body: dict) -> tuple[str, dict]:
        key = "generation_config" if "generation_config" in body else "generationConfig"
        return key, dict(body.get(key) or {})

    @classmethod
    def schema(cls, body: dict) -> dict | None:
        config = cls._config(body)[1]
        for key in (
            "responseJsonSchema",
            "response_json_schema",
            "responseSchema",
            "response_schema",
        ):
            if isinstance(config.get(key), dict):
                return config[key]
        return None

    @classmethod
    def with_schema(cls, body: dict, schema: dict) -> dict:
        name, config = cls._config(body)
        for key in (
            "responseJsonSchema",
            "response_json_schema",
            "responseSchema",
            "response_schema",
        ):
            if isinstance(config.get(key), dict):
                return {**body, name: {**config, key: schema}}
        return body

    @classmethod
    def without_schema(cls, body: dict) -> dict:
        name, config = cls._config(body)
        shaped = (
            "responseJsonSchema",
            "response_json_schema",
            "responseSchema",
            "response_schema",
            "responseMimeType",
            "response_mime_type",
        )
        if not any(key in config for key in shaped):
            return body
        return {**body, name: {k: v for k, v in config.items() if k not in shaped}}

    @staticmethod
    def after_tools(body: dict, said: dict, results: dict) -> dict:
        answers = [
            {
                "functionResponse": {
                    "name": "reopen",
                    "id": call_id,
                    "response": {"output": text},
                }
            }
            for call_id, text in results.items()
        ]
        turn = said["candidates"][0]["content"]
        return {
            **body,
            "contents": [*body["contents"], turn, {"role": "user", "parts": answers}],
        }

    @classmethod
    def follow_up(cls, body: dict, answer: str, text: str, limit: int) -> dict:
        name, config = cls._config(body)
        return {
            **body,
            "contents": [
                *body["contents"],
                {"role": "model", "parts": [{"text": answer}]},
                {"role": "user", "parts": [{"text": text}]},
            ],
            name: {**config, "maxOutputTokens": limit},
        }

    @staticmethod
    def _parts(body: dict) -> list:
        candidates = body.get("candidates") or ()
        return (
            list((candidates[0].get("content") or {}).get("parts") or ())
            if candidates
            else []
        )

    @classmethod
    def reply(cls, body: dict) -> Reply:
        candidates = body.get("candidates") or ({},)
        reply = Reply(
            usage=body.get("usageMetadata"),
            model=body.get("modelVersion"),
            cut=candidates[0].get("finishReason") == "MAX_TOKENS",
        )
        texts = []
        for j, part in enumerate(cls._parts(body)):
            if isinstance(part.get("text"), str) and not part.get("thought"):
                texts.append(part["text"])
                reply.paths.append(("candidates", 0, "content", "parts", j, "text"))
            elif isinstance(part.get("functionCall"), dict):
                call = part["functionCall"]
                reply.calls.append(
                    {
                        "id": call.get("id") or call.get("name"),
                        "name": call.get("name"),
                        "arguments": _arguments(call.get("args")),
                    }
                )
        reply.text = "\n\n".join(texts)
        return reply

    @classmethod
    def without_tool(cls, body: dict, name: str) -> dict:
        kept = [
            p
            for p in cls._parts(body)
            if (p.get("functionCall") or {}).get("name") != name
        ]
        return put(body, ("candidates", 0, "content", "parts"), kept)

    @staticmethod
    def prune(body: dict) -> dict:
        return body

    @classmethod
    def delta(cls, data: dict) -> tuple[tuple | None, str | None]:
        for j, part in enumerate(cls._parts(data)):
            if isinstance(part.get("text"), str) and not part.get("thought"):
                return ("candidates", 0, "content", "parts", j, "text"), part["text"]
        return None, None

    @staticmethod
    def stream_usage(data: dict) -> dict | None:
        return data.get("usageMetadata")

    @staticmethod
    def stream_end(data: dict) -> str | None:
        candidates = data.get("candidates") or ()
        return candidates[0].get("finishReason") if candidates else None

    @classmethod
    def stream_calls(cls, data: dict) -> list:
        return cls.reply(data).calls


DIALECTS = (OpenAIChat, Anthropic, OpenAIResponses, Gemini)


def dialect_for(path: str):
    """The dialect of a request path, or None for an endpoint compache leaves alone."""
    path = path.split("?", 1)[0].rstrip("/")
    for dialect in DIALECTS:
        if path.endswith(dialect.suffixes):
            return dialect
    return None


def streams(path: str) -> bool:
    """Whether a path alone says that its reply streams, as Gemini's does."""
    return ":streamGenerateContent" in path


def set_text(dialect, body: dict, path: tuple, text: str) -> dict:
    """`body` with the text at `path` replaced."""
    value = dialect.text_value(path, text) if hasattr(dialect, "text_value") else text
    return put(body, path, value)
