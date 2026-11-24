"""A scripted model behind the wire formats compache speaks, with a prompt cache that behaves.

`FakeModel` answers Chat Completions and Anthropic Messages bodies, whole or
streamed, with usage from a simulated prefix cache: what a request shares with
one sent inside the lifetime is read, the rest is written. `serve` puts it
behind a local HTTP port, so SDK clients and the proxy can be run against it.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from compache.live import Result

NOTE_ASKED = "<!-- refs: LABELS -->"
# A notice in a request: the reference of an archived part.
NOTICE = re.compile(r"ctx:[0-9a-f]{24}")


def noticed(value) -> bool:
    """Whether what was sent or shown holds the notice of an archived part."""
    return (
        NOTICE.search(
            value if isinstance(value, str) else json.dumps(value, default=str)
        )
        is not None
    )


def tokens(value) -> int:
    return len(json.dumps(value, ensure_ascii=False).encode()) // 4


class FakeModel:
    """Replies from a script, and meters a prompt cache.

    `script(request)` returns a dict with `text`, and optionally `tool` as
    (name, arguments) for a tool call or `tools` for several, `refs` for the labels of a note, `cut`
    for a reply the output limit ended, `reasoning` for a reply of the Responses API that
    reasons first, and `status` with `error` for a refusal. `request` has the `body`, its `texts`
    (every text of the conversation in order), the `last` text and `n`, the
    number of the call. Without a script every reply is "Answer <n>.".
    """

    def __init__(self, script=None, *, ttl=300, clock=None, min_tokens=64):
        self.script, self.ttl, self.min_tokens = script, ttl, min_tokens
        self.clock = clock or (lambda: 0.0)
        self.bodies: list[dict] = []
        self.paths: list[str] = []
        self.headers: list[dict] = []
        self._cache: dict[str, float] = {}
        self._lock = threading.Lock()

    # -- the cache -----------------------------------------------------------

    def _meter(self, head, blocks, marked: int | None, marks=None) -> dict:
        """Read and written tokens for a request of `blocks` behind `head`, cached up to block `marked`.

        With `marks`, the blocks that carry a cache mark, a provider's rules for
        several of them: an entry is written at each, and looked for at most
        twenty blocks back from each.
        """
        now = self.clock()
        sums, running = [], hashlib.sha256(json.dumps(head, sort_keys=True).encode())
        size = tokens(head)
        for block in blocks:
            running = running.copy()
            running.update(json.dumps(block, sort_keys=True).encode())
            size += tokens(block)
            sums.append((running.hexdigest(), size))
        total = size
        end = len(sums) if marked is None else marked + 1
        read = 0
        looked = list(reversed(sums[:end]))
        if marks is not None:
            places = {i for m in marks for i in range(max(0, m - 19), m + 1)}
            looked = [sums[i] for i in sorted(places, reverse=True)]
        with self._lock:
            for key, upto in looked:
                if self._cache.get(key, -1e18) + self.ttl > now:
                    read = upto
                    break
            for key, upto in sums:
                # Reading an entry keeps alive what it is made of.
                if upto <= read and self._cache.get(key, -1e18) + self.ttl > now:
                    self._cache[key] = now
            written = 0
            if end and sums[end - 1][1] >= self.min_tokens:
                written = sums[end - 1][1] - read
                keys = sums[:end] if marked is None else sums[end - 1 : end]
                if marks is not None:
                    keys = [sums[m] for m in marks]
                for key, _ in keys:
                    self._cache[key] = now
        return {"total": total, "read": read, "write": max(0, written)}

    # -- scripts -------------------------------------------------------------

    def _decide(self, body, texts) -> dict:
        request = {
            "body": body,
            "texts": texts,
            "last": texts[-1] if texts else "",
            "n": len(self.bodies),
        }
        made = dict(
            self.script(request)
            if self.script
            else {"text": f"Answer {len(self.bodies)}."}
        )
        made.setdefault("text", "")
        made["tools"] = made.get("tools") or (
            [made["tool"]] if made.get("tool") else []
        )
        if made.get("refs") is not None and NOTE_ASKED in request["last"]:
            made["text"] += f"\n<!-- refs: {made['refs']} -->"
        return made

    # -- Chat Completions ------------------------------------------------------

    def chat(self, body: dict, path="/v1/chat/completions", headers=None):
        self.bodies.append(body)
        self.paths.append(path)
        self.headers.append(dict(headers or {}))
        texts = []
        for message in body["messages"]:
            content = message.get("content")
            if isinstance(content, str):
                texts.append(content)
            else:
                texts += [
                    p.get("text", "") for p in content or () if isinstance(p, dict)
                ]
        made = self._decide(body, texts)
        if made.get("status"):
            return made["status"], {"error": {"message": made.get("error", "refused")}}
        marked, plain = None, []
        for i, message in enumerate(body["messages"]):
            if "cache_control" in json.dumps(message):
                marked = i
            # A cache mark is not part of what is cached, and text is text however it is wrapped.
            content = message.get("content")
            if isinstance(content, list):
                content = [
                    {k: v for k, v in p.items() if k != "cache_control"}
                    for p in content
                ]
                if all(p.get("type") == "text" for p in content):
                    content = "".join(p["text"] for p in content)
            plain.append({**message, "content": content})
        if "cache_control" in body and plain:
            # A mark for the whole request: everything up to its last message.
            marked = len(plain) - 1
        head = [body.get("tools"), body.get("response_format")]
        meter = self._meter(
            head, plain, marked if "claude" in str(body.get("model")) else None
        )
        if "claude" in str(body.get("model")) and marked is None:
            meter["read"] = meter["write"] = 0
        usage = {
            "prompt_tokens": meter["total"],
            "completion_tokens": max(1, len(made["text"]) // 4),
            "prompt_tokens_details": {
                "cached_tokens": meter["read"],
                "cache_write_tokens": meter["write"],
            },
            "cost": round(meter["total"] * 1e-7, 9),
        }
        usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
        message = {"role": "assistant", "content": made["text"] or None}
        finish = "length" if made.get("cut") else "stop"
        if made["tools"]:
            message["tool_calls"] = [
                {
                    "id": f"call_{len(self.bodies)}" + (f"_{k}" if k else ""),
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(arguments)},
                }
                for k, (name, arguments) in enumerate(made["tools"])
            ]
            finish = "tool_calls"
        reply = {
            "id": f"chatcmpl-{len(self.bodies)}",
            "object": "chat.completion",
            "model": body.get("model"),
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": usage,
        }
        if not body.get("stream"):
            return 200, reply
        return 200, list(self._chat_stream(reply, made))

    @staticmethod
    def _chat_stream(reply, made):
        base = {
            "id": reply["id"],
            "object": "chat.completion.chunk",
            "model": reply["model"],
        }

        def chunk(delta, finish=None, **more):
            data = {
                **base,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
                **more,
            }
            return ("data: " + json.dumps(data) + "\n\n").encode()

        yield chunk({"role": "assistant", "content": ""})
        for piece in re.findall(r".{1,7}", made["text"], re.S):
            yield chunk({"content": piece})
        calls = reply["choices"][0]["message"].get("tool_calls", ())
        for at, call in enumerate(calls):
            arguments = call["function"]["arguments"]
            opening = {
                **call,
                "index": at,
                "function": {"name": call["function"]["name"], "arguments": ""},
            }
            yield chunk({"tool_calls": [opening]})
            for piece in re.findall(r".{1,9}", arguments, re.S):
                yield chunk(
                    {"tool_calls": [{"index": at, "function": {"arguments": piece}}]}
                )
        yield chunk({}, reply["choices"][0]["finish_reason"])
        yield chunk({}, reply["choices"][0]["finish_reason"], usage=reply["usage"])
        yield b"data: [DONE]\n\n"

    # -- Anthropic Messages ----------------------------------------------------

    def messages(self, body: dict, path="/v1/messages", headers=None):
        self.bodies.append(body)
        self.paths.append(path)
        self.headers.append(dict(headers or {}))
        texts, blocks, marks = [], [], []
        for message in body["messages"]:
            content = message.get("content")
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]
            for block in content:
                if block.get("type") == "text":
                    texts.append(block["text"])
                elif block.get("type") == "tool_result":
                    inner = block.get("content")
                    texts.append(inner if isinstance(inner, str) else json.dumps(inner))
                blocks.append(
                    [
                        message["role"],
                        {k: v for k, v in block.items() if k != "cache_control"},
                    ]
                )
                if "cache_control" in block:
                    marks.append(len(blocks) - 1)
        made = self._decide(body, texts)
        if made.get("status"):
            return made["status"], {
                "type": "error",
                "error": {"message": made.get("error", "refused")},
            }
        head = [body.get("tools"), body.get("system"), body.get("output_config")]
        if "cache_control" in body and blocks:
            marks.append(len(blocks) - 1)
        meter = self._meter(head, blocks, marks[-1] if marks else None, marks)
        if not marks:
            meter["read"] = meter["write"] = 0
        usage = {
            "input_tokens": meter["total"] - meter["read"] - meter["write"],
            "cache_read_input_tokens": meter["read"],
            "cache_creation_input_tokens": meter["write"],
            "output_tokens": max(1, len(made["text"]) // 4),
        }
        content = [{"type": "text", "text": made["text"]}] if made["text"] else []
        stop = "max_tokens" if made.get("cut") else "end_turn"
        for k, (name, arguments) in enumerate(made["tools"]):
            content.append(
                {
                    "type": "tool_use",
                    "id": f"toolu_{len(self.bodies)}" + (f"_{k}" if k else ""),
                    "name": name,
                    "input": arguments,
                }
            )
            stop = "tool_use"
        reply = {
            "id": f"msg_{len(self.bodies)}",
            "type": "message",
            "role": "assistant",
            "model": body.get("model"),
            "content": content,
            "stop_reason": stop,
            "usage": usage,
        }
        if not body.get("stream"):
            return 200, reply
        return 200, list(self._message_stream(reply))

    @staticmethod
    def _message_stream(reply):
        def event(name, data):
            return f"event: {name}\ndata: {json.dumps({'type': name, **data})}\n\n".encode()

        opening = {**reply, "content": [], "stop_reason": None}
        opening["usage"] = {**reply["usage"], "output_tokens": 1}
        yield event("message_start", {"message": opening})
        for index, block in enumerate(reply["content"]):
            if block["type"] == "text":
                yield event(
                    "content_block_start",
                    {"index": index, "content_block": {"type": "text", "text": ""}},
                )
                for piece in re.findall(r".{1,7}", block["text"], re.S):
                    yield event(
                        "content_block_delta",
                        {
                            "index": index,
                            "delta": {"type": "text_delta", "text": piece},
                        },
                    )
            else:
                shell = {**block, "input": {}}
                yield event(
                    "content_block_start", {"index": index, "content_block": shell}
                )
                for piece in re.findall(r".{1,9}", json.dumps(block["input"]), re.S):
                    yield event(
                        "content_block_delta",
                        {
                            "index": index,
                            "delta": {
                                "type": "input_json_delta",
                                "partial_json": piece,
                            },
                        },
                    )
            yield event("content_block_stop", {"index": index})
        yield event(
            "message_delta",
            {
                "delta": {"stop_reason": reply["stop_reason"], "stop_sequence": None},
                "usage": {"output_tokens": reply["usage"]["output_tokens"]},
            },
        )
        yield event("message_stop", {})

    # -- OpenAI Responses -------------------------------------------------------

    def responses(self, body: dict, path="/v1/responses", headers=None):
        self.bodies.append(body)
        self.paths.append(path)
        self.headers.append(dict(headers or {}))
        items = body.get("input")
        if isinstance(items, str):
            items = [{"role": "user", "content": items}]
        texts = []
        for item in items:
            content = item.get("content", item.get("output"))
            if isinstance(content, str):
                texts.append(content)
            else:
                texts += [
                    p.get("text", "") for p in content or () if isinstance(p, dict)
                ]
        made = self._decide(body, texts)
        if made.get("status"):
            return made["status"], {"error": {"message": made.get("error", "refused")}}
        meter = self._meter(
            [body.get("tools"), body.get("instructions"), body.get("text")], items, None
        )
        usage = {
            "input_tokens": meter["total"],
            "input_tokens_details": {
                "cached_tokens": meter["read"],
                "cache_write_tokens": meter["write"],
            },
            "output_tokens": max(1, len(made["text"]) // 4),
            "output_tokens_details": {"reasoning_tokens": 0},
        }
        usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
        n, output = len(self.bodies), []
        if made.get("reasoning"):
            output.append(
                {
                    "type": "reasoning",
                    "id": f"rs_{n}",
                    "summary": [{"type": "summary_text", "text": made["reasoning"]}],
                    "encrypted_content": f"enclosed-{n}",
                }
            )
        if made["text"]:
            part = {"type": "output_text", "text": made["text"], "annotations": []}
            output.append(
                {
                    "type": "message",
                    "id": f"msg_{n}",
                    "role": "assistant",
                    "status": "completed",
                    "content": [part],
                }
            )
        for k, (name, arguments) in enumerate(made["tools"]):
            more = f"_{k}" if k else ""
            output.append(
                {
                    "type": "function_call",
                    "id": f"fc_{n}{more}",
                    "call_id": f"call_{n}{more}",
                    "name": name,
                    "arguments": json.dumps(arguments),
                    "status": "completed",
                }
            )
        reply = {
            "id": f"resp_{n}",
            "object": "response",
            "created_at": 0,
            "status": "completed",
            "model": body.get("model"),
            "output": output,
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": body.get("tools") or [],
            "usage": usage,
        }
        if not body.get("stream"):
            return 200, reply
        events = []

        def event(name, **data):
            line = json.dumps({"type": name, "sequence_number": len(events), **data})
            events.append(f"event: {name}\ndata: {line}\n\n".encode())

        event(
            "response.created",
            response={**reply, "status": "in_progress", "output": [], "usage": None},
        )
        for index, item in enumerate(output):
            event("response.output_item.added", output_index=index, item=item)
            if item["type"] == "message":
                for piece in re.findall(r".{1,7}", made["text"], re.S):
                    event(
                        "response.output_text.delta",
                        output_index=index,
                        content_index=0,
                        item_id=item["id"],
                        delta=piece,
                    )
                event(
                    "response.output_text.done",
                    output_index=index,
                    content_index=0,
                    item_id=item["id"],
                    text=made["text"],
                )
            if item["type"] == "function_call":
                event(
                    "response.function_call_arguments.delta",
                    output_index=index,
                    item_id=item["id"],
                    delta=item["arguments"],
                )
            event("response.output_item.done", output_index=index, item=item)
        event("response.completed", response=reply)
        return 200, events

    # -- Gemini -----------------------------------------------------------------

    def gemini(self, body: dict, path, headers=None):
        self.bodies.append(body)
        self.paths.append(path)
        self.headers.append(dict(headers or {}))
        texts = []
        for content in body["contents"]:
            for part in content.get("parts") or ():
                if "text" in part:
                    texts.append(part["text"])
                elif "functionResponse" in part:
                    texts.append(json.dumps(part["functionResponse"].get("response")))
        made = self._decide(body, texts)
        if made.get("status"):
            return made["status"], {
                "error": {
                    "code": made["status"],
                    "message": made.get("error", "refused"),
                }
            }
        head = [
            body.get("tools"),
            body.get("systemInstruction"),
            body.get("generationConfig"),
        ]
        meter = self._meter(head, body["contents"], None)
        parts = [{"text": made["text"]}] if made["text"] else []
        for name, arguments in made["tools"]:
            parts.append({"functionCall": {"name": name, "args": arguments}})
        usage = {
            "promptTokenCount": meter["total"],
            "cachedContentTokenCount": meter["read"],
            "candidatesTokenCount": max(1, len(made["text"]) // 4),
        }
        usage["totalTokenCount"] = (
            usage["promptTokenCount"] + usage["candidatesTokenCount"]
        )
        model = path.split("/models/")[-1].split(":")[0]

        def chunk(parts, last):
            data = {
                "candidates": [
                    {"content": {"role": "model", "parts": parts}, "index": 0}
                ],
                "modelVersion": model,
            }
            if last:
                data["candidates"][0]["finishReason"] = "STOP"
                data["usageMetadata"] = usage
            return data

        if ":streamGenerateContent" not in path:
            return 200, chunk(parts, True)
        pieces = [[{"text": t}] for t in re.findall(r".{1,7}", made["text"], re.S)]
        pieces += [[p] for p in parts if "functionCall" in p]
        pieces = pieces or [[]]
        return 200, [
            (
                "data: " + json.dumps(chunk(piece, i == len(pieces) - 1)) + "\n\n"
            ).encode()
            for i, piece in enumerate(pieces)
        ]

    # -- as a provider --------------------------------------------------------

    def handle(self, path: str, body: dict, headers=None):
        plain = path.split("?")[0].rstrip("/")
        if plain.endswith("/messages"):
            return self.messages(body, path, headers)
        if plain.endswith("/responses"):
            return self.responses(body, path, headers)
        if "GenerateContent" in plain or "generateContent" in plain:
            return self.gemini(body, path, headers)
        return self.chat(body, path, headers)

    def send(self, path="/v1/chat/completions"):
        """A `send` for `Live.exchange`: the fake as the provider behind `path`."""

        def send(body):
            status, reply = self.handle(path, json.loads(json.dumps(body)))
            if isinstance(reply, list):
                return Result(status, events=iter(reply))
            return Result(status, json=reply, raw=json.dumps(reply).encode())

        return send


def read_stream(chunks) -> dict:
    """What an application gathers from a stream: its text, tool calls and how it ended."""
    text, calls, ended, raw = "", {}, None, b"".join(chunks)
    for block in raw.decode().split("\n\n"):
        data = next(
            (
                line[5:].strip()
                for line in block.splitlines()
                if line.startswith("data:")
            ),
            None,
        )
        if not data or data == "[DONE]":
            continue
        event = json.loads(data)
        if event.get("object") == "chat.completion.chunk":
            for choice in event.get("choices") or ():
                delta = choice.get("delta") or {}
                text += delta.get("content") or ""
                for call in delta.get("tool_calls") or ():
                    entry = calls.setdefault(
                        call.get("index", 0), {"name": None, "arguments": ""}
                    )
                    function = call.get("function") or {}
                    entry["name"] = function.get("name") or entry["name"]
                    entry["arguments"] += function.get("arguments") or ""
                ended = choice.get("finish_reason") or ended
        elif event.get("type") == "content_block_start":
            block_ = event["content_block"]
            if block_["type"] == "tool_use":
                calls[event["index"]] = {"name": block_["name"], "arguments": ""}
        elif event.get("type") == "content_block_delta":
            delta = event["delta"]
            text += delta.get("text") or ""
            if delta.get("type") == "input_json_delta":
                calls[event["index"]]["arguments"] += delta["partial_json"]
        elif event.get("type") == "message_delta":
            ended = event["delta"].get("stop_reason") or ended
    return {
        "text": text,
        "calls": [calls[k] for k in sorted(calls)],
        "ended": ended,
        "raw": raw,
    }


def serve(model: FakeModel):
    """The fake behind a local HTTP port. Returns its base URL and a function that stops it."""

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(
                self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}"
            )
            status, reply = model.handle(self.path, body, dict(self.headers.items()))
            if isinstance(reply, list):
                self.send_response(status)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                for chunk in reply:
                    self.wfile.write(f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n")
                    self.wfile.flush()
                self.wfile.write(b"0\r\n\r\n")
                return
            payload = json.dumps(reply).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    class Server(ThreadingHTTPServer):
        daemon_threads = True

        def handle_error(self, request, client_address):
            # A client that hangs up is not worth a traceback.
            pass

    server = Server(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def stop():
        server.shutdown()
        server.server_close()

    return f"http://127.0.0.1:{server.server_address[1]}", stop
