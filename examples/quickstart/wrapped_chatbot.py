"""Send a chatbot conversation through Compache to an in-memory Anthropic endpoint."""

import json

import anthropic
import httpx2

import compache

RECALL = "Which replacement hinge did we record for ticket PKG-104?"


def scripted_reply(body):
    """The fake model can read only the messages and tools in this request."""
    content = body["messages"][-1]["content"]
    blocks = [{"type": "text", "text": content}] if isinstance(content, str) else content
    for block in blocks:
        if block["type"] == "tool_result" and block["tool_use_id"] == "lookup_ticket":
            result = json.loads(block["content"])
            if result["status"] == "ok":
                return {"type": "text", "text": result["text"].splitlines()[0]}
            return {"type": "text", "text": "Ticket not found."}
        if block.get("text") == RECALL:
            if any(tool["name"] == "reopen" for tool in body.get("tools", [])):
                return {
                    "type": "tool_use",
                    "id": "lookup_ticket",
                    "name": "reopen",
                    "input": {"reference": "PKG-104"},
                }
            return {"type": "text", "text": "Ticket unavailable."}
    return {"type": "text", "text": "Ticket recorded."}


def main():
    # Advance a local clock so a two-hour pause takes no wall time.
    now = 1_800_000_000
    live = compache.Live("policy.json", clock=lambda: now)
    seen = []

    def local_model(request):
        body = json.loads(request.content)
        seen.append(body)
        reply = scripted_reply(body)
        return httpx2.Response(
            200,
            json={
                "id": f"msg_{len(seen)}",
                "type": "message",
                "role": "assistant",
                "model": body["model"],
                "content": [reply],
                "stop_reason": "tool_use" if reply["type"] == "tool_use" else "end_turn",
                "usage": {
                    "input_tokens": len(json.dumps(body["messages"])) // 4,
                    "output_tokens": 8,
                },
            },
        )

    transport = httpx2.MockTransport(local_model)
    sdk = anthropic.Anthropic(
        api_key="offline-example",
        base_url="https://offline.invalid",
        http_client=httpx2.Client(transport=transport),
        max_retries=0,
    )
    inspection = "Inspect the packaging. " * 300
    questions = ["Ticket PKG-104 needs hinge H-772.\n" + inspection]
    questions.extend(f"Inspection {n}: {inspection}" for n in range(1, 11))
    questions.append(RECALL)

    history = []
    with sdk, compache.wrap(sdk, live=live, pings=False) as client:
        with compache.thread("ticket-chat"):
            for question in questions:
                history.append({"role": "user", "content": question})
                first_request = len(seen)
                reply = client.messages.create(
                    model="claude-sonnet-5-5", max_tokens=64, messages=history
                )
                history.append({"role": "assistant", "content": reply.content[0].text})
                now += 7200

    recall_messages = seen[first_request]["messages"]
    print(f"Application history: {len(history) // 2} user turns")
    print(f"Hinge number in original message: {'H-772' in history[0]['content']}")
    print(f"Hinge number in recall request before reopen: {'H-772' in json.dumps(recall_messages)}")
    print(f"Reopen calls: {live.stats['reopened']}")
    print(f"Provider requests: {len(seen)} observed, {live.stats['physical_calls']} counted by Compache")
    print(f"Answer: {history[-1]['content']}")


if __name__ == "__main__":
    main()
