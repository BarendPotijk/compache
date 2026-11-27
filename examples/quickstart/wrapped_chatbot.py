"""Send a chatbot conversation through Compache to an in-memory Anthropic endpoint."""

import json

import anthropic
import httpx2

import compache


def main():
    # Hold a deterministic clock so a two-hour pause takes no wall time.
    now = 1_800_000_000
    live = compache.Live("policy.json", clock=lambda: now)
    seen = []

    def local_model(request):
        body = json.loads(request.content)
        seen.append(body)
        # Synthetic receipts use a simple character-based token estimate.
        input_tokens = len(json.dumps(body["messages"])) // 4
        return httpx2.Response(
            200,
            json={
                "id": f"msg_{len(seen)}",
                "type": "message",
                "role": "assistant",
                "model": body["model"],
                "content": [{"type": "text", "text": "Ticket recorded."}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": input_tokens, "output_tokens": 4},
            },
        )

    transport = httpx2.MockTransport(local_model)
    sdk = anthropic.Anthropic(
        api_key="offline-example",
        base_url="https://offline.invalid",
        http_client=httpx2.Client(transport=transport),
        max_retries=0,
    )
    history = []
    with compache.wrap(sdk, live=live, pings=False) as client:
        with compache.thread("ticket-chat"):
            for n in range(1, 13):
                question = f"Ticket {n:02}: " + "Inspect the packaging. " * 300
                history.append({"role": "user", "content": question})
                reply = client.messages.create(
                    model="claude-sonnet-5-5", max_tokens=64, messages=history
                )
                history.append({"role": "assistant", "content": reply.content[0].text})
                now += 7200

    final_questions = json.dumps(seen[-1]["messages"])
    final_tools = [tool["name"] for tool in seen[-1].get("tools", [])]
    print(f"Application history: {len(history) // 2} user turns")
    print(f"Requests received by the in-memory model: {len(seen)}")
    print(f"First question in application history: {'Ticket 01:' in history[0]['content']}")
    print(f"First question in final provider request: {'Ticket 01:' in final_questions}")
    print(f"Recovery tool offered on final request: {'reopen' in final_tools}")


if __name__ == "__main__":
    main()
