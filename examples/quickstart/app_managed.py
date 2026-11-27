"""Run from the repository root after 'compache demo -o policy.json'.

The application owns every send. An in-memory provider returns synthetic usage
receipts so the example can run without credentials or a network connection.
"""

import compache


class DemoProvider:
    def __init__(self):
        self.calls = []

    def send(self, kind, turn, output_tokens):
        usage = {
            "prompt_tokens": turn.tokens,
            "completion_tokens": output_tokens,
            "prompt_tokens_details": {"cached_tokens": 0},
        }
        self.calls.append((kind, usage))
        return usage


def main():
    policy = compache.Policy.load("policy.json")
    session = policy.session("demo-order")
    provider = DemoProvider()
    due = None
    for index, question in enumerate(
        ("When does my order ship?", "Can I change it?"), 1
    ):
        at = 1_800_000_000 + index * 7200
        # A single host timer owns this conversation. A user return supersedes it.
        while due is not None and due < at:
            ping = session.ping(at=due)
            if ping is None:
                break
            receipt = provider.send("keep-warm", ping, output_tokens=1)
            session.ping_sent(at=due, usage=receipt, completed_at=due + 1)
            due = session.next_ping_at

        turn = session.prepare(question, at=at)
        first = provider.send("lookup request", turn, output_tokens=8)
        session.replied(turn, calls=[{"order": "example"}], at=at + 1, usage=first)
        # A stable ID identifies this occurrence, even if the same record is fetched again.
        part_id = f"order-lookup-{index}"
        record = {"part_id": part_id, "text": "Ships Friday. Changes close Thursday."}
        turn = session.extend(turn, [record], at=at + 2)
        last = provider.send("answer", turn, output_tokens=7)
        answer = "It ships Friday." if index == 1 else "Changes close Thursday."
        due = session.complete(
            turn,
            answer,
            at=at + 3,
            usage=first,
            last_usage=last,
            cited_part_ids=[part_id],
        )
        print(answer)
    print(f"{len(provider.calls)} physical calls recorded by the application.")


if __name__ == "__main__":
    main()
