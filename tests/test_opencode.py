import unittest

from compache.optimizer.contracts import ModelSpec
from compache.optimizer.parts import Request, Source
from compache.optimizer.opencode import (
    OpenCodeBaseline,
    Settings,
    Usage,
    serialize,
    Message,
)
from compache.traces import ContractError


def request(parts, index=2):
    return Request("t", "r", index, "model", tuple(parts))


class OpenCodeTests(unittest.TestCase):
    def test_threshold_and_reservation(self):
        s = Settings(context=100000, output_limit=16000, input_limit=80000)
        self.assertEqual(s.usable, 64000)
        self.assertFalse(s.overflow(Usage(input=63999)))
        self.assertTrue(s.overflow(Usage(input=1, total=64000)))
        self.assertEqual(Settings(context=100000).usable, 68000)
        self.assertEqual(Settings(context=100000).tail_budget, 15000)
        for kw in (
            {"context": -1},
            {"context": True},
            {"context": 5, "reserved": -1},
            {"context": 5, "output_limit": 2.5},
            {"context": 5, "auto": 1},
        ):
            with self.assertRaises(ContractError):
                Settings(**kw)
        with self.assertRaises(ContractError):
            Usage(input=-1)

    def test_tail_split_and_repeated_summary(self):
        parts = [
            Source("u1", "input", 1, "one"),
            Source("r1", "retrieval", 1, "long"),
            Source("a1", "output", 1, "answer"),
            Source("u2", "input", 2, "two"),
        ]
        r = request(parts)
        baseline = OpenCodeBaseline(
            Settings(context=10, output_limit=1, preserve_recent_tokens=2),
            ModelSpec("m"),
            lambda ms: len(ms),
        )
        p = baseline.prepare(r, Usage(total=9))
        self.assertEqual([m.id for m in p.tail], ["a1", "u2"])
        self.assertIn("[Tool result]: long", p.prompt)
        projection = baseline.complete(
            r, p, "actual summary", Usage(input=12, output=3)
        )
        self.assertEqual(projection[0].text, "What did we do so far?")
        self.assertEqual(projection[1].text, "actual summary")
        self.assertTrue(projection[-1].synthetic)
        p2 = baseline.prepare(r, Usage(total=9))
        self.assertIn("<prior-summary>\nactual summary\n</prior-summary>", p2.prompt)
        self.assertEqual(baseline.summary_usage[0].output, 3)
        next_request = request(
            parts
            + [
                Source("a2", "output", 2, "second answer"),
                Source("u3", "input", 3, "new question"),
            ],
            3,
        )
        projected = baseline.project(next_request)
        ids = [m.id for m in projected]
        self.assertNotIn("u1", ids)
        self.assertEqual(ids.count("a1"), 1)
        self.assertEqual(ids[-2:], ["a2", "u3"])

    def test_truncates_tool_output(self):
        self.assertTrue(
            serialize(Message("x", "tool", "a" * 2001)).endswith(
                "a" * 2000 + "\n[truncated]"
            )
        )


if __name__ == "__main__":
    unittest.main()
