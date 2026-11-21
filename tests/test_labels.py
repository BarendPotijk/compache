"""Labels: what a labeller sees, the simple form, and what the validator accepts."""

import unittest

from compache.errors import ContractError
from compache.labels import build_packet, label_index, normalize_label, validate_label
from compache.threads import Thread, normalize_thread

from test_threads import sample


def two_turns():
    return Thread(
        normalize_thread(
            {
                "thread_id": "t",
                "runs": [
                    {
                        "started_at": "2026-01-05T09:00:00Z",
                        "input": "What is the fee?",
                        "retrievals": ["The fee is 40 euros."],
                        "output": "40 euros.",
                    },
                    {
                        "started_at": "2026-01-05T09:05:00Z",
                        "input": "Double it.",
                        "retrievals": ["Unrelated notice."],
                        "output": "80 euros.",
                    },
                ],
            }
        )
    )


class PacketTests(unittest.TestCase):
    def test_masked_text_and_split_names_stay_out(self):
        data = sample()
        data["scenario_id"] = "secret"
        data["runs"][0]["retrievals"][0].update(content_status="masked", text=None)
        packet = build_packet(data, "r")
        self.assertEqual(packet["request_coverage"], "partial")
        self.assertNotIn("secret", str(packet))
        self.assertIsNone(packet["request_parts"][1]["text"])

    def test_a_packet_holds_the_request_and_nothing_later(self):
        packet = build_packet(two_turns(), "t:1")
        self.assertEqual(
            [p["part_id"] for p in packet["request_parts"]],
            ["t:1:input", "t:1:retrieval:1"],
        )
        self.assertEqual(packet["target_output"]["text"], "40 euros.")
        with self.assertRaises(ContractError):
            build_packet(two_turns(), "missing")


class SimpleLabelTests(unittest.TestCase):
    def test_needs_becomes_the_canonical_form(self):
        thread = two_turns()
        label = normalize_label(
            {
                "thread_id": "t",
                "run": 2,
                "needs": [
                    ["t:1:output"],
                    [{"part_id": "t:1:retrieval:1", "quote": "40 euros"}],
                ],
            },
            thread,
        )
        requirement = label["task_requirements"][0]
        self.assertEqual((label["target_run_id"], label["status"]), ("t:2", "complete"))
        self.assertEqual(requirement["historical_context_needed"], "yes")
        self.assertEqual(len(requirement["alternatives"]), 2)
        self.assertEqual(validate_label(label, build_packet(thread, "t:2")), [])

    def test_nothing_needed_and_unknown(self):
        thread = two_turns()
        nothing = normalize_label(
            {"thread_id": "t", "run_id": "t:2", "needs": []}, thread
        )
        self.assertEqual(
            nothing["task_requirements"][0]["historical_context_needed"], "no"
        )
        self.assertEqual(validate_label(nothing, build_packet(thread, "t:2")), [])
        # Own parts that suffice mean no history is needed, whatever else is listed.
        own = normalize_label(
            {"thread_id": "t", "run": 2, "needs": [["t:1:output"], "t:2:retrieval:1"]},
            thread,
        )
        self.assertEqual(own["task_requirements"][0]["historical_context_needed"], "no")
        unknown = normalize_label({"thread_id": "t", "run": 2, "needs": None}, thread)
        self.assertEqual(unknown["status"], "unassessable")
        self.assertEqual(
            unknown["task_requirements"][0]["historical_context_needed"], "unknown"
        )
        self.assertEqual(validate_label(unknown, build_packet(thread, "t:2")), [])

    def test_bad_simple_rows(self):
        thread = two_turns()
        for row in (
            {"thread_id": "t", "run": 9, "needs": []},
            {"thread_id": "t", "run": 2},
            {"thread_id": "t", "run": 2, "needs": "t:1:output"},
            {"thread_id": "t", "run": 2, "needs": [[]]},
            {"thread_id": "t", "run": 2, "needs": [[{"id": "t:1:output"}]]},
            "not a row",
        ):
            with self.subTest(row=row), self.assertRaises(ContractError):
                normalize_label(row, thread)

    def test_a_canonical_row_passes_through(self):
        row = {
            "thread_id": "t",
            "target_run_id": "t:2",
            "status": "complete",
            "task_requirements": [],
        }
        self.assertIs(normalize_label(row, two_turns()), row)


class ValidatorTests(unittest.TestCase):
    def label(self, sources, needed="yes"):
        return {
            "thread_id": "t",
            "target_run_id": "t:2",
            "status": "complete",
            "task_requirements": [
                {
                    "requirement_id": "a",
                    "historical_context_needed": needed,
                    "alternatives": [{"sources": sources}],
                }
            ],
        }

    def test_quotes_visibility_and_consistency(self):
        packet = build_packet(two_turns(), "t:2")
        self.assertEqual(
            validate_label(
                self.label([{"part_id": "t:1:output", "quote": "40 euros."}]), packet
            ),
            [],
        )
        cases = {
            "verbatim": self.label([{"part_id": "t:1:output", "quote": "invented"}]),
            "absent, future": self.label([{"part_id": "t:2:output"}]),
            "conflicts": self.label([{"part_id": "t:2:input"}]),
            "current-source": self.label([{"part_id": "t:1:output"}], needed="no"),
            "duplicate sources": self.label(
                [{"part_id": "t:1:output"}, {"part_id": "t:1:output"}]
            ),
            "empty alternative": self.label([]),
            "does not match": {
                **self.label([{"part_id": "t:1:output"}]),
                "thread_id": "other",
            },
            "status": {**self.label([{"part_id": "t:1:output"}]), "status": "done"},
            "yes, no or unknown": self.label(
                [{"part_id": "t:1:output"}], needed="maybe"
            ),
        }
        for message, label in cases.items():
            with self.subTest(message):
                self.assertTrue(
                    any(message in e for e in validate_label(label, packet)),
                    validate_label(label, packet),
                )
        self.assertEqual(validate_label("row", packet), ["label: expected object"])
        self.assertEqual(
            validate_label(self.label([]), {"thread_id": "t"}),
            ["packet: missing required fields"],
        )

    def test_the_full_form_with_extra_sections_is_accepted(self):
        label = self.label(
            [{"part_id": "t:1:retrieval:1", "quote": "The fee is 40 euros."}]
        )
        label.update(
            schema_version="compache-dependency-label-v1",
            coverage_notes=[],
            claims=[{"claim_id": "c"}],
            historical_dependencies=[],
            unresolved=[],
        )
        label["task_requirements"][0].update(
            description="fee",
            source_kind="retrieval",
            necessity="inferred",
            question_quote="Double it.",
            fulfillment="fulfilled",
        )
        self.assertEqual(validate_label(label, build_packet(two_turns(), "t:2")), [])

    def test_unavailable_text_cannot_be_evidence(self):
        data = two_turns().to_dict()
        data["runs"][0]["retrievals"][0].update(content_status="masked", text=None)
        packet = build_packet(data, "t:2")
        errors = validate_label(self.label([{"part_id": "t:1:retrieval:1"}]), packet)
        self.assertTrue(any("unavailable" in e for e in errors))


class IndexTests(unittest.TestCase):
    def test_index_validates_and_keys_by_run(self):
        thread = two_turns()
        index = label_index(
            [thread], [{"thread_id": "t", "run": 2, "needs": [["t:1:output"]]}]
        )
        self.assertEqual(list(index), ["t:2"])
        with self.assertRaisesRegex(ContractError, "unique runs"):
            label_index([thread], [{"thread_id": "t", "run": 2, "needs": []}] * 2)
        with self.assertRaisesRegex(ContractError, "unique runs"):
            label_index([thread], [{"thread_id": "elsewhere", "run": 2, "needs": []}])
        with self.assertRaisesRegex(ContractError, "invalid label"):
            label_index(
                [thread], [{"thread_id": "t", "run": 2, "needs": [["t:2:output"]]}]
            )


if __name__ == "__main__":
    unittest.main()
