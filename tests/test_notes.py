"""Dependency notes: what is asked, what comes back, and that the reply survives it."""

import json
import unittest

from compache import notes
from compache.notes import (
    KEY,
    Note,
    extend_schema,
    index,
    instruction,
    leaked,
    read,
    resolve,
    sampled,
    split_json,
)

TURNS = [
    {
        "input": "How long is the warranty on the pump?",
        "retrievals": ["The warranty covers parts and labour for two years."],
        "output": "Two years.",
    },
    {
        "input": "And how do I claim?",
        "retrievals": [
            "A claim must be filed within thirty days.",
            "Send the form to the service desk.",
        ],
        "output": "File within thirty days.",
    },
    {"input": "Does that also cover the motor?", "retrievals": ["Motors: see pumps."]},
]


class AskTests(unittest.TestCase):
    def test_the_list_names_every_earlier_part_and_this_turns_retrieved_text(self):
        listed, cut = index(TURNS)
        self.assertFalse(cut)
        self.assertEqual(
            [label for label, _ in listed],
            ["Q1", "R1.1", "A1", "Q2", "R2.1", "R2.2", "A2", "R3.1"],
        )
        self.assertEqual(dict(listed)["Q1"], "How long is the warranty on the pump?")
        long = [{"input": "word " * 50, "retrievals": [None], "output": "x"}, {}]
        words = dict(index(long)[0])
        self.assertTrue(words["Q1"].endswith("…"))
        self.assertLessEqual(len(words["Q1"]), 65)
        self.assertEqual(words["R1.1"], "(archived)")

    def test_a_long_conversation_lists_its_latest_parts_and_says_so(self):
        many = [
            {"input": f"q{n}", "retrievals": [], "output": f"a{n}"} for n in range(60)
        ]
        listed, cut = index(many + [{"input": "now"}], limit=20)
        self.assertTrue(cut)
        self.assertEqual(len(listed), 20)
        self.assertEqual(listed[-1][0], "A60")
        self.assertIn("older", instruction(many + [{"input": "now"}], limit=20))
        self.assertNotIn("older", instruction(TURNS))

    def test_the_request_asks_for_labels_only(self):
        text = instruction(TURNS)
        self.assertIn("<!-- refs: LABELS -->", text)
        self.assertIn("{R2.2} Send the form to the service desk.", text)
        # The current question is not something to name.
        self.assertNotIn("{Q3}", text)
        self.assertIn(f'"{KEY}"', instruction(TURNS, carrier="json"))
        self.assertNotIn("<!--", instruction(TURNS, carrier="json"))

    def test_sampling_is_a_fixed_share_and_the_same_for_a_retry(self):
        hits = sum(sampled(f"t{i}", 4, 0.25) for i in range(4000))
        self.assertTrue(880 < hits < 1120, hits)
        self.assertEqual(sampled("t", 7, 0.5), sampled("t", 7, 0.5))
        self.assertFalse(sampled("t", 1, 1.0))
        self.assertTrue(sampled("t", 2, 1.0))
        self.assertFalse(sampled("t", 2, 0.0))
        self.assertNotEqual(
            [sampled(f"t{i}", 3, 0.5) for i in range(64)],
            [sampled(f"t{i}", 3, 0.5, salt="other") for i in range(64)],
        )


class ReadTests(unittest.TestCase):
    def test_a_reply_without_a_note_is_untouched(self):
        for text in (
            "Two years.",
            "Two years.  \n\n",
            "Keep <!-- this --> comment.",
            "",
            "a <!-",
        ):
            self.assertEqual(read(text), (text, None))
        self.assertEqual(read(None), (None, None))

    def test_the_note_is_taken_out_and_returned(self):
        self.assertEqual(
            read("Two years.\n<!-- refs: {R1.1} {A1} -->"),
            ("Two years.", "{R1.1} {A1}"),
        )
        self.assertEqual(
            read("Two years.\n\n<!--refs:none-->\n"), ("Two years.", "none")
        )
        self.assertEqual(read("<!-- REFS : R1.1 -->"), ("", "R1.1"))
        # Cut off by the output limit, at any point.
        for tail in ("<!-- refs: {R1.1} {A", "<!-- refs", "<!-- re", "<!--"):
            clean, found = read("Two years.\n" + tail)
            self.assertEqual(clean, "Two years.")
            self.assertIsNotNone(found)
        # Another comment at the end is the reply's own.
        self.assertEqual(read("Code:\n<!-- todo")[1], None)

    def test_labels_become_part_ids(self):
        note = resolve("{R1.1} {A1} {R3.1}", TURNS, "t")
        self.assertEqual(note.needs, ("t:1:retrieval:1", "t:1:output"))
        self.assertEqual(note.cited, ("t:1:retrieval:1", "t:3:retrieval:1"))
        self.assertEqual((note.unknown, note.older), (0, False))
        # Without braces, lower case, repeated, and a turn's retrieved text as a whole.
        note = resolve("q1 R2 r2.2 R2", TURNS, "t")
        self.assertEqual(
            note.needs, ("t:1:input", "t:2:retrieval:1", "t:2:retrieval:2")
        )
        self.assertEqual(resolve("none", TURNS, "t"), Note())
        self.assertEqual(resolve(["R1.1", "older"], TURNS, "t").older, True)
        # Nothing by these names: a later turn, a missing item, the current question and answer.
        for label in ("R9.1", "R1.4", "Q3", "A3", "Q1.2", "A7"):
            note = resolve(label, TURNS, "t")
            self.assertEqual((note.needs, note.cited, note.unknown), ((), (), 1), label)

    def test_a_label_in_the_visible_reply_is_noticed(self):
        self.assertTrue(leaked("As {R1.1} says, two years."))
        self.assertFalse(leaked("Q3 revenue rose and the A4 sheet lists R2 values."))
        self.assertFalse(leaked("Use {name} and {A} here."))


class JsonTests(unittest.TestCase):
    SCHEMA = {
        "type": "object",
        "properties": {"answer": {"type": "string"}, "sure": {"type": "boolean"}},
        "required": ["answer", "sure"],
        "additionalProperties": False,
    }

    def test_a_schema_gains_one_last_property(self):
        extended = extend_schema(self.SCHEMA)
        self.assertEqual(list(extended["properties"]), ["answer", "sure", KEY])
        self.assertEqual(extended["required"], ["answer", "sure", KEY])
        self.assertEqual(self.SCHEMA["required"], ["answer", "sure"])
        loose = extend_schema({"type": "object", "properties": {}})
        self.assertNotIn("required", loose)
        for unfit in ({"type": "array"}, {"type": "object"}, extended, None, "x"):
            self.assertIsNone(extend_schema(unfit))

    def test_the_property_is_cut_out_and_the_rest_is_as_written(self):
        cases = [
            (
                '{"answer": "Two years.", "sure": true, "_refs": ["R1.1", "A1"]}',
                '{"answer": "Two years.", "sure": true}',
            ),
            ('{"_refs":["R1.1"],"answer":"x"}', '{"answer":"x"}'),
            (
                '{\n  "answer": "a, \\"b\\" {c}",\n  "_refs": []\n}',
                '{\n  "answer": "a, \\"b\\" {c}"\n}',
            ),
            ('{"_refs": ["A1"]}', "{}"),
            (
                '{"a": {"_refs": 1, "k": [1, {"x": "}"}]}, "_refs": "R1.1", "z": null}',
                '{"a": {"_refs": 1, "k": [1, {"x": "}"}]}, "z": null}',
            ),
        ]
        for text, clean in cases:
            got, labels = split_json(text)
            self.assertEqual(got, clean)
            self.assertIsInstance(labels, list)
            json.loads(got)
        self.assertEqual(split_json(cases[0][0])[1], ["R1.1", "A1"])
        self.assertEqual(split_json(cases[4][0])[1], ["R1.1"])

    def test_anything_else_comes_back_as_it_was(self):
        for text in (
            '{"answer": "x"}',
            '{"answer": "mentions _refs"}',
            '{"a": {"_refs": [1]}}',
            '{"_refs": ["R1.1"]',
            "not json _refs",
            '["_refs"]',
            None,
        ):
            self.assertEqual(split_json(text), (text, None))


class ModuleTests(unittest.TestCase):
    def test_the_note_carries_labels_and_no_text(self):
        # What the model is asked to write back holds no words of the conversation.
        reply = "File a claim.\n<!-- refs: {R2.1} {A1} -->"
        _, found = read(reply)
        note = resolve(found, TURNS, "t")
        stored = json.dumps([note.needs, note.cited])
        for turn in TURNS:
            for text in (turn["input"], *turn["retrievals"]):
                self.assertNotIn(text[:12], stored)
        self.assertEqual(notes.KEY, "_refs")


if __name__ == "__main__":
    unittest.main()
