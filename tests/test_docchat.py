"""DocChat builder: the deterministic parts, and the whole pipeline with stand-in models."""

import re
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks" / "docchat"))

import docchat as d  # noqa: E402

DOC = {"id": "doc-a", "title": "Handbook", "version": "v1"}
SENTENCE = "Staff in group {n} may carry over {n} days of leave when a supervisor approves it in writing before the year ends. "
SECTIONS = [(f"Page {n // 3} › Rule {n}", SENTENCE.format(n=n) * 12) for n in range(30)]
CFG = {
    "top_k": 4,
    "assistant": "You answer from the handbook.",
    "personas": [{"who": "a new employee", "goal": "learn the rules"}],
    "mix": {
        "lookup": 0.3,
        "follow_up": 0.2,
        "recall": 0.1,
        "compare": 0.1,
        "user_context": 0.1,
        "reuse_output": 0.1,
        "synthesis": 0.1,
    },
}


def stand_in(judge_passes):
    """Models that write numbered turns, echo the history they were shown, and grade by a rule."""
    count = Counter()

    def run(model, system, prompt, schema, effort):
        if schema is d.WRITER_SCHEMA:
            count["turn"] += 1
            planted = "I am in group 7" if "PLANT A FACT" in prompt else ""
            return {
                "skip": False,
                "skip_reason": "",
                "question": f"What is rule {count['turn']} about leave?",
                "answer": f"Answer {count['turn']}.",
                "planted_fact": planted,
                "quotes": [],
            }, {"model": model}
        if schema is d.JUDGE_SCHEMA:
            shown = re.findall(
                r"Candidate ([A-Z]):\n(.*?)(?=\n\nCandidate|\Z)",
                prompt.split("CANDIDATES:\n")[1],
                re.S,
            )
            return {
                "reference_supported": True,
                "problem": "",
                "key_facts": ["a fact"],
                "verdicts": [
                    {"id": letter, "correct": judge_passes(text)}
                    for letter, text in shown
                ],
            }, {"model": model}
        return prompt.split("RETRIEVED FOR")[0], {"model": model}

    return run


def build(judge_passes, tier="medium"):
    chunks = d.chunk(DOC, SECTIONS)
    plan = d.plan_thread("doc-a-m1", "hr", CFG, DOC, chunks, tier, 7)
    with tempfile.TemporaryDirectory() as folder:
        models = d.Models(Path(folder), runner=stand_in(judge_passes))
        written = d.write_thread(plan, chunks, d.Index(chunks), CFG, models, "writer")
        done = {r["turn"]: r for r in written}
        checks = [
            d.verify_turn(
                plan["thread_id"], r, done, chunks, models, "answerer", "judge"
            )
            for r in written
        ]
        again = d.write_thread(plan, chunks, d.Index(chunks), CFG, models, "writer")
    return plan, chunks, written, checks, again


class Corpus(unittest.TestCase):
    def test_pieces_fit_the_limit_and_lose_no_words(self):
        text = "First paragraph. " * 300 + "\n\n" + "word " * 900 + "\n" + "x" * 7000
        pieces = d.split(text, 2800)
        self.assertTrue(all(len(p) <= 2800 for p in pieces))
        self.assertEqual("".join(text.split()), "".join("".join(pieces).split()))

    def test_short_neighbours_on_one_page_share_a_chunk(self):
        sections = [
            ("Page › A", "short text."),
            ("Page › B", "more short text."),
            ("Other › C", "prose " * 300),
        ]
        chunks = d.chunk(DOC, sections)
        self.assertEqual(len(chunks), 2)
        self.assertIn("B\nmore short text.", chunks[0]["text"])
        self.assertEqual([c["evidence"] for c in chunks], [False, True])
        self.assertEqual(chunks[1]["chunk_id"], "doc-a-c001")

    def test_tables_of_numbers_are_not_evidence(self):
        chunks = d.chunk(DOC, [("Numbers", "1,204 3,880 (12.5) 44 " * 200)])
        self.assertFalse(any(c["evidence"] for c in chunks))

    def test_markdown_keeps_code_comments_inside_their_section(self):
        with tempfile.TemporaryDirectory() as folder:
            page = Path(folder) / "page.md"
            page.write_text(
                '---\ntitle: "Volumes"\n---\nIntro {{< glossary_tooltip text="Pod" term_id="pod" >}}.\n## Use {#use}\n```sh\n# a comment\nkubectl get pv\n```\n[docs](/x) <br>\n',
                encoding="utf-8",
            )
            sections = d.parse_markdown([page])
        self.assertEqual([h for h, _ in sections], ["Volumes", "Volumes › Use"])
        self.assertIn("Intro Pod.", sections[0][1])
        self.assertIn("# a comment", sections[1][1])
        self.assertIn("docs", sections[1][1])
        self.assertNotIn("<br>", sections[1][1])

    def test_search_ranks_the_matching_chunk_and_places_a_required_one(self):
        chunks = d.chunk(DOC, SECTIONS)
        index = d.Index(chunks)
        target = next(i for i, c in enumerate(chunks) if "group 17 " in c["text"])
        ranked = index.search("leave for group 17", 4)
        self.assertEqual(ranked[0], target)
        unranked = next(i for i in range(len(chunks)) if i not in ranked)
        self.assertEqual(
            index.search("leave for group 17", 4, must=unranked),
            ranked[:3] + [unranked],
        )
        self.assertEqual(index.search("leave for group 17", 4, must=target), ranked)

    def test_every_pinned_document_has_a_hash_and_a_unique_id(self):
        documents = [
            doc
            for domain in d.read_json(d.HERE / "sources.json")["domains"].values()
            for doc in domain["documents"]
        ]
        self.assertEqual(len({doc["id"] for doc in documents}), len(documents))
        self.assertTrue(all(len(doc.get("sha256", "")) == 64 for doc in documents))
        self.assertEqual(
            set(d.read_json(d.HERE / "sources.json")["domains"]),
            set(d.read_json(d.HERE / "domains.json")["domains"]),
        )


class Plan(unittest.TestCase):
    def test_a_plan_is_reproducible_and_only_builds_on_earlier_turns(self):
        chunks = d.chunk(DOC, SECTIONS)
        plan = d.plan_thread("doc-a-l1", "hr", CFG, DOC, chunks, "long", 7)
        self.assertEqual(
            plan, d.plan_thread("doc-a-l1", "hr", CFG, DOC, chunks, "long", 7)
        )
        self.assertNotEqual(
            plan["turns"],
            d.plan_thread("doc-a-l2", "hr", CFG, DOC, chunks, "long", 7)["turns"],
        )
        self.assertTrue(70 <= len(plan["turns"]) <= 90)
        self.assertEqual(plan["turns"][0]["intent"], "lookup")
        for turn in plan["turns"]:
            self.assertTrue(all(ref < turn["turn"] for ref in turn["refs"]))
            self.assertEqual(
                turn["chunk"] is not None,
                turn["intent"] in ("lookup", "compare", "user_context"),
            )
            self.assertEqual(bool(turn["refs"]), turn["intent"] != "lookup")
            if turn["intent"] == "follow_up":
                self.assertEqual(turn["refs"], [turn["turn"] - 1])
            if turn["intent"] == "user_context":
                self.assertEqual(
                    plan["turns"][turn["refs"][0] - 1]["plant_for"], turn["turn"]
                )
        self.assertGreater(len({turn["intent"] for turn in plan["turns"]}), 5)


class Labels(unittest.TestCase):
    def test_history_is_needed_only_when_the_current_turn_alone_fails(self):
        sets = {
            "cur": [],
            "prev": [[2, "input"], [2, "output"]],
            "in": [[1, "input"]],
            "out": [[1, "output"]],
            "full": [[1, "input"], [1, "output"]],
        }
        check = lambda **passes: {
            "supported": True,
            "sets": sets,
            "passes": {name: False for name in sets} | passes,
        }  # noqa: E731
        self.assertEqual(d.needed(check(cur=True, out=True)), ("no", [[]]))
        self.assertEqual(d.needed(check(out=True, full=True)), ("yes", [sets["out"]]))
        self.assertEqual(
            d.needed(check(prev=True, out=True, full=True)),
            ("yes", [sets["prev"], sets["out"]]),
        )
        self.assertEqual(d.needed(check(full=True)), ("yes", [sets["full"]]))
        self.assertEqual(d.needed(check()), ("unknown", []))
        self.assertEqual(
            d.needed(check(cur=True) | {"supported": False}), ("unknown", [])
        )

    def test_threads_and_labels_pass_the_contract_when_history_matters(self):
        plan, chunks, written, checks, again = build(lambda answer: "[turn " in answer)
        self.assertEqual(written, again)
        thread, labels, notes = d.assemble_thread(
            plan, written, checks, chunks, clock="mixed", seed=7, model="replay"
        )
        d.check_contract(thread, labels)
        need = [note["history_needed"] for note in notes]
        self.assertEqual(need[0], "unknown")
        self.assertEqual(set(need[1:]), {"yes"})
        self.assertTrue(all(min(note["lags"]) >= 1 for note in notes[1:]))
        kinds = Counter(
            label["task_requirements"][0]["source_kind"] for label in labels
        )
        self.assertGreater(kinds["mixed"], 0)
        first = thread["runs"][0]["generation"]["request_part_ids"]
        self.assertEqual(
            first,
            [
                thread["runs"][0]["input"]["part_id"],
                *(r["part_id"] for r in thread["runs"][0]["retrievals"]),
            ],
        )
        self.assertLess(
            thread["runs"][0]["started_at"], thread["runs"][-1]["started_at"]
        )

    def test_threads_and_labels_pass_the_contract_when_nothing_needs_history(self):
        plan, chunks, written, checks, _ = build(lambda answer: True, tier="short")
        thread, labels, notes = d.assemble_thread(
            plan, written, checks, chunks, clock="slow", seed=7, model="replay"
        )
        d.check_contract(thread, labels)
        self.assertEqual({note["history_needed"] for note in notes}, {"no"})
        self.assertEqual(
            {label["task_requirements"][0]["source_kind"] for label in labels}
            - {"retrieval", "input"},
            set(),
        )

    def test_a_planted_fact_is_recorded_on_the_turn_that_states_it(self):
        plan, _, written, _, _ = build(lambda answer: True, tier="long")
        planted = {turn["turn"] for turn in plan["turns"] if turn["plant_for"]}
        self.assertTrue(planted)
        self.assertEqual({r["turn"] for r in written if r["planted_fact"]}, planted)
        self.assertTrue(all(r["intent"] == r["planned"] for r in written))


class Run(unittest.TestCase):
    def test_a_compacted_turn_reopens_through_the_package_and_a_full_turn_has_nothing_to_reopen(
        self,
    ):
        import run
        from compache.optimizer.contracts import ModelSpec
        from compache.optimizer.retention import RetentionRecipe

        plan, chunks, written, checks, _ = build(lambda answer: "[turn " in answer)
        thread, _, _ = d.assemble_thread(
            plan, written, checks, chunks, clock="mixed", seed=7, model="m"
        )
        spec = ModelSpec("m", context_tokens=1_000_000)
        seen = []

        def model(name, system, prompt, schema, effort):
            seen.append(prompt)
            if "YOU CALLED" in prompt:
                return "final answer", {"model": name}
            found = re.search(r"ctx:[0-9a-f]+", prompt)
            return (f"REOPEN {found.group(0)}" if found else "plain answer"), {
                "model": name
            }

        with tempfile.TemporaryDirectory() as folder:
            models = d.Models(Path(folder), runner=model)
            recipe = RetentionRecipe(input_runs=0, output_runs=0, retrieval_runs=0)
            cut = run.answer(
                models, "m", "system", thread, thread["runs"][5], recipe, spec
            )
            full = run.answer(
                models, "m", "system", thread, thread["runs"][5], None, spec
            )
        self.assertEqual(
            (cut["answer"], cut["reopen_calls"], cut["reopened"]),
            ("final answer", 1, 1),
        )
        self.assertGreater(cut["archived"], 10)
        self.assertIn('"status": "ok"', seen[1])
        self.assertIn(thread["runs"][0]["input"]["text"], seen[1])
        self.assertEqual(
            (full["answer"], full["reopen_calls"], full["archived"]),
            ("plain answer", 0, 0),
        )
        self.assertGreater(full["prompt_tokens"], cut["prompt_tokens"])
        self.assertIn(thread["runs"][5]["input"]["text"], seen[0])
        self.assertNotIn(thread["runs"][5]["output"]["text"], seen[2])


if __name__ == "__main__":
    unittest.main()
