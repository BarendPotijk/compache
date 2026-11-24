"""The thread contract, and the simple shapes that are turned into it."""

import copy
import gzip
import json
import tempfile
import unittest
from pathlib import Path

from compache.threads import (
    ContractError,
    Thread,
    canonical_usage,
    load_threads,
    normalize_thread,
    recorded,
    system_prompt_of,
    write_threads,
)


def sample():
    return {
        "company_id": "c",
        "thread_id": "t",
        "runs": [
            {
                "run_id": "r",
                "run_index": 1,
                "started_at": "2025-01-01T00:00:00Z",
                "ended_at": "2025-01-01T00:00:03Z",
                "history_policy": "full_canonical",
                "input": {"part_id": "i", "text": "Question"},
                "output": {"part_id": "o", "text": "Answer"},
                "retrievals": [
                    {
                        "part_id": "p",
                        "text": "Evidence",
                        "document_id": "d",
                        "version": "v",
                        "chunk_id": "k",
                        "exposed_to_model": True,
                    }
                ],
                "generation": {
                    "observation_id": "g",
                    "model": "m",
                    "started_at": "2025-01-01T00:00:01Z",
                    "ended_at": "2025-01-01T00:00:02Z",
                    "usage": {
                        "input": 12,
                        "output": 4,
                        "cache_read": 7,
                        "cache_write": 2,
                    },
                    "request_part_ids": ["i", "p"],
                },
            }
        ],
    }


def chat():
    return {
        "thread_id": "chat",
        "messages": [
            {"role": "system", "content": "Answer from the handbook."},
            {
                "role": "user",
                "content": "How many leave days?",
                "at": "2026-01-05T09:00:00Z",
            },
            {
                "role": "retrieval",
                "content": "Staff get 25 days.",
                "id": "handbook#leave",
            },
            {"role": "assistant", "content": "25 days.", "at": "2026-01-05T09:00:04Z"},
            {
                "role": "user",
                "content": "And for part-timers?",
                "at": "2026-01-05T09:03:00Z",
            },
            {"role": "tool", "content": "Part-time staff get days pro rata."},
            {"role": "assistant", "content": "Pro rata.", "at": "2026-01-05T09:03:05Z"},
        ],
    }


class CanonicalThreadTests(unittest.TestCase):
    def test_a_thread_is_a_private_copy(self):
        data = sample()
        thread = Thread.from_dict(data)
        data["runs"][0]["input"]["text"] = "changed"
        self.assertEqual(thread.runs[0].input.text, "Question")
        self.assertEqual(
            thread.part_origin,
            {"i": ("input", 1), "o": ("output", 1), "p": ("retrieval", 1)},
        )

    def test_bad_contracts(self):
        for mutation in (
            "timestamp",
            "id",
            "future",
            "output",
            "decoy",
            "history",
            "field",
        ):
            data = sample()
            run = data["runs"][0]
            if mutation == "timestamp":
                run["started_at"] = "bad"
            if mutation == "id":
                run["run_id"] = ""
            if mutation == "future":
                run["generation"]["request_part_ids"].append("foreign")
            if mutation == "output":
                run["generation"]["request_part_ids"].append("o")
            if mutation == "decoy":
                run["retrievals"][0]["exposed_to_model"] = False
            if mutation == "history":
                run["generation"]["request_part_ids"] = ["i"]
            if mutation == "field":
                data["title"] = "extra"
            with self.subTest(mutation=mutation), self.assertRaises(ContractError):
                Thread.from_dict(data)

    def test_usage_must_be_consistent(self):
        data = sample()
        data["runs"][0]["generation"]["usage"]["cache_read"] = 99
        with self.assertRaises(ContractError):
            Thread(data)
        data = sample()
        del data["runs"][0]["generation"]["usage"]
        with self.assertRaisesRegex(ContractError, "explicit"):
            Thread(data)

    def test_masked_text_must_be_null(self):
        data = sample()
        data["runs"][0]["retrievals"][0]["content_status"] = "masked"
        with self.assertRaisesRegex(ContractError, "null text"):
            Thread(data)
        data = sample()
        data["runs"][0]["input"]["content_status"] = "missing"
        with self.assertRaisesRegex(ContractError, "null text"):
            Thread.from_dict(data)


class SimpleShapeTests(unittest.TestCase):
    def test_a_canonical_thread_is_unchanged(self):
        self.assertEqual(normalize_thread(sample()), sample())

    def test_chat_log_citations_resolve_to_exposed_occurrences(self):
        messages = [
            {"role": "user", "at": 1, "content": "How many?"},
            {"role": "tool", "part_id": "inv-1", "content": "Five"},
            {"role": "assistant", "content": "Five", "cited_part_ids": ["inv-1"]},
            {"role": "tool", "content": "Refreshed result"},
            {
                "role": "assistant",
                "content": "Confirmed",
                "cited_part_ids": ["inv-1", "chat:1:retrieval:2"],
            },
        ]
        raw = {"thread_id": "chat", "messages": messages}
        th = Thread(normalize_thread(raw))
        self.assertEqual(th.runs[0].cited_part_ids, ("inv-1", "chat:1:retrieval:2"))
        self.assertEqual(Thread(th.to_dict()).runs, th.runs)
        for invalid in (["chat:1:retrieval:2"], ["inv-1", "inv-1"], "inv-1", [None]):
            messages[2]["cited_part_ids"] = invalid
            with self.assertRaises(ContractError):
                Thread(normalize_thread(raw))

    def test_a_chat_log_becomes_runs(self):
        thread = Thread(normalize_thread(chat()))
        self.assertEqual(len(thread.runs), 2)
        first, second = thread.runs
        self.assertEqual(thread.company_id, "default")
        self.assertEqual(
            (first.run_id, first.input.part_id), ("chat:1", "chat:1:input")
        )
        self.assertEqual(first.input.text, "How many leave days?")
        self.assertEqual(first.output.text, "25 days.")
        self.assertEqual(first.retrievals[0].document_id, "handbook#leave")
        self.assertEqual(second.retrievals[0].part_id, "chat:2:retrieval:1")
        self.assertEqual(
            second.generation.request_part_ids,
            (
                "chat:1:input",
                "chat:1:retrieval:1",
                "chat:1:output",
                "chat:2:input",
                "chat:2:retrieval:1",
            ),
        )
        self.assertEqual(first.started_at, "2026-01-05T09:00:00Z")
        self.assertEqual(first.ended_at, "2026-01-05T09:00:04Z")
        self.assertEqual(system_prompt_of(chat()), "Answer from the handbook.")
        self.assertIsNone(system_prompt_of(sample()))

    def test_tool_results_inside_blocks_and_split_messages(self):
        data = {
            "thread_id": "a",
            "messages": [
                {"role": "tool", "content": "Loaded before the first question."},
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "First part."}],
                    "at": 1767603600,
                },
                {"role": "user", "content": "Second part.", "at": 1767603601},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Let me look."},
                        {"type": "tool_use", "name": "search", "input": {}},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "content": [{"type": "text", "text": "Found it."}],
                        }
                    ],
                },
                {
                    "role": "assistant",
                    "content": "Here it is.",
                    "at": 1767603610,
                    "model": "m",
                    "usage": {
                        "input_tokens": 5,
                        "output_tokens": 3,
                        "cache_read_input_tokens": 20,
                        "cache_creation_input_tokens": 0,
                    },
                },
                {"role": "user", "content": "Never answered.", "at": 1767603700},
            ],
        }
        thread = Thread(normalize_thread(data))
        self.assertEqual(len(thread.runs), 1)
        run = thread.runs[0]
        self.assertEqual(run.input.text, "First part.\n\nSecond part.")
        self.assertEqual(
            [p.text for p in run.retrievals],
            ["Loaded before the first question.", "Found it."],
        )
        self.assertEqual(run.output.text, "Let me look.\n\nHere it is.")
        self.assertEqual(run.started_at, "2026-01-05T09:00:00Z")
        self.assertEqual(run.generation.model, "m")
        self.assertEqual(
            dict(run.generation.usage),
            {"input": 25, "output": 3, "cache_read": 20, "cache_write": 0},
        )

    def test_model_calls_per_turn_are_recorded(self):
        usage = {
            "input_tokens": 5,
            "output_tokens": 3,
            "cache_read_input_tokens": 20,
            "cache_creation_input_tokens": 0,
        }
        chat = {
            "thread_id": "loop",
            "messages": [
                {"role": "user", "content": "Find it.", "at": 1767603600},
                {
                    "role": "assistant",
                    "content": [{"type": "tool_use", "name": "search", "input": {}}],
                    "usage": usage,
                },
                {"role": "tool", "content": "Found."},
                {
                    "role": "assistant",
                    "content": "Here.",
                    "at": 1767603610,
                    "usage": {
                        **usage,
                        "output_tokens": 7,
                        "cache_read_input_tokens": 120,
                    },
                },
                {"role": "user", "content": "Thanks.", "at": 1767603700},
                {"role": "assistant", "content": "Welcome.", "at": 1767603702},
            ],
        }
        first, second = Thread(normalize_thread(chat)).runs
        # Every assistant message is one call. The turn's usage is the last call's input, all
        # the output, and the cache read of the first call: what the turn found on arrival.
        self.assertEqual((first.calls, second.calls), (2, 1))
        self.assertEqual(
            dict(first.generation.usage),
            {"input": 125, "output": 10, "cache_read": 20, "cache_write": 0},
        )
        explicit = {
            "thread_id": "e",
            "runs": [
                {"started_at": 1767603600, "input": "q", "output": "a", "calls": 3}
            ],
        }
        self.assertEqual(Thread(normalize_thread(explicit)).runs[0].calls, 3)
        self.assertIsNone(Thread(sample()).runs[0].calls)
        for bad in (0, "2", 1.5, True):
            explicit["runs"][0]["calls"] = bad
            with (
                self.subTest(calls=bad),
                self.assertRaisesRegex(ContractError, "calls"),
            ):
                Thread(normalize_thread(explicit))

    def test_a_record_says_which_model_prompt_and_tools_it_was_made_with(self):
        chat = {
            "thread_id": "c",
            "tools": [{"name": "search"}],
            "observed_until": "2026-01-05T10:00:00Z",
            "messages": [
                {"role": "system", "content": "Be brief."},
                {"role": "user", "content": "Hi", "at": "2026-01-05T09:00:00Z"},
                {"role": "assistant", "content": "Hello.", "model": "model-a"},
            ],
        }
        runs = {
            "thread_id": "r",
            "model": "model-b",
            "observed_until": "2026-01-05T11:00:00+00:00",
            "runs": [
                {"started_at": 1767603600, "input": "q", "output": "a"},
                {"started_at": 1767603700, "input": "q", "output": "a"},
            ],
        }
        found = recorded([chat, runs, Thread(sample())])
        self.assertEqual(found["system_prompt"], "Be brief.")
        self.assertEqual(found["tools"], [{"name": "search"}])
        self.assertEqual((found["model"], found["models"]["model-a"]), ("model-b", 1))
        self.assertEqual(found["observed_until"], "2026-01-05T11:00:00+00:00")
        # What the record says about itself is not part of the conversation.
        self.assertNotIn("tools", normalize_thread(chat))
        self.assertEqual(
            recorded([{"thread_id": "x", "runs": []}]),
            {
                "system_prompt": None,
                "tools": None,
                "model": None,
                "models": {},
                "observed_until": None,
            },
        )

    def test_explicit_runs_with_plain_text(self):
        data = {
            "thread_id": "b",
            "model": "recorded",
            "runs": [
                {
                    "started_at": "2026-01-05T09:00:00",
                    "input": "Q1",
                    "retrievals": ["R1", {"text": "R2", "id": "doc-2"}],
                    "output": "A1",
                },
                {
                    "started_at": "2026-01-05T10:05:00+01:00",
                    "ended_at": "2026-01-05T10:05:09+01:00",
                    "input": "Q2",
                    "output": "A2",
                },
            ],
        }
        thread = Thread(normalize_thread(data))
        first, second = thread.runs
        self.assertEqual(first.started_at, "2026-01-05T09:00:00Z")
        self.assertEqual(first.ended_at, first.started_at)
        self.assertEqual(
            [p.document_id for p in first.retrievals], ["b:1:retrieval:1", "doc-2"]
        )
        self.assertEqual(second.retrievals, ())
        self.assertEqual(first.generation.model, "recorded")
        self.assertEqual(
            Thread(normalize_thread(data, model="priced")).runs[1].generation.model,
            "priced",
        )

    def test_what_cannot_be_read_is_refused(self):
        cases = {
            "no time": {
                "thread_id": "x",
                "messages": [
                    {"role": "user", "content": "q"},
                    {"role": "assistant", "content": "a"},
                ],
            },
            "no answer": {
                "thread_id": "x",
                "messages": [{"role": "user", "content": "q", "at": 1}],
            },
            "odd role": {
                "thread_id": "x",
                "messages": [{"role": "narrator", "content": "q", "at": 1}],
            },
            "both": {"thread_id": "x", "messages": [], "runs": []},
            "no runs": {"thread_id": "x", "runs": []},
            "no id": {"runs": [{"input": "q"}]},
        }
        for name, data in cases.items():
            with self.subTest(name), self.assertRaises(ContractError):
                normalize_thread(data)

    def test_provider_usage_shapes(self):
        self.assertEqual(
            canonical_usage(
                {
                    "input_tokens": 10,
                    "output_tokens": 4,
                    "cache_read_input_tokens": 70,
                    "cache_creation_input_tokens": 20,
                }
            ),
            {"input": 100, "output": 4, "cache_read": 70, "cache_write": 20},
        )
        self.assertEqual(
            canonical_usage(
                {
                    "prompt_tokens": 100,
                    "completion_tokens": 4,
                    "prompt_tokens_details": {"cached_tokens": 70},
                }
            ),
            {"input": 100, "output": 4, "cache_read": 70, "cache_write": 0},
        )
        self.assertEqual(
            canonical_usage(
                {
                    "input_tokens": 100,
                    "output_tokens": 4,
                    "input_tokens_details": {
                        "cached_tokens": 70,
                        "cache_write_tokens": 20,
                    },
                }
            ),
            {"input": 100, "output": 4, "cache_read": 70, "cache_write": 20},
        )
        # LiteLLM reports both names: its prompt total counts what was cached already.
        self.assertEqual(
            canonical_usage(
                {
                    "prompt_tokens": 100,
                    "completion_tokens": 4,
                    "total_tokens": 104,
                    "cache_read_input_tokens": 70,
                    "cache_creation_input_tokens": 20,
                    "prompt_tokens_details": {
                        "cached_tokens": 70,
                        "cache_write_tokens": 20,
                    },
                }
            ),
            {"input": 100, "output": 4, "cache_read": 70, "cache_write": 20},
        )
        same = {"input": 9, "output": 1, "cache_read": 2, "cache_write": 3}
        self.assertEqual(canonical_usage(same), same)
        self.assertIsNone(canonical_usage({"tokens": 5}))
        self.assertIsNone(canonical_usage(None))

        class Usage:
            def model_dump(self):
                return {
                    "input_tokens": 3,
                    "output_tokens": 2,
                    "cache_read_input_tokens": 5,
                    "cache_creation_input_tokens": 0,
                }

        self.assertEqual(canonical_usage(Usage())["input"], 8)
        self.assertIsNone(canonical_usage(object()))


class FileTests(unittest.TestCase):
    def test_files_folders_and_gzip(self):
        thread = Thread(sample())
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            write_threads([thread], base / "t.jsonl")
            self.assertEqual(
                load_threads(base / "t.jsonl")[0].to_dict(), thread.to_dict()
            )
            folder = base / "traces"
            folder.mkdir()
            write_threads([thread], folder / "a.jsonl.gz")
            with gzip.open(folder / "b.jsonl.gz", "wt", encoding="utf-8") as stream:
                stream.write(json.dumps(chat()) + "\n")
            loaded = load_threads(folder, model="priced")
            self.assertEqual([t.thread_id for t in loaded], ["t", "chat"])
            self.assertTrue(
                all(r.generation.model == "priced" for t in loaded for r in t.runs)
            )
            (base / "bad.jsonl").write_text("{not json}\n")
            with self.assertRaisesRegex(ContractError, "invalid JSON"):
                load_threads(base / "bad.jsonl")

    def test_identities_are_unique_across_threads(self):
        other = copy.deepcopy(sample())
        other["thread_id"] = "t2"
        with self.assertRaisesRegex(ContractError, "duplicate run_id"):
            load_threads([sample(), other])
        with self.assertRaisesRegex(ContractError, "thread 2"):
            load_threads([sample(), {"thread_id": "x"}])


if __name__ == "__main__":
    unittest.main()
