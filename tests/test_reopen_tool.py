import json
import unittest
from dataclasses import replace

from compache.optimizer import (
    REOPEN_INSTRUCTIONS,
    REOPEN_TOOL,
    Request,
    Source,
    handle,
    reopen,
)
from compache.threads import ContractError, Thread
from helpers import thread


def retrieval(part_id, text, *, run=1, version="v1"):
    return Source(part_id, "retrieval", run, text, "register", version, "entry")


class ReopenTests(unittest.TestCase):
    def test_host_instructions_match_the_registered_tool_and_require_grounding(self):
        name = REOPEN_TOOL["function"]["name"]
        self.assertIn(
            f"call {name} with its exact external reference", REOPEN_INSTRUCTIONS
        )
        self.assertIn("current message or retrieval", REOPEN_INSTRUCTIONS)
        self.assertIn("Read the returned source before answering", REOPEN_INSTRUCTIONS)

    def test_schema_matches_public_dispatch(self):
        tool = json.loads(json.dumps(REOPEN_TOOL))
        self.assertEqual(tool["type"], "function")
        self.assertEqual(tool["function"]["name"], reopen.__name__)
        self.assertIn("visible evidence", tool["function"]["description"])
        schema = tool["function"]["parameters"]
        self.assertEqual(set(schema["properties"]), {"reference", "offset"})
        self.assertEqual(schema["required"], ["reference"])
        self.assertFalse(schema["additionalProperties"])
        self.assertIn(
            "missing earlier subject", schema["properties"]["reference"]["description"]
        )

    def test_failed_container_lookup_gives_same_hint_as_absent_id_and_enables_retry(
        self,
    ):
        original = retrieval("r1-source", "Order KIND-88 was dispatched in May.")
        current = retrieval("r2-source", "Order TREE-17 is backordered.", run=2)
        question = Source(
            "r2-input", "input", 2, "Compare TREE-17 with the earlier order KIND-88."
        )
        request = Request("t", "r2", 2, "m", (original, question, current))
        failure = reopen(request, {"reference": "never-introduced"})
        self.assertEqual(set(failure), {"status", "hint"})
        self.assertEqual(failure["status"], "not_found")
        self.assertIn(
            "exact identifier of the missing earlier subject", failure["hint"]
        )
        self.assertIn("distinctive phrase", failure["hint"])
        for reference in (
            question.part_id,
            current.part_id,
            handle("another-thread", original),
        ):
            self.assertEqual(reopen(request, {"reference": reference}), failure)
        self.assertNotIn("KIND-88", failure["hint"])
        self.assertNotIn("TREE-17", failure["hint"])
        self.assertEqual(
            reopen(request, {"reference": "KIND-88"})["text"], original.text
        )

    def test_literal_opens_near_deep_evidence_and_handle_pages_losslessly(self):
        fact = "Case Q.42+[A] was approved."
        body = "☀🌷" * 7000 + "\n" + fact + "\n" + "λ" * 7000
        source = retrieval("record", body)
        request = Request("thread", "r2", 2, "model", (source,))
        result = reopen(request, {"reference": "q.42+[a]", "offset": 0})
        self.assertEqual(result["status"], "ok")
        self.assertGreater(result["offset"], 4096)
        self.assertEqual(result["length"], len(body))
        self.assertIn(fact, result["text"])
        self.assertEqual(result["text"], body[result["offset"] : result["next_offset"]])
        self.assertEqual(result["next_reference"], handle("thread", source))
        self.assertEqual(result["part_id"], "record")
        self.assertEqual(result["document_id"], "register")
        self.assertEqual(result["version"], "v1")
        self.assertEqual(result["chunk_id"], "entry")
        pieces = []
        reference, offset = result["handle"], 0
        while reference is not None:
            page = reopen(request, {"reference": reference, "offset": offset})
            self.assertEqual(page["offset"], offset)
            self.assertLessEqual(len(page["text"]), 4096)
            pieces.append(page["text"])
            reference, offset = page["next_reference"], page["next_offset"]
        self.assertEqual("".join(pieces), body)

    def test_scope_with_same_id_stale_handle_and_host_archive(self):
        archived = retrieval("a", "REF-17 is blue.")
        visible = retrieval("visible", "REF-17 is green.")
        masked = retrieval("masked", None)
        updated = replace(archived, version="v2")
        request = Request("one", "r2", 2, "m", (archived, visible, masked))
        eligible = {"a"}
        self.assertEqual(
            reopen(request, {"reference": "REF-17"}, archived_ids=eligible)["text"],
            archived.text,
        )
        miss = reopen(request, {"reference": "not-present"})
        for ref in (
            "visible",
            "masked",
            handle("one", visible),
            handle("two", archived),
            handle("one", updated),
        ):
            with self.subTest(reference=ref):
                self.assertEqual(
                    reopen(request, {"reference": ref}, archived_ids=eligible), miss
                )
        self.assertEqual(
            reopen(request, {"reference": "REF-17"}, archived_ids=set()), miss
        )
        self.assertEqual(reopen(request, {"reference": "masked"}), miss)

    def test_validated_trace_excludes_unexposed_current_and_future_even_if_named(self):
        data = thread(n=3).to_dict()
        unavailable = dict(
            data["runs"][0]["retrievals"][0],
            part_id="undisclosed",
            text="REF-SUPPRESSED",
            exposed_to_model=False,
        )
        data["runs"][0]["retrievals"].append(unavailable)
        trace = Thread.from_dict(data)
        request = Request.from_thread(trace, trace.runs[1].run_id)
        future = Request.from_thread(trace, trace.runs[2].run_id).parts[-1]
        current = request.parts[-1]
        for ref in (
            "undisclosed",
            "REF-SUPPRESSED",
            current.part_id,
            future.part_id,
            handle(request.thread_id, current),
            handle(request.thread_id, future),
        ):
            with self.subTest(reference=ref):
                self.assertEqual(
                    reopen(request, {"reference": ref})["status"], "not_found"
                )

    def test_input_and_output_require_scoped_handle_or_exact_part_id(self):
        for kind in ("input", "output"):
            with self.subTest(kind=kind):
                part = Source("prior-" + kind, kind, 1, "REF-98")
                request = Request("t", "r2", 2, "m", (part,))
                for ref in (handle("t", part), part.part_id):
                    self.assertEqual(reopen(request, {"reference": ref})["kind"], kind)
                self.assertEqual(
                    reopen(request, {"reference": "REF-98"})["status"], "not_found"
                )

    def test_ambiguity_and_candidate_pages_preserve_occurrence_and_version(self):
        parts = tuple(
            retrieval(
                f"source-{i}",
                "x" * 8000 + f"\nREF-5: resolved on day {i}",
                version=f"v{i}",
            )
            for i in range(10)
        )
        request = Request("t", "r2", 2, "m", parts)
        for reference in ("REF-5", "register", "entry"):
            listing = reopen(request, {"reference": reference})
            self.assertEqual(listing["status"], "matches")
            self.assertEqual(listing["total"], len(parts))
            self.assertEqual(len(listing["matches"]), 8)
            self.assertNotIn("text", listing)
            more = reopen(
                request,
                {
                    "reference": listing["next_reference"],
                    "offset": listing["next_offset"],
                },
            )
            self.assertIsNone(more["next_offset"])
            for candidate, source in zip(
                listing["matches"] + more["matches"], parts, strict=True
            ):
                self.assertEqual(candidate["version"], source.version)
                self.assertLessEqual(len(candidate["excerpt"]), 512)
                result = reopen(
                    request,
                    {"reference": candidate["handle"], "offset": candidate["offset"]},
                )
                self.assertEqual(
                    result["text"],
                    source.text[result["offset"] : result["offset"] + 4096],
                )
                if reference == "REF-5":
                    self.assertIn(f"day {source.part_id.split('-')[1]}", result["text"])
        for partial in ("REF", "EF-5", "REF-50"):
            self.assertEqual(
                reopen(request, {"reference": partial})["status"], "not_found"
            )

    def test_invalid_arguments_and_failed_reads_do_not_modify_history(self):
        part = retrieval("small", "original")
        request = Request("t", "r2", 2, "m", (part,))
        for arguments in (
            None,
            [],
            {},
            {"reference": "  "},
            {"reference": 7},
            {"reference": "x" * 257},
            {"reference": "small", "offset": -1},
            {"reference": "small", "offset": True},
            {"reference": "small", "offset": 9},
            {"reference": "small", "offset": 10**1000},
            {"reference": "small", "archived_ids": ["small"]},
        ):
            with self.subTest(arguments=arguments), self.assertRaises(ContractError):
                reopen(request, arguments)
        self.assertEqual(reopen(request, {"reference": "small"})["text"], "original")
