"""Threads from the sessions that coding agents keep on disk."""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import compache
from compache import transcripts
from compache.cli import main
from compache.record import Recorder, load, thread_name

START = datetime.now(timezone.utc) - timedelta(days=1)


def at(seconds):
    moment = START + timedelta(seconds=seconds)
    return moment.isoformat().replace("+00:00", "Z")


def claude_rows(session="s-1", calls=4, name="claude-haiku-5-5"):
    """A session as Claude Code writes it: one line per block, and lines about other things in between."""
    rows = [
        {"type": "mode", "mode": "normal", "sessionId": session},
        {
            "type": "user",
            "sessionId": session,
            "timestamp": at(0),
            "message": {"role": "user", "content": "Fix the bug"},
        },
        {
            "type": "user",
            "sessionId": session,
            "timestamp": at(0),
            "isMeta": True,
            "message": {"role": "user", "content": "caveat"},
        },
    ]
    size, clock = 20_000, 1

    def reply(k, blocks, usage):
        for j, block in enumerate(blocks):
            rows.append(
                {
                    "type": "assistant",
                    "sessionId": session,
                    "timestamp": at(clock + 0.1 * j),
                    "message": {
                        "id": f"msg-{session}-{k}",
                        "role": "assistant",
                        "model": name,
                        "content": [block],
                        "usage": usage,
                    },
                }
            )

    for k in range(calls):
        usage = {
            "input_tokens": 2,
            "cache_read_input_tokens": size if k else 0,
            "cache_creation_input_tokens": 500 if k else size,
            "output_tokens": 90,
            "output_tokens_details": {"thinking_tokens": 40},
        }
        said = f"result line number {k - 2} that only it holds" if k >= 3 else "look"
        tools = [
            {
                "type": "tool_use",
                "id": f"t{k}{j}",
                "name": "Bash",
                "input": {"command": said},
            }
            for j in range(2)
        ]
        reply(k, [{"type": "thinking", "thinking": "", "signature": "x" * 1100}], usage)
        reply(k, tools[:1], usage)
        for j, tool in enumerate(tools):
            if j:
                # The second call of the same reply, after the first one's result.
                reply(k, tools[1:], usage)
            rows.append(
                {
                    "type": "user",
                    "sessionId": session,
                    "timestamp": at(clock + 1),
                    "message": {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": tool["id"],
                                "content": f"result line number {k} that only it holds\n"
                                + "more " * 300,
                            }
                        ],
                    },
                }
            )
        size, clock = size + 500, clock + 10
    reply(
        "last",
        [{"type": "text", "text": "Fixed."}],
        {"input_tokens": size, "output_tokens": 5},
    )
    return rows


def codex_rows(session="c-1", calls=3):
    """A session as Codex writes it: items, and the usage of a call after its items."""

    def row(seconds, kind, **payload):
        return {"timestamp": at(seconds), "type": kind, "payload": payload}

    rows = [
        row(0, "session_meta", id=session),
        row(0, "turn_context", model="gpt-6-luna"),
        row(0, "response_item", type="message", role="developer", content="You are"),
        row(
            1,
            "response_item",
            type="message",
            role="user",
            content=[{"type": "input_text", "text": "Fix the bug"}],
        ),
    ]
    clock, size = 2, 10_000
    for k in range(calls):
        usage = {
            "input_tokens": size,
            "cached_input_tokens": size - 600 if k else 0,
            "cache_write_input_tokens": 600,
            "output_tokens": 70,
            "reasoning_output_tokens": 30,
        }
        code = f'await tools.apply_patch("a\\nresult line number {k - 1} that only it holds\\n")'
        rows += [
            row(clock, "response_item", type="reasoning", encrypted_content="e" * 2000),
            row(
                clock + 1,
                "response_item",
                type="custom_tool_call",
                call_id=f"c{k}",
                name="exec",
                input=code,
            ),
            row(clock + 1, "token_usage_record", usage=usage),
            row(
                clock + 2,
                "response_item",
                type="custom_tool_call_output",
                call_id=f"c{k}",
                output=[
                    {"type": "input_text", "text": "Script completed"},
                    {
                        "type": "input_text",
                        "text": f"result line number {k} that only it holds\n"
                        + "x " * 900,
                    },
                ],
            ),
        ]
        clock, size = clock + 10, size + 1_000
    rows += [
        row(
            clock,
            "response_item",
            type="message",
            role="assistant",
            content=[{"type": "output_text", "text": "Fixed."}],
        ),
        row(
            clock,
            "token_usage_record",
            usage={"input_tokens": size, "output_tokens": 5},
        ),
    ]
    return rows


def written(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows) + '{"cut off')


class ReadTests(unittest.TestCase):
    def test_a_session_of_claude_code(self):
        with tempfile.TemporaryDirectory() as folder:
            written(Path(folder) / "p" / "s-1.jsonl", claude_rows())
            rows, report = transcripts.pull("claude", folder)
        self.assertEqual(report["sessions"], 1)
        self.assertEqual(report["models"], {"claude-haiku-5-5": 5})
        thread = compache.load_threads(rows)[0]
        self.assertEqual(thread.thread_id, "s-1")
        (run,) = thread.runs
        self.assertEqual(run.input.text, "Fix the bug")
        self.assertEqual(len(run.steps), 5)
        self.assertEqual([s.results for s in run.steps], [0, 2, 4, 6, 8])
        self.assertEqual([s.thought_tokens for s in run.steps], [40] * 4 + [0])
        self.assertEqual(run.steps[1].input_tokens, 20_000 + 500 + 500 + 2)
        self.assertTrue(run.steps[3].needs, "it repeats a line of an earlier result")
        self.assertGreater(rows[0]["system_bytes"], 70_000)

    def test_a_session_of_codex(self):
        with tempfile.TemporaryDirectory() as folder:
            written(Path(folder) / "2026" / "rollout-c-1.jsonl", codex_rows())
            rows, report = transcripts.pull("codex", folder)
        self.assertEqual(report["model_calls"], 4)
        thread = compache.load_threads(rows)[0]
        (run,) = thread.runs
        self.assertEqual([s.results for s in run.steps], [0, 1, 2, 3])
        self.assertEqual([s.thought_tokens for s in run.steps], [30, 30, 30, 0])
        self.assertEqual(
            [s.input_tokens for s in run.steps], [10_000, 11_000, 12_000, 13_000]
        )
        # A patch inside a script quotes the file it patches.
        self.assertTrue(run.steps[2].needs)

    def test_a_compacted_session_goes_on_as_another_thread(self):
        rows = claude_rows()
        more = claude_rows(calls=2)
        for k, row in enumerate(more):
            message = row.get("message")
            if isinstance(message, dict) and message.get("id"):
                message["id"] += "-more"
        more[1]["isCompactSummary"] = True
        with tempfile.TemporaryDirectory() as folder:
            written(Path(folder) / "s-1.jsonl", rows + more)
            threads, _ = transcripts.pull("claude", folder)
        self.assertEqual([t["thread_id"] for t in threads], ["s-1", "s-1#2"])

    def test_what_an_agent_started_is_a_thread_of_its_own(self):
        rows = claude_rows()
        side = [dict(r, agentId="a7", isSidechain=True) for r in claude_rows(calls=2)]
        for row in side:
            message = row.get("message")
            if isinstance(message, dict) and message.get("id"):
                row["message"] = {**message, "id": message["id"] + "-side"}
        with tempfile.TemporaryDirectory() as folder:
            written(Path(folder) / "s-1.jsonl", rows + side)
            threads, _ = transcripts.pull("claude", folder)
        self.assertEqual({t["thread_id"] for t in threads}, {"s-1", "s-1~a7"})

    def test_what_is_older_is_left_out(self):
        with tempfile.TemporaryDirectory() as folder:
            written(Path(folder) / "s-1.jsonl", claude_rows())
            self.assertEqual(transcripts.pull("claude", folder, days=0.5)[0], [])
            with self.assertRaises(compache.ContractError):
                transcripts.pull("claude", Path(folder) / "nowhere")
            with self.assertRaises(compache.ContractError):
                transcripts.pull("someone", folder)


class ImportTests(unittest.TestCase):
    def test_past_sessions_join_a_record_as_sizes_and_times(self):
        with tempfile.TemporaryDirectory() as folder:
            home, kept = Path(folder) / "sessions", Path(folder) / "record"
            for k in range(5):
                written(home / f"s-{k}.jsonl", claude_rows(f"s-{k}", calls=45))
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                status = main(
                    ["learn", str(kept), "--from", "claude", "--sessions", str(home)]
                )
            self.assertEqual(status, 0, out.getvalue())
            self.assertIn("Added 5 of 5 sessions, 230 model calls", out.getvalue())
            self.assertIn("Fitted claude-haiku-5-5.agent on 230 calls", out.getvalue())
            text = (kept / "turns.jsonl").read_text()
            self.assertNotIn("result line", text)
            self.assertNotIn("s-1", text, "a session goes by a hash")
            threads, _ = load(kept)
            steps = threads[0]["runs"][0]["steps"]
            self.assertTrue(any(s["needs"] for s in steps))
            self.assertEqual(steps[0]["thought_tokens"], 40)
            # Once more adds nothing.
            found = transcripts.imported(kept, "claude", home)
            self.assertEqual(found["added"], 0)

    def test_a_session_that_went_through_compache_is_left_as_it_is(self):
        with tempfile.TemporaryDirectory() as folder:
            home, kept = Path(folder) / "sessions", Path(folder) / "record"
            written(home / "s-1.jsonl", claude_rows())
            Recorder(kept).turn(
                "s-1",
                1,
                started_at=1.0,
                ended_at=2.0,
                model="claude-haiku-5-5",
                sizes={"input": 1, "retrievals": [], "output": 1, "tools": 9},
                cut=True,
            )
            before = (kept / "turns.jsonl").read_text()
            self.assertEqual(transcripts.imported(kept, "claude", home)["added"], 0)
            self.assertEqual((kept / "turns.jsonl").read_text(), before)
            self.assertIn(thread_name("s-1"), before)

    def test_a_threads_file_to_fit_on(self):
        with tempfile.TemporaryDirectory() as folder:
            home, out = Path(folder) / "sessions", io.StringIO()
            for k in range(3):
                written(home / f"s-{k}.jsonl", claude_rows(f"s-{k}", calls=30))
            path = Path(folder) / "threads.jsonl.gz"
            with contextlib.redirect_stdout(out):
                self.assertEqual(
                    main(["pull", "claude", str(home), "-o", str(path)]), 0
                )
                status = main(
                    ["fit", str(path), "-o", str(Path(folder) / "policy.json")]
                )
            self.assertEqual(status, 0, out.getvalue())
            self.assertIn("Read 3 files: 93 model calls in 3 sessions", out.getvalue())
            policy = compache.Policy.load(Path(folder) / "policy.json")
            self.assertTrue(policy.agentic)


if __name__ == "__main__":
    unittest.main()
