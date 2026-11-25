"""A policy that follows from the record it is applied to."""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import compache
from compache import learn
from compache.cli import main
from compache.learn import Learner
from compache.record import Recorder, load
from fakes import FakeModel, noticed
from test_agent_live import Agent, Chat, Claude, Gemini, worker
from test_live import Clock

MODEL = "claude-haiku-5-5"


class Haiku(Claude):
    model = MODEL


def quoting(every=3, back=2, count=24):
    """A model that repeats a line of an earlier file now and then."""

    def extra(request, seen):
        if seen >= back + 1 and seen % every == 0 and seen < count:
            line = f"a line of file number {seen - back} that only it holds"
            return {"text": line, "tool": ("read", {"file": seen})}

    return worker(count, extra)


class LearnerTests(unittest.TestCase):
    def test_it_starts_from_the_default_and_fits_once_there_is_enough(self):
        clock = Clock()
        with tempfile.TemporaryDirectory() as folder:
            live = compache.Live(
                learn=Learner(folder, clock=clock, background=False), clock=clock
            )
            self.assertIsNone(live.learner.fitted_on(MODEL))
            begun = live.learner.policy(MODEL, True)
            self.assertEqual(
                begun.recipe, compache.default_policy(MODEL, agent=True).recipe
            )
            for k in range(22):
                clock.at += 4_000
                model = FakeModel(quoting(), clock=lambda: clock.at)
                app = Agent(live, Haiku(), model, stream=k % 2 == 0, clock=clock)
                app.run(f"Task {k}")
            found = live.learner.fitted_on(MODEL)
            self.assertGreaterEqual(found["calls"], learn.FIRST_CALLS)
            self.assertTrue(
                (Path(folder) / "policies" / f"{MODEL}.agent.json").is_file()
            )
            fitted = live.learner.policy(MODEL, True)
            self.assertIn("solve", fitted.artifact)
            # It is there for the next process too.
            again = Learner(folder)
            self.assertEqual(again.policy(MODEL, True).recipe, fitted.recipe)
            self.assertEqual(again.fitted_on(MODEL), found)

    def test_one_conversation_in_ten_is_left_alone(self):
        with tempfile.TemporaryDirectory() as folder:
            learner = Learner(folder)
            given = [learner.policy(MODEL, True) is None for _ in range(30)]
            self.assertEqual(sum(given), 3)
            self.assertFalse(given[0], "never the first")

    def test_the_count_goes_on_in_the_next_process(self):
        clock = Clock()
        with tempfile.TemporaryDirectory() as folder:
            alone = []
            for k in range(20):
                # One process per conversation, as `compache wrap` makes them.
                clock.at += 4_000
                learner = Learner(folder, clock=clock, background=False)
                live = compache.Live(learn=learner, clock=clock)
                model = FakeModel(worker(2), clock=lambda: clock.at)
                Agent(live, Haiku(), model, clock=clock).run(f"Task {k}")
                alone.append("reopen" not in json.dumps(model.bodies))
            self.assertEqual([k for k, a in enumerate(alone) if a], [9, 19])

    def test_what_is_left_alone_is_recorded_whole(self):
        clock = Clock()
        with tempfile.TemporaryDirectory() as folder:
            live = compache.Live(
                learn=Learner(folder, clock=clock, background=False), clock=clock
            )
            for k in range(10):
                clock.at += 4_000
                model = FakeModel(worker(40), clock=lambda: clock.at)
                Agent(live, Haiku(), model, clock=clock).run(f"Task {k}")
                kept = not any(noticed(b["messages"]) for b in model.bodies)
                self.assertEqual(kept, k == 9)
            threads, _ = load(folder)
            known = [t["runs"][0]["steps"][-1]["needs"] is not None for t in threads]
            self.assertEqual(sum(known), 1)

    def test_a_conversation_without_tools_is_only_recorded_until_labels_teach_one(self):
        with tempfile.TemporaryDirectory() as folder:
            learner = Learner(folder)
            self.assertIsNone(learner.policy(MODEL, False))

    def test_a_model_without_a_price_sheet_is_trimmed_at_what_most_charge(self):
        class Other(Chat):
            model = "maker/some-model:free"

        with tempfile.TemporaryDirectory() as folder:
            live = compache.Live(learn=folder)
            model = FakeModel(worker(40))
            Agent(live, Other(), model).run("Read forty files")
            self.assertTrue(any(noticed(b["messages"]) for b in model.bodies))
            self.assertEqual(live.learner.policy(Other.model, True).max_pings, 0)
            self.assertIsNone(live.learner.header_lifetime(Other.model, 0.0))
            self.assertIsNone(live.learner.policy(None, True))

    def test_a_model_without_a_price_sheet_is_fitted_on_too(self):
        class Other(Haiku):
            model = "maker/some-model:free"

        clock = Clock()
        with tempfile.TemporaryDirectory() as folder:
            learner = Learner(folder, clock=clock, background=False)
            live = compache.Live(learn=learner, clock=clock)
            for k in range(9):
                clock.at += 4_000
                model = FakeModel(quoting(), clock=lambda: clock.at)
                Agent(live, Other(), model, clock=clock).run(f"Task {k}")
            self.assertTrue(
                (
                    Path(folder) / "policies" / "maker_some-model_free.agent.json"
                ).is_file()
            )
            self.assertIsNotNone(learner.fitted_on(Other.model))

    def test_a_model_that_only_the_path_names(self):
        class Named(Gemini):
            path = f"/v1beta/models/{MODEL}:generateContent"

        with tempfile.TemporaryDirectory() as folder:
            live = compache.Live(learn=folder)
            model = FakeModel(worker(40))
            Agent(live, Named(), model).run("Read forty files")
            self.assertTrue(any(noticed(b["contents"]) for b in model.bodies))

    def test_the_header_is_cached_for_longer_once_that_would_have_helped(self):
        clock = Clock()
        with tempfile.TemporaryDirectory() as folder:

            def task(live, app=None):
                model = FakeModel(worker(2), clock=lambda: clock.at)
                app = app or Agent(live, Haiku(), model, clock=clock)
                app.model = model
                app.run("Read two files")
                marked = ['"1h"' in json.dumps(b["system"]) for b in model.bodies]
                return app, marked

            live = compache.Live(learn=Learner(folder, clock=clock), clock=clock)
            app, marked = task(live)
            self.assertFalse(any(marked), "nothing speaks for it yet")
            clock.at += 100
            _, marked = task(live, app)
            self.assertFalse(any(marked), "the short entry was there")
            clock.at += 900
            app, marked = task(live, app)
            self.assertTrue(all(marked), "it had to be written again")
            # From then on every conversation begins that way, in the next process too.
            clock.at += 50_000
            live = compache.Live(learn=Learner(folder, clock=clock), clock=clock)
            _, marked = task(live)
            self.assertTrue(all(marked))

    def test_a_session_that_begins_after_a_pause_counts_too(self):
        clock = Clock()
        with tempfile.TemporaryDirectory() as folder:
            for k, pause in enumerate((0, 600)):
                clock.at += pause
                live = compache.Live(learn=Learner(folder, clock=clock), clock=clock)
                model = FakeModel(worker(2), clock=lambda: clock.at)
                Agent(live, Haiku(), model, clock=clock).run("Read two files")
                marked = ['"1h"' in json.dumps(b["system"]) for b in model.bodies]
                self.assertEqual(set(marked), {k == 1})

    def test_a_provider_that_refuses_the_lifetime_is_remembered(self):
        with tempfile.TemporaryDirectory() as folder:
            learner = Learner(folder)
            hour = {
                **learner.policy(MODEL, True).artifact,
                "standard_spec": compache.model_spec(MODEL).to_dict(),
            }
            hour["model_spec"] = {
                **hour["model_spec"],
                "cache_ttl_seconds": 3600,
                "write_multiplier": 2.0,
            }
            compache.Policy(hour).save(
                Path(folder) / "policies" / f"{MODEL}.agent.json"
            )
            again = Learner(folder)
            self.assertEqual(again.policy(MODEL, True).cache_lifetime, 3600)
            again.default_lifetime = True
            self.assertIsNone(again.policy(MODEL, True).cache_lifetime)

    def test_the_command_that_fits_again(self):
        clock = Clock()
        with tempfile.TemporaryDirectory() as folder:
            live = compache.Live(
                learn=Learner(folder, clock=clock, background=True), clock=clock
            )
            live.learner.refit = lambda **_: None
            for k in range(12):
                clock.at += 4_000
                model = FakeModel(quoting(), clock=lambda: clock.at)
                Agent(live, Haiku(), model, clock=clock).run(f"Task {k}")
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(main(["learn", folder]), 0)
            self.assertIn(f"Fitted {MODEL}.agent on 300 calls", out.getvalue())
            with contextlib.redirect_stdout(out):
                self.assertEqual(main(["learn", folder]), 0)
            self.assertIn("Nothing has grown enough", out.getvalue())

    def test_a_line_that_was_never_finished_is_passed_over(self):
        clock = Clock()
        with tempfile.TemporaryDirectory() as folder:

            def session(k):
                clock.at += 4_000
                learner = Learner(folder, clock=clock, background=False)
                live = compache.Live(learn=learner, clock=clock)
                model = FakeModel(quoting(), clock=lambda: clock.at)
                Agent(live, Haiku(), model, clock=clock).run(f"Task {k}")
                return "reopen" not in json.dumps(model.bodies)

            alone = [session(k) for k in range(5)]
            turns = Path(folder) / "turns.jsonl"
            last = turns.read_text().splitlines()[-1]
            with open(turns, "a") as stream:
                stream.write(last[: len(last) // 2])
            alone += [session(k) for k in range(5, 15)]
            self.assertEqual([k for k, a in enumerate(alone) if a], [9])
            learner = Learner(folder, clock=clock)
            learner.refit(wait=True)
            self.assertEqual(learner.fitted_on(MODEL)["threads"], 15)

    def test_a_conversation_that_does_not_load_is_left_out_of_the_fit(self):
        clock = Clock()
        with tempfile.TemporaryDirectory() as folder:
            learner = Learner(folder, clock=clock, background=False)
            live = compache.Live(learn=learner, clock=clock)
            for k in range(12):
                clock.at += 4_000
                model = FakeModel(quoting(), clock=lambda: clock.at)
                Agent(live, Haiku(), model, clock=clock).run(f"Task {k}")
            turns = Path(folder) / "turns.jsonl"
            rows = [json.loads(line) for line in turns.read_text().splitlines()]
            rows[0]["steps"][3][3] = -1
            turns.write_text("".join(json.dumps(r) + "\n" for r in rows))
            again = Learner(folder, clock=clock)
            again._state = {}
            with self.assertLogs("compache.learn", "WARNING"):
                again.refit(wait=True)
            self.assertEqual(again.fitted_on(MODEL)["calls"], 300)
            self.assertIn("solve", again.policy(MODEL, True).artifact)

    def test_replies_may_name_the_model_otherwise(self):
        """OpenAI answers with the snapshot that an alias stands for."""

        class Mini(Chat):
            model = "some-model-mini"

        class Dated(FakeModel):
            def chat(self, body, path="/v1/chat/completions", headers=None):
                status, reply = super().chat(body, path, headers)
                if isinstance(reply, dict):
                    reply["model"] = "some-model-mini-2026-08-07"
                return status, reply

        clock = Clock()
        with tempfile.TemporaryDirectory() as folder:
            alone = []
            for k in range(12):
                clock.at += 4_000
                learner = Learner(folder, clock=clock, background=False)
                live = compache.Live(learn=learner, clock=clock)
                model = Dated(quoting(), clock=lambda: clock.at)
                Agent(live, Mini(), model, clock=clock).run(f"Task {k}")
                alone.append("reopen" not in json.dumps(model.bodies))
            self.assertEqual([k for k, a in enumerate(alone) if a], [9])
            learner.refit(wait=True)
            again = Learner(folder, clock=clock)
            self.assertIn("solve", again.policy(Mini.model, True).artifact)
            self.assertIsNotNone(again.fitted_on(Mini.model))

    def test_one_that_is_left_alone_stays_so_where_the_host_branches_off(self):
        for named in (False, True):
            clock = Clock()
            with tempfile.TemporaryDirectory() as folder:
                learner = Learner(folder, clock=clock, background=False)
                live = compache.Live(learn=learner, clock=clock)
                for k in range(10):
                    clock.at += 4_000
                    model = FakeModel(quoting(count=40), clock=lambda: clock.at)
                    name = f"session {k}" if named else None
                    app = Agent(live, Haiku(), model, clock=clock, thread=name)
                    app.run(f"Task {k}")
                kept = list(app.shape.said)
                app.shape.ask("Suggest what the user may say next")
                model.script = lambda request: {"text": "run the tests"}
                clock.at += 1
                app.call()
                app.shape.said = kept
                model.script = quoting(back=45, count=70)
                clock.at += 20
                app.run("Go on")
                self.assertNotIn("reopen", json.dumps(model.bodies))
                rows, _ = load(folder)
                known = [
                    r for r in rows if r["runs"][0]["steps"][0]["needs"] is not None
                ]
                self.assertEqual(len(known), 1)
                # What was read before it branched is known to be quoted after.
                (_, run) = known[0]["runs"]
                first = [n for s in run["steps"] for n in s["needs"] if ":1:" in n]
                self.assertTrue(first)

    def test_no_pings_are_fitted_where_the_cache_is_a_guess(self):
        model = "maker/some-model:free"
        with tempfile.TemporaryDirectory() as folder:
            recorder = Recorder(folder)
            for k in range(10):
                begun = at = 1.79e9 + 1e6 * k
                steps = []
                for j in range(30):
                    step = {"started_at": at, "ended_at": at + 5, "results": j}
                    step |= {"bytes": 200, "output_tokens": 50, "input_tokens": None}
                    steps.append({**step, "thought_tokens": 0, "needs": []})
                    # A test suite of seven minutes.
                    at += 420
                recorder.turn(
                    f"t{k}",
                    1,
                    started_at=begun,
                    ended_at=at,
                    model=model,
                    sizes={
                        "input": 400,
                        "retrievals": [8_000] * 29,
                        "output": 200,
                        "system": 8_000,
                        "tools": 4_000,
                    },
                    calls=30,
                    steps=steps,
                )
            learner = Learner(folder)
            learner.refit(wait=True)
            self.assertIsNotNone(learner.fitted_on(model))
            self.assertEqual(learner.policy(model, True).max_pings, 0)

    def test_a_refused_lifetime_is_not_asked_for_in_a_chat_either(self):
        with tempfile.TemporaryDirectory() as folder:
            chat = compache.default_policy(MODEL).artifact
            hour = {"cache_ttl_seconds": 3600, "write_multiplier": 2.0}
            hour = {**chat["model_spec"], **hour}
            policy = {**chat, "standard_spec": chat["model_spec"], "model_spec": hour}
            learner = Learner(folder)
            path = Path(folder) / "policies" / f"{MODEL}.chat.json"
            compache.Policy(policy).save(path)
            self.assertEqual(learner.policy(MODEL, False).cache_lifetime, 3600)
            learner.default_lifetime = True
            self.assertIsNone(learner.policy(MODEL, False).cache_lifetime)

    def test_processes_share_what_they_learned_about_the_header(self):
        models = (MODEL, "claude-sonnet-5-5")
        with tempfile.TemporaryDirectory() as folder:
            for learner, model in zip([Learner(folder), Learner(folder)], models):
                learner.policy(model, True)
                learner.opened(model, 1_000.0)
                learner.header_lifetime(model, 1_600.0)
            for model in models:
                third = Learner(folder)
                third.policy(model, True)
                self.assertEqual(third.header_lifetime(model, 99_999.0), 3_600)

    def test_two_learners_may_write_at_once(self):
        with tempfile.TemporaryDirectory() as folder:
            learners = [Learner(folder), Learner(folder)]
            failed = []

            def write(learner):
                for k in range(100):
                    try:
                        learner._write("state.json", json.dumps({"k": "x" * 2_000}))
                    except Exception as error:
                        failed.append(error)

            threads = [threading.Thread(target=write, args=(x,)) for x in learners]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            self.assertFalse(failed)
            left = [p.name for p in (Path(folder) / "policies").iterdir()]
            self.assertEqual(left, ["state.json"])

    def test_a_file_that_cannot_be_had_at_this_moment(self):
        """As on Windows, while another process reads or replaces it."""
        with tempfile.TemporaryDirectory() as folder:
            learner, refused = Learner(folder), []

            def busy(action):
                def tried(*arguments):
                    if len(refused) < 3:
                        refused.append(arguments)
                        raise PermissionError(13, "Access is denied")
                    return action(*arguments)

                return tried

            with mock.patch.object(learn, "_AGAIN", 9):
                for name in ("replace", "read_text"):
                    refused.clear()
                    whole = getattr(Path, name)
                    with mock.patch.object(Path, name, autospec=True) as patched:
                        patched.side_effect = busy(whole)
                        if name == "replace":
                            learner._write("state.json", '{"k": 1}')
                        else:
                            self.assertEqual(learner._read_state(), {"k": 1})
                    self.assertEqual(len(refused), 3)

    def test_a_broken_policy_file_does_not_stop_anything(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "policies"
            path.mkdir()
            (path / f"{MODEL}.agent.json").write_text("{")
            (path / "state.json").write_text("[]")
            self.assertTrue(Learner(folder).policy(MODEL, True).agentic)

    def test_the_folder_is_all_there_is_to_it(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                compache.Live("policy.json", learn=folder)
            with self.assertRaises(ValueError):
                compache.Live(learn=folder, record=folder)


if __name__ == "__main__":
    unittest.main()
