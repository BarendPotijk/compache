"""A policy that follows from the record it is applied to."""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import compache
from compache import learn
from compache.cli import main
from compache.learn import Learner
from compache.record import load
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

    def test_a_model_without_a_price_sheet_is_only_recorded(self):
        with tempfile.TemporaryDirectory() as folder:
            live = compache.Live(learn=folder)
            model = FakeModel(worker(20))
            app = Agent(live, Chat(), model)
            app.run("Read twenty files")
            self.assertFalse(any(noticed(b["messages"]) for b in model.bodies))
            self.assertNotIn("reopen", json.dumps(model.bodies))
            self.assertEqual(len(load(folder)[0]), 1)

    def test_a_model_that_only_the_path_names(self):
        class Named(Gemini):
            path = f"/v1beta/models/{MODEL}:generateContent"

        with tempfile.TemporaryDirectory() as folder:
            live = compache.Live(learn=folder)
            model = FakeModel(worker(40))
            Agent(live, Named(), model).run("Read forty files")
            self.assertTrue(any(noticed(b["contents"]) for b in model.bodies))

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
