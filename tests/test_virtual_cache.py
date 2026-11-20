import unittest

from compache.optimizer.virtual_cache import DEFAULT_TTLS, CacheRequest, VirtualCache
from compache.optimizer.usage import response_usage
from compache.traces import ContractError


def message(role, content):
    return {"role": role, "content": content}


def request(*items, scope="lane-1", **options):
    return CacheRequest.from_payload(
        {"model": "gpt-6-luna", "input": list(items), **options}, scope=scope
    )


def meter(total, read, write, output=5):
    return {
        "input_tokens": total,
        "cached_input_tokens": read,
        "cache_write_input_tokens": write,
        "output_tokens": output,
    }


class VirtualCacheTests(unittest.TestCase):
    header = message("developer", "pinned instructions")
    question = message("user", "source and question")
    answer = message("assistant", "answer")
    ping = message("user", "reply OK")
    followup = message("user", "different question")

    def test_published_endpoint_reachability_for_training(self):
        source = request(self.header, self.question)
        extra = [message("user", str(i)) for i in range(21)]
        self.assertIs(source.published_endpoint_reachable_in(
            request(self.header, self.question, *extra[:20])), True)
        self.assertIs(source.published_endpoint_reachable_in(
            request(self.header, self.question, *extra)), False)
        self.assertIs(source.published_endpoint_reachable_in(
            request(self.header, self.followup)), False)
        self.assertIsNone(source.published_endpoint_reachable_in(
            request(self.header, self.question, scope="another-lane")))
        self.assertIs(request(self.answer).published_endpoint_reachable_in(source), False)

    def test_receipt_capacity_and_native_ping_sibling(self):
        seed = request(self.header, self.question)
        ping = request(self.header, self.question, self.answer, self.ping)
        followup = request(self.header, self.question, self.answer, self.followup)
        cache = VirtualCache()
        # An already hot physical namespace still starts with a compulsory virtual write.
        first = cache.apply(seed, at=0, usage=meter(8873, 8798, 0), state="warm")
        self.assertEqual(first["modeled_usage"], meter(8873, 0, 8798))
        self.assertEqual(first["observed_usage"], meter(8873, 8798, 0))
        self.assertEqual(first["cache_status"], "first")
        self.assertEqual(cache.available(ping, at=255), 8798)
        # Also model warm reads correctly if the actual request was physically cold.
        first_ping = cache.apply(ping, at=255, usage=meter(8921, 0, 8846))
        self.assertEqual(first_ping["modeled_usage"], meter(8921, 8798, 48))
        self.assertEqual(first_ping["matched_item_count"], 2)
        repeat = cache.apply(ping, at=510, usage=meter(8921, 8846, 0))
        self.assertEqual(repeat["modeled_usage"], meter(8921, 8846, 0))
        self.assertEqual(repeat["matched_item_count"], 4)
        result = cache.apply(followup, at=750, usage=meter(8980, 8798, 100))
        self.assertEqual(result["modeled_usage"], meter(8980, 8798, 100))
        self.assertEqual(result["matched_item_count"], 2)

    def test_expiry_and_overrides_consider_only_previously_written_boundaries(self):
        seed = request(self.question)
        branches = [
            request(self.question, message("user", text)) for text in ("a", "b", "c")
        ]
        for ttl in DEFAULT_TTLS:
            with self.subTest(ttl=ttl):
                cache = VirtualCache(ttl)
                cache.apply(seed, at=0, usage=meter(1203, 0, 1200))
                self.assertEqual(cache.available(seed, at=ttl - 1), 1200)
                expired = cache.apply(branches[0], at=ttl, usage=meter(1253, 1200, 50))
                self.assertEqual(expired["cache_status"], "expired")
                self.assertEqual(expired["modeled_usage"], meter(1253, 0, 1250))
                self.assertEqual(cache.available(branches[1], at=ttl + 1), 0)
                forced = cache.apply(
                    branches[1], at=ttl + 1, usage=meter(1253, 0, 1250), state="warm"
                )
                self.assertEqual(forced["modeled_usage"], meter(1253, 1200, 50))
                cold = cache.apply(
                    branches[2], at=ttl + 2, usage=meter(1253, 1200, 50), state="cold"
                )
                self.assertEqual(cold["modeled_usage"], meter(1253, 0, 1250))
                self.assertEqual(cache.available(seed, at=ttl + 3), 0)
                self.assertEqual(cache.available(branches[0], at=ttl + 3), 1250)

    def test_latest_write_eligible_lookback_and_native_tool_groups(self):
        cache = VirtualCache()
        cache.apply(request(self.header), at=0, usage=meter(1203, 0, 1200))
        source = request(self.header, self.question)
        cache.apply(source, at=1, usage=meter(1403, 1200, 200))
        self.assertEqual(
            cache.available(
                request(self.header, self.question, *[self.ping] * 20), at=2
            ),
            1400,
        )
        self.assertEqual(
            cache.available(
                request(self.header, self.question, *[self.ping] * 21), at=2
            ),
            1200,
        )
        multi = request(
            self.header,
            self.question,
            self.answer,
            message("developer", "late instructions"),
        )
        self.assertEqual(multi.boundaries, source.boundaries)
        result = {"type": "function_call_output", "call_id": "a", "output": "value"}
        tool_branch = request(self.header, self.question, result)
        cache.apply(tool_branch, at=2, usage=meter(1603, 1400, 200))
        # Appending a second consecutive tool result moves the eligible group ending.
        self.assertEqual(
            cache.available(
                request(self.header, self.question, result, {**result, "call_id": "b"}),
                at=3,
            ),
            1400,
        )
        self.assertEqual(
            cache.available(
                request(self.header, self.question, result, self.ping), at=3
            ),
            1600,
        )
        # A long unchanged text inside a rewritten user item cannot match its old ending.
        self.assertEqual(
            cache.available(
                request(self.header, message("user", "source and question plus more")),
                at=4,
            ),
            1200,
        )

    def test_scope_model_rendering_settings_and_ordered_tools(self):
        settings = {
            "reasoning": {"effort": "low"},
            "tools": [{"name": "a"}, {"name": "b"}],
            "prompt_cache_key": "k",
        }
        cache = VirtualCache()
        cache.apply(
            request(self.question, **settings), at=1, usage=meter(1203, 0, 1200)
        )
        for edits in (
            {"scope": "another"},
            {"model": "different"},
            {"reasoning": {"effort": "high"}},
            {"tools": list(reversed(settings["tools"]))},
            {"prompt_cache_key": "other"},
        ):
            self.assertEqual(
                cache.available(
                    request(self.question, **(settings | edits)), at=2, state="warm"
                ),
                0,
            )
        prewarm = request(
            self.question, **settings, prompt_cache_options={"prewarm": True}
        )
        projected = cache.project_ping(
            prewarm,
            at=200,
            input_tokens=1203,
            cacheable_input_tokens=1200,
            output_tokens=0,
        )
        self.assertIsNone(projected["observed_usage"])
        self.assertEqual(projected["modeled_usage"], meter(1203, 1200, 0, 0))
        self.assertEqual(
            cache.available(request(self.question, self.followup, **settings), at=450),
            1200,
        )

    def test_output_capped_generation_still_pays_for_its_input_and_output(self):
        cache = VirtualCache()
        seed = request(self.question)
        usage = response_usage({
            "status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"},
            "usage": {"input_tokens": 1503, "output_tokens": 512,
                      "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 1500}},
        })
        result = cache.apply(seed, at=0, usage=usage)
        self.assertEqual(result["modeled_usage"], meter(1503, 0, 1500, 512))
        self.assertEqual(cache.available(seed, at=1), 1500)

    def test_no_receipt_invention_or_mutation_on_invalid_usage(self):
        cache = VirtualCache()
        seed = request(self.question)
        for bad in (
            None,
            {"input_tokens": 2000},
            meter(1500, 1001, 500),
            meter(True, 0, 0),
            {**meter(1500, 0, 1400), "status": "incomplete"},
            {**meter(1500, 0, 1400), "mixed_cache_write_ttls": True},
        ):
            with self.assertRaises(ContractError):
                cache.apply(seed, at=8, usage=bad)
        self.assertEqual(cache.available(seed, at=0), 0)
        row = cache.apply(seed, at=2, usage=meter(720, 0, 0))
        self.assertEqual(row["cache_status"], "uncacheable")
        with self.assertRaises(ContractError):
            cache.apply(seed, at=1, usage=meter(1200, 0, 1200))
        with self.assertRaises(ContractError):
            cache.apply(request(self.answer), at=2, usage=meter(1200, 0, 1200))
        for extras in (
            {"previous_response_id": "resp"},
            {"prompt_cache_options": {"mode": "explicit"}},
        ):
            with self.assertRaises(ContractError):
                request(self.question, **extras)
        with self.assertRaises(ContractError):
            request(
                message(
                    "user",
                    [
                        {
                            "type": "input_text",
                            "text": "sample",
                            "prompt_cache_breakpoint": {"mode": "explicit"},
                        }
                    ],
                )
            )

    def test_archive_can_supply_proven_prefixes_and_explicit_token_estimates(self):
        cache = VirtualCache(1800)
        first = CacheRequest("pinned", ((1, "seed-digest"),))
        followup = CacheRequest("pinned", ((1, "seed-digest"), (3, "return-digest")))
        for req, at, size, expected in (
            (first, 0, 1024, 0),
            (followup, 300, 1100, 1024),
        ):
            modeled = cache.project(
                req,
                at=at,
                input_tokens=size + 3,
                cacheable_input_tokens=size,
                output_tokens=8,
            )
            self.assertIsNone(modeled["observed_usage"])
            self.assertEqual(modeled["usage_basis"], "supplied_estimates")
            self.assertEqual(
                modeled["modeled_usage"], meter(size + 3, expected, size - expected, 8)
            )
        with self.assertRaises(ContractError):
            CacheRequest("pinned", ((2, "late"), (1, "earlier")))


if __name__ == "__main__":
    unittest.main()
