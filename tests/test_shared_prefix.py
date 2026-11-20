import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

from compache.optimizer.cache_breaks import CacheScope
from compache.optimizer.contracts import ModelSpec
from compache.optimizer.shared_prefix import SharedPrefixCatalog
from compache.traces import ContractError


class SharedPrefixes(unittest.TestCase):
    def setUp(self):
        self.scope = CacheScope("tenant", "provider", "model", "A", "route")
        self.spec = ModelSpec(
            "model", cache_min_tokens=4, cache_block_tokens=4, cache_ttl_seconds=100
        )
        self.catalog = SharedPrefixCatalog()

    def prefix(self, thread="A", scope=None, **opts):
        scope = replace(self.scope, thread_id=thread) if scope is None else scope
        args = dict(
            system_prompt=b"12345678",
            default_tools=b"abcdefgh",
            spec=self.spec,
            tokenize=list,
        )
        args.update(opts)
        return self.catalog.measure(scope, **args)

    def test_warm_on_first_request_of_second_thread(self):
        first, second = self.prefix(), self.prefix("B", dynamic_tools="dynamic")
        self.assertEqual(
            (
                second.system_tokens,
                second.default_tool_tokens,
                second.dynamic_tool_tokens,
            ),
            (8, 8, 7),
        )
        self.assertEqual(second.static_tokens, 16)
        self.assertEqual(second.total_tokens, 23)
        self.assertEqual(second.wire_bytes, 23)
        self.assertEqual(self.catalog.lookup(second, at=0).status, "cold")
        self.assertTrue(
            self.catalog.record(first, at=10, request_tokens=100, cached_input_tokens=0)
        )
        hit = self.catalog.lookup(second, at=11, request_tokens=200)
        self.assertEqual(
            (hit.eligible_tokens, hit.system_tokens, hit.default_tool_tokens),
            (16, 8, 8),
        )
        self.assertTrue(hit.cross_thread)
        self.assertEqual(hit.age_seconds, 1)
        self.assertEqual(hit.source, "modeled_native_token_ids")
        self.assertEqual(hit.status, "full_static_hit")
        self.assertEqual(self.catalog.lookup(first, at=111).status, "expired")

    def test_tools_first_or_system_first_block_charge_preserves_only_matching_part(
        self,
    ):
        for order in [
            ("system_prompt", "default_tools"),
            ("default_tools", "system_prompt"),
        ]:
            with self.subTest(order=order):
                a = self.prefix(component_order=order)
                self.catalog.record(a, at=0)
                drift = self.prefix(
                    "B", default_tools="YYYYYYYY", component_order=order
                )
                hit = self.catalog.lookup(drift, at=1)
                self.assertEqual(
                    hit.eligible_tokens, 8 if order[0] == "system_prompt" else 0
                )
                self.assertEqual(hit.default_tool_tokens, 0)
                other = self.prefix(
                    "B", system_prompt="YYYYYYYY", component_order=order
                )
                hit = self.catalog.lookup(other, at=1)
                self.assertEqual(
                    hit.eligible_tokens, 8 if order[0] == "default_tools" else 0
                )
                self.assertEqual(hit.system_tokens, 0)
        partial = self.prefix("C", default_tools="abcdNEW!")
        hit = self.catalog.lookup(partial, at=2)
        self.assertEqual(
            (hit.eligible_tokens, hit.system_tokens, hit.default_tool_tokens),
            (12, 8, 4),
        )

    def test_parser_counts_wrappers_and_partial_tool_boundaries(self):
        p = self.prefix(default_tools=[{"a": 1}, {"b": 2}])
        self.assertEqual(p.default_tool_tokens, len(b'[{"a":1},{"b":2}]'))
        self.assertEqual(p.tool_breakdown, (len(b'[{"a":1}'), len(b',{"b":2}]')))
        self.assertEqual(p.unattributed_tool_tokens, 0)
        opaque = self.prefix(default_tools=b'[{"a":1},{"b":2}]')
        self.assertEqual(opaque.tool_breakdown, ())
        self.assertEqual(opaque.unattributed_tool_tokens, opaque.default_tool_tokens)
        framed = self.prefix(
            default_tools=b"tools-A|tools-B", tool_fragments=[b"tools-A|", b"tools-B"]
        )
        self.assertEqual(framed.tool_breakdown, (8, 7))
        with self.assertRaises(ContractError):
            self.prefix(default_tools=b"ab", tool_fragments=[b"a", b"x"])

    def test_native_cumulative_tokenization_handles_boundaries_and_dynamic_tail(self):
        # One token spans system/default tools; a second merged token crosses static/dynamic.
        def encode(data):
            return list(data.replace(b"abc", b"Z").replace(b"xy", b"W"))

        p = self.prefix(
            system_prompt=b"ab",
            default_tools=b"c12x",
            dynamic_tools=b"y!!",
            tokenize=encode,
        )
        self.assertEqual(encode(b"abc12xy!!"), list(b"Z12W!!"))
        self.assertEqual(
            (p.system_tokens, p.default_tool_tokens, p.dynamic_tool_tokens), (0, 3, 3)
        )
        self.assertEqual(p.total_tokens, len(encode(b"abc12xy!!")))
        q = self.prefix(
            system_prompt=b"ab",
            default_tools=b"c",
            tokenize=None,
            count_tokens=lambda data: len(encode(data)),
        )
        self.assertEqual(q.total_tokens, 1)
        self.assertGreaterEqual(q.default_tool_tokens, 0)
        self.assertEqual(q.system_tokens + q.default_tool_tokens, q.total_tokens)

    def test_minimum_block_capacity_expiry_and_success(self):
        p = self.prefix(spec=replace(self.spec, cache_min_tokens=8))
        self.assertFalse(self.catalog.record(p, at=0, request_tokens=7))
        self.assertFalse(self.catalog.record(p, at=0, successful=False))
        self.assertEqual(self.catalog.lookup(p, at=1).status, "cold")
        self.assertTrue(self.catalog.record(p, at=10, request_tokens=9))
        self.assertEqual(
            self.catalog.lookup(p, at=10, request_tokens=7).status, "ineligible_request"
        )
        self.assertEqual(
            self.catalog.lookup(p, at=20, request_tokens=16).eligible_tokens, 8
        )
        self.assertEqual(self.catalog.lookup(p, at=9).eligible_tokens, 0)
        self.assertEqual(self.catalog.lookup(p, at=110).status, "expired")
        changed = self.prefix(
            default_tools=b"",
            system_prompt=b"123456",
            spec=replace(self.spec, cache_min_tokens=8),
        )
        self.assertEqual(
            self.catalog.lookup(changed, at=20, request_tokens=100).status,
            "static_prefix_below_minimum",
        )

    def test_shared_catalog_scopes_and_private_opt_out(self):
        self.catalog.record(self.prefix(share=False), at=10)
        self.assertFalse(
            self.catalog.lookup(self.prefix("B", share=False), at=20).eligible_tokens
        )
        self.assertTrue(
            self.catalog.lookup(self.prefix(share=False), at=20).eligible_tokens
        )
        self.assertFalse(self.catalog.lookup(self.prefix(), at=20).eligible_tokens)
        self.catalog.record(self.prefix(), at=20)
        for f in ("provider", "installation", "cache_key"):
            p = self.prefix(scope=replace(self.scope, **{f: "other"}))
            self.assertEqual(self.catalog.lookup(p, at=21).eligible_tokens, 0)
        p = self.prefix(
            scope=replace(self.scope, model="other"),
            spec=replace(self.spec, model_id="other"),
        )
        self.assertEqual(self.catalog.lookup(p, at=21).eligible_tokens, 0)
        with self.assertRaises(ContractError):
            SharedPrefixCatalog().lookup(self.prefix(), at=1)

    def test_shorter_reported_input_does_not_erase_an_older_full_length_entry(self):
        first, second = self.prefix(), self.prefix()
        self.catalog.record(first, at=5, request_tokens=16)
        self.catalog.record(second, at=70, request_tokens=8)
        self.assertEqual(self.catalog.lookup(first, at=80).eligible_tokens, 16)
        self.assertEqual(self.catalog.lookup(first, at=110).eligible_tokens, 8)

    def test_host_cap_on_own_entry_does_not_override_another_publishers_entry(self):
        a, b = self.prefix(), self.prefix("B")
        self.catalog.record(b, at=0)
        self.catalog.record(a, at=5)
        for limit in (0, 7):
            found = self.catalog.lookup(a, at=30, same_thread_upper_bound=limit)
            self.assertTrue(found.cross_thread)
            self.assertEqual((found.system_tokens, found.default_tool_tokens), (8, 8))
        # B expires at 100. Only A remains; cap before block rounding/attribution.
        found = self.catalog.lookup(a, at=100, same_thread_upper_bound=11)
        self.assertFalse(found.cross_thread)
        self.assertEqual(
            (found.eligible_tokens, found.system_tokens, found.default_tool_tokens),
            (8, 8, 0),
        )
        self.assertEqual(
            self.catalog.lookup(a, at=104, same_thread_upper_bound=3).eligible_tokens, 0
        )
        self.assertEqual(
            self.catalog.lookup(b, at=104, same_thread_upper_bound=0).eligible_tokens,
            16,
        )
        with self.assertRaises(ContractError):
            self.catalog.lookup(a, at=30, same_thread_upper_bound=-1)

    def test_estimated_and_count_only_paths_use_actual_ordered_component_boundaries(
        self,
    ):
        for callback in (None, len):
            spec = replace(self.spec, bytes_per_token=1)
            a = self.prefix(tokenize=None, count_tokens=callback, spec=spec)
            self.catalog.record(a, at=0)
            b = self.prefix(
                "B",
                default_tools="new tools",
                tokenize=None,
                count_tokens=callback,
                spec=spec,
            )
            hit = self.catalog.lookup(b, at=10)
            self.assertEqual((hit.system_tokens, hit.default_tool_tokens), (8, 0))
            self.assertIn(
                hit.source, ("modeled_estimated_from_bytes", "modeled_host_token_count")
            )

    def test_concurrent_records_keep_freshest_timestamp_and_no_raw_messages(self):
        prefix = self.prefix(
            system_prompt="CONFIDENTIAL instructions",
            default_tools=[{"secret": "TOOL-SCHEMA"}],
        )
        self.assertNotIn("CONFIDENTIAL", repr(prefix))
        self.assertNotIn("TOOL-SCHEMA", repr(prefix))
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(
                pool.map(
                    lambda t: self.catalog.record(prefix, at=t), reversed(range(1, 16))
                )
            )
        self.assertEqual(self.catalog.lookup(prefix, at=18).age_seconds, 3)
        self.assertEqual(len(self.catalog._entries), 1)
        self.assertNotIn("CONFIDENTIAL", repr(self.catalog._entries))
