import re
import unittest
from types import SimpleNamespace

from compache.baselines.headroom_live import HeadroomLive
from compache.optimizer import Request, Source
from compache.traces import ContractError


class Router:
    def __init__(self):
        self.calls = []
        self.fail = False
    def compress(self, text, **kwargs):
        self.calls.append((text, kwargs))
        if self.fail:
            raise RuntimeError("compressor failure")
        return SimpleNamespace(compressed="small <<ccr:abcdef123456>>", strategy_used=SimpleNamespace(value="smart_crusher"), unrecoverable_sections=False)

class Store:
    def retrieve(self, key):
        return SimpleNamespace(original_content="native fragment") if key == "abcdef123456" else None

def request(thread, parts, n=1):
    return Request(thread, f"run-{n}", n, "model", tuple(parts))

def source(part, text, kind="retrieval", n=1):
    return Source(part, kind, n, text, "doc", "v1", "chunk")

class HeadroomLiveTests(unittest.TestCase):
    def test_stable_projection_and_recovery_is_session_scoped(self):
        router = Router()
        adapter = HeadroomLive("thread", router=router, store=Store())
        original = source("p", "evidence " * 150)
        first, meta = adapter.prepare(request("thread", [original]), 1)
        self.assertEqual(meta["new_sources"][0]["status"], "compressed")
        self.assertEqual(first[0].document_id, "doc")
        key = re.search(r'hash=([a-f0-9]+)', first[0].text).group(1)
        self.assertEqual(adapter.recover(key), original.text)
        self.assertEqual(adapter.recover("abcdef123456"), "native fragment")
        updated, meta = adapter.prepare(request("thread", [original, source("q", "question", "input", 2)], 2), 2)
        self.assertEqual(updated[0], first[0])
        self.assertEqual(len(router.calls), 1)
        self.assertEqual(meta["frozen_sources"], 1)
        other = HeadroomLive("other", router=Router(), store=Store())
        with self.assertRaises(ContractError): other.recover(key)
        with self.assertRaises(ContractError): adapter.prepare(request("other", [original]), 3)
        with self.assertRaises(ContractError): adapter.prepare(request("thread", [source("p", "replaced")], 2), 3)

    def test_failures_and_unresolved_native_markers_preserve_original(self):
        router = Router()
        adapter = HeadroomLive("thread", router=router, store=None)
        original = source("p", "x" * 1000)
        result, meta = adapter.prepare(request("thread", [original]), 1)
        self.assertEqual(result, (original,))
        self.assertEqual(meta["new_sources"][-1]["status"], "error_preserved_original")
        self.assertEqual(meta["recovery_handles"], 0)
        router.fail = True
        second = source("p2", "x" * 1000, n=2)
        result, meta = adapter.prepare(request("thread", [original, second], 2), 2)
        self.assertEqual(result[-1], second)
        self.assertEqual(meta["new_sources"][-1]["status"], "error_preserved_original")

    def test_input_output_and_missing_text_are_preserved(self):
        router = Router()
        adapter = HeadroomLive("thread", router=router, store=Store())
        sources = [source("a", "q" * 1000, "input"), source("b", None), source("c", "a" * 1000, "output")]
        result, _ = adapter.prepare(request("thread", sources, 2), 1)
        self.assertEqual(result, tuple(sources))
        self.assertEqual(router.calls, [])
