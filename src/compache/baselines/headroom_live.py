"""Per-session Headroom content routing for live, subscription-backed comparisons.

This integrates the library router, not Headroom's provider proxy.  Model calls
and recovery round trips are the caller's responsibility.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import inspect
import os
import re
from importlib.metadata import version
from pathlib import Path

from compache.traces import ContractError

_NATIVE_MARKERS = re.compile(r'(?:<<ccr:|\bhash=)([a-f0-9]{12,24})')
RECOVERY_SCHEMA = {
    "name": "headroom_retrieve",
    "description": "Retrieve exact original content using a hash from a Headroom compression marker.",
    "parameters": {"type": "object", "properties": {"hash": {"type": "string"}}, "required": ["hash"]},
}


class HeadroomLive:
    """Cache-stable source projections and session-scoped exact recovery.

    One instance belongs to one thread/replica. Only newly appended retrieved
    sources are routed. Input, output, absent text and old projections stay
    byte-identical. Every changed source gets an adapter-owned full-source
    handle; native CCR handles are copied into this session before use.
    """

    def __init__(self, thread_id, *, enable_kompress=True, router=None, store=None):
        if not isinstance(thread_id, str) or not thread_id:
            raise ContractError("Headroom thread identity required")
        self.thread_id = thread_id
        self._items = []
        self._fingerprints = []
        self._recovery = {}
        self._last_now = None
        self._test_router = router is not None
        self._kompress_calls = 0
        if router is None:
            if os.environ.get("HEADROOM_OFFLINE") != "1":
                raise ContractError("HEADROOM_OFFLINE=1 is required before importing Headroom")
            if version("headroom-ai") != "0.44.0":
                raise ContractError("Headroom live adapter requires headroom-ai==0.44.0")
            os.environ.setdefault("HEADROOM_CCR_BACKEND", "memory")
            from headroom.transforms.content_router import ContentRouter, ContentRouterConfig
            from headroom.cache.compression_store import get_compression_store
            router = ContentRouter(ContentRouterConfig(enable_kompress=enable_kompress,
                relevance_split=False, ccr_enabled=True, ccr_inject_marker=True))
            store = get_compression_store()
            backend = None
            if enable_kompress:
                compressor = router._get_kompress()
                if compressor is None:
                    raise ContractError("Kompress unavailable; install its local model and dependencies")
                try:
                    backend = compressor.preload(allow_download=False)
                    original_compress = compressor.compress
                    def counted_compress(*args, **kwargs):
                        self._kompress_calls += 1
                        return original_compress(*args, **kwargs)
                    compressor.compress = counted_compress
                except Exception as exc:
                    raise ContractError(f"Kompress local preload failed: {type(exc).__name__}: {exc}") from exc
            path = inspect.getsourcefile(ContentRouter)
            self.provenance = {"package": "headroom-ai", "version": version("headroom-ai"),
                "content_router_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                "kompress_enabled": enable_kompress, "kompress_backend": backend,
                "ccr_enabled": True, "relevance_split": False,
                "scope": "Direct ContentRouter per new retrieval; adapter-frozen projections and session-scoped recovery. Not Headroom proxy, client pipeline, cold hook, or native provider cache."}
        else:
            self.provenance = {"package": "test-double"}
        self._router = router
        self._store = store

    def prepare(self, request, now):
        if request.thread_id != self.thread_id:
            raise ContractError("Headroom session cannot cross threads")
        if not isinstance(now, (int, float)) or isinstance(now, bool) or not __import__('math').isfinite(now):
            raise ContractError("Headroom requires finite logical time")
        if self._last_now is not None and now < self._last_now:
            raise ContractError("Headroom logical time moved backwards")
        fingerprints = [p.fingerprint for p in request.parts]
        if fingerprints[:len(self._fingerprints)] != self._fingerprints:
            raise ContractError("Headroom canonical prefix changed or was removed")
        question = next((p.text for p in reversed(request.parts) if p.kind == "input" and p.text), "")
        new_items, new_fingerprints, pending_handles, events = [], [], {}, []
        for p in request.parts[len(self._items):]:
            projected = p
            status = "not_retrieval" if p.kind != "retrieval" else "unchanged"
            strategy = None
            chain = []
            native_before = native_after = None
            kompress_before = self._kompress_calls
            if p.kind == "retrieval" and p.text is not None and len(p.text) >= 500:
                try:
                    result = self._router.compress(p.text, context=question, question=question)
                    candidate = result.compressed
                    strategy = getattr(result.strategy_used, "value", str(result.strategy_used))
                    chain = list(getattr(result, "strategy_chain", []))
                    native_before = getattr(result, "total_original_tokens", None)
                    native_after = getattr(result, "total_compressed_tokens", None)
                    if not isinstance(candidate, str) or result.unrecoverable_sections:
                        status = "unsafe_router_output"
                    elif candidate != p.text:
                        native = {}
                        for key in set(_NATIVE_MARKERS.findall(candidate)):
                            entry = self._store.retrieve(key) if self._store is not None else None
                            if entry is None or not isinstance(entry.original_content, str):
                                raise ContractError("native CCR marker unresolved")
                            native[key] = entry.original_content
                        # Separate per-source handle makes recovery exact even if a
                        # Headroom transform is a lossless fold without a CCR marker.
                        key = hashlib.sha256((self.thread_id + "\0" + p.part_id + "\0" + p.fingerprint).encode()).hexdigest()[:24]
                        marker = f"\n[Original source available via headroom_retrieve hash={key}]"
                        if len((candidate + marker).encode()) < len(p.text.encode()):
                            handles = {**native, key: p.text}
                            for h, original in handles.items():
                                old = pending_handles.get(h, self._recovery.get(h))
                                if old is not None and old != original:
                                    raise ContractError("CCR hash collision")
                            pending_handles.update(handles)
                            projected = replace(p, text=candidate + marker)
                            status = "compressed"
                        else:
                            status = "inflation_guard"
                except Exception as exc:
                    status = "error_preserved_original"
                    events.append({"source_id": p.part_id, "error": f"{type(exc).__name__}: {exc}"})
            new_items.append(projected)
            new_fingerprints.append(p.fingerprint)
            events.append({"source_id": p.part_id, "status": status, "strategy": strategy,
                           "strategy_chain": chain, "kompress_invocations": self._kompress_calls - kompress_before,
                           "router_tokens_before": native_before, "router_tokens_after": native_after,
                           "original_bytes": len((p.text or "").encode()),
                           "projected_bytes": len((projected.text or "").encode())})
        self._items.extend(new_items)
        self._fingerprints.extend(new_fingerprints)
        self._recovery.update(pending_handles)
        self._last_now = now
        return tuple(self._items), {"new_sources": events, "frozen_sources": len(self._items) - len(new_items),
                                    "recovery_handles": len(self._recovery),
                                    "errors": sum(e.get("status") in ("error_preserved_original", "unsafe_router_output") for e in events)}

    def recover(self, key):
        if not isinstance(key, str) or key not in self._recovery:
            raise ContractError("Headroom recovery hash not found in this session")
        return self._recovery[key]
