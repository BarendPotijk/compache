"""Prepared Responses API transport. No calls occur until run() is invoked."""

from __future__ import annotations
import hashlib
import json
import os
import time
import threading
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

MODEL = "gpt-6-luna"
NORMAL_TASK = "Answer the latest substantive user question."


def make_payload(view, cache_key, model=MODEL):
    if model != MODEL:
        raise ValueError("unknown model and tariff")
    if not cache_key:
        raise ValueError("a stable cache key is required")
    messages = []
    if "\nCONVERSATION\n" in view:
        header, rest = view.split("\nCONVERSATION\n", 1)
        body, suffix = rest.rsplit("\nTASK\n", 1)
        messages.append(
            {
                "role": "developer",
                "content": header
                + "\nAnswer the latest substantive user question unless the final user message requests another task.",
            }
        )
        for line in body.splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            role = obj.get("role")
            if role not in ("user", "assistant", "tool"):
                raise ValueError("unknown conversation role")
            messages.append(
                {
                    "role": "assistant" if role == "assistant" else "user",
                    "content": line,
                }
            )
        if suffix.strip() != NORMAL_TASK:
            messages.append({"role": "user", "content": "TASK\n" + suffix})
    else:
        messages = [{"role": "user", "content": view}]
    return {
        "model": model,
        "input": messages,
        "reasoning": {"effort": "low"},
        "service_tier": "default",
        "store": False,
        "truncation": "disabled",
        "max_output_tokens": 32000,
        "prompt_cache_key": cache_key,
        "prompt_cache_options": {"mode": "implicit", "ttl": "30m"},
    }


def usage_cost(usage):
    if not isinstance(usage, dict) or not isinstance(
        usage.get("input_tokens_details"), dict
    ):
        return None, None
    details = usage["input_tokens_details"]
    values = (
        usage.get("input_tokens"),
        details.get("cached_tokens"),
        details.get("cache_write_tokens"),
        usage.get("output_tokens"),
    )
    if any(type(x) is not int or x < 0 for x in values):
        return None, None
    inp, cached, write, out = values
    if cached + write > inp or inp == 0:
        return None, None
    long = inp > 272000
    m = Decimal(2) if long else Decimal(1)
    om = Decimal("1.5") if long else Decimal(1)
    cost = (
        (inp - cached - write) * Decimal(".1") * m
        + cached * Decimal(".01") * m
        + write * Decimal(".125") * m
        + out * Decimal(".5") * om
    ) / Decimal(1000000)
    return {
        "input_tokens": inp,
        "cached_input_tokens": cached,
        "cache_write_input_tokens": write,
        "output_tokens": out,
    }, float(cost)


def interpret(response):
    usage, cost = usage_cost(response.get("usage"))
    chunks = [
        x.get("text", "")
        for item in response.get("output", [])
        if item.get("type") == "message"
        for x in item.get("content", [])
        if x.get("type") == "output_text"
    ]
    answer = "".join(chunks)
    errors = []
    if response.get("status") != "completed":
        errors.append("response not completed")
    if response.get("service_tier") not in (None, "default"):
        errors.append("unexpected service tier")
    if usage is None:
        errors.append("missing or invalid usage including cache write")
    if not answer:
        errors.append("missing output text")
    if any(
        i.get("type") not in ("message", "reasoning")
        for i in response.get("output", [])
    ):
        errors.append("unexpected output item")
    return {
        "ok": not errors,
        "errors": errors,
        "answer": answer or None,
        "usage": usage,
        "cost_usd": cost if not errors else None,
    }


class Budget:
    """Atomic reservation; unresolved calls retain their full worst-case charge."""

    def __init__(self, limit_usd):
        self.limit = Decimal(str(limit_usd))
        self.spent = Decimal(0)
        self.reserved = Decimal(0)
        self.lock = threading.Lock()
        if self.limit <= 0:
            raise ValueError("budget must be positive")

    def reserve(self):
        amount = Decimal(".26")
        with self.lock:
            if self.spent + self.reserved + amount > self.limit:
                raise RuntimeError("budget exhausted")
            self.reserved += amount
        return amount

    def reconcile(self, reservation, actual):
        if actual is None:
            return
        value = Decimal(str(actual))
        with self.lock:
            self.reserved -= reservation
            self.spent += value


class OpenAISession:
    def __init__(self, cache_key, api_key=None, model=MODEL, budget=None):
        self.api_key = (
            api_key if api_key is not None else os.environ.get("OPENAI_API_KEY")
        )
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY or explicit api_key required")
        if model != MODEL:
            raise ValueError("unknown model and tariff")
        self.model = model
        self.cache_key = cache_key
        self.budget = budget

    def run(self, view, output_dir, timeout=900):
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=False)
        payload = make_payload(view, self.cache_key, self.model)
        (path / "prompt.txt").write_text(view, encoding="utf8")
        (path / "request.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf8"
        )
        started = datetime.now(timezone.utc).isoformat()
        clock = time.monotonic()
        result = {
            "ok": False,
            "errors": [],
            "answer": None,
            "usage": None,
            "cost_usd": None,
        }
        reservation = None
        try:
            if self.budget is not None:
                reservation = self.budget.reserve()
            req = Request(
                "https://api.openai.com/v1/responses",
                data=json.dumps(payload).encode(),
                headers={
                    "Authorization": "Bearer " + self.api_key,
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            with urlopen(req, timeout=timeout) as resp:
                response = json.load(resp)
            # Response may echo sensitive user input; fixture data must be synthetic.
            (path / "response.json").write_text(
                json.dumps(response, ensure_ascii=False, indent=2) + "\n",
                encoding="utf8",
            )
            result = interpret(response)
            result["response_id"] = response.get("id")
            if self.budget is not None:
                self.budget.reconcile(reservation, result["cost_usd"])
        except (
            HTTPError,
            URLError,
            TimeoutError,
            OSError,
            ValueError,
            RuntimeError,
        ) as exc:
            # Do not persist arbitrary error bodies, headers, or exception strings that may contain credentials.
            result["errors"].append(
                "request failed: "
                + type(exc).__name__
                + (" status " + str(exc.code) if isinstance(exc, HTTPError) else "")
            )
        result.update(
            {
                "model": self.model,
                "reasoning_effort": "low",
                "started_at_utc": started,
                "ended_at_utc": datetime.now(timezone.utc).isoformat(),
                "wall_seconds": time.monotonic() - clock,
                "view_sha256": hashlib.sha256(view.encode()).hexdigest(),
                "model_context_window": 1050000,
                "native_runtime_overhead": None,
                "usage_scope": "Responses API reported usage; no Codex runtime overhead",
                "price_scope": "API tariff calculation, not subscription billing; excludes unobserved retries",
            }
        )
        (path / "result.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf8"
        )
        return result

    def close(self):
        pass
