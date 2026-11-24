"""compache as a Hermes Agent context engine.

Hermes lets a context engine choose what a request holds (`select_context`), and nothing
else has to change: the conversation Hermes keeps is untouched. compache's engine replaces
archived parts by their notice, gives the model a `reopen` tool for them, and leaves
everything else to Hermes. When a conversation outgrows the model, Hermes' own compressor
does what it always did.

    pip install compache

    # ~/.hermes/plugins/compache/__init__.py  (a ContextEngine for Hermes)
    from compache.hermes import register

    # ~/.hermes/config.yaml
    context:
      engine: compache

The policy is read from COMPACHE_POLICY, a record without any text is kept in
COMPACHE_RECORD. Both are optional. Set COMPACHE_RECORD first, with no policy: compache
then only records, and `compache fit` on the folder gives the policy.

Notes are not asked here, and no keep-warm pings are sent: Hermes sends the requests, so
compache has no request of its own to send. `compache proxy` does both.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from . import wire
from .live import Live, _REOPEN_TOOL

__all__ = ["make", "register"]

logger = logging.getLogger(__name__)

_PATH = "/v1/chat/completions"
_REOPEN = _REOPEN_TOOL["function"]["name"]


def _from_environment() -> Live:
    return Live(
        os.environ.get("COMPACHE_POLICY") or None,
        record=os.environ.get("COMPACHE_RECORD") or None,
    )


def _same_lifetime(live: Live) -> None:
    """Say so where Hermes' cache marks ask for another lifetime than the policy counts on.

    Hermes marks the requests, so what it asks for is its own setting.
    """
    spec = live.policy.spec if live.policy is not None else None
    if spec is None or "claude" not in spec.model_id:
        return
    wanted = wire.TTLS.get(spec.cache_ttl_seconds)
    if wanted is None:
        return
    try:
        from hermes_cli.config import load_config_readonly

        asked = (load_config_readonly().get("prompt_caching") or {}).get(
            "cache_ttl", "auto"
        )
    except Exception:
        return
    # Its default, "auto", moves to an hour after a conversation's first pause.
    if str(asked) != wanted:
        logger.warning(
            "compache: the policy counts on %s cache entries and Hermes asks for %s. "
            "Set prompt_caching.cache_ttl to %s in config.yaml",
            wanted,
            asked,
            wanted,
        )


def make(live: Live | None = None):
    """The engine, for Hermes' own `ctx.register_context_engine`.

    Hermes is imported only here, so that the package installs without it.
    """
    from agent.context_engine import ContextEngine

    # Every engine Hermes makes, one per agent, shares one policy and one record.
    shared: dict[str, Any] = {"live": live, "checked": False}

    class Compache(ContextEngine):
        """Chooses what a request holds by a fitted policy. `compress` stays Hermes' own."""

        # What the host reads off every engine.
        last_prompt_tokens = 0
        last_completion_tokens = 0
        last_total_tokens = 0
        threshold_tokens = 0
        context_length = 0
        compression_count = 0

        def __init__(self):
            super().__init__()
            if shared["live"] is None:
                shared["live"] = _from_environment()
            if not shared["checked"]:
                shared["checked"] = True
                _same_lifetime(shared["live"])
            self.live = shared["live"]
            self.session = "session"
            self.model = ""
            self._route: dict[str, Any] = {}

        @property
        def name(self) -> str:
            return "compache"

        def clone_for_agent(self):
            """One engine per agent. The policy, the record and what was seen are shared."""
            return type(self)()

        # -- what the request holds ------------------------------------------------

        def on_session_start(self, session_id: str, **kwargs) -> None:
            self.session = str(session_id or "session")

        def on_session_reset(self) -> None:
            super().on_session_reset()

        def select_context(
            self,
            request_messages: list,
            *,
            conversation_messages=None,
            incoming_message=None,
            budget_tokens: int = 0,
        ):
            """The messages with archived parts as their notice. None when nothing is archived."""
            body = {"model": self.model, "messages": request_messages}
            shown = self.live.view(_PATH, body, thread=self.session)
            messages = shown.get("messages") if isinstance(shown, dict) else None
            return (
                messages
                if messages is not None and messages is not request_messages
                else None
            )

        # -- the model fetching an archived part back -------------------------------

        def get_tool_schemas(self) -> list:
            """`reopen`, which reads an archived part. It is shown only when there can be one."""
            if self.live.policy is None:
                return []
            function = _REOPEN_TOOL["function"]
            return [
                {
                    "name": function["name"],
                    "description": function["description"],
                    "parameters": function["parameters"],
                }
            ]

        def handle_tool_call(self, name: str, args: dict, **kwargs) -> str:
            if name != _REOPEN:
                return super().handle_tool_call(name, args, **kwargs)
            args = args if isinstance(args, dict) else {}
            answer = self.live.reopen(
                self.session, str(args.get("reference", "")), args.get("offset")
            )
            return json.dumps(answer, ensure_ascii=False)

        # -- what came back --------------------------------------------------------

        def update_from_response(self, usage: dict) -> None:
            self.last_prompt_tokens = int(usage.get("prompt_tokens") or 0)
            self.last_completion_tokens = int(usage.get("completion_tokens") or 0)
            self.last_total_tokens = int(usage.get("total_tokens") or 0)
            self.live.heard(self.session, _usage(usage), model=self.model or None)

        def on_turn_complete(
            self, messages: list, usage: dict | None = None, **kwargs
        ) -> None:
            self.live.conclude(
                _PATH, {"model": self.model, "messages": messages}, thread=self.session
            )

        # -- when a conversation outgrows the model, Hermes' own compressor -------------------

        def update_model(
            self,
            model: str,
            context_length: int,
            base_url: str = "",
            api_key: str = "",
            provider: str = "",
            api_mode: str = "",
        ) -> None:
            self.model = str(model or "")
            self.context_length = int(context_length or 0)
            self._route = {
                "model": self.model,
                "base_url": base_url,
                "api_key": api_key,
                "provider": provider,
                "api_mode": api_mode,
                "config_context_length": self.context_length or None,
                "quiet_mode": True,
            }
            try:
                super().update_model(
                    model, context_length, base_url, api_key, provider, api_mode
                )
            except Exception:
                # A Hermes of another version: its threshold is not essential here.
                self.threshold_tokens = int(
                    self.context_length * self.threshold_percent
                )

        def should_compress(self, prompt_tokens: int | None = None) -> bool:
            """Only when a conversation is about to outgrow the model."""
            tokens = self.last_prompt_tokens if prompt_tokens is None else prompt_tokens
            return self.threshold_tokens > 0 and tokens >= self.threshold_tokens

        def compress(
            self,
            messages: list,
            current_tokens=None,
            focus_topic=None,
            force: bool = False,
            memory_context: str = "",
        ) -> list:
            fallback = self._default()
            if fallback is None:
                return messages
            self.compression_count += 1
            return fallback.compress(
                messages,
                current_tokens,
                focus_topic=focus_topic,
                force=force,
                memory_context=memory_context,
            )

        def _default(self):
            """Hermes' built-in compressor, made for the model that is in use."""
            if not self._route:
                return None
            try:
                from agent.context_compressor import ContextCompressor

                return ContextCompressor(**self._route)
            except Exception:
                logger.warning(
                    "compache: Hermes' own compressor could not be made, nothing was compressed",
                    exc_info=True,
                )
                return None

    return Compache()


def _usage(usage: dict) -> dict:
    """Hermes' token counts in the OpenAI form: its prompt total counts what was read or written.

    What Hermes does not report is left out: a missing count is not a zero.
    """
    details = {
        ours: int(usage[theirs] or 0)
        for ours, theirs in (
            ("cached_tokens", "cache_read_tokens"),
            ("cache_write_tokens", "cache_write_tokens"),
        )
        if usage.get(theirs) is not None
    }
    return {
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        **({"prompt_tokens_details": details} if details else {}),
    }


def register(ctx) -> None:
    """Hermes' plugin entry: hands it the engine (`register_context_engine`)."""
    ctx.register_context_engine(make())
