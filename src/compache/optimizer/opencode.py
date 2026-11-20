"""Projection of pinned OpenCode RAG compaction; see docs/opencode-baseline.md."""

from dataclasses import dataclass
import json
from compache.optimizer.contracts import number
from compache.optimizer.gate import Source
from compache.traces import ContractError

TEMPLATE = 'Output exactly the Markdown structure shown inside <template> and keep the section order unchanged. Do not include the <template> tags in your response.\n<template>\n## Objective\n- [one or two brief sentences describing what the user is trying to accomplish]\n\n## Important Details\n- [constraints/preferences, decisions and why, important facts/assumptions, exact context needed to continue, or "(none)"]\n\n## Work State\n### Completed\n- [finished work, verified facts, or changes made; otherwise "(none)"]\n\n### Active\n- [current work, partial changes, or investigation state; otherwise "(none)"]\n\n### Blocked\n- [blockers, failing commands, or unknowns; otherwise "(none)"]\n\n## Next Move\n1. [immediate concrete action, or "(none)"]\n2. [next action if known, or "(none)"]\n\n## Relevant Files\n- [file or directory path: why it matters, or "(none)"]\n</template>\n\nRules:\n- Keep every section, even when empty.\n- Use terse bullets, not prose paragraphs.\n- Preserve exact file paths, symbols, commands, error strings, URLs, and identifiers when known.\n- Do not mention the summary process or that context was compacted.'
UPDATE = 'The <prior-summary> summarizes everything that happened before the <conversation>. Construct a new summary that combines both. The <prior-summary> is discarded after this: anything you do not carry into the new summary is lost.\n\nWhen combining:\n- Carry forward objectives, constraints, user directives, decisions, and parallel workstreams from the <prior-summary> even when the <conversation> does not mention them. Drop only what is finished and no longer needed.\n- The <conversation> is more recent than the <prior-summary>. Where they conflict, the conversation wins: state the corrected fact and drop the old claim.\n- Add new progress, decisions, constraints, and context from the conversation.\n- Move completed work from "Active" to "Completed".\n- If a blocker has been resolved, update the summary to reflect that while keeping any details still needed to continue the work.\n- Update "Objective" and "Next Move" to reflect the current work state.'
OPENCODE_RELEASE = "v1.19.5"
OPENCODE_COMMIT = "810c0eba1a7348036f8bdf61e6632d0c36c6876c"
OPENCODE_LICENSE = "MIT"
CONTINUE = "Continue if you have next steps, or stop and ask for clarification if you are unsure how to proceed."


def build_prompt(conversation, previous_summary=None):
    intro = f"Here is the conversation so far:\n\n<conversation>\n{conversation}\n</conversation>"
    if not previous_summary:
        return "\n\n".join(
            (
                intro,
                "Create a new anchored summary from the conversation history in the <conversation> tags above so another coding agent can continue the work.",
                TEMPLATE,
            )
        )
    prior = f"Here is the summary of the conversation before the <conversation> above:\n\n<prior-summary>\n{previous_summary}\n</prior-summary>"
    return "\n\n".join((intro, prior, UPDATE, TEMPLATE))


@dataclass(frozen=True)
class Usage:
    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_write: int = 0
    total: int = 0

    def __post_init__(self):
        for name in ("input", "output", "cache_read", "cache_write", "total"):
            number(getattr(self, name), name, integer=True)

    @property
    def count(self):
        return (
            self.total or self.input + self.output + self.cache_read + self.cache_write
        )


@dataclass(frozen=True)
class Settings:
    context: int
    output_limit: int = 32000
    input_limit: int = 0
    output_token_max: int = 32000
    reserved: int | None = None
    preserve_recent_tokens: int | None = None
    tail_turns: int | None = None
    auto: bool = True

    def __post_init__(self):
        for name in ("context", "output_limit", "input_limit"):
            number(getattr(self, name), name, integer=True)
        number(self.output_token_max, "output_token_max", 1, integer=True)
        for name in ("reserved", "preserve_recent_tokens", "tail_turns"):
            if getattr(self, name) is not None:
                number(getattr(self, name), name, integer=True)
        if type(self.auto) is not bool:
            raise ContractError("auto must be boolean")

    @property
    def max_output(self):
        return min(self.output_limit, self.output_token_max) or self.output_token_max

    @property
    def usable(self):
        if not self.context:
            return 0
        reserved = (
            self.reserved if self.reserved is not None else min(20000, self.max_output)
        )
        return (
            max(0, self.input_limit - reserved)
            if self.input_limit
            else max(0, self.context - self.max_output)
        )

    @property
    def tail_budget(self):
        return (
            self.preserve_recent_tokens
            if self.preserve_recent_tokens is not None
            else min(15000, max(2000, self.usable // 4))
        )

    def overflow(self, usage):
        return bool(self.auto and self.context and usage.count >= self.usable)


@dataclass(frozen=True)
class Message:
    id: str
    role: str
    text: str
    source: Source | None = None
    synthetic: bool = False


def to_messages(request):
    if any(p.text is None for p in request.parts):
        raise ContractError("OpenCode projection requires available source text")
    return tuple(
        Message(
            p.part_id,
            {"input": "user", "output": "assistant", "retrieval": "tool"}[p.kind],
            p.text or "",
            p,
        )
        for p in request.parts
    )


def serialize(m):
    if m.role == "user":
        return f"[User]: {m.text}" if m.text else ""
    if m.role == "assistant":
        return f"[Assistant]: {m.text}" if m.text else ""
    if m.role == "tool":
        data = m.text.encode("utf-16-le", "surrogatepass")
        text = (
            data[:4000].decode("utf-16-le", "surrogatepass") + "\n[truncated]"
            if len(data) > 4000
            else m.text
        )
        return f"[Assistant tool call]: retrieval({{}})\n[Tool result]: {text}"
    raise ContractError("unsupported role")


@dataclass(frozen=True)
class Prepared:
    prompt: str
    head: tuple
    tail: tuple
    previous_summary: str | None
    tail_start_id: str | None


class OpenCodeBaseline:
    def __init__(self, settings, spec, token_counter=None):
        self.settings, self.spec = settings, spec
        self.token_counter = token_counter or self._estimate
        self.summary = None
        self.retained = None
        self.seen_ids = set()
        self.summary_usage = []

    def _estimate(self, messages):
        return self.spec.text_tokens(
            json.dumps(
                [{"role": m.role, "content": m.text} for m in messages],
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )

    def project(self, request):
        incoming = to_messages(request)
        if self.retained is None:
            return incoming
        return self.retained + tuple(m for m in incoming if m.id not in self.seen_ids)

    def prepare(self, request, usage):
        if not self.settings.overflow(usage):
            return None
        messages = tuple(
            m
            for m in self.project(request)
            if m.id not in ("__opencode_summary__", "__opencode_marker__")
        )
        all_starts = [i for i, m in enumerate(messages) if m.role == "user"]
        starts = all_starts
        limit = self.settings.tail_turns
        if limit is not None:
            starts = starts[-limit:] if limit > 0 else []
        keep = None
        total = 0
        for start in reversed(starts):
            end = next((i for i in all_starts if i > start), len(messages))
            size = self.token_counter(messages[start:end])
            if total + size <= self.settings.tail_budget:
                total += size
                keep = start
                continue
            remaining = self.settings.tail_budget - total
            if remaining > 0:
                for split in range(start + 1, end):
                    if self.token_counter(messages[split:end]) <= remaining:
                        keep = split
                        break
            break
        head, tail = (
            (messages, ())
            if keep is None or keep == 0
            else (messages[:keep], messages[keep:])
        )
        conversation = "\n\n".join(filter(None, map(serialize, head)))
        return Prepared(
            build_prompt(conversation, self.summary),
            head,
            tail,
            self.summary,
            tail[0].id if tail else None,
        )

    def complete(self, request, prepared, summary, summary_usage):
        if not summary.strip():
            raise ContractError("completed compaction requires actual summary text")
        self.summary = summary.strip()
        self.summary_usage.append(summary_usage)
        self.seen_ids.update(m.id for m in to_messages(request))
        self.retained = (
            (
                Message(
                    "__opencode_marker__",
                    "user",
                    "What did we do so far?",
                    synthetic=True,
                ),
                Message(
                    "__opencode_summary__", "assistant", self.summary, synthetic=True
                ),
            )
            + prepared.tail
            + (
                Message(
                    f"__opencode_continue_{len(self.summary_usage)}__",
                    "user",
                    CONTINUE,
                    synthetic=True,
                ),
            )
        )
        return self.retained
