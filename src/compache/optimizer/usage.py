"""Read token counters already returned by an OpenAI Responses API model call."""

from collections.abc import Mapping


def _field(value, name):
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _tokens(value, name):
    n = _field(value, name)
    return n if type(n) is int and n >= 0 else None


def response_usage(response):
    """Return content-free counters from a response, SDK object, or terminal event.

    input_tokens includes cache reads. output_tokens includes reasoning. Optional
    breakdowns remain None when absent. Only a completed response for the exact
    submitted view is an occupancy anchor. The caller owns event deduplication.
    """
    if _field(response, "type") in (
        "response.completed",
        "response.incomplete",
        "response.failed",
    ):
        response = _field(response, "response")
    usage = _field(response, "usage")
    incoming = _field(usage, "input_tokens_details")
    outgoing = _field(usage, "output_tokens_details")
    write_details = _field(incoming, "cache_write_tokens_details")
    incomplete_details = _field(response, "incomplete_details")
    return {
        "response_id": _field(response, "id"),
        "model": _field(response, "model"),
        "status": _field(response, "status"),
        "incomplete_reason": _field(incomplete_details, "reason"),
        "input_tokens": _tokens(usage, "input_tokens"),
        "cached_input_tokens": _tokens(incoming, "cached_tokens"),
        "cache_write_input_tokens": _tokens(incoming, "cache_write_tokens"),
        "mixed_cache_write_ttls": bool(
            _tokens(write_details, "ttl_30m_tokens")
            and _tokens(write_details, "ttl_12h_tokens")
        ),
        "output_tokens": _tokens(usage, "output_tokens"),
        "reasoning_output_tokens": _tokens(outgoing, "reasoning_tokens"),
    }
