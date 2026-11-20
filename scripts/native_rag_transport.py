"""One fresh Codex CLI session per supplied view; no session resumption.

The CLI's usage is total respondent turn usage, including runtime overhead; it
cannot isolate supplied-view tokens or prove billing. Persist the JSONL events
and stderr for auditing. Other processes and platform overhead are excluded.
"""

from __future__ import annotations

import argparse
import hashlib
from datetime import datetime, timezone
import re
import json
import subprocess
import time
from pathlib import Path

REQUIRED_USAGE = ("input_tokens", "cached_input_tokens", "output_tokens")


def interpret(stdout: str, returncode: int) -> dict:
    events = []
    errors = []
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
            if not isinstance(event, dict):
                raise ValueError("event is not an object")
            events.append(event)
        except (ValueError, json.JSONDecodeError):
            errors.append("invalid JSONL event")
    completed = [e for e in events if e.get("type") == "turn.completed"]
    if returncode != 0 or any(
        e.get("type") in ("error", "turn.failed") for e in events
    ):
        errors.append("process or turn failed")
    if len(completed) != 1:
        errors.append("expected exactly one completed turn")
    messages = [
        e.get("item", {}).get("text")
        for e in events
        if e.get("type") == "item.completed"
        and e.get("item", {}).get("type") == "agent_message"
    ]
    tools = [
        e
        for e in events
        if e.get("type", "").startswith("item.")
        and e.get("item", {}).get("type") not in ("agent_message", "reasoning")
    ]
    if any(
        "compact" in str(e.get("type", "")).lower()
        or "compact" in str(e.get("item", {}).get("type", "")).lower()
        for e in events
    ):
        errors.append("compaction event detected")
    if tools:
        errors.append("non-message item detected; tool-free execution not established")
    if not messages or not isinstance(messages[-1], str) or not messages[-1]:
        errors.append("missing answer")
    usage = completed[0].get("usage") if len(completed) == 1 else None
    if not isinstance(usage, dict) or any(
        type(usage.get(k)) is not int or usage[k] < 0 for k in REQUIRED_USAGE
    ):
        errors.append("missing or invalid usage counters")
    elif (
        usage["input_tokens"] <= 0
        or usage["cached_input_tokens"] > usage["input_tokens"]
    ):
        errors.append("implausible usage counters")
    return {
        "ok": not errors,
        "errors": errors,
        "answer": messages[-1] if messages else None,
        "usage": usage,
        "event_count": len(events),
    }


def redact_stderr(text: str) -> str:
    text = re.sub(
        r"(?i)(authorization\s*[:=]\s*(?:bearer\s+)?)[^\s,;]+", r"\1[REDACTED]", text
    )
    text = re.sub(
        r"(?i)((?:api[_-]?key|access[_-]?token|refresh[_-]?token|password|secret)\s*[:=]\s*[\"']?)[^\s,;\"']+",
        r"\1[REDACTED]",
        text,
    )
    return re.sub(r"\bsk-[A-Za-z0-9_-]{12,}\b", "[REDACTED]", text)


def run(
    view: str,
    output_dir: Path,
    *,
    working_dir: Path,
    codex: str = "codex",
    model: str = "gpt-6-luna",
    timeout: float = 300,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=False)
    working_dir = working_dir.resolve(strict=True)
    if not working_dir.is_dir():
        raise ValueError("working_dir must be an existing directory")
    (output_dir / "prompt.txt").write_text(view, encoding="utf-8")
    # A fresh session prevents accumulated hidden conversation history.
    cmd = [
        codex,
        "exec",
        "--json",
        "--ephemeral",
        "--ignore-user-config",
        "--skip-git-repo-check",
        "--sandbox",
        "read-only",
        "-c",
        "model_context_window=872000",
        "-c",
        "model_auto_compact_token_limit=872000",
        "-c",
        'model_reasoning_effort="low"',
        "-c",
        'service_tier="default"',
        "-m",
        model,
        "-C",
        str(working_dir),
        "-",
    ]
    try:
        version_run = subprocess.run(
            [codex, "--version"], text=True, capture_output=True, timeout=10
        )
        cli_version = (
            version_run.stdout.strip() if version_run.returncode == 0 else None
        )
    except (OSError, subprocess.TimeoutExpired):
        cli_version = None
    disabled_features = (
        "apps",
        "plugins",
        "hooks",
        "memories",
        "multi_agent",
        "shell_tool",
        "unified_exec",
        "view_image",
        "image_generation",
        "tool_suggest",
        "computer_use",
        "browser_use",
        "goals",
        "sleep_tool",
        "workspace_dependencies",
    )
    for feature in disabled_features:
        cmd[2:2] = ["--disable", feature]
    started_at = datetime.now(timezone.utc).isoformat()
    start = time.monotonic()
    try:
        p = subprocess.run(
            cmd, input=view, text=True, capture_output=True, timeout=timeout
        )
        stdout, stderr, code = p.stdout, p.stderr, p.returncode
    except subprocess.TimeoutExpired as exc:

        def as_text(value):
            return (
                value.decode("utf-8", errors="replace")
                if isinstance(value, bytes)
                else (value or "")
            )

        stdout, stderr, code = as_text(exc.stdout), as_text(exc.stderr), None
        stderr += "\ntransport timeout\n"
    except OSError as exc:
        stdout, stderr, code = "", str(exc), None
    wall = time.monotonic() - start
    ended_at = datetime.now(timezone.utc).isoformat()
    (output_dir / "events.jsonl").write_text(stdout, encoding="utf-8")
    (output_dir / "stderr.txt").write_text(redact_stderr(stderr), encoding="utf-8")
    result = interpret(stdout, code if code is not None else -1)
    session_ids = []
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
            if isinstance(event, dict) and event.get("type") == "thread.started":
                session_ids.append(event.get("thread_id"))
        except (ValueError, TypeError):
            pass
    result.update(
        {
            "session_id": session_ids[0] if len(session_ids) == 1 else None,
            "started_at_utc": started_at,
            "ended_at_utc": ended_at,
            "cli_args": cmd,
            "cli_version": cli_version,
            "working_dir": str(working_dir),
            "reasoning_effort": "low",
            "requested_service_tier": "default",
            "effective_service_tier": None,
            "requested_context_window": 872000,
            "requested_auto_compact_limit": 872000,
            "probe_observed_usable_context_window": 828400,
            "provider_prompt_cache_key": None,
            "prompt_cache_key_note": "not observable or controlled",
            "returncode": code,
            "wall_seconds": wall,
            "model": model,
            "view_sha256": hashlib.sha256(view.encode()).hexdigest(),
            "usage_scope": "CLI reported respondent session total; runtime overhead included, split unavailable; controller and platform overhead excluded",
            "cost_usd": None,
            "cache_state_inferred_from_elapsed_time": False,
        }
    )
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("view", type=Path)
    p.add_argument("output_dir", type=Path)
    p.add_argument(
        "--working-dir",
        type=Path,
        required=True,
        help="Stable existing working directory shared by comparable calls",
    )
    p.add_argument("--codex", default="codex")
    p.add_argument("--model", default="gpt-6-luna")
    p.add_argument("--timeout", type=float, default=300)
    args = p.parse_args()
    result = run(
        args.view.read_text(encoding="utf-8"),
        args.output_dir,
        working_dir=args.working_dir,
        codex=args.codex,
        model=args.model,
        timeout=args.timeout,
    )
    print(json.dumps(result))
    raise SystemExit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
