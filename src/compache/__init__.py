"""compache: fit context compaction and prompt-cache keep-warm to your own conversations.

policy = compache.fit("threads.jsonl", "labels.jsonl", model="claude-sonnet-5-5")
session = policy.session("thread-1")
turn = session.prepare(question, retrieved)   # send turn.messages
session.complete(turn, answer)
"""

__version__ = "0.11.0"

from . import langfuse
from .api import Policy, Session, Turn, fit, replay
from .errors import ContractError
from .models import known_models, model_spec
from .optimizer.contracts import ModelSpec
from .optimizer.recovery import REOPEN_INSTRUCTIONS, REOPEN_TOOL
from .profile import profile
from .threads import Thread, load_threads

__all__ = [
    "REOPEN_INSTRUCTIONS",
    "REOPEN_TOOL",
    "ContractError",
    "ModelSpec",
    "Policy",
    "Session",
    "Thread",
    "Turn",
    "fit",
    "known_models",
    "langfuse",
    "load_threads",
    "model_spec",
    "profile",
    "replay",
]
