"""compache: fit context compaction and prompt-cache keep-warm to your own conversations.

policy = compache.fit("threads.jsonl", "labels.jsonl", model="claude-sonnet-5-5")
client = compache.wrap(client, policy)          # or: compache proxy --policy policy.json
with compache.thread("conversation-17"):
    client.messages.create(...)                  # the policy is applied on the way out

Without past traces to fit on, record going forward and fit on the record:

client = compache.wrap(client, record="compache-record", sample=0.1)
policy = compache.fit("compache-record")
"""

__version__ = "0.15.2"

from . import langchain, langfuse, litellm, mlflow, otel
from .api import Policy, Session, Turn, fit, replay
from .defaults import default_policy
from .errors import ContractError
from .live import Live
from .models import known_models, model_spec
from .optimizer.contracts import ModelSpec
from .optimizer.recovery import REOPEN_INSTRUCTIONS, REOPEN_TOOL
from .profile import profile
from .threads import Thread, load_threads
from .wrap import async_http_client, http_client, thread, wrap

__all__ = [
    "REOPEN_INSTRUCTIONS",
    "REOPEN_TOOL",
    "ContractError",
    "Live",
    "ModelSpec",
    "Policy",
    "Session",
    "Thread",
    "Turn",
    "async_http_client",
    "default_policy",
    "fit",
    "http_client",
    "known_models",
    "langchain",
    "langfuse",
    "litellm",
    "load_threads",
    "mlflow",
    "model_spec",
    "otel",
    "profile",
    "replay",
    "thread",
    "wrap",
]
