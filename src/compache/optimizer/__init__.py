"""The engine behind `compache.fit`, `compache.replay` and `compache.Session`.

Use these directly for what the top-level API leaves out: a persistent cache-break
checker, a tokenizer callback, shared headers across threads or a host call ledger.
"""

from .cache_breaks import CacheBreakChecker, CacheScope
from .contracts import ModelSpec
from .edit_accounting import InputAnchor, InputEstimate, PartMeter, estimate_input
from .host_ledger import count_foreground_calls
from .parts import Request, Source, expand, handle
from .recovery import REOPEN_INSTRUCTIONS, REOPEN_TOOL, reopen
from .replay import compare_host_ledger, replay_workload
from .retention import RetentionRecipe
from .session import (
    WorkloadAdmissionError,
    WorkloadPolicy,
    WorkloadSession,
    fit_workload,
)
from .shared_prefix import SharedPrefixCatalog
from .solve import solve_workload
from .usage import response_usage

__all__ = [
    "REOPEN_INSTRUCTIONS",
    "REOPEN_TOOL",
    "CacheBreakChecker",
    "CacheScope",
    "InputAnchor",
    "InputEstimate",
    "ModelSpec",
    "PartMeter",
    "Request",
    "RetentionRecipe",
    "SharedPrefixCatalog",
    "Source",
    "WorkloadAdmissionError",
    "WorkloadPolicy",
    "WorkloadSession",
    "compare_host_ledger",
    "count_foreground_calls",
    "estimate_input",
    "expand",
    "fit_workload",
    "handle",
    "reopen",
    "replay_workload",
    "response_usage",
    "solve_workload",
]
