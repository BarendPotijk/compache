"""Offline optimization and pure RAG request/retention decisions."""

from .contracts import Constraints, ModelSpec, Policy
from .calibration import OnlineTokenCalibrator
from .fit import fit_profile, freeze_profile
from .gate import Prepared, Request, Source, ThreadState, expand, handle, prepare
from .recovery import REOPEN_INSTRUCTIONS, REOPEN_TOOL, reopen
from .planner import RetentionDecision, end_run
from .position import fit_position_growth, position_risk, position_risk_policy
from .replay import replay
from .solve import optimize, split_threads
from .cache_breaks import CacheBreakChecker, CacheScope
from .shared_prefix import SharedPrefixCatalog
from .usage import response_usage
from .virtual_cache import CacheRequest, VirtualCache
from .host_ledger import count_foreground_calls
from .edit_accounting import InputAnchor, InputEstimate, PartMeter, estimate_input
from .retention import RetentionRecipe
from .workload import WorkloadAdmissionError, WorkloadPolicy, WorkloadSession, fit_workload
from .workload_replay import compare_host_ledger, replay_workload

__all__ = [
    "CacheRequest",
    "VirtualCache",
    "response_usage",
    "count_foreground_calls",
    "InputAnchor",
    "InputEstimate",
    "PartMeter",
    "estimate_input",
    "CacheBreakChecker",
    "CacheScope",
    "SharedPrefixCatalog",
    "RetentionRecipe",
    "WorkloadAdmissionError",
    "WorkloadPolicy",
    "WorkloadSession",
    "fit_workload",
    "replay_workload",
    "compare_host_ledger",
    "Constraints",
    "ModelSpec",
    "Policy",
    "OnlineTokenCalibrator",
    "fit_profile",
    "freeze_profile",
    "Prepared",
    "Request",
    "Source",
    "ThreadState",
    "expand",
    "REOPEN_INSTRUCTIONS",
    "REOPEN_TOOL",
    "reopen",
    "handle",
    "prepare",
    "RetentionDecision",
    "end_run",
    "fit_position_growth",
    "position_risk",
    "position_risk_policy",
    "replay",
    "optimize",
    "split_threads",
]
