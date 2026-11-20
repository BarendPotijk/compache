"""Training-only distributions of peak context growth at a fixed run position."""

from __future__ import annotations

from bisect import bisect_right
from collections import Counter, defaultdict
from dataclasses import replace
from fractions import Fraction
from math import ceil, floor

from compache.traces import ContractError
from .contracts import digest, number

SOURCES = {'single_generation_proxy', 'single_generation_reported_output', 'observed_peak'}
STATUSES = {'complete', 'upper_bound', 'censored', 'missing'}
PHASES = ('remaining_tokens', 'whole_run_tokens')
JOINT_PHASE = 'next_admission_or_peak'


def fit_position_growth(rows, min_threads=5):
    """Fit one row per thread reaching a position, including missing/censored rows."""
    number(min_threads, 'min_threads', 1, integer=True)
    grouped = defaultdict(dict)
    for row in rows:
        if not isinstance(row, dict) or set(row) != {
            'thread_id', 'run_index', 'arrival_tokens', 'remaining_tokens', 'source', 'status'
        }:
            raise ContractError('invalid position growth observation fields')
        tid, pos = row['thread_id'], row['run_index']
        if not isinstance(tid, str) or not tid.strip():
            raise ContractError('position growth requires a thread ID')
        number(pos, 'run_index', 1, integer=True)
        if tid in grouped[pos]:
            raise ContractError('duplicate thread/run position in growth observations')
        status, source = row['status'], row['source']
        if status not in STATUSES or source not in SOURCES:
            raise ContractError('invalid growth status/source')
        for key in ('arrival_tokens', 'remaining_tokens'):
            if row[key] is not None:
                number(row[key], key, integer=True)
        if (status == 'missing') != (row['remaining_tokens'] is None):
            raise ContractError('missing growth must be null, other statuses require a value')
        grouped[pos][tid] = row
    positions = {}
    for pos, observations in sorted(grouped.items()):
        values = list(observations.values())
        bounded = [r for r in values if r['status'] in ('complete', 'upper_bound')]
        positions[str(pos)] = {
            'threads': len(values),
            'sources': dict(sorted(Counter(r['source'] for r in values).items())),
            'statuses': dict(sorted(Counter(r['status'] for r in values).items())),
            'remaining_tokens': sorted(r['remaining_tokens'] for r in bounded),
            'whole_run_tokens': sorted(r['arrival_tokens'] + r['remaining_tokens']
                                      for r in bounded if r['arrival_tokens'] is not None),
            'arrival_and_whole_tokens': sorted([r['arrival_tokens'], r['arrival_tokens'] + r['remaining_tokens']]
                                               for r in bounded if r['arrival_tokens'] is not None),
        }
    return {'schema_version': 'compache-position-growth-v2', 'min_threads': min_threads,
            'by_run_index': positions}


def validate_growth(model):
    if not isinstance(model, dict) or set(model) != {'schema_version', 'min_threads', 'by_run_index'} or (
        model['schema_version'] not in ('compache-position-growth-v1', 'compache-position-growth-v2')
        or not isinstance(model['by_run_index'], dict)
    ):
        raise ContractError('invalid position growth model')
    number(model['min_threads'], 'min_threads', 1, integer=True)
    for key, row in model['by_run_index'].items():
        if not isinstance(key, str) or not key.isascii() or not key.isdigit() or key.startswith('0'):
            raise ContractError('invalid growth run position')
        fields = {'threads', 'sources', 'statuses', *PHASES}
        if model['schema_version'] == 'compache-position-growth-v2':
            fields.add('arrival_and_whole_tokens')
        if not isinstance(row, dict) or set(row) != fields:
            raise ContractError('invalid growth position row')
        n = row['threads']
        number(n, 'threads', 1, integer=True)
        for field, allowed in [('sources', SOURCES), ('statuses', STATUSES)]:
            counts = row[field]
            if not isinstance(counts, dict) or set(counts) - allowed:
                raise ContractError('invalid growth evidence counts')
            for count in counts.values():
                number(count, 'evidence count', 1, integer=True)
            if sum(counts.values()) != n:
                raise ContractError('growth evidence counts must include every reached thread')
        supported = sum(row['statuses'].get(s, 0) for s in ('complete', 'upper_bound'))
        for phase in PHASES:
            vals = row[phase]
            if not isinstance(vals, (list, tuple)) or len(vals) > supported:
                raise ContractError('invalid growth sample count')
            if phase == 'remaining_tokens' and len(vals) != supported:
                raise ContractError('remaining growth sample count differs from evidence counts')
            for val in vals:
                number(val, phase, integer=True)
            if any(b < a for a, b in zip(vals, vals[1:])):
                raise ContractError('growth samples must be sorted')
        if 'arrival_and_whole_tokens' in row:
            pairs = row['arrival_and_whole_tokens']
            if not isinstance(pairs, (list, tuple)):
                raise ContractError('invalid growth pairs')
            for pair in pairs:
                if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                    raise ContractError('invalid arrival/whole growth pair')
                number(pair[0], 'arrival_tokens', integer=True)
                number(pair[1], 'whole_run_tokens', pair[0], integer=True)
            if sorted(w for _, w in pairs) != list(row['whole_run_tokens']):
                raise ContractError('paired whole growth differs from fitted whole growth')
            if Counter(w-a for a, w in pairs) - Counter(row['remaining_tokens']):
                raise ContractError('paired residuals differ from fitted remaining growth')


def position_risk_policy(base_policy, profile, *, max_overflow_probability=.05):
    """Bind a training profile to a standalone policy. All retrievals enter verbatim."""
    if base_policy.schema_version != 'compache-policy-v1':
        raise ContractError('position risk requires a verbatim-ingress v1 base policy')
    if base_policy.profile_hash != digest(profile) or profile.get('model_spec_hash') != digest(base_policy.model_spec):
        raise ContractError('position risk training profile/model mismatch')
    if 'position_growth' not in profile:
        raise ContractError('training profile lacks position growth')
    return replace(base_policy, schema_version='compache-policy-v3',
                   position_growth=profile['position_growth'], max_overflow_probability=max_overflow_probability)


def position_risk(policy, run_index, base_tokens, phase):
    """Estimate a strict exceedance. The rank guard counts one additional exceedance."""
    number(run_index, 'run_index', 1, integer=True)
    number(base_tokens, 'base_tokens', integer=True)
    if phase not in (*PHASES, JOINT_PHASE) or policy.position_growth is None:
        raise ContractError('position risk requires a fitted policy and known phase')
    model = policy.position_growth
    alpha = Fraction(str(policy.max_overflow_probability))
    row = model['by_run_index'].get(str(run_index), {})
    n = row.get('threads', 0)
    window = policy.model_spec['context_tokens']
    room = window - base_tokens
    details = {}
    if phase == JOINT_PHASE:
        input_limit = floor((1 - policy.reserve) * window)
        reserved = window - input_limit
        pairs = row.get('arrival_and_whole_tokens')
        if pairs is None:
            # Older artifacts store only whole-run growth. Since arrival <=
            # whole, charge the entire increment against the tighter input cap.
            values = [w + reserved for w in row.get('whole_run_tokens', [])]
            details['joint_basis'] = 'whole_run_against_input_cap_conservative'
        else:
            # A sample crosses if *either* cap fails. Counting the maximum
            # preserves correlation and counts a double breach exactly once.
            values = sorted(max(a + reserved, w) for a, w in pairs)
            details.update(joint_basis='paired_arrival_and_peak',
                           bounded_pairs=len(pairs),
                           bounded_pair_admission_exceedances=sum(base_tokens+a > input_limit for a, _ in pairs),
                           bounded_pair_peak_exceedances=sum(base_tokens+w > window for _, w in pairs),
                           bounded_pair_joint_exceedances=len(pairs)-bisect_right(values, room))
        details.update(input_limit_tokens=input_limit, admission_headroom_tokens=input_limit-base_tokens)
    else:
        values = row.get(phase, [])
    # Treat each missing, unbounded or censored maximum as +infinity. The rank
    # uses n+1 to account for a new sample. It is not a confidence bound on p.
    exceeds = n - bisect_right(values, room)
    rank = ceil((1 - alpha) * (n + 1))
    supported = n >= model['min_threads'] and rank <= n
    fallback = None
    if not supported:
        fallback = 'insufficient_position_threads' if n else 'unseen_position'
    unresolved = n - len(values)
    # An unbounded rank is evidence of unresolved risk. Never turn missing
    # training peaks into zero growth or substitute a pooled later position.
    quantile = values[rank - 1] if supported and rank <= len(values) else None
    return {
        'phase': phase, 'run_index': run_index, 'base_tokens': base_tokens, **details,
        'headroom_tokens': room, 'threads': n, 'unbounded_observations': unresolved,
        'sources': row.get('sources', {}), 'statuses': row.get('statuses', {}),
        'sample_exceedance_fraction_conservative': exceeds / n if n else None,
        'rank_adjusted_exceedance': (exceeds + 1) / (n + 1) if n else None,
        'max_overflow_probability': float(alpha), 'growth_quantile_tokens': quantile,
        'warm_ceiling_tokens': policy.model_spec['context_tokens'] - quantile if quantile is not None else None,
        'fallback': fallback,
        'triggered': ((exceeds + 1) > alpha * (n + 1)) if supported else None,
    }
