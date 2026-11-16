"""Validated offline trace contract and scoped Langfuse envelope adapter."""
from __future__ import annotations
import copy
import json
import hashlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

class ContractError(ValueError):
    """Invalid trace contract."""

def _id(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ContractError(f'{name}: expected nonempty trimmed string')
    return value

def _time(value: Any, name: str) -> datetime:
    _id(value, name)
    try:
        result = datetime.fromisoformat(value.replace('Z', '+00:00'))
        if result.tzinfo is None or result.utcoffset() is None: raise ValueError()
        return result
    except ValueError as e:
        raise ContractError(f'{name}: expected timezone-aware ISO timestamp') from e

def _part(p: Any, kind: str) -> None:
    if not isinstance(p, dict): raise ContractError(f'{kind}: expected object')
    allowed = {'part_id', 'text', 'content_status'} | ({'document_id', 'version', 'chunk_id', 'exposed_to_model'} if kind == 'retrieval' else set())
    if set(p) - allowed: raise ContractError(f'{kind}: unknown fields: {sorted(set(p) - allowed)}')
    _id(p.get('part_id'), 'part_id')
    text = p.get('text')
    if text is not None and not isinstance(text, str): raise ContractError('part text must be string or null')
    status = p.get('content_status', 'available' if text is not None else 'missing')
    if status not in ('available', 'missing', 'masked'): raise ContractError('invalid content_status')
    if status == 'available' and text is None: raise ContractError('available content must have text')
    if status != 'available' and text is not None: raise ContractError('masked or missing content must have null text')
    if kind == 'retrieval':
        for k in ('document_id', 'version', 'chunk_id'): _id(p.get(k), k)
        if type(p.get('exposed_to_model')) is not bool: raise ContractError('exposed_to_model must be boolean')

@dataclass(frozen=True)
class Part:
    part_id: str
    text: str | None
    content_status: str

@dataclass(frozen=True)
class RetrievalPart(Part):
    document_id: str
    version: str
    chunk_id: str
    exposed_to_model: bool

@dataclass(frozen=True)
class Generation:
    observation_id: str
    model: str
    started_at: str
    ended_at: str
    usage: tuple[tuple[str, int], ...] | None
    request_part_ids: tuple[str, ...]

@dataclass(frozen=True)
class Run:
    run_id: str
    run_index: int
    started_at: str
    ended_at: str
    history_policy: str
    input: Part
    output: Part
    retrievals: tuple[RetrievalPart, ...]
    generation: Generation

def _typed_part(p: dict, retrieval: bool = False) -> Part:
    args = (p['part_id'], p.get('text'), p.get('content_status', 'available' if p.get('text') is not None else 'missing'))
    return RetrievalPart(*args, p['document_id'], p['version'], p['chunk_id'], p['exposed_to_model']) if retrieval else Part(*args)

class Thread:
    def __init__(self, data: dict):
        self._data = self._validate(data)
        d = self._data
        self.company_id, self.thread_id = d['company_id'], d['thread_id']
        records = []
        for r in d['runs']:
            g = r['generation']; usage = g['usage']
            generation = Generation(g['observation_id'], g['model'], g['started_at'], g['ended_at'], tuple(sorted(usage.items())) if usage is not None else None, tuple(g['request_part_ids']))
            records.append(Run(r['run_id'], r['run_index'], r['started_at'], r['ended_at'], r['history_policy'], _typed_part(r['input']), _typed_part(r['output']), tuple(_typed_part(p, True) for p in r['retrievals']), generation))
        self.runs = tuple(records)
    @classmethod
    def from_dict(cls, data: dict) -> 'Thread':
        return cls(data)
    @staticmethod
    def _validate(data: dict) -> dict:
        if not isinstance(data, dict): raise ContractError('thread must be object')
        if set(data) - {'company_id', 'thread_id', 'runs', 'scenario_id', 'provenance'}: raise ContractError('unknown thread fields')
        if 'provenance' in data: _id(data['provenance'], 'provenance')
        _id(data.get('company_id'), 'company_id'); _id(data.get('thread_id'), 'thread_id')
        runs = data.get('runs')
        if not isinstance(runs, list) or not runs: raise ContractError('runs must be a nonempty array')
        seen: dict[str, tuple[int, str, dict]] = {}; run_ids = set(); observations = set(); last_end = None; expected_history = []
        for i, run in enumerate(runs, 1):
            if not isinstance(run, dict): raise ContractError('run must be object')
            if set(run) - {'run_id','run_index','started_at','ended_at','history_policy','input','output','retrievals','generation'}: raise ContractError('unknown run fields')
            rid = _id(run.get('run_id'), 'run_id')
            if rid in run_ids: raise ContractError('duplicate run_id')
            run_ids.add(rid)
            if type(run.get('run_index')) is not int or run['run_index'] != i: raise ContractError('run_index must be consecutive starting at 1')
            start, end = _time(run.get('started_at'), 'run.started_at'), _time(run.get('ended_at'), 'run.ended_at')
            if start > end or (last_end is not None and start < last_end): raise ContractError('run timestamps not chronological')
            last_end = end
            if run.get('history_policy') != 'full_canonical': raise ContractError('unsupported history_policy')
            retrievals = run.get('retrievals')
            if not isinstance(retrievals, list): raise ContractError('retrievals must be array')
            for kind, parts in (('input', [run.get('input')]), ('retrieval', retrievals), ('output', [run.get('output')])):
                for p in parts:
                    _part(p, kind); pid = p['part_id']
                    if pid in seen: raise ContractError('duplicate part_id: ' + pid)
                    seen[pid] = (i, kind, p)
            g = run.get('generation')
            if not isinstance(g, dict): raise ContractError('exactly one generation object required')
            if set(g) - {'observation_id','started_at','ended_at','model','usage','request_part_ids'}: raise ContractError('unknown generation fields')
            oid = _id(g.get('observation_id'), 'observation_id'); _id(g.get('model'), 'model')
            if oid in observations: raise ContractError('duplicate observation_id')
            observations.add(oid)
            gs, ge = _time(g.get('started_at'), 'generation.started_at'), _time(g.get('ended_at'), 'generation.ended_at')
            if not start <= gs <= ge <= end: raise ContractError('generation timestamps outside run')
            if 'usage' not in g: raise ContractError('generation usage must be explicit, null if unavailable')
            usage = g.get('usage')
            if usage is not None and (not isinstance(usage, dict) or set(usage) != {'input','output','cache_read','cache_write'} or any(type(v) is not int or v < 0 for v in usage.values()) or usage['cache_read'] + usage['cache_write'] > usage['input']): raise ContractError('usage must contain nonnegative integer input/output/cache_read/cache_write, with cache within input')
            ids = g.get('request_part_ids')
            if not isinstance(ids, list) or any(not isinstance(x,str) or not x for x in ids) or len(ids) != len(set(ids)): raise ContractError('request_part_ids must be unique IDs')
            expected = expected_history + [run['input']['part_id']] + [p['part_id'] for p in retrievals if p['exposed_to_model']]
            if ids != expected: raise ContractError('full_canonical request does not match ordered visible history and current sources')
            expected_history = expected + [run['output']['part_id']]
            if run['input']['part_id'] not in ids: raise ContractError('current input missing from request')
            for pid in ids:
                if pid not in seen: raise ContractError('unknown, foreign or future source: ' + pid)
                origin, kind, part = seen[pid]
                if kind == 'output' and origin == i: raise ContractError('current output cannot be visible')
                if kind == 'retrieval' and not part['exposed_to_model']: raise ContractError('unexposed retrieval in request')
        return copy.deepcopy(data)
    def to_dict(self) -> dict: return copy.deepcopy(self._data)

def load_threads(path: str | Path) -> list[Thread]:
    rows = []
    for n, line in enumerate(Path(path).read_text(encoding='utf-8').splitlines(), 1):
        if line.strip():
            try: rows.append(Thread.from_dict(json.loads(line)))
            except (ValueError, TypeError) as e: raise ContractError(f'line {n}: {e}') from e
    _unique(rows)
    return rows

def _unique(threads: Iterable[Thread]) -> None:
    tids=set(); rids=set(); pids=set(); oids=set()
    for t in threads:
        d=t.to_dict(); tid=d['thread_id']
        if tid in tids: raise ContractError('duplicate thread_id: '+tid)
        tids.add(tid)
        for r in d['runs']:
            for value, seen, name in [(r['run_id'],rids,'run_id'),(r['generation']['observation_id'],oids,'observation_id')]:
                if value in seen: raise ContractError('duplicate '+name+': '+value)
                seen.add(value)
            for p in [r['input'],r['output'],*r['retrievals']]:
                if p['part_id'] in pids: raise ContractError('duplicate part_id: '+p['part_id'])
                pids.add(p['part_id'])

def write_threads(threads: Iterable[Thread], path: str | Path) -> None:
    ts=list(threads); _unique(ts)
    Path(path).write_text(''.join(json.dumps(t.to_dict(), ensure_ascii=False, sort_keys=True)+'\n' for t in ts), encoding='utf-8')

def _export_usage(usage: dict | None) -> dict | None:
    if usage is None: return None
    return {'input': usage['input'] - usage['cache_read'] - usage['cache_write'], 'output': usage['output'], 'cache_read_input_tokens': usage['cache_read'], 'cache_creation_input_tokens': usage['cache_write']}

def _import_usage(usage: Any) -> dict | None:
    if usage is None: return None
    keys = {'input', 'output', 'cache_read_input_tokens', 'cache_creation_input_tokens'}
    if not isinstance(usage, dict) or set(usage) != keys or any(type(v) is not int or v < 0 for v in usage.values()): raise ContractError('unsupported usageDetails: expected four exclusive nonnegative integer buckets')
    return {'input': usage['input'] + usage['cache_read_input_tokens'] + usage['cache_creation_input_tokens'], 'output': usage['output'], 'cache_read': usage['cache_read_input_tokens'], 'cache_write': usage['cache_creation_input_tokens']}

def export_langfuse(threads: Iterable[Thread]) -> list[dict]:
    ts=list(threads); _unique(ts); rows=[]; observation_ids=set()
    for thread in ts:
        t=thread.to_dict(); parts={}
        for r in t['runs']:
            for kind, ps in [('input',[r['input']]),('retrieval',r['retrievals']),('output',[r['output']])]:
                for p in ps: parts[p['part_id']] = (kind,p)
            root=hashlib.sha256(('root:'+r['run_id']).encode()).hexdigest()[:16]; trace=hashlib.sha256(('trace:'+r['run_id']).encode()).hexdigest()[:32]; ret=hashlib.sha256(('retrieval:'+r['run_id']).encode()).hexdigest()[:16]
            for oid in (root, ret, r['generation']['observation_id']):
                if oid in observation_ids: raise ContractError('export observation ID collision: ' + oid)
                observation_ids.add(oid)
            meta={'company_id':t['company_id'],'thread_id':t['thread_id'],'run_id':r['run_id'],'run_index':r['run_index'],'history_policy':r['history_policy']}
            if 'provenance' in t: meta['provenance'] = t['provenance']
            base={'traceId':trace,'sessionId':t['thread_id'],'metadata':meta}
            rows.append({**base,'id':root,'type':'SPAN','name':'synthetic-rag-run','parentObservationId':None,'isRootObservation':True,'startTime':r['started_at'],'endTime':r['ended_at'],'input':r['input'],'output':r['output']})
            rows.append({**base,'id':ret,'type':'SPAN','name':'retrieval','parentObservationId':root,'isRootObservation':False,'startTime':r['started_at'],'endTime':r['started_at'],'output':{'candidates':r['retrievals']}})
            g=r['generation']; msgs=[]
            for pid in g['request_part_ids']:
                kind,p=parts[pid]; msg={'part_id':pid,'role':'assistant' if kind=='output' else 'user','content':p.get('text')}
                if kind=='retrieval': msg['context_kind']='retrieval'
                msgs.append(msg)
            rows.append({**base,'id':g['observation_id'],'type':'GENERATION','name':'answer-generation','parentObservationId':root,'isRootObservation':False,'startTime':g['started_at'],'endTime':g['ended_at'],'model':g['model'],'usageDetails':_export_usage(g['usage']),'input':{'schema_version':'compache-rag-message-envelope-v1','messages':msgs,'request_part_ids':g['request_part_ids']},'output':r['output']})
    return copy.deepcopy(rows)

def _decoded(x: Any) -> Any:
    if isinstance(x,str):
        try: return json.loads(x)
        except json.JSONDecodeError as e: raise ContractError('invalid JSON encoded observation payload') from e
    return x

def normalize_langfuse(rows: Iterable[dict]) -> list[Thread]:
    unique={}
    for original in rows:
        if not isinstance(original,dict): raise ContractError('observation must be object')
        row=copy.deepcopy(original); oid=_id(row.get('id'),'observation id')
        if oid in unique and unique[oid] != row: raise ContractError('conflicting duplicate observation: '+oid)
        unique[oid]=row
    groups={}
    for row in unique.values():
        sid=_id(row.get('sessionId'),'sessionId'); trace=_id(row.get('traceId'),'traceId'); m=row.get('metadata')
        if not isinstance(m,dict): raise ContractError('metadata required')
        company=_id(m.get('company_id'),'company_id'); tid=_id(m.get('thread_id'),'thread_id'); rid=_id(m.get('run_id'),'run_id')
        if sid != tid: raise ContractError('sessionId and thread_id mismatch')
        groups.setdefault((company,tid,rid,trace),[]).append(row)
    thread_runs={}; envelopes={}; provenance={}
    for (company,tid,rid,trace), rs in groups.items():
        roots=[r for r in rs if r.get('name')=='synthetic-rag-run' and r.get('type')=='SPAN']; rets=[r for r in rs if r.get('name')=='retrieval' and r.get('type')=='SPAN']; gens=[r for r in rs if r.get('type')=='GENERATION']
        if len(roots)!=1 or len(rets)!=1 or len(gens)!=1 or len(rs)!=3: raise ContractError('expected one root, retrieval and generation per run; unsupported observation dialect')
        root,ret,gen=roots[0],rets[0],gens[0]
        for row in rs:
            meta = row['metadata']
            for key in ('run_index', 'history_policy'):
                if key in meta and (key not in root['metadata'] or meta[key] != root['metadata'][key]): raise ContractError('observation metadata mismatch: ' + key)
            start = _time(row.get('startTime'), 'observation.startTime'); end = _time(row.get('endTime'), 'observation.endTime')
            if not _time(root.get('startTime'), 'root.startTime') <= start <= end <= _time(root.get('endTime'), 'root.endTime'): raise ContractError('observation timestamps outside root')
            if 'isRootObservation' in row and row['isRootObservation'] is not (row is root): raise ContractError('isRootObservation mismatch')
        if root.get('parentObservationId') is not None or any(r.get('parentObservationId')!=root['id'] for r in (ret,gen)): raise ContractError('invalid observation ancestry')
        env=_decoded(gen.get('input')); ro=_decoded(ret.get('output'))
        if not isinstance(env,dict) or env.get('schema_version')!='compache-rag-message-envelope-v1': raise ContractError('unsupported generation envelope')
        if not isinstance(ro,dict) or not isinstance(ro.get('candidates'),list): raise ContractError('invalid retrieval candidates')
        inp=_decoded(root.get('input')); out=_decoded(root.get('output'))
        if _decoded(gen.get('output')) != out: raise ContractError('generation output differs from root output')
        m=root['metadata']; run={'run_id':rid,'run_index':m.get('run_index'), 'started_at':root.get('startTime'),'ended_at':root.get('endTime'),'history_policy':m.get('history_policy','full_canonical'),'input':inp,'output':out,'retrievals':ro['candidates'],'generation':{'observation_id':gen['id'],'started_at':gen.get('startTime'),'ended_at':gen.get('endTime'),'model':gen.get('model'),'usage':_import_usage(gen.get('usageDetails')),'request_part_ids':env.get('request_part_ids')}}
        prov = root['metadata'].get('provenance')
        if any(r['metadata'].get('provenance') != prov for r in rs): raise ContractError('observation provenance mismatch')
        if (company,tid) in provenance and provenance[(company,tid)] != prov: raise ContractError('thread provenance mismatch')
        provenance[(company,tid)] = prov
        thread_runs.setdefault((company,tid),[]).append(run); envelopes[(company,tid,rid)]=env
    result=[]
    for (company,tid), runs in sorted(thread_runs.items()):
        for r in runs:
            if r['run_index'] is not None and (type(r['run_index']) is not int or r['run_index'] < 1): raise ContractError('invalid run_index')
        starts = [_time(r['started_at'], 'run.started_at') for r in runs]
        if len(starts) != len(set(starts)) and any(r['run_index'] is None for r in runs): raise ContractError('ambiguous tied run timestamps without run_index')
        runs.sort(key=lambda r: (_time(r['started_at'],'run.started_at'), r['run_index'] if r['run_index'] is not None else 0))
        for i,r in enumerate(runs,1):
            if r['run_index'] is None: r['run_index']=i
        data={'company_id':company,'thread_id':tid,'runs':runs}
        if provenance[(company,tid)] is not None: data['provenance'] = provenance[(company,tid)]
        t=Thread.from_dict(data); parts={}
        for r in runs:
            for kind,ps in [('input',[r['input']]),('retrieval',r['retrievals']),('output',[r['output']])]:
                for p in ps: parts[p['part_id']]=(kind,p)
            env=envelopes[(company,tid,r['run_id'])]; msgs=env.get('messages'); ids=r['generation']['request_part_ids']
            if not isinstance(msgs,list) or len(msgs)!=len(ids): raise ContractError('message/request count mismatch')
            for msg,pid in zip(msgs,ids):
                kind,p=parts[pid]
                if not isinstance(msg,dict) or msg.get('part_id')!=pid or msg.get('role') != ('assistant' if kind=='output' else 'user') or msg.get('content') != p.get('text') or (msg.get('context_kind')=='retrieval') != (kind=='retrieval'): raise ContractError('message role/source/content mismatch: '+pid)
        result.append(t)
    _unique(result); return result
