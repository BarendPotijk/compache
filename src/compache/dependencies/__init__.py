"""Truth-free labeling packets and evidence validation."""
from __future__ import annotations
import copy
import json
from importlib import resources
from compache.traces import Thread, ContractError

def load_label_schema() -> dict:
    return json.loads(resources.files(__package__).joinpath("label.schema.json").read_text(encoding="utf-8"))

def _safe_part(part: dict) -> dict:
    result = {k: v for k, v in part.items() if k in ('part_id', 'text', 'content_status')}
    if part.get('content_status', 'available' if part.get('text') is not None else 'missing') != 'available': result['text'] = None
    return result

def build_packet(thread: Thread | dict, target_run_id: str) -> dict:
    t = Thread.from_dict(thread) if isinstance(thread, dict) else thread
    if not isinstance(t, Thread): raise ContractError("thread must be Thread or dict")
    # Read the validated thread in place. Only the packet returned below is copied.
    data = t._data; parts = {}
    for run in data["runs"]:
        for kind, items in (("input", [run["input"]]), ("retrieval", run["retrievals"]), ("output", [run["output"]])):
            for part in items:
                p = {"part_id": part["part_id"], "origin_run_id": run["run_id"], "origin_run_index": run["run_index"], "kind": kind, "text": part.get("text") if part.get("content_status", "available" if part.get("text") is not None else "missing") == "available" else None, "content_status": part.get("content_status", "available" if part.get("text") is not None else "missing")}
                if kind == "retrieval": p.update({k: part[k] for k in ("document_id", "version", "chunk_id")})
                parts[part["part_id"]] = p
        if run["run_id"] == target_run_id:
            visible = [parts[x] for x in run["generation"]["request_part_ids"]]
            notes = ["Visible source text unavailable: " + p["part_id"] for p in visible if p["text"] is None or p["content_status"] != "available"]
            if run["output"].get("text") is None or run["output"].get("content_status", "available") != "available": notes.append("Target output is unavailable.")
            return copy.deepcopy({"schema_version": "compache-label-input-v1", "thread_id": data["thread_id"], "target_run_id": target_run_id, "target_run_index": run["run_index"], "target_question": _safe_part(run["input"]), "target_output": _safe_part(run["output"]), "request_parts": visible, "request_coverage": "partial" if notes else "complete", "coverage_notes": notes})
    raise ContractError("target run not found: " + str(target_run_id))

def shape_errors(value, schema, path='$'):
    errors = []
    kind = schema.get('type')
    types = {'object': dict, 'array': list, 'string': str}
    if kind not in types:
        raise ValueError(f'unsupported schema type in bundled validator: {kind}')
    if not isinstance(value, types[kind]):
        return [f'{path}: expected {kind}']
    if 'enum' in schema and value not in schema['enum']:
        errors.append(f'{path}: invalid enum value')
    if kind == 'object':
        props = schema['properties']
        for key in schema['required']:
            if key not in value:
                errors.append(f'{path}.{key}: missing')
        for key in value:
            if key not in props:
                errors.append(f'{path}.{key}: unexpected')
            else:
                errors.extend(shape_errors(value[key], props[key], path + '.' + key))
    elif kind == 'array':
        for i, item in enumerate(value):
            errors.extend(shape_errors(item, schema['items'], f'{path}[{i}]'))
    return errors[:100]


def validate_label(label, packet):
    if not isinstance(packet, dict): return ['packet: expected object']
    required = ('thread_id', 'target_run_id', 'target_run_index', 'target_question', 'target_output', 'request_parts', 'request_coverage', 'coverage_notes')
    if any(k not in packet for k in required): return ['packet: missing required fields']
    if not isinstance(packet['target_question'], dict) or not isinstance(packet['target_output'], dict) or not isinstance(packet['request_parts'], list): return ['packet: invalid question, output, or request_parts']
    if type(packet['target_run_index']) is not int or packet['request_coverage'] not in ('complete', 'partial') or not isinstance(packet['coverage_notes'], list): return ['packet: invalid index or coverage']
    parts_list = packet['request_parts']
    if any(not isinstance(p, dict) or not isinstance(p.get('part_id'), str) or p.get('kind') not in ('input', 'retrieval', 'output') or type(p.get('origin_run_index')) is not int or not isinstance(p.get('origin_run_id'), str) or p.get('content_status') not in ('available','missing','masked') or (p.get('text') is not None and not isinstance(p.get('text'),str)) for p in parts_list): return ['packet: invalid request part']
    ids = [p['part_id'] for p in parts_list]
    if len(ids) != len(set(ids)): return ['packet: duplicate request part']
    question = packet['target_question']; output = packet['target_output']
    if question.get('part_id') not in ids or output.get('part_id') in ids: return ['packet: current input/output visibility mismatch']
    current = next(p for p in parts_list if p['part_id'] == question['part_id'])
    if current['kind'] != 'input' or current['origin_run_index'] != packet['target_run_index'] or current['origin_run_id'] != packet['target_run_id'] or current.get('text') != question.get('text'): return ['packet: current question mismatch']
    if any(p['origin_run_index'] > packet['target_run_index'] or (p['origin_run_index'] == packet['target_run_index'] and (p['origin_run_id'] != packet['target_run_id'] or p['kind'] == 'output')) for p in parts_list): return ['packet: future or foreign source']
    unavailable = any(p.get('text') is None or p['content_status'] != 'available' for p in parts_list) or output.get('text') is None or output.get('content_status', 'available') != 'available'
    if packet['request_coverage'] == 'complete' and unavailable: return ['packet: false complete coverage']
    schema = load_label_schema()
    errors = shape_errors(label, schema)
    if errors:
        return errors[:100]
    def check(condition, message):
        if not condition:
            errors.append(message)
    for key in ('thread_id', 'target_run_id'):
        check(label[key] == packet[key], f'{key}: does not match packet')
    parts = {p['part_id']: p for p in packet['request_parts']}
    output = packet['target_output'].get('text')
    question = packet['target_question'].get('text')
    if label['status'] == 'complete':
        if question:
            check(bool(label['task_requirements']), 'complete assessment needs task requirements')
        if output:
            check(bool(label['claims']), 'complete assessment needs observed-answer claims')
    if packet['request_coverage'] != 'complete' or output is None:
        check(label['status'] != 'complete', 'missing content cannot yield complete coverage')
    if label['status'] != 'complete':
        check(bool(label['coverage_notes']), 'partial/unassessable needs coverage notes')
    def exact(quote, text, name):
        check(bool(quote) and isinstance(text, str) and quote in text, name + ': quote not found verbatim')
    def requirement(req, name):
        check(req['necessity'] != 'tested', name + ': tested requires an intervention artifact; unsupported by v1')
        kinds = set()
        for alt in req['alternatives']:
            check(bool(alt['sources']), name + ': empty alternative is not sufficient evidence')
            source_ids = [s['part_id'] for s in alt['sources']]
            check(len(source_ids) == len(set(source_ids)), name + ': duplicate sources in alternative')
            for source in alt['sources']:
                p = parts.get(source['part_id'])
                check(p is not None, name + ': source is absent, future, or not visible: ' + source['part_id'])
                if p:
                    kinds.add(p['kind'])
                    check(p['content_status'] == 'available', name + ': unavailable source content')
                    exact(source['quote'], p['text'], name + '/' + source['part_id'])
        expected_kind = next(iter(kinds)) if len(kinds) == 1 else ('mixed' if kinds else 'unknown')
        check(req['source_kind'] == expected_kind, name + ': source_kind does not match cited sources')
    task_ids = [r['requirement_id'] for r in label['task_requirements']]
    check(len(task_ids) == len(set(task_ids)), 'duplicate task requirement IDs')
    for req in label['task_requirements']:
        exact(req['question_quote'], question, 'task/' + req['requirement_id'])
        requirement(req, 'task/' + req['requirement_id'])
        current_alternative = any(alt['sources'] and all(s['part_id'] in parts and parts[s['part_id']]['origin_run_index'] == packet['target_run_index'] for s in alt['sources']) for alt in req['alternatives'])
        if req['historical_context_needed'] == 'no':
            check(current_alternative, 'task/' + req['requirement_id'] + ': no history requires a sufficient current-source alternative')
        elif req['historical_context_needed'] == 'yes':
            check(bool(req['alternatives']) and not current_alternative, 'task/' + req['requirement_id'] + ': history needed conflicts with sufficient alternatives')
    claim_ids = [c['claim_id'] for c in label['claims']]
    check(len(claim_ids) == len(set(claim_ids)), 'duplicate claim IDs')
    for claim in label['claims']:
        exact(claim['target_quote'], output, 'claim/' + claim['claim_id'])
        if claim['relation'] == 'supported':
            check(bool(claim['requirements']) and all(r['alternatives'] for r in claim['requirements']),
                  'supported claim needs nonempty evidence requirements')
        req_ids = [r['requirement_id'] for r in claim['requirements']]
        check(len(req_ids) == len(set(req_ids)), 'duplicate requirement IDs within claim')
        for req in claim['requirements']:
            requirement(req, 'claim/' + claim['claim_id'] + '/' + req['requirement_id'])
    for edge in label['historical_dependencies']:
        p = parts.get(edge['source_part_id'])
        check(p is not None, 'historical edge source not visible')
        if p:
            check(p['origin_run_index'] < packet['target_run_index'], 'historical edge source is not earlier')
            check(edge['source_run_id'] == p['origin_run_id'], 'historical edge run identity mismatch')
            check(edge['source_kind'] == p['kind'], 'historical edge source_kind mismatch')
            if p['text'] is None or p['content_status'] != 'available':
                check(edge['relation'] == 'possibly_used', 'unavailable source can only be possibly_used')
        check(bool(edge['target_claim_ids'] or edge['target_task_requirement_ids']), 'unlinked historical edge')
        check(set(edge['target_claim_ids']).issubset(claim_ids), 'edge references unknown claim')
        check(set(edge['target_task_requirement_ids']).issubset(task_ids), 'edge references unknown task requirement')
    return errors[:100]


def label_index(threads: list[Thread], labels: list[dict]) -> dict:
    """Validated labels by target run. A packet is built only for a labelled run."""
    # Python callers can construct individually valid threads without the
    # cross-thread identity checks made by load_threads(). Labels are keyed by
    # run ID, so a collision could leak a held-out observation.
    thread_ids = [t.thread_id for t in threads]
    run_ids = [r.run_id for t in threads for r in t.runs]
    if len(set(thread_ids)) != len(thread_ids) or len(set(run_ids)) != len(run_ids):
        raise ContractError("training/evaluation requires globally unique thread and run IDs")
    owner = {r.run_id: t for t in threads for r in t.runs}
    result = {}
    for label in labels:
        rid = label.get("target_run_id") if isinstance(label, dict) else None
        if rid not in owner or rid in result:
            raise ContractError("label targets must be unique runs in the supplied traces")
        errors = validate_label(label, build_packet(owner[rid], rid))
        if errors:
            raise ContractError(f"invalid label for {rid}: " + "; ".join(errors[:3]))
        result[rid] = label
    return result
