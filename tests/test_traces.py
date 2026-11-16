import copy
import json
import tempfile
import unittest
from pathlib import Path
from compache.traces import ContractError, Thread, export_langfuse, normalize_langfuse, load_threads, write_threads

def sample():
    return {'company_id':'c','thread_id':'t','runs':[{'run_id':'r','run_index':1,'started_at':'2025-01-01T00:00:00Z','ended_at':'2025-01-01T00:00:03Z','history_policy':'full_canonical','input':{'part_id':'i','text':'Question'},'output':{'part_id':'o','text':'Answer'},'retrievals':[{'part_id':'p','text':'Evidence','document_id':'d','version':'v1','chunk_id':'x','exposed_to_model':True}], 'generation':{'observation_id':'g','model':'m','started_at':'2025-01-01T00:00:01Z','ended_at':'2025-01-01T00:00:02Z','usage':{'input':12,'output':4,'cache_read':7,'cache_write':2},'request_part_ids':['i','p']}}]}

class TraceTests(unittest.TestCase):
    def test_roundtrip_and_copy(self):
        d=sample(); t=Thread.from_dict(d); d['runs'][0]['input']['text']='changed'
        rows=export_langfuse([t]); rows[2]['input']=json.dumps(rows[2]['input']); rows[0]['input']=json.dumps(rows[0]['input'])
        self.assertEqual(normalize_langfuse(list(reversed(rows))+[copy.deepcopy(rows[0])])[0].to_dict(), t.to_dict())
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'t.jsonl'; write_threads([t],p); self.assertEqual(load_threads(p)[0].to_dict(),t.to_dict())
    def test_conflicts_and_roles(self):
        rows=export_langfuse([Thread.from_dict(sample())]); bad=copy.deepcopy(rows[0]); bad['name']='other'
        with self.assertRaises(ContractError): normalize_langfuse(rows+[bad])
        rows[2]['input']['messages'][1]['role']='assistant'
        with self.assertRaises(ContractError): normalize_langfuse(rows)
    def test_bad_contracts(self):
        for mutation in ('timestamp','id','future','output','decoy'):
            d=sample(); r=d['runs'][0]
            if mutation=='timestamp': r['started_at']='bad'
            if mutation=='id': r['run_id']=''
            if mutation=='future': r['generation']['request_part_ids'].append('foreign')
            if mutation=='output': r['generation']['request_part_ids'].append('o')
            if mutation=='decoy': r['retrievals'][0]['exposed_to_model']=False
            with self.subTest(mutation=mutation), self.assertRaises(ContractError): Thread.from_dict(d)
    def test_missing_session(self):
        rows=export_langfuse([Thread.from_dict(sample())]); rows[0]['sessionId']=None
        with self.assertRaisesRegex(ContractError,'sessionId'): normalize_langfuse(rows)

    def test_explicit_usage_conversion(self):
        t=Thread(sample()); rows=export_langfuse([t])
        self.assertEqual(rows[2]['usageDetails'], {'input':3,'output':4,'cache_read_input_tokens':7,'cache_creation_input_tokens':2})
        rows[2]['usageDetails']['input_tokens']=1
        with self.assertRaises(ContractError): normalize_langfuse(rows)
        d=sample(); d['runs'][0]['generation']['usage']['cache_read']=99
        with self.assertRaises(ContractError): Thread(d)

    def test_full_history_and_ancestry_timestamps(self):
        d=sample(); d['runs'][0]['generation']['request_part_ids']=['i']
        with self.assertRaises(ContractError): Thread(d)
        rows=export_langfuse([Thread(sample())]); rows[1]['endTime']='2025-01-01T00:01:00Z'
        with self.assertRaises(ContractError): normalize_langfuse(rows)
        rows=export_langfuse([Thread(sample())]); rows[1]['metadata']=dict(rows[1]['metadata'],run_index=3)
        with self.assertRaises(ContractError): normalize_langfuse(rows)

    def test_redacted_residual_text_rejected_at_both_boundaries(self):
        d=sample(); d['runs'][0]['retrievals'][0]['content_status']='masked'
        with self.assertRaisesRegex(ContractError, 'null text'): Thread(d)
        rows=export_langfuse([Thread(sample())]); rows[1]['output']['candidates'][0]['content_status']='masked'
        with self.assertRaisesRegex(ContractError, 'null text'): normalize_langfuse(rows)
        d=sample(); d['runs'][0]['input']['content_status']='missing'
        with self.assertRaisesRegex(ContractError, 'null text'): Thread.from_dict(d)

    def test_tied_timestamps_and_export_collision(self):
        import hashlib
        first=sample()['runs'][0]
        first['ended_at']=first['started_at']; first['generation']['started_at']=first['started_at']; first['generation']['ended_at']=first['started_at']
        second=copy.deepcopy(first); second['run_id']='r2'; second['run_index']=2; second['input']['part_id']='i2'; second['output']['part_id']='o2'; second['retrievals'][0]['part_id']='p2'; second['generation']['observation_id']='g2'; second['generation']['request_part_ids']=['i','p','o','i2','p2']
        thread=Thread({'company_id':'c','thread_id':'t','runs':[first,second]}); rows=export_langfuse([thread])
        self.assertEqual(normalize_langfuse(list(reversed(rows)))[0].to_dict(),thread.to_dict())
        for row in rows: row['metadata'].pop('run_index', None)
        with self.assertRaisesRegex(ContractError, 'ambiguous'): normalize_langfuse(rows)
        d=sample(); d['runs'][0]['generation']['observation_id']=hashlib.sha256(b'root:r').hexdigest()[:16]
        with self.assertRaisesRegex(ContractError, 'collision'): export_langfuse([Thread(d)])
