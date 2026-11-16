import unittest
from compache.dependencies import build_packet, load_label_schema, validate_label
from test_traces import sample

class DependencyTests(unittest.TestCase):
    def test_masked_and_truth_free(self):
        d=sample(); d['scenario_id']='secret'; d['runs'][0]['retrievals'][0]['content_status']='masked'; d['runs'][0]['retrievals'][0]['text']=None
        packet=build_packet(d,'r'); self.assertEqual(packet['request_coverage'],'partial'); self.assertNotIn('secret',str(packet)); self.assertIsNone(packet['request_parts'][1]['text'])
    def test_schema_and_invalid_label(self):
        schema=load_label_schema(); self.assertEqual(schema['type'],'object')
        packet=build_packet(sample(),'r'); errors=validate_label({'thread_id':3,'extra':True},packet)
        self.assertTrue(any('unexpected' in e for e in errors)); self.assertTrue(any('expected string' in e for e in errors))

    def test_quotes_necessity_and_coverage(self):
        packet=build_packet(sample(),'r')
        req={'requirement_id':'a','description':'evidence','source_kind':'retrieval','necessity':'inferred','alternatives':[{'sources':[{'part_id':'p','quote':'Evidence'}]}],'evidence_note':'note'}
        label={'schema_version':'compache-dependency-label-v1','thread_id':'t','target_run_id':'r','status':'complete','coverage_notes':[],'task_requirements':[], 'claims':[{'claim_id':'c','target_quote':'Answer','relation':'supported','requirements':[req]}], 'historical_dependencies':[], 'unresolved':[]}
        self.assertEqual(validate_label(label,packet),[])
        req['alternatives'][0]['sources'][0]['quote']='invented'
        req['necessity']='tested'; packet['request_coverage']='partial'
        errors=validate_label(label,packet)
        self.assertTrue(any('verbatim' in e for e in errors))
        self.assertTrue(any('tested' in e for e in errors))
        self.assertTrue(any('complete coverage' in e for e in errors))
        req['alternatives'][0]['sources'][0]['part_id']='foreign'
        self.assertTrue(any('absent, future' in e for e in validate_label(label,packet)))
