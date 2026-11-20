import unittest
from petlm.pilot_gate import assess


class PilotGateTests(unittest.TestCase):
    def protocol(self):
        return {'data_gates':{'minimum_train_tokens_including_eos':8,'minimum_train_source_groups':1,'minimum_train_topic_groups':1,
                'maximum_train_source_token_share':1.0,'maximum_train_document_token_share':1.0,'minimum_tokens_each_train_topic':8,
                'minimum_independent_validation_tokens_including_eos':8,'minimum_independent_validation_rows':1,
                'minimum_independent_validation_documents':1,'minimum_independent_validation_source_groups':1,'minimum_independent_validation_topics':1,
                'minimum_rows_each_validation_source':1,'minimum_tokens_each_validation_source':8,'minimum_rows_each_validation_topic':1,
                'minimum_tokens_each_validation_topic':8,'maximum_validation_source_token_share':1.0,'maximum_validation_blocks_per_document':3,'allowed_topics':['daily']}}
    def rows(self):
        return [{'split':s,'human_approved':True,'coverage_source_group':s,'coverage_topic':'daily','document_group':s,
                 'tokens_with_eos':8,'primary_independent_validation':s=='validation'} for s in ['train','validation']]
    def test_pending_rows_do_not_count(self):
        rows=self.rows();rows[0]['human_approved']=False
        result=assess(rows,self.protocol(),True)
        self.assertFalse(result['training_ready']);self.assertEqual(result['train']['tokens_with_eos'],0)
    def test_legacy_validation_is_not_independent(self):
        rows=self.rows();rows[1]['primary_independent_validation']=False
        result=assess(rows,self.protocol(),True)
        self.assertEqual(result['primary_independent_validation']['rows'],0)
        self.assertEqual(result['legacy_diagnostic_validation_rows'],1)
    def test_cross_source_leakage_blocks(self):
        rows=self.rows();rows[1]['coverage_source_group']='train'
        self.assertFalse(assess(rows,self.protocol(),True)['checks']['independent_validation_authors_not_in_training'])
    def test_unfrozen_data_blocks_even_with_sufficient_counts(self):
        self.assertFalse(assess(self.rows(),self.protocol())['training_ready'])
        self.assertTrue(assess(self.rows(),self.protocol(),True)['training_ready'])


if __name__=='__main__':unittest.main()
