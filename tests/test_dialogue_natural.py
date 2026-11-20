import unittest

from petlm.dialogue_natural import (chain_to, high_quality_assistant,
                                    history_eligible, partition_for_tree,
                                    safe_history_labels, safe_labels)


def row(**overrides):
    base={'role':'assistant','lang':'zh','text':'这是一个有帮助的回答。','deleted':False,
          'review_result':True,'synthetic':False,'model_name':None,
          'labels':{'quality':{'value':0.75},'helpfulness':{'value':0.75},
                    'spam':{'value':0.0},'lang_mismatch':{'value':0.0},
                    'pii':{'value':0.0},'not_appropriate':{'value':0.0},
                    'hate_speech':{'value':0.0},'sexual_content':{'value':0.0},
                    'toxicity':{'value':0.0}}}
    base.update(overrides)
    return base


class NaturalDialogueTests(unittest.TestCase):
    def test_quality_and_risk_thresholds(self):
        self.assertTrue(high_quality_assistant(row()))
        bad=row()
        bad['labels']['quality']['value']=0.25
        self.assertFalse(high_quality_assistant(bad))
        risky=row()
        risky['labels']['pii']['value']=0.5
        self.assertFalse(safe_labels(risky))

    def test_missing_legacy_risk_labels_are_allowed_only_for_history(self):
        legacy=row(labels={'quality':{'value':0.75},'helpfulness':{'value':0.75}})
        self.assertTrue(safe_history_labels(legacy))
        self.assertFalse(safe_labels(legacy))

    def test_history_can_lack_review_result_but_final_answer_cannot(self):
        history=row(review_result=None)
        self.assertTrue(history_eligible(history))
        self.assertFalse(high_quality_assistant(history))

    def test_chain_requires_existing_alternating_ancestors(self):
        messages={
            'u1':{'role':'prompter','parent_id':None},
            'a1':{'role':'assistant','parent_id':'u1'},
            'u2':{'role':'prompter','parent_id':'a1'},
        }
        self.assertEqual([item['role'] for item in chain_to(messages,messages['u2'])],
                         ['prompter','assistant','prompter'])
        messages['u2']['parent_id']='missing'
        self.assertIsNone(chain_to(messages,messages['u2']))

    def test_tree_partition_is_deterministic(self):
        self.assertEqual(partition_for_tree('same-tree'),partition_for_tree('same-tree'))
        self.assertIn(partition_for_tree('same-tree'),('train','validation','blind'))


if __name__=='__main__':
    unittest.main()
