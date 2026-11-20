import unittest

from petlm.dialogue_oasst2 import label_value, safe, split_for_tree


class Oasst2DialogueTests(unittest.TestCase):
    def test_label_value_supports_sequence_schema(self):
        row={'labels':[{'name':'quality','value':0.75,'count':4}]}
        self.assertEqual(label_value(row,'quality'),0.75)
        self.assertIsNone(label_value(row,'helpfulness'))

    def test_explicit_detoxify_risk_is_rejected(self):
        self.assertFalse(safe({'detoxify':{'toxicity':0.9}}))
        self.assertTrue(safe({'detoxify':{'toxicity':0.1}}))

    def test_tree_split_is_deterministic(self):
        self.assertEqual(split_for_tree('tree-a'),split_for_tree('tree-a'))


if __name__=='__main__':
    unittest.main()
