import unittest

from petlm.prepare_capability_expansion import EXPANSION_FAMILIES, make_case


class CapabilityExpansionTests(unittest.TestCase):
    def test_every_family_is_tagged_and_rule_checked(self):
        for capability,families in EXPANSION_FAMILIES.items():
            for index,family in enumerate(families):
                row=make_case(capability,family,index,20261109)
                self.assertEqual(row['adapter_capability'],capability)
                self.assertEqual(row['training_group'],'rule_authored_capability_expansion_v2')
                self.assertEqual(row['review_status'],'rule_checked')
                self.assertTrue(all(fragment in row['answer'] for fragment in row['review_checks']))


if __name__=='__main__':
    unittest.main()
