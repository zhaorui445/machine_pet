import unittest

from petlm.dialogue_quality import keep


class DialogueQualityTests(unittest.TestCase):
    def test_only_high_score_oasst2_is_kept(self):
        accepted={'training_group':'natural_human_oasst2','source_quality':0.75,
                  'source_helpfulness':0.8,'source_rank':0}
        self.assertTrue(keep(accepted)[0])
        self.assertFalse(keep({**accepted,'source_quality':0.74})[0])
        self.assertFalse(keep({**accepted,'source_helpfulness':None})[0])

    def test_reviewed_manual_and_synthetic_groups_are_kept(self):
        for group in ('legacy','diverse_synthetic','natural_human_rule_assisted_item_audit'):
            self.assertTrue(keep({'training_group':group})[0])
        self.assertFalse(keep({'training_group':'natural_human_openassistant'})[0])


if __name__=='__main__':
    unittest.main()
