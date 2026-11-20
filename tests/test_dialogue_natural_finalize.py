import unittest

from petlm.dialogue_natural_finalize import keep_base_row, require_reviewed


class NaturalDialogueFinalizeTests(unittest.TestCase):
    def test_keeps_synthetic_manual_and_all_test_rows(self):
        self.assertTrue(keep_base_row({'training_group':'diverse_synthetic'},'train'))
        self.assertTrue(keep_base_row({'review_status':'approved','tree_id':'manual:1'},'train'))
        self.assertFalse(keep_base_row({'training_group':'legacy','tree_id':'oasst'},'train'))
        self.assertTrue(keep_base_row({'training_group':'legacy'},'test'))

    def test_requires_complete_review_metadata(self):
        require_reviewed([{'id':'a','review_status':'approved','reviewer':'u',
                           'reviewed_at':'2026-11-03'}],'rows')
        with self.assertRaisesRegex(RuntimeError,'pending'):
            require_reviewed([{'id':'b','review_status':'pending'}],'rows')


if __name__=='__main__':
    unittest.main()
