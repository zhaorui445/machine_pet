import unittest

from petlm.final_sft_acceptance import diagnostic_flags, normalized


class FinalSftAcceptanceTests(unittest.TestCase):
    def test_normalized_ignores_spacing_and_punctuation(self):
        self.assertEqual(normalized('好的，明白。'),normalized('好的 明白'))

    def test_detects_exact_thread_repetition(self):
        flags=diagnostic_flags('先检查桌面。',['先检查桌面。'])
        self.assertIn('exact_answer_repeated_in_conversation',flags)

    def test_detects_known_bad_patterns(self):
        flags=diagnostic_flags('请先检索论文检索论文检索论文','')
        self.assertIn('repeated_phrase',flags)
        self.assertIn('unrelated_research_redirect',flags)

    def test_clean_short_answer_has_no_flags(self):
        self.assertEqual(diagnostic_flags('可以，先检查沙发缝隙。',[]),[])


if __name__=='__main__':
    unittest.main()
