import unittest

from petlm.foundation_quality_gate import mechanical_flags, normalized


class FoundationQualityGateTests(unittest.TestCase):
    def test_normalization(self):
        self.assertEqual(normalized('自然，中文。'),normalized('自然 中文'))

    def test_clean_continuation(self):
        self.assertEqual(mechanical_flags('可以先检查窗框，再修补松动的密封条。','eos'),[])

    def test_short_and_incomplete_continuation(self):
        flags=mechanical_flags('好的','eos')
        self.assertIn('too_few_chinese_characters',flags)
        self.assertIn('incomplete_sentence_ending',flags)

    def test_repetition_and_truncation(self):
        flags=mechanical_flags('继续整理继续整理继续整理继续整理','maximum_tokens')
        self.assertIn('repeated_phrase',flags)
        self.assertIn('maximum_token_truncation',flags)


if __name__=='__main__':
    unittest.main()
