import unittest

from petlm.generation_repair_cycle_r2 import normalized, raw_flags


class TinyTokenizer:
    def encode(self,text):
        return list(range(len(text)))


class GenerationRepairCycleR2Tests(unittest.TestCase):
    def test_normalized_text(self):
        self.assertEqual(normalized('计划，更新。'),normalized('计划 更新'))

    def test_raw_repetition_flags(self):
        self.assertIn('repeated_character',raw_flags('好好好好'))
        self.assertIn('repeated_phrase',raw_flags('继续学习，继续学习，继续学习。'))

    def test_clean_raw_text(self):
        self.assertEqual(raw_flags('可以先检查窗框的密封条。'),[])


if __name__=='__main__':
    unittest.main()
