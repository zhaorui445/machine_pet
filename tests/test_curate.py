import unittest

from petlm.curate import screen


class CurationTests(unittest.TestCase):
    def test_rejects_obvious_spam_and_classical_source(self):
        good = '今天我们学习科学，讨论自然现象。' * 40
        self.assertEqual(screen(good,'wikisource')[1], 'classical_source')
        self.assertEqual(screen(good+'皇冠现金网','fineweb2')[1], 'spam_pattern')

    def test_normalizes_traditional_and_accepts_modern_text(self):
        text = ''.join(f'這是第{i}次科學練習，我們分析不同的問題和現象。' for i in range(60))
        result, reason = screen(text,'fineweb2')
        self.assertIsNone(reason)
        self.assertIn('这是第',result)
        self.assertNotIn('這',result)

    def test_rejects_repeated_blocks(self):
        repeated = '这个网站重复发布相同内容，读者无法得到新的知识。' * 120
        self.assertEqual(screen(repeated,'fineweb2')[1], 'repeated_blocks')

    def test_rejects_unexpected_scripts_and_mojibake(self):
        korean = ('这是一段中文正文，用来说明正常内容。' * 30) + ('한글' * 30)
        broken = ('这是一段中文正文，用来说明正常内容。' * 30) + '\ufffd' * 10
        self.assertEqual(screen(korean,'fineweb2')[1], 'unexpected_script_or_symbols')
        self.assertEqual(screen(broken,'fineweb2')[1], 'unexpected_script_or_symbols')


if __name__=='__main__':
    unittest.main()
