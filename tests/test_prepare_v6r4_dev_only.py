from pathlib import Path
import tempfile
import unittest

from petlm.prepare_v6r4_dev_only import build


class PrepareV6R4DevelopmentOnlyTests(unittest.TestCase):
    def test_real_reviewed_data_uses_development_only(self):
        root = Path(__file__).parents[1]
        reviewed = root / 'data' / 'petlm' / 'dialogue' / 'human-review-v6r4-final'
        tokenizer = root / 'data' / 'petlm' / 'tokens-chinese-v17-stage7-24k' / 'tokenizer.model'
        if not (reviewed / 'manifest.json').is_file() or not tokenizer.is_file():
            self.skipTest('Requires the local reviewed corpus and tokenizer')
        with tempfile.TemporaryDirectory() as directory:
            report = build(reviewed, tokenizer, Path(directory) / 'prepared')
            self.assertIn('Internal-test files were not parsed', report['isolation'])
            for capability in report['capabilities'].values():
                self.assertEqual(capability['counts'], {'train': 64, 'validation': 16})
                self.assertFalse(capability['internal_test_used'])


if __name__ == '__main__':
    unittest.main()
